
import functools
import os
import time
import typing
import json
import threading
import queue

from jax import profiler as jax_profiler
import torch
from torch.distributed import fsdp
import torch.nn.functional as F
import torch_tpu

import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import utils as tpu_utils

import torchtitan.experiments.tpu.llama3
import torchtitan.experiments.tpu.qwen3
from torchtitan.experiments.tpu.rl import grpo_job_config
import torchtitan.experiments.tpu.rl.grpo_utils as grpo_utils
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.tools import utils
import torchtitan.tools.logging

TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
ParallelDims = torchtitan.distributed.ParallelDims
logger = torchtitan.tools.logging.logger

request_queue = queue.Queue()
response_queue = queue.Queue()

def fastapi_server():
    try:
        print("\n[FASTAPI THREAD]: Starting Trainer HTTP server on port 8000...", flush=True)
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse
        
        app = FastAPI(title="Trainer Server")
        
        @app.post("/train")
        async def train_endpoint(req: Request):
            data = await req.json()
            request_queue.put(("train", data))
            return response_queue.get()
            
        @app.post("/export_weights")
        def export_weights():
            request_queue.put(("export", {}))
            return response_queue.get()
            
        @app.get("/weights")
        def get_weights():
            return FileResponse("/tmp/trainer_weights.pt")
            
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        print("[FASTAPI THREAD]: Calling server.run() on port 8000...", flush=True)
        server.run()
        print("[FASTAPI THREAD]: server.run() has EXITED on port 8000!", flush=True)
    except Exception as e:
        print(f"\n[FATAL ERROR IN TRAINER FASTAPI THREAD]: {e}\n", flush=True)
        import traceback
        traceback.print_exc()

def start_trainer(job_config: grpo_job_config.GRPOJobConfig) -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        torchtitan.tools.logging.init_logger()
        threading.Thread(target=fastapi_server, daemon=True).start()

    device = tpu_utils.get_device()

    if world_size > 1:
        dist_utils.init_distributed(
            job_config.comm, enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.dump_folder,
        )
        torch.distributed.barrier()

    dp_shard = job_config.parallelism.data_parallel_shard_degree
    if dp_shard == -1:
        dp_shard = world_size
    parallel_dims = ParallelDims(
        dp_shard=dp_shard, dp_replicate=world_size // dp_shard,
        cp=1, tp=1, pp=1, ep=1, world_size=world_size,
    )

    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(trainer_config=job_config)

    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))
        ref_model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))

    if world_size > 1:
        for m in [model, ref_model]:
            try:
                train_spec.parallelize_fn(
                    m,
                    parallel_dims=parallel_dims,
                    training=job_config.training,
                    parallelism=job_config.parallelism,
                    compile_config=job_config.compile,
                    ac_config=job_config.activation_checkpoint,
                    dump_folder=job_config.dump_folder,
                )
            except TypeError:
                train_spec.parallelize_fn(m, parallel_dims, job_config)

    model = model.to_empty(device=device)
    ref_model = ref_model.to_empty(device=device)

    with torch.no_grad():
        model.init_weights()
        ref_model.init_weights()

    try:
        with fsdp.FullyShardedDataParallel.summon_full_params(ref_model, recurse=True, writeback=True):
            ref_model.load_state_dict(model.state_dict())
    except Exception:
        ref_model.load_state_dict(model.state_dict())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=job_config.optimizer.lr, eps=job_config.optimizer.eps, foreach=True)

    # SPMD Execution Loop
    logger.info("Trainer models initialized. Entering SPMD control loop.")
    while True:
        cmd_list = [None, None]
        if rank == 0:
            cmd, data = request_queue.get()
            cmd_list = [cmd, data]
        
        if world_size > 1:
            torch.distributed.broadcast_object_list(cmd_list, src=0)
            
        cmd, data = cmd_list
        
        if cmd == "train":
            if rank == 0:
                logger.info("[SPMD LOOP RANK 0]: Received 'train' command! Executing forward/backward/opt (Step 0 XLA compilation takes ~60s)...")
            t0 = time.time()
            prompt_ids = torch.tensor(data["prompt_ids"], device=device)
            completed_ids = torch.tensor(data["completed_ids"], device=device)
            advantages = torch.tensor(data["advantages"], device=device)
            
            prompt_len = prompt_ids.shape[1]
            gen_targets = completed_ids[:, prompt_len:]

            model.eval()
            with torch.no_grad():
                outputs = model(completed_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                gen_logits = logits[:, prompt_len - 1 : -1, :]
                token_log_probs = F.log_softmax(gen_logits, dim=-1).gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
            model.train()

            ref_model.eval()
            with torch.no_grad():
                outputs = ref_model(completed_ids)
                ref_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                gen_ref_logits = ref_logits[:, prompt_len - 1 : -1, :]
                ref_token_log_probs = F.log_softmax(gen_ref_logits, dim=-1).gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
                torch_tpu._internal.sync.synchronize(ref_token_log_probs, wait=True)

            for epoch in range(job_config.grpo.ppo_epochs):
                optimizer.zero_grad()
                loss = grpo_utils.compute_grpo_loss(
                    model, prompt_ids, completed_ids, ref_log_probs=ref_token_log_probs,
                    advantages=advantages, ppo_clip_eps=job_config.grpo.ppo_clip_eps,
                    grpo_beta=job_config.grpo.grpo_beta,
                )
                loss.backward()
                optimizer.step()
                torch_tpu._internal.sync.synchronize(loss, wait=True)
            
            if rank == 0:
                logger.info(f"[SPMD LOOP RANK 0]: 'train' completed in {time.time() - t0:.2f}s! Loss: {loss.cpu().item():.4f}")
                response_queue.put({"status": "ok", "loss": loss.cpu().item()})

        elif cmd == "export":
            if rank == 0:
                logger.info("[SPMD LOOP RANK 0]: Received 'export_weights' command! Saving checkpoint...")
            t0 = time.time()
            with fsdp.FullyShardedDataParallel.summon_full_params(model, recurse=True, writeback=False):
                if rank == 0:
                    torch.save(model.state_dict(), "/tmp/trainer_weights.pt")
            if rank == 0:
                logger.info(f"[SPMD LOOP RANK 0]: 'export_weights' completed in {time.time() - t0:.2f}s!")
                response_queue.put({"status": "ok"})

if __name__ == "__main__":
    gmain.handle_main(start_trainer)
