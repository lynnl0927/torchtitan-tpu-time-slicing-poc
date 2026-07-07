
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

import socket
class CPUCommandBroadcast:
    """Broadcasts SPMD commands over CPU TCP sockets on localhost to keep TPU 100% idle while waiting!"""
    def __init__(self, rank: int, world_size: int, port: int = 18000):
        self.rank = rank
        self.world_size = world_size
        self.port = port
        if world_size <= 1:
            return
            
        if rank == 0:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(("127.0.0.1", port))
            self.server.listen(world_size - 1)
            self.clients = []
            for _ in range(world_size - 1):
                conn, _ = self.server.accept()
                self.clients.append(conn)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for _ in range(600):
                try:
                    self.sock.connect(("127.0.0.1", port))
                    break
                except ConnectionRefusedError:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"Rank {rank} failed to connect to CPU broadcast server on port {port}")
                
    def broadcast(self, data: typing.Any) -> typing.Any:
        if self.world_size <= 1:
            return data
        if self.rank == 0:
            payload = (json.dumps(data) + "\n").encode("utf-8")
            for conn in self.clients:
                conn.sendall(payload)
            return data
        else:
            buffer = ""
            while "\n" not in buffer:
                chunk = self.sock.recv(4096).decode("utf-8")
                if not chunk:
                    raise ConnectionResetError("CPU broadcast server closed connection")
                buffer += chunk
            line, _, _ = buffer.partition("\n")
            return json.loads(line)

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

    if job_config.checkpoint.initial_load_path:
        job_config.checkpoint.initial_load_path = os.path.expanduser(job_config.checkpoint.initial_load_path)

    if job_config.checkpoint.initial_load_path or (job_config.checkpoint.enable and os.path.exists(os.path.join(job_config.dump_folder, job_config.checkpoint.folder))):
        logger.info("Loading initial checkpoint into trainer model using checkpointer...")
        checkpointer = job_config.checkpoint.build(
            dataloader=None,
            model_parts=[model],
            optimizers=None,
            lr_schedulers=None,
            states={},
            sd_adapter=(
                train_spec.state_dict_adapter(model_args, job_config.checkpoint.initial_load_path)
                if hasattr(train_spec, "state_dict_adapter") and train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.dump_folder,
        )
        checkpointer.enable = True
        checkpointer.load(step=job_config.checkpoint.load_step)

    logger.info("Syncing weights from model to ref_model...")
    grpo_utils.sync_model_weights(model, ref_model, parallel_dims)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=job_config.optimizer.lr, eps=job_config.optimizer.eps, foreach=True)

    # SPMD Execution Loop
    logger.info("Trainer models initialized. Entering CPU-idle SPMD control loop.")
    cpu_broadcast = CPUCommandBroadcast(rank, world_size, port=18000)
    while True:
        cmd_list = [None, None]
        if rank == 0:
            cmd, data = request_queue.get()
            cmd_list = [cmd, data]
        
        cmd_list = cpu_broadcast.broadcast(cmd_list)
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
            
            # Action mask to ignore Qwen3 pad tokens (151643) and chat eos/pad tokens (151645)
            action_mask = ((gen_targets != 151643) & (gen_targets != 151645)).float()

            model.eval()
            with torch.no_grad():
                outputs = model(completed_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                gen_logits = logits[:, prompt_len - 1 : -1, :]
                token_log_probs = F.log_softmax(gen_logits, dim=-1).gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
                torch_tpu._internal.sync.synchronize(token_log_probs, wait=True)
            model.train()

            ref_model.eval()
            with torch.no_grad():
                outputs = ref_model(completed_ids)
                ref_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                gen_ref_logits = ref_logits[:, prompt_len - 1 : -1, :]
                ref_token_log_probs = F.log_softmax(gen_ref_logits, dim=-1).gather(2, gen_targets.unsqueeze(-1)).squeeze(-1)
                torch_tpu._internal.sync.synchronize(ref_token_log_probs, wait=True)

            total_loss = 0.0
            total_kl = 0.0
            for epoch in range(job_config.grpo.ppo_epochs):
                optimizer.zero_grad()
                loss, kl = grpo_utils.compute_grpo_loss(
                    model, prompt_ids, completed_ids, ref_log_probs=ref_token_log_probs,
                    advantages=advantages, old_log_probs=token_log_probs,
                    action_mask=action_mask,
                    ppo_clip_eps=job_config.grpo.ppo_clip_eps,
                    grpo_beta=job_config.grpo.grpo_beta,
                    return_kl=True,
                )
                loss.backward()
                optimizer.step()
                torch_tpu._internal.sync.synchronize(loss, wait=True)
                torch_tpu._internal.sync.synchronize(kl, wait=True)
                total_loss += loss.detach().cpu().item()
                total_kl += kl.detach().cpu().item()
            
            if rank == 0:
                avg_loss = total_loss / max(job_config.grpo.ppo_epochs, 1)
                avg_kl = total_kl / max(job_config.grpo.ppo_epochs, 1)
                logger.info(f"[SPMD LOOP RANK 0]: 'train' completed in {time.time() - t0:.2f}s! Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}")
                response_queue.put({"status": "ok", "loss": avg_loss, "kl": avg_kl})

        elif cmd == "export":
            if rank == 0:
                logger.info("[SPMD LOOP RANK 0]: Received 'export_weights' command! Saving checkpoint...")
            t0 = time.time()
            # All ranks must participate in full_tensor() collective (all-gather)
            sd = model.state_dict()
            cpu_sd = {}
            for k, v in sd.items():
                if hasattr(v, 'full_tensor'):
                    cpu_sd[k] = v.full_tensor().cpu()
                elif hasattr(v, 'cpu'):
                    cpu_sd[k] = v.cpu()
                else:
                    cpu_sd[k] = v
            if rank == 0:
                torch.save(cpu_sd, "/tmp/trainer_weights.pt")
                logger.info(f"[SPMD LOOP RANK 0]: 'export_weights' completed in {time.time() - t0:.2f}s!")
                response_queue.put({"status": "ok"})

if __name__ == "__main__":
    gmain.handle_main(start_trainer)
