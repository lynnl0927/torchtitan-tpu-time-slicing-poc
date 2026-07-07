
import functools
import os
import time
import typing
import json
import threading
import queue
import requests

from jax import profiler as jax_profiler
import torch
from torch.distributed import fsdp
import torch_tpu

import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import gmain
from torchtitan.experiments.tpu import utils as tpu_utils

import torchtitan.experiments.tpu.llama3
import torchtitan.experiments.tpu.qwen3
from torchtitan.experiments.tpu.rl import grpo_job_config
import torchtitan.experiments.tpu.rl.grpo_sampler as grpo_sampler
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
    def __init__(self, rank: int, world_size: int, port: int = 18001):
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
        print("\n[FASTAPI THREAD]: Starting Sampler HTTP server on port 8001...", flush=True)
        import uvicorn
        from fastapi import FastAPI, Request
        
        app = FastAPI(title="Sampler Server")
        
        @app.post("/generate")
        async def generate_endpoint(req: Request):
            data = await req.json()
            request_queue.put(("generate", data))
            return response_queue.get()
            
        @app.post("/update_weights")
        async def update_weights(req: Request):
            data = await req.json()
            # Download weights on API thread before telling SPMD workers to load
            trainer_ip = data.get("trainer_ip", "localhost")
            r = requests.get(f"http://{trainer_ip}:8000/weights")
            with open("/tmp/trainer_weights.pt", "wb") as f:
                f.write(r.content)
            
            request_queue.put(("update_weights", {}))
            return response_queue.get()
            
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        print("[FASTAPI THREAD]: Calling server.run() on port 8001...", flush=True)
        server.run()
        print("[FASTAPI THREAD]: server.run() has EXITED on port 8001!", flush=True)
    except Exception as e:
        print(f"\n[FATAL ERROR IN SAMPLER FASTAPI THREAD]: {e}\n", flush=True)
        import traceback
        traceback.print_exc()

def start_sampler(job_config: grpo_job_config.GRPOJobConfig) -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        torchtitan.tools.logging.init_logger()
        threading.Thread(target=fastapi_server, daemon=True).start()

    device = tpu_utils.get_device()

    if world_size > 1:
        dist_utils.init_distributed(job_config.comm, enable_cpu_backend=job_config.training.enable_cpu_offload, base_folder=job_config.dump_folder)
        torch.distributed.barrier()

    dp_shard = job_config.parallelism.data_parallel_shard_degree
    if dp_shard == -1:
        dp_shard = world_size
    parallel_dims = ParallelDims(dp_shard=dp_shard, dp_replicate=world_size // dp_shard, cp=1, tp=1, pp=1, ep=1, world_size=world_size)

    train_spec = train_spec_module.get_train_spec(job_config.model.name)
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(trainer_config=job_config)

    with torch.device("meta"), utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]):
        sampler_model = typing.cast(torch.nn.Module, train_spec.model_cls(model_args))

    if world_size > 1:
        try:
            train_spec.parallelize_fn(
                sampler_model,
                parallel_dims=parallel_dims,
                training=job_config.training,
                parallelism=job_config.parallelism,
                compile_config=job_config.compile,
                ac_config=job_config.activation_checkpoint,
                dump_folder=job_config.dump_folder,
            )
        except TypeError:
            train_spec.parallelize_fn(sampler_model, parallel_dims, job_config)

    sampler_model = sampler_model.to_empty(device=device)

    with torch.no_grad():
        sampler_model.init_weights()

    if job_config.checkpoint.initial_load_path:
        job_config.checkpoint.initial_load_path = os.path.expanduser(job_config.checkpoint.initial_load_path)

    if job_config.checkpoint.initial_load_path or (job_config.checkpoint.enable and os.path.exists(os.path.join(job_config.dump_folder, job_config.checkpoint.folder))):
        logger.info("Loading initial checkpoint into sampler model using checkpointer...")
        checkpointer = job_config.checkpoint.build(
            dataloader=None,
            model_parts=[sampler_model],
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

    for p in sampler_model.parameters():
        p.requires_grad = False

    logger.info("Sampler model initialized. Entering CPU-idle SPMD control loop.")
    cpu_broadcast = CPUCommandBroadcast(rank, world_size, port=18001)
    while True:
        cmd_list = [None, None]
        if rank == 0:
            cmd, data = request_queue.get()
            cmd_list = [cmd, data]
        
        cmd_list = cpu_broadcast.broadcast(cmd_list)
        cmd, data = cmd_list
        
        if cmd == "generate":
            if rank == 0:
                logger.info("[SPMD LOOP RANK 0]: Received 'generate' command! Executing autoregressive sampling (Step 0 XLA compilation takes ~60s)...")
            t0 = time.time()
            prompt_ids = torch.tensor(data["prompt_ids"], device=device)
            sampler_model.eval()
            with torch.no_grad():
                with fsdp.FullyShardedDataParallel.summon_full_params(sampler_model, recurse=True, writeback=False):
                    completed_ids, _ = grpo_sampler.generate(
                        sampler_model, prompt_ids, max_seq_len=job_config.training.seq_len,
                        max_new_tokens=job_config.sampler.max_new_tokens,
                        temperature=job_config.sampler.temperature, top_k=None,
                    )
            torch_tpu._internal.sync.synchronize(completed_ids, wait=True)
            if rank == 0:
                logger.info(f"[SPMD LOOP RANK 0]: 'generate' completed in {time.time() - t0:.2f}s! Returning completions to orchestrator.")
                response_queue.put({"status": "ok", "completed_ids": completed_ids.cpu().tolist()})

        elif cmd == "update_weights":
            if rank == 0:
                logger.info("[SPMD LOOP RANK 0]: Received 'update_weights' command! Reloading state dict...")
            t0 = time.time()
            state_dict = torch.load("/tmp/trainer_weights.pt", map_location="cpu")
            # Convert plain CPU tensors to DTensors matching the sampler model's mesh
            from torch.distributed.tensor import DTensor, distribute_tensor
            model_sd = sampler_model.state_dict()
            dtensor_sd = {}
            for k, v in state_dict.items():
                if k in model_sd and isinstance(model_sd[k], DTensor):
                    mesh = model_sd[k].device_mesh
                    placements = model_sd[k].placements
                    dtensor_sd[k] = distribute_tensor(v.to(device), device_mesh=mesh, placements=placements)
                else:
                    dtensor_sd[k] = v.to(device) if hasattr(v, 'to') else v
            sampler_model.load_state_dict(dtensor_sd)
            if rank == 0:
                logger.info(f"[SPMD LOOP RANK 0]: 'update_weights' completed in {time.time() - t0:.2f}s!")
                response_queue.put({"status": "ok"})

if __name__ == "__main__":
    gmain.handle_main(start_sampler)
