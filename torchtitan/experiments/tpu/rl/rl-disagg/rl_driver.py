"""rl_driver.py — Disaggregated GRPO Orchestrator with Real GSM8K Dataset & Rule-Based Rewards

This orchestrator coordinates the synchronous disaggregated RL loop:
  1. Downloads and samples from the real GSM8K math reasoning dataset.
  2. Uses TorchTitan's HuggingFaceTokenizer to encode few-shot XML reasoning prompts.
  3. Sends prompt_ids to the Sampler VM (server_sampler.py) for autoregressive generation.
  4. Decodes generated token IDs to text and grades them using rule-based GSM8K rewards (reward.py).
  5. Computes GRPO advantages across prompt groups.
  6. Sends prompt_ids, completed_ids, and real advantages to the Trainer VM (server_trainer.py).
  7. Triggers synchronous FSDP checkpoint export and weight reloading.
"""

import json
import os
import random
import sys
import time
import urllib.request
import requests
import uuid

# Import TorchTitan tokenizer and local GSM8K reward functions
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from reward import compute_rewards, compute_advantages, extract_answer

TRAINER_IP = os.environ.get("TRAINER_IP", "127.0.0.1")
TRAINER_PORT = int(os.environ.get("TRAINER_PORT", "8000"))
SAMPLER_IP = os.environ.get("SAMPLER_IP", "127.0.0.1")
SAMPLER_PORT = int(os.environ.get("SAMPLER_PORT", "8001"))
ORCHESTRATOR_IP = os.environ.get("ORCHESTRATOR_IP", "127.0.0.1")
JOB_ID = os.environ.get("JOB_ID", str(uuid.uuid4()))
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
DATASET_PATH = os.environ.get("DATASET_PATH", "/tmp/gsm8k_prompts.json")


def download_dataset():
    """Download GSM8K dataset from OpenAI GitHub repository if not present locally."""
    if os.path.exists(DATASET_PATH):
        return
    print(f"Downloading GSM8K dataset from {GSM8K_URL}...")
    data = urllib.request.urlopen(GSM8K_URL, timeout=60).read().decode("utf-8")
    records = [json.loads(line) for line in data.strip().split("\n") if line.strip()]
    with open(DATASET_PATH, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} GSM8K prompts to {DATASET_PATH}")


def load_dataset():
    """Load GSM8K prompt records from disk."""
    download_dataset()
    with open(DATASET_PATH) as f:
        return json.load(f)


def sample_batch(dataset, n, tokenizer, prompt_len=128):
    """Sample n prompts from GSM8K, format with XML reasoning instructions, and tokenize.
    
    Pads or truncates on the left to maintain a static tensor shape (prompt_len) for XLA compilation.
    """
    batch = random.sample(dataset, min(n, len(dataset)))
    prompts_text, prompt_ids_list, answers = [], [], []
    pad_id = tokenizer.eos_id if tokenizer.eos_id is not None else 0
    
    for item in batch:
        question = item.get("question", item.get("prompt", ""))
        answer = item.get("answer", item.get("ground_truth", ""))
        gt = extract_answer(str(answer)) or str(answer)
        
        # Few-shot chat XML reasoning prompt format
        prompt = (
            "<|im_start|>system\n"
            "Respond in the following format:\n\n"
            "<reasoning>\n...\n</reasoning>\n"
            "<answer>\n...\n</answer>"
            "<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<reasoning>\n"
        )
        ids = tokenizer.encode(prompt)
        
        # Left-pad or truncate to static prompt_len to prevent XLA recompilation across steps
        if len(ids) > prompt_len:
            ids = ids[-prompt_len:]
        else:
            ids = [pad_id] * (prompt_len - len(ids)) + ids
            
        prompts_text.append(prompt)
        prompt_ids_list.append(ids)
        answers.append(gt)
        
    return prompts_text, prompt_ids_list, answers


def run_rl_loop():
    print("=" * 70)
    print(f"Disaggregated TPU GRPO RL Orchestrator (GSM8K Real Dataset)")
    print(f"  Trainer VM: http://{TRAINER_IP}:{TRAINER_PORT}")
    print(f"  Sampler VM: http://{SAMPLER_IP}:{SAMPLER_PORT}")
    print("=" * 70)
    
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} GSM8K prompts.")
    
    tokenizer_path = os.environ.get("TOKENIZER_PATH")
    if not tokenizer_path:
        for p in [
            os.path.expanduser("~/qwen3_checkpoint"),
            os.path.expanduser("~/qwen_checkpoint"),
            "tests/assets/tokenizer",
            os.path.expanduser("~/torchtitan/tests/assets/tokenizer"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../tests/assets/tokenizer")),
        ]:
            if os.path.exists(p):
                tokenizer_path = p
                break
        if not tokenizer_path:
            tokenizer_path = os.path.expanduser("~/qwen3_checkpoint")
            
    print(f"Loading TorchTitan tokenizer from '{tokenizer_path}'...")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_path)
    
    group_size = int(os.environ.get("GROUP_SIZE", "4"))
    num_prompts = int(os.environ.get("PROMPTS_PER_STEP", "1"))
    prompt_len = int(os.environ.get("PROMPT_LEN", "128"))
    num_steps = int(os.environ.get("N_RL_STEPS", "1000"))
    target_reward = float(os.environ.get("TARGET_REWARD", "1.90"))
    window_size = int(os.environ.get("CONVERGENCE_WINDOW", "20"))
    
    print(f"Config: {num_steps} max steps | {num_prompts} prompts/step | group_size={group_size} | prompt_len={prompt_len}")
    print(f"Convergence Target: Moving Avg Reward >= {target_reward:.2f} over {window_size} steps")
    print("-" * 70)
    
    print("Registering workloads with Orchestrator...", flush=True)
    requests.post(f"http://{ORCHESTRATOR_IP}:9000/register", json={
        "workload_id": f"{JOB_ID}_sampler",
        "pool": "sampler",
        "pids": [],
        "url": f"http://{SAMPLER_IP}:{SAMPLER_PORT}",
        "checkpointed": False
    }).raise_for_status()
    requests.post(f"http://{ORCHESTRATOR_IP}:9000/register", json={
        "workload_id": f"{JOB_ID}_trainer",
        "pool": "trainer",
        "pids": [],
        "url": f"http://{TRAINER_IP}:{TRAINER_PORT}",
        "checkpointed": False
    }).raise_for_status()
    
    recent_rewards = []
    holding_sampler = False
    
    for step in range(num_steps):
        print(f"\n--- Step {step} ---")
        
        # Sample prompts from GSM8K and encode to static tensor shapes
        _, prompt_ids_list, answers = sample_batch(dataset, num_prompts, tokenizer, prompt_len=prompt_len)
        
        # Repeat each prompt group_size times for GRPO exploration
        prompt_ids_repeated = [ids for ids in prompt_ids_list for _ in range(group_size)]
        expanded_gt = [gt for gt in answers for _ in range(group_size)]
        
        micro_batch_size = int(os.environ.get("MICRO_BATCH_SIZE", "4"))
        rollout_batch_size = int(os.environ.get("ROLLOUT_BATCH_SIZE", micro_batch_size))
        train_batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", micro_batch_size))
        
        # 1. Autoregressive Generation on Sampler VM (chunked into safe micro-batches to prevent TPU HBM OOM!)
        t0 = time.time()
        if not holding_sampler:
            print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Acquiring Sampler lock...", flush=True)
            requests.post(f"http://{ORCHESTRATOR_IP}:9000/acquire", json={"workload_id": f"{JOB_ID}_sampler"}).raise_for_status()
            requests.post(f"http://{SAMPLER_IP}:{SAMPLER_PORT}/start").raise_for_status()
            print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Sampler restored. Starting generation...", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Sampler lock already held. Starting generation...", flush=True)

        completed_ids = []
        for i in range(0, len(prompt_ids_repeated), rollout_batch_size):
            chunk = prompt_ids_repeated[i : i + rollout_batch_size]
            res = requests.post(f"http://{SAMPLER_IP}:{SAMPLER_PORT}/generate", json={"prompt_ids": chunk})
            if res.status_code != 200:
                print(f"Error generating micro-batch {i//rollout_batch_size} from Sampler ({res.status_code}):", res.text)
                break
            completed_ids.extend(res.json()["completed_ids"])
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Yielding Sampler lock...", flush=True)
        requests.post(f"http://{ORCHESTRATOR_IP}:9000/yield", json={"workload_id": f"{JOB_ID}_sampler"}).raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Sampler checkpointed and yielded.", flush=True)
        holding_sampler = False
        
        if len(completed_ids) < len(prompt_ids_repeated):
            print("Skipping step due to generation errors in micro-batching.")
            continue
        gen_time = time.time() - t0
        
        # Decode generated completion tokens back to text strings
        completions_text = [tokenizer.decode(ids[prompt_len:]) for ids in completed_ids]
        
        # 2. Compute Rule-Based GSM8K Rewards & GRPO Advantages
        rewards, reward_stats = compute_rewards(completions_text, expanded_gt)
        advantages, _ = compute_advantages(rewards, group_size)
        mean_reward = sum(rewards) / max(len(rewards), 1)
        
        recent_rewards.append(mean_reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
        moving_avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        print(f"Generated {len(completed_ids)} completions ({len(completed_ids)//rollout_batch_size} micro-batches) in {gen_time:.2f}s | "
              f"Reward: {mean_reward:.3f} (Moving Avg: {moving_avg_reward:.3f}) | Accuracy (Correct Rate): {reward_stats['correct_rate']:.2%} | Format Rate: {reward_stats['format_rate']:.2%}")
        
        # Print sample rollout for visibility
        if completions_text:
            sample_preview = completions_text[0].replace("\n", " ")[:100]
            print(f"  [Sample Rollout 0 (gt={expanded_gt[0]} | r={rewards[0]})]: {sample_preview}...")
        
        # Check convergence criteria
        if len(recent_rewards) == window_size and moving_avg_reward >= target_reward:
            print("\n" + "=" * 70)
            print(f"🎉 CONVERGENCE REACHED! Moving average reward ({moving_avg_reward:.3f}) over the last {window_size} steps reached target ({target_reward:.3f})!")
            print("=" * 70)
            break
        
        train_epochs = int(os.environ.get("TRAIN_EPOCHS", "4"))
        
        # 3. Policy Gradient Backpropagation on Trainer VM (chunked into micro-batches across multiple PPO/GRPO epochs!)
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Acquiring Trainer lock...", flush=True)
        requests.post(f"http://{ORCHESTRATOR_IP}:9000/acquire", json={"workload_id": f"{JOB_ID}_trainer"}).raise_for_status()
        requests.post(f"http://{TRAINER_IP}:{TRAINER_PORT}/start").raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Trainer restored. Starting training...", flush=True)
        total_loss = 0.0
        total_kl = 0.0
        n_chunks = 0
        for epoch in range(train_epochs):
            for i in range(0, len(prompt_ids_repeated), train_batch_size):
                p_chunk = prompt_ids_repeated[i : i + train_batch_size]
                c_chunk = completed_ids[i : i + train_batch_size]
                a_chunk = advantages[i : i + train_batch_size]
                res = requests.post(f"http://{TRAINER_IP}:{TRAINER_PORT}/train", json={
                    "prompt_ids": p_chunk,
                    "completed_ids": c_chunk,
                    "advantages": a_chunk
                })
                if res.status_code != 200:
                    print(f"Error training epoch {epoch} micro-batch {i//micro_batch_size} on Trainer ({res.status_code}):", res.text)
                    break
                res_data = res.json()
                total_loss += res_data["loss"]
                total_kl += res_data.get("kl", 0.0)
                n_chunks += 1
            
        if n_chunks == 0:
            print("Skipping step due to training errors.")
            continue
        loss = total_loss / n_chunks
        kl = total_kl / n_chunks
        print(f"Trained {n_chunks} micro-batches ({train_epochs} epochs) in {time.time() - t0:.2f}s | "
              f"Reward: {mean_reward:.3f} | Accuracy: {reward_stats['correct_rate']:.2%} | KL: {kl:.4f} | Loss: {loss:.4f}")
        
        # 4. Synchronous FSDP Checkpoint Export & Weight Sync
        t0_sync = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Exporting weights from Trainer...", flush=True)
        requests.post(f"http://{TRAINER_IP}:{TRAINER_PORT}/export_weights").raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Yielding Trainer lock...", flush=True)
        requests.post(f"http://{ORCHESTRATOR_IP}:9000/yield", json={"workload_id": f"{JOB_ID}_trainer"}).raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Trainer checkpointed and yielded.", flush=True)
        
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Acquiring Sampler lock for weight update...", flush=True)
        requests.post(f"http://{ORCHESTRATOR_IP}:9000/acquire", json={"workload_id": f"{JOB_ID}_sampler"}).raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Updating Sampler weights...", flush=True)
        requests.post(f"http://{SAMPLER_IP}:{SAMPLER_PORT}/update_weights", json={"trainer_ip": TRAINER_IP, "trainer_port": TRAINER_PORT}).raise_for_status()
        print(f"[{time.strftime('%H:%M:%S')}] [{JOB_ID}] Sampler holding lock for next iteration.", flush=True)
        holding_sampler = True
        print(f"Synced weights across FSDP meshes in {time.time() - t0_sync:.2f}s")

    print("\n" + "=" * 70)
    print("Disaggregated GSM8K GRPO training loop completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_rl_loop()
