
import requests
import json
import time
import os

TRAINER_IP = os.environ.get("TRAINER_IP", "127.0.0.1")
SAMPLER_IP = os.environ.get("SAMPLER_IP", "127.0.0.1")

def run_rl_loop():
    print(f"Connecting to Trainer ({TRAINER_IP}:8000) and Sampler ({SAMPLER_IP}:8001)")
    
    # Dummy prompt (pad 0s, length 128)
    prompt_ids = [[0]*124 + [1, 2, 3, 4]]
    group_size = 4
    prompt_ids_repeated = prompt_ids * group_size
    
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        # 1. Generate
        t0 = time.time()
        res = requests.post(f"http://{SAMPLER_IP}:8001/generate", json={"prompt_ids": prompt_ids_repeated})
        if res.status_code != 200:
            print("Error generating:", res.text)
            continue
        completed_ids = res.json()["completed_ids"]
        print(f"Generated {len(completed_ids)} completions in {time.time() - t0:.2f}s")
        
        # 2. Simulated Non-Zero Advantages (net positive reward to drive policy gradient updates)
        advantages = [1.0, 0.5, -0.2, -0.3]
        
        # 3. Train
        t0 = time.time()
        res = requests.post(f"http://{TRAINER_IP}:8000/train", json={
            "prompt_ids": prompt_ids_repeated,
            "completed_ids": completed_ids,
            "advantages": advantages
        })
        if res.status_code != 200:
            print("Error training:", res.text)
            continue
        loss = res.json()["loss"]
        print(f"Trained step with loss {loss:.4f} in {time.time() - t0:.2f}s")
        
        # 4. Sync
        t0 = time.time()
        requests.post(f"http://{TRAINER_IP}:8000/export_weights")
        requests.post(f"http://{SAMPLER_IP}:8001/update_weights", json={"trainer_ip": TRAINER_IP})
        print(f"Synced weights in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_rl_loop()
