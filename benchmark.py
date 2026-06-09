import subprocess
import time
import torch


def run_single():
    start = time.time()
    subprocess.run(['python', 'train_single.py'])
    return time.time() - start


def run_ddp():
    start = time.time()
    subprocess.run(['python', 'train_ddp.py'])
    return time.time() - start


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"GPUs available: {world_size}")

    print("\nrunning single GPU training...")
    single_time = run_single()
    print(f"single GPU time: {single_time:.2f}s")

    if world_size > 1:
        print("\nrunning DDP training...")
        ddp_time = run_ddp()
        print(f"DDP time: {ddp_time:.2f}s")
        print(f"speedup: {single_time/ddp_time:.2f}x")
    else:
        print("\nonly 1 GPU available — can't benchmark DDP speedup")
        print("run on a multi-GPU machine to see speedup")
        print("running DDP with 1 GPU anyway to verify correctness...")
        run_ddp()
