import numpy as np
import time
import tqdm
import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

n = 16384

print(f"n = {n}")
print(f"Size of a single matrix: {n * n * 2 / 1024 / 1024 / 1024} GB")

with nf.inference_mode():
    a = nf.Tensor.empty((n, n), nf.float16, nf.Device.cuda(0))
    b = nf.Tensor.empty((n, n), nf.float16, nf.Device.cuda(0))
    start_time = time.time()
    for i in tqdm.tqdm(range(10)):
        a = a @ b
    end_time = time.time()
    
    flop = 2 * n * n * n * 10
    flops = flop / (end_time - start_time)
    print(f"TFLOPS: {flops / 1e12}")
