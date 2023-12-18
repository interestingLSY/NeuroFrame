"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task2自动微分部分的函数
"""

import numpy as np
import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf
from test_task1_backward import gradient_check

def test_compute_gradient():
    gradient_check(
        lambda A, B, C: nf.ops.tensor_reduction_sum((A @ B + C) * (A @ B)),
        nf.Tensor(np.random.randn(10, 9)),
        nf.Tensor(np.random.randn(9, 8)),
        nf.Tensor(np.random.randn(10, 8))
    )
    gradient_check(
        lambda A, B: nf.ops.tensor_reduction_sum(nf.ops.broadcast_to(A, (10, 9)) * B),
        nf.Tensor(np.random.randn(10, 1)),
        nf.Tensor(np.random.randn(10, 9))
    )
    gradient_check(
        lambda A, B, C: nf.ops.tensor_reduction_sum(
            nf.ops.tensor_divs(nf.ops.reshape(A, (10, 10)) @ B, 5) + C
        ),
        nf.Tensor(np.random.randn(100)),
        nf.Tensor(np.random.randn(10, 5)),
        nf.Tensor(np.random.randn(10, 5))
    )

    nf.cgraph.clear_graph()
    x2 = nf.Tensor([6], nf.float32)
    x3 = nf.Tensor([0], nf.float32)
    y = x2 * x2 + x2 * x3
    nf.cgraph.perform_backward(y, nf.Tensor.fill(nf.Scalar(1.0), y.shape, y.dtype))
    grad_x2 = nf.cgraph.get_computed_grad(x2)
    grad_x3 = nf.cgraph.get_computed_grad(x3)
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    
    # nf.cgraph.perform_backward(grad_x2, nf.Tensor.fill(nf.Scalar(1.0), y.shape, y.dtype))
    # grad_x2_x2 = nf.cgraph.get_computed_grad(grad_x2)
    # grad_x2_x3 = nf.cgraph.get_computed_grad(grad_x3)
    # assert grad_x2_x2.numpy() == 2
    # assert grad_x2_x3.numpy() == 1

if __name__ == "__main__":
    np.random.seed(0)
    nf.set_random_seed(0)
    devices = nf.Device.get_available_devices()
    print("Found devices:", devices)
    for device in devices:
        print(f"---------------------------------------")
        print(f"Testing device {repr(device)} ({device.get_hardware_name()})")
        nf.Device.set_default_device(device)
        
        test_compute_gradient()
        
        print(f"Pass")
