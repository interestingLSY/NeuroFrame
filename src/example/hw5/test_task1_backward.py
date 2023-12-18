"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task1梯度计算部分的函数
"""

import os, sys
import numpy as np
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

def gradient_check(f, *tensors_, tol=3e-2):
    eps = 1/2**14   # This number can be exactly represented in float32
    tensors = list(tensors_)
    
    with nf.inference_mode():
        numerical_grads = [np.zeros(a.shape) for a in tensors]
        for i in range(len(tensors)):
            for j in range(tensors[i].numel()):
                old_tensor = nf.ops.copy(tensors[i])
                raw_data = tensors[i].numpy()
                
                raw_data.flat[j] += eps
                tensors[i] = nf.Tensor(raw_data)
                f1 = float(f(*tensors).numpy().sum())
                
                raw_data.flat[j] -= 2*eps
                tensors[i] = nf.Tensor(raw_data)
                f2 = float(f(*tensors).numpy().sum())
                
                tensors[i] = old_tensor
                numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    
    nf.cgraph.clear_graph()
    output = nf.ops.tensor_reduction_sum(f(*tensors))
    nf.cgraph.perform_backward(output, nf.Tensor.fill(nf.Scalar(1.0), output.shape, output.dtype))
    computed_grads = [nf.cgraph.get_computed_grad(a).numpy() for a in tensors]

    max_error = max(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(tensors))
    )
    if max_error > tol:
        print(f"Gradient check failed on function {f.__name__} ({f})")
        print(f"Sum of absolute errors: {max_error} > tol = {tol}")
        for i in range(len(tensors)):
            err = np.linalg.norm(computed_grads[i] - numerical_grads[i])
            if err > tol:
                print(f"  Tensor {i} error: {err}")
                print(f"  Computed gradient: {computed_grads[i]}")
                print(f"  Numerical gradient: {numerical_grads[i]}")
                print()
        
        print("Test failed. Aborted")
        sys.exit(1)
                
    return computed_grads


def test_power_scalar_backward():
    gradient_check(
        lambda x: nf.ops.tensor_pows(x, 2), nf.Tensor(np.random.randn(5, 4))
    )

def test_divide_backward():
    gradient_check(
        lambda x, y: nf.ops.tensor_div(x, y),
        nf.Tensor(np.random.randn(5, 4)),
        nf.Tensor(5 + np.random.randn(5, 4)),
    )


def test_divide_scalar_backward():
    gradient_check(
        lambda x: nf.ops.tensor_divs(x, 0.114514),
        nf.Tensor(np.random.randn(5, 4), nf.float32),
        tol = 0.02
    )


def test_matmul_simple_backward():
    gradient_check(
        lambda x, y: nf.ops.matmul(x, y),
        nf.Tensor(np.random.randn(5, 4)),
        nf.Tensor(np.random.randn(4, 5)),
        tol = 0.02
    )


def test_matmul_batched_backward():
    gradient_check(
        nf.ops.matmul,
        nf.Tensor(np.random.randn(5, 4)),
        nf.Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        nf.ops.matmul,
        nf.Tensor(np.random.randn(6, 5, 4)),
        nf.Tensor(np.random.randn(6, 4, 3)),
    )
    gradient_check(
        nf.ops.matmul,
        nf.Tensor(np.random.randn(6, 5, 4)),
        nf.Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        nf.ops.matmul,
        nf.Tensor(np.random.randn(5, 4)),
        nf.Tensor(np.random.randn(6, 4, 3)),
    )


def test_reshape_backward():
    gradient_check(
        lambda x: nf.ops.reshape(x, (4, 5)),
        nf.Tensor(np.random.randn(5, 4))
    )


def test_negate_backward():
    gradient_check(
        nf.ops.tensor_negate,
        nf.Tensor(np.random.randn(5, 4))
    )


def test_transpose_backward():
    gradient_check(
        lambda x: nf.ops.transpose(x, 1, 2),
        nf.Tensor(np.random.randn(3, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.transpose(x, 0, 2),
        nf.Tensor(np.random.randn(3, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.transpose(x, 1, 0),
        nf.Tensor(np.random.randn(3, 5, 4))
    )


def test_broadcast_to_backward():
    gradient_check(
        lambda x: nf.ops.broadcast_to(x, (3, 3)),
        nf.Tensor(np.random.randn(3, 1))
    )
    gradient_check(
        lambda x: nf.ops.broadcast_to(x, (3, 3)),
        nf.Tensor(np.random.randn(1, 3))
    )
    gradient_check(
        lambda x: nf.ops.broadcast_to(x, (3, 3, 3)),
        nf.Tensor(
            np.random.randn(
                1,
            )
        )
    )
    gradient_check(
        lambda x: nf.ops.broadcast_to(x, (3, 3, 3)),
        nf.Tensor(np.random.randn())
    )
    gradient_check(
        lambda x: nf.ops.broadcast_to(x, (5, 4, 3)),
        nf.Tensor(np.random.randn(5, 4, 1))
    )


def test_summation_backward():
    gradient_check(
        lambda x: nf.ops.tensor_reduction_sum(x),
        nf.Tensor(np.random.randn(2, 4, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.tensor_reduction_sum(x, 0),
        nf.Tensor(np.random.randn(2, 4, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.tensor_reduction_sum(x, 1),
        nf.Tensor(np.random.randn(2, 4, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.tensor_reduction_sum(x, 2),
        nf.Tensor(np.random.randn(2, 4, 5, 4))
    )
    gradient_check(
        lambda x: nf.ops.tensor_reduction_sum(x, 3),
        nf.Tensor(np.random.randn(2, 4, 5, 4))
    )


def test_log_backward():
    gradient_check(
        lambda x: nf.ops.tensor_log(x),
        nf.Tensor(np.random.randn(5, 4)),
    )


def test_exp_backward():
    gradient_check(
        lambda x: nf.ops.tensor_exp(x),
        nf.Tensor(np.random.randn(5, 4)),
    )

if __name__ == "__main__":
    np.random.seed(0)
    nf.set_random_seed(0)
    devices = nf.Device.get_available_devices()
    print("Found devices:", devices)
    for device in devices:
        print(f"---------------------------------------")
        print(f"Testing device {repr(device)} ({device.get_hardware_name()})")
        nf.Device.set_default_device(device)
        
        ## 可以分别测试每个函数
        print("Testing power_scalar")
        test_power_scalar_backward()
        print("Testing divide")
        test_divide_backward()
        print("Testing divide_scalar")
        test_divide_scalar_backward()
        print("Testing matmul_simple")
        test_matmul_simple_backward()
        print("Testing matmul_batched")
        test_matmul_batched_backward()
        print("Testing summation")
        test_summation_backward()
        print("Testing broadcast_to")
        test_broadcast_to_backward()
        print("Testing reshape")
        test_reshape_backward()
        print("Testing negate")
        test_negate_backward()
        print("Testing transpose")
        test_transpose_backward()
        print("Testing log")
        test_log_backward()
        print("Testing exp")
        test_exp_backward()
        ## log 和 exp 的测试没写...（我帮您写了，就在上一行）
        ## 交作业的时候也是会测试的...（我帮您写了，就在上一行）
        
        print(f"Pass")
