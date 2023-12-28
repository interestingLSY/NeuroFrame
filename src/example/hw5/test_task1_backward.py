"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task1梯度计算部分的函数
"""

import os, sys
import numpy as np
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import torch
import neuroframe as nf

def gradient_check(f, *tensors_, tol=1e-4, step=1/2**14, enable_inference_mode=True):
    eps = step
    tensors = list(tensors_)
    class PlaceHolder:
        def __init__(self): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    with (nf.inference_mode() if enable_inference_mode else PlaceHolder()):
        tensors_fp64 = [nf.Tensor(a.numpy(), nf.float64) for a in tensors]
        numerical_grads = [np.zeros(a.shape) for a in tensors]
        for i in range(len(tensors_fp64)):
            for j in range(tensors_fp64[i].numel()):
                old_tensor = nf.ops.copy(tensors_fp64[i])
                raw_data = tensors_fp64[i].numpy()
                
                raw_data.flat[j] += eps
                tensors_fp64[i] = nf.Tensor(raw_data, nf.float64)
                f1 = float(f(*tensors_fp64).numpy().sum())
                
                raw_data.flat[j] -= 2*eps
                tensors_fp64[i] = nf.Tensor(raw_data, nf.float64)
                f2 = float(f(*tensors_fp64).numpy().sum())
                
                tensors_fp64[i] = old_tensor
                numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)

    nf.cgraph.clear_graph()
    output = nf.ops.tensor_reduction_sum(f(*tensors))
    nf.cgraph.perform_backward(output, nf.Tensor.fill(nf.Scalar(1.0), output.shape, output.dtype))
    computed_grads = [nf.cgraph.get_computed_grad(a).numpy() for a in tensors]

    max_error = max(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(tensors))
    )
    if max_error > tol or np.isnan(max_error):
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
        lambda x: nf.ops.tensor_pows(x, 2), nf.Tensor(np.random.randn(5, 4)+10)
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
        nf.Tensor(np.random.rand(5, 4)+10),
    )


def test_exp_backward():
    gradient_check(
        lambda x: nf.ops.tensor_exp(x),
        nf.Tensor(np.random.randn(5, 4)),
    )
    

def test_conv_backward():
    gradient_check(
        lambda x, y: nf.ops.tensor_pows(nf.ops.batched_convolution(x, y), 3),
        nf.Tensor(np.random.rand(2, 2, 12, 12)),
        nf.Tensor(np.random.rand(2, 2, 3, 3)),
        tol=4e-3
    )


def test_cross_entropy_backward():
    ground_truth = nf.Tensor(np.random.randint(0, 10, (10)), nf.int32)
    gradient_check(
        lambda x: nf.ops.cross_entropy_loss(x, ground_truth),
        nf.Tensor(np.random.rand(10, 10)),
        tol=4e-3
    )
    

def test_batch_norm_backward():
    C = 8
    x_raw = np.random.rand(4, C, 4, 4)
    gamma_raw = np.random.rand(C)
    beta_raw = np.random.rand(C)
    mean_raw = np.random.rand(C)
    var_raw = np.random.rand(C)
    output_grad_raw = np.random.rand(4, C, 4, 4)
    
    x = nf.Tensor(x_raw)
    gamma = nf.Tensor(gamma_raw)
    beta = nf.Tensor(beta_raw)
    mean = nf.Tensor(mean_raw)
    var = nf.Tensor(var_raw)
    output_grad = nf.Tensor(output_grad_raw)
    
    x_th = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
    gamma_th = torch.tensor(gamma_raw, dtype=torch.float32, requires_grad=True)
    beta_th = torch.tensor(beta_raw, dtype=torch.float32, requires_grad=True)
    mean_th = torch.tensor(mean_raw, dtype=torch.float32)
    var_th = torch.tensor(var_raw, dtype=torch.float32)
    output_grad_th = torch.tensor(output_grad_raw, dtype=torch.float32)
    
    std_result = torch.nn.functional.batch_norm(
        x_th, mean_th, var_th, weight=gamma_th, bias=beta_th,
        momentum=0.5, eps=1e-5, training=True
    )
    std_result.backward(gradient=output_grad_th)
    x_grad_th = x_th.grad
    gamma_grad_th = gamma_th.grad
    beta_grad_th = beta_th.grad
    
    nf.cgraph.clear_graph()
    my_result = nf.ops.batch_norm(x, gamma, beta, mean, var, 0.5, 1e-5)
    nf.cgraph.perform_backward(my_result, output_grad)
    x_grad = nf.cgraph.get_computed_grad(x)
    gamma_grad = nf.cgraph.get_computed_grad(gamma)
    beta_grad = nf.cgraph.get_computed_grad(beta)
    
    def check(name: str, std: torch.Tensor, ans: nf.Tensor):
        std = std.detach().numpy()
        ans = ans.numpy()
        max_error = np.linalg.norm(std - ans)
        if max_error > 1e-4:
            print(f"Error: {max_error} on batch_norm {name}")
            print(f"Std: {std}")
            print(f"Ans: {ans}")
            print()
            print("Test failed. Aborted")
            sys.exit(1)
    
    check("result", std_result, my_result)
    check("x_grad", x_grad_th, x_grad)
    check("gamma_grad", gamma_grad_th, gamma_grad)
    check("beta_grad", beta_grad_th, beta_grad)

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
        if device.is_cuda():
            print("Testing conv")
            test_conv_backward()
        print("Testing batch_norm")
        test_batch_norm_backward()
        print("Testing cross_entropy")
        test_cross_entropy_backward()
        ## log 和 exp 的测试没写...（我帮您写了，就在上一行）
        ## 交作业的时候也是会测试的...（我帮您写了，就在上一行）
        
        print(f"Pass")
