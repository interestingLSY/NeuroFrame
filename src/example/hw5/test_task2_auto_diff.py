"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task2自动微分部分的函数
"""

from test_task1_backward import gradient_check
from test_task1_forward import matmul, reshape, negate, transpose, broadcast_to, summation
import numpy as np
from tensor import TensorFull as Tensor

def test_compute_gradient():
    gradient_check(
        lambda A, B, C: summation((A @ B + C) * (A @ B), axes=None),
        Tensor(np.random.randn(10, 9)),
        Tensor(np.random.randn(9, 8)),
        Tensor(np.random.randn(10, 8)),
        backward=True,
    )
    gradient_check(
        lambda A, B: summation(broadcast_to(A, shape=(10, 9)) * B, axes=None),
        Tensor(np.random.randn(10, 1)),
        Tensor(np.random.randn(10, 9)),
        backward=True,
    )
    gradient_check(
        lambda A, B, C: summation(
            reshape(A, shape=(10, 10)) @ B / 5 + C, axes=None
        ),
        Tensor(np.random.randn(100)),
        Tensor(np.random.randn(10, 5)),
        Tensor(np.random.randn(10, 5)),
        backward=True,
    )

    # check gradient of gradient
    x2 = Tensor([6])
    x3 = Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1

if __name__ == "__main__":
    test_compute_gradient()
