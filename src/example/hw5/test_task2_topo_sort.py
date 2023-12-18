"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task2拓扑排序部分的函数
"""

import numpy as np
# from task2_autodiff import *
import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

def test_topo_sort():
    # Test case 1
    print("Running testcase 1")
    a1, b1 = nf.Tensor(np.asarray([[0.88282157]])), nf.Tensor(
        np.asarray([[0.90170084]])
    )
    c1 = nf.ops.tensor_muls(a1, 3) * a1 + nf.ops.tensor_muls(b1, 4) * a1 - a1
    np.testing.assert_allclose(
        c1.numpy(), np.array([[4.639464008328271]])
    )

    soln = np.array([
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[0.88282157]]),
        np.array([[0.88282157]]),
        np.array([[7.90373278]]),
        np.array([[3.531286]]),
    ])

    grads = nf.cgraph.perform_backward(c1, nf.Tensor([[1.0]]), True)
    assert len(soln) == len(grads)
    np.testing.assert_allclose(
        [grad.numpy() for grad in grads],
        soln, rtol=1e-06, atol=1e-06
    )

    # Test case 2
    print("Running testcase 2")
    a1, b1 = nf.Tensor(np.asarray([[0.20914675], [0.65264178]])), nf.Tensor(
        np.asarray([[0.65394286, 0.08218317]])
    )
    c1 = nf.ops.tensor_adds(nf.ops.tensor_muls((b1 @ a1) + nf.ops.tensor_muls(b1, 2.3412) @ a1, 3), 1.5)
    np.testing.assert_allclose(
        c1.numpy(), np.array([[3.4085555076599121]])
    )

    soln = [
        np.array([[1.000000]]),
        np.array([[1.000000]]),
        np.array([[3.000000]]),
        np.array([[3.000000]]),
        np.array([[3.000000]]),
        np.array([[0.627440, 1.957925]]),
        np.array([[6.554862], [0.823771]]),
        np.array([[2.096404, 6.541821]]),
    ]

    grads = nf.cgraph.perform_backward(c1, nf.Tensor([[1.0]]), True)
    assert len(soln) == len(grads)
    for ans, std in zip(soln, grads):
        np.testing.assert_allclose(ans, std.numpy(), rtol=1e-06, atol=1e-06)

    # Test case 3
    print("Running testcase 3")
    a = nf.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = nf.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = nf.ops.matmul((nf.ops.matmul(a, b) + b - a), a)
    np.testing.assert_allclose(
        e.numpy(), np.array([[ 1.9889806509,  3.5122723579],
        [ 0.3428499997, -1.1873207092]])
    )

    grads = nf.cgraph.perform_backward(e, nf.Tensor([[1.0, 1.0], [1.0, 1.0]]), True)

    soln = np.array([
        np.array([[1.000000, 1.000000], [1.000000, 1.000000]]),
        np.array([[1.739101, -1.069422], [1.739101, -1.069422]]),
        np.array([[1.739101, -1.069422], [1.739101, -1.069422]]),
        np.array([[1.739101, -1.069422], [1.739101, -1.069422]]),
        np.array([[3.338857, 1.098642], [0.058579, -2.181636]]),
        np.array([[4.373498, -2.689386], [0.269345, -0.165628]]),
    ])

    assert len(soln) == len(grads)
    np.testing.assert_allclose([grad.numpy() for grad in grads], soln, rtol=1e-06, atol=1e-06)

if __name__ == "__main__":
    devices = nf.Device.get_available_devices()
    print("Found devices:", devices)
    for device in devices:
        print(f"---------------------------------------")
        print(f"Testing device {repr(device)} ({device.get_hardware_name()})")
        nf.Device.set_default_device(device)
        test_topo_sort()
        print("Pass")