import sys, time
from functools import reduce
import random
sys.path.append("/home/intlsy/projects/programming-in-ai/NeuroFrame/build/src/pybind")

import neuroframe
import torch

def generate_random_list(shape: list[int]):
    return torch.rand(shape).tolist()

def test_batched_conv():
    print(f"\n---- Batched convolution:")
    input_shape = [2, 3, 5, 5]	# [batch, channel, height, width]
    weight_shape = [2, 3, 3, 3]	# [out_channel, in_channel, kernel_height, kernel_width]
    input_vals = generate_random_list(input_shape)
    weight_vals = generate_random_list(weight_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    weight = neuroframe.Tensor(weight_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.batched_convolution(input, weight)
    print(f"Input: {input}")
    print(f"Weight: {weight}")
    print(f"Output: {output}")

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    weight_th = torch.tensor(weight_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.nn.functional.conv2d(input_th, weight_th, padding='same')
    print(f"Output(th): {output_th}")

def test_cross_entropy_loss():
    print(f"\n---- Cross entropy loss:")
    input_shape = [16384, 32]	# [batch, class]
    ground_truth_shape = [16384] # [batch]
    input_vals = generate_random_list(input_shape)
    ground_truth_vals = [random.randint(0, input_shape[1]-1) for _ in range(input_shape[0])]
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    ground_truth = neuroframe.Tensor(ground_truth_vals, neuroframe.int32, device_cuda)
    loss = neuroframe.ops.cross_entropy_loss(input, ground_truth)
    print(f"Input: {input}")
    print(f"Ground truth: {ground_truth}")
    print(f"Loss: {loss}")

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    ground_truth_th = torch.tensor(ground_truth_vals, dtype=torch.int64, device=torch.device("cuda:0"))
    loss_th = torch.nn.functional.cross_entropy(input_th, ground_truth_th)
    print(f"Loss(th): {loss_th}")

def test_matmul_1():
    print(f"\n---- Matrix multiplication (unbatched - unbatched):")
    input1_shape = [5, 6]
    input2_shape = [6, 7]
    input1_vals = generate_random_list(input1_shape)
    input2_vals = generate_random_list(input2_shape)
    input1 = neuroframe.Tensor(input1_vals, neuroframe.float32, device_cuda)
    input2 = neuroframe.Tensor(input2_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.matmul(input1, input2)
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Output: {output}")

    input1_th = torch.tensor(input1_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    input2_th = torch.tensor(input2_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.matmul(input1_th, input2_th)
    print(f"Output(th): {output_th}")

def test_matmul_2():
    print(f"\n---- Matrix multiplication (unbatched - batched):")
    input1_shape = [5, 6]
    input2_shape = [3, 6, 7]
    input1_vals = generate_random_list(input1_shape)
    input2_vals = generate_random_list(input2_shape)
    input1 = neuroframe.Tensor(input1_vals, neuroframe.float32, device_cuda)
    input2 = neuroframe.Tensor(input2_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.matmul(input1, input2)
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Output: {output}")

    input1_th = torch.tensor(input1_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    input2_th = torch.tensor(input2_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.matmul(input1_th, input2_th)
    print(f"Output(th): {output_th}")

def test_matmul_3():
    print(f"\n---- Matrix multiplication (batched - batched):")
    input1_shape = [3, 5, 6]
    input2_shape = [3, 6, 7]
    input1_vals = generate_random_list(input1_shape)
    input2_vals = generate_random_list(input2_shape)
    input1 = neuroframe.Tensor(input1_vals, neuroframe.float32, device_cuda)
    input2 = neuroframe.Tensor(input2_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.matmul(input1, input2)
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Output: {output}")

    input1_th = torch.tensor(input1_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    input2_th = torch.tensor(input2_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.matmul(input1_th, input2_th)
    print(f"Output(th): {output_th}")

def test_pooling():
    print(f"\n---- Pooling:")
    input_shape = [3, 6, 6]	# [batch, height, width]
    input_vals = generate_random_list(input_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.pool(input, 2)

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.nn.functional.max_pool2d(input_th, (2, 2))
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Output(th): {output_th}")

def test_relu():
    print(f"\n---- ReLU:")
    input_shape = [32]
    input_vals = generate_random_list(input_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.relu(input)

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.nn.functional.relu(input_th)
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Output(th): {output_th}")

def test_sigmoid():
    print(f"\n---- Sigmoid:")
    input_shape = [32]
    input_vals = generate_random_list(input_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.sigmoid(input)

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.nn.functional.sigmoid(input_th)
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Output(th): {output_th}")

def test_tensor_binary_op():
    print(f"\n---- Tensor add:")
    input_shape = [3, 4]
    input1_vals = generate_random_list(input_shape)
    input2_vals = generate_random_list(input_shape)
    input1 = neuroframe.Tensor(input1_vals, neuroframe.float32, device_cuda)
    input2 = neuroframe.Tensor(input2_vals, neuroframe.float32, device_cuda)
    output = input1 + input2

    input1_th = torch.tensor(input1_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    input2_th = torch.tensor(input2_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.add(input1_th, input2_th)
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"input1+input2: {output}")
    print(f"input1+input2(th): {output_th}")

    print(f"\n---- Tensor sub:")
    output = input1 - input2
    output_th = torch.sub(input1_th, input2_th)
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"input1-input2: {output}")
    print(f"input1-input2(th): {output_th}")
    
def test_tensor_unary_op():
    print(f"\n---- Tensor neg:")
    input_shape = [3, 4]
    input_vals = generate_random_list(input_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    output = -input

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.neg(input_th)
    print(f"Input: {input}")
    print(f"-input: {output}")
    print(f"-input(th): {output_th}")
    
def test_transpose():
    print(f"\n---- Tensor transpose:")
    input_shape = [3, 4]
    input_vals = generate_random_list(input_shape)
    input = neuroframe.Tensor(input_vals, neuroframe.float32, device_cuda)
    output = neuroframe.ops.transpose(input)

    input_th = torch.tensor(input_vals, dtype=torch.float32, device=torch.device("cuda:0"))
    output_th = torch.transpose(input_th, 1, 0)
    print(f"Input: {input}")
    print(f"input.transpose(1, 0): {output}")
    print(f"input.transpose(1, 0)(th): {output_th}")

neuroframe.set_random_seed(0)
random.seed(0)
torch.set_printoptions(sci_mode=False)

print(f"-------- Testing basic components --------")

# Device
device_cpu = neuroframe.Device.cpu()
device_cuda = neuroframe.Device.cuda(0)
print(f"Device CPU: {device_cpu}")
print(f"Device CUDA: {device_cuda}")

# dtype
print(f"Float16: {neuroframe.dtype.float32}")
print(f"Float32: {neuroframe.dtype.float32}")
print(f"Float64: {neuroframe.float64}")	# Both neuroframe.dtype.XX and neuroframe.XX are valid
print(f"Int8: {neuroframe.dtype.int8}")
print(f"Int16: {neuroframe.dtype.int16}")
print(f"Int32: {neuroframe.int32}")
print(f"Int64: {neuroframe.int64}")

# Scalar
print(f"Scalar: {neuroframe.Scalar(1.0).__repr__()}")
print(f"Scalar: {neuroframe.Scalar(10).__repr__()}")
print(f"Scalar: {neuroframe.Scalar(1.0, neuroframe.float16).__repr__()}")
print(f"Scalar: {neuroframe.Scalar(10, neuroframe.int8).__repr__()}")

print(f"\n\n")
print(f"-------- Testing Tensor --------")

# Tensor creation
tensor1 = neuroframe.Tensor(list(range(12)), [3, 4], neuroframe.int32, device_cpu)
print(f"Tensor1: {tensor1}")

tensor2 = neuroframe.Tensor(generate_random_list(120), [1, 2, 3, 4, 5], neuroframe.float16, device_cpu)
print(f"Tensor2: {tensor2}")

tensor3 = neuroframe.Tensor.randu([3, 4], neuroframe.float32, device_cuda)
print(f"Tensor3: {tensor3}")
print(f"Tensor3's info:")
print(f"    device: {tensor3.device}")
print(f"    dtype: {tensor3.dtype}")
print(f"    shape: {tensor3.shape}")
print(f"    stride: {tensor3.stride}")
print(f"    numel: {tensor3.numel()}")
print(f"    dim: {tensor3.dim()}")
print(f"Tensor3.cuda(): {tensor3.cuda()}")
print(f"Tensor3 == Tensor3.cuda(): {tensor3 == tensor3.cuda()}")

tensor4 = neuroframe.Tensor.zeros([1], neuroframe.float64, device_cuda)
print(f"Tensor4: {tensor4}")

tensor5 = neuroframe.Tensor([
    [1, 2],
    [3, 4]
], neuroframe.float16, device_cuda)
print(f"Tensor5: {tensor5}")

print(f"\n\n")
print(f"-------- Testing Tensor ops --------")

test_batched_conv()
test_cross_entropy_loss()
test_matmul_1()
test_matmul_2()
test_matmul_3()
test_pooling()
test_relu()
test_sigmoid()
test_tensor_binary_op()
test_tensor_unary_op()
test_transpose()
