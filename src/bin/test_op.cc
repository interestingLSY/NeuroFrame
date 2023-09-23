#include <bits/stdc++.h>

#include "src/tensor/tensor.h"
#include "src/op/relu.h"
#include "src/op/sigmoid.h"

using NeuroFrame::Tensor, NeuroFrame::dtype_t, NeuroFrame::Device;

int main() {
	{
		// Test ReLU (forward) on CPU
		printf("ReLU\n");
		Tensor tensor = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		tensor.print();
		Tensor relu_tensor_cpu = NeuroFrame::relu(tensor);
		relu_tensor_cpu.print();
		Tensor relu_tensor_gpu = NeuroFrame::relu(tensor.to(Device::cuda()));
		relu_tensor_gpu.print();
	}

	{
		// Test sigmoid (forward)
		printf("Sigmoid\n");
		Tensor tensor = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		tensor.print();
		Tensor sigmoid_tensor_cpu = NeuroFrame::sigmoid(tensor);
		sigmoid_tensor_cpu.print();
		Tensor sigmoid_tensor_gpu = NeuroFrame::sigmoid(tensor.to(Device::cuda()));
		sigmoid_tensor_gpu.print();
	}
}