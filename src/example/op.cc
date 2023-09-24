#include <bits/stdc++.h>

#include "src/tensor/tensor.h"
#include "src/op/relu.h"
#include "src/op/sigmoid.h"

using NeuroFrame::Tensor, NeuroFrame::dtype_t, NeuroFrame::Device;

int main() {
	{
		// ReLU (forward) on CPU & CUDA
		printf("ReLU\n");
		Tensor input = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		input.print();
		Tensor output_cpu = NeuroFrame::relu(input);
		output_cpu.print();
		Tensor output_gpu = NeuroFrame::relu(input.to(Device::cuda()));
		output_gpu.print();
	}

	{
		// Sigmoid (forward) on CPU & CUDA
		printf("Sigmoid\n");
		Tensor input = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		input.print();
		Tensor output_cpu = NeuroFrame::sigmoid(input);
		output_cpu.print();
		Tensor output_gpu = NeuroFrame::sigmoid(input.to(Device::cuda()));
		output_gpu.print();
	}
}