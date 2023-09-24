#include <bits/stdc++.h>

#include "src/tensor/tensor.h"

using NeuroFrame::Tensor, NeuroFrame::dtype_t, NeuroFrame::Device;

int main() {
	// Test tensor creation & display on CPU
	{
		printf("1\n");
		Tensor tensor1 = Tensor::zeros({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		Tensor tensor2 = Tensor::randu({4, 3, 1}, dtype_t::FLOAT16, Device::cpu());
		Tensor tensor3 = Tensor::from_vector({
			1., 2., 3., 4., 5.,
			-1., -2., -3., -4., -5.
		}, {2, 5}, dtype_t::FLOAT64, Device::cpu());
		tensor1.print();
		tensor2.print();
		tensor3.print();
	}

	// Test scalar tensor on CPU
	{
		Tensor tensor = Tensor::randu({}, dtype_t::FLOAT32, Device::cpu());
		tensor.print();
	}

	// Test tensor creation & display on CUDA
	{
		printf("2\n");
		Tensor tensor1 = Tensor::zeros({2, 3, 4}, dtype_t::FLOAT32, Device::cuda(0));
		Tensor tensor2 = Tensor::randu({4, 3, 1}, dtype_t::FLOAT16, Device::cuda(0));
		Tensor tensor3 = Tensor::from_vector({
			1., 2., 3., 4., 5.,
			-1., -2., -3., -4., -5.
		}, {2, 5}, dtype_t::FLOAT64, Device::cuda(0));
		tensor1.print();
		tensor2.print();
		tensor3.print();
	}

	// Test scalar tensor on CUDA
	{
		Tensor tensor = Tensor::randu({}, dtype_t::FLOAT32, Device::cuda(0));
		tensor.print();
	}

	// Test moving tensor between devices
	{
		printf("3\n");
		Tensor tensor = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		tensor.print();
		tensor.to(Device::cuda(0)).print();
		for (int i = 0; i < 1000; ++i) {
			int t = rand()%2;
			if (t == 1) tensor = tensor.to(Device::cpu());
			else tensor = tensor.to(Device::cuda(0));
		}
		tensor.print();
	}

	// Test scalar tensors
	{
		printf("4\n");
		Tensor tensor1 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cpu());
		tensor1.print();
		for (int i = 0; i < tensor1.numel(); ++i) {
			tensor1.get_elem({i}) = (float)i;
		}
		tensor1.print();
		Tensor tensor2 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cuda());
		tensor2.print();
		for (int i = 0; i < tensor2.numel(); ++i) {
			tensor2.get_elem({i}) = (float)i;
		}
		tensor2.print();
	}

	// Test tensor comparison
	{
		printf("5\n");
		Tensor tensor1 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cpu());
		Tensor tensor2 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cpu());
		Tensor tensor3 = Tensor::randu({10}, dtype_t::FLOAT32, Device::cpu());
		Tensor tensor4 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cuda());
		Tensor tensor5 = Tensor::zeros({10}, dtype_t::FLOAT32, Device::cuda());
		Tensor tensor6 = Tensor::randu({10}, dtype_t::FLOAT32, Device::cuda());
		for (int i = 0; i < tensor1.numel(); ++i) {
			tensor1.get_elem({i}) = (float)i;
			tensor2.get_elem({i}) = (float)i;
			tensor4.get_elem({i}) = (float)i;
			tensor5.get_elem({i}) = (float)i;
		}
		printf("%d\n", tensor1 == tensor2);
		printf("%d\n", tensor1 == tensor3);
		printf("%d\n", tensor4 == tensor5);
		printf("%d\n", tensor4 == tensor6);
	}
}
