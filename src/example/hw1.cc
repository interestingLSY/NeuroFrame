#include <bits/stdc++.h>

#include "src/tensor/tensor.h"

#include "src/op/sigmoid.h"
#include "src/op/relu.h"

using NeuroFrame::Tensor, NeuroFrame::dtype_t, NeuroFrame::Device;

int main() {
	{
		printf("Tensor tests\n");
		// Create a tensor on CPU with dtype FLOAT32
		// "randu" = "random uniformly"
		Tensor tensor1 = Tensor::randu({2, 3, 4}, dtype_t::FLOAT32, Device::cpu());
		printf("Tensor1: ");
		tensor1.print();

		// Create a tensor on GPU with dtype FLOAT16
		Tensor tensor2 = Tensor::randu({4, 3, 1}, dtype_t::FLOAT16, Device::cuda(0));
		printf("Tensor2: ");
		tensor2.print();

		// Move tensor1 to GPU
		tensor1 = tensor1.cuda();	// The same as tensor1.to(Device::cuda(device_index))
		printf("Tensor1: ");
		tensor1.print();

		// Move tensor2 to CPU
		tensor2 = tensor2.cpu();	// The same as tensor2.to(Device::cpu())
		printf("Tensor2: ");
		tensor2.print();

		// Create a tensor with literal values
		Tensor tensor3 = Tensor::from_vector({
			1., 2., 3., 4.,
			5., 6., 7., 8.,
			-1., -2., -3., -4.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());
		tensor3.print();
		tensor3.cuda().print();
		tensor3.cuda().cpu().cpu().cpu().cuda().cuda().cpu().cuda().cpu().cpu().cuda().print();
	}

	{
		// Test ReLU
		// For a much stronger unit test, please refer to src/unittest/op/activations.cc
		printf("\n\nReLU tests\n");
		// Input to the relu function
		Tensor relu_input = Tensor::from_vector({
			-1., 2., -3., 4.,
			5., -6., 7., -8.,
			-1., 2., -3., 4.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());
		relu_input.print();

		// Gradient of the output of the relu function (it will be used as the input of the backward function)
		Tensor relu_output_grad = Tensor::from_vector({
			2., 3., 4., 5.,
			6., 7., 8., 9.,
			10., 11., 12., 13.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());
		relu_output_grad.print();

		// Reference output of the relu function
		Tensor relu_output_reference = Tensor::from_vector({
			0., 2., 0., 4.,
			5., 0., 7., 0.,
			0., 2., 0., 4.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());

		// Reference gradient of the input of the relu function
		Tensor relu_input_grad_reference = Tensor::from_vector({
			0., 3., 0., 5.,
			6., 0., 8., 0.,
			0., 11., 0., 13.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());

		// Run ReLU on CPU
		NeuroFrame::OpContext ctx_h;
		Tensor relu_output_h = NeuroFrame::relu_forward_manual(relu_input, ctx_h);
		relu_output_h.print();
		assert (relu_output_h == relu_output_reference);

		// Run ReLU (backward) on CPU
		Tensor relu_input_grad_h = NeuroFrame::relu_backward_manual(relu_output_grad, ctx_h);
		relu_input_grad_h.print();
		assert (relu_input_grad_h == relu_input_grad_reference);

		// Run ReLU on GPU
		NeuroFrame::OpContext ctx_d;
		Tensor relu_output_d = NeuroFrame::relu_forward_manual(relu_input.cuda(), ctx_d);
		relu_output_d.print();
		assert (relu_output_d == relu_output_reference.cuda());

		// Run ReLU (backward) on GPU
		Tensor relu_input_grad_d = NeuroFrame::relu_backward_manual(relu_output_grad.cuda(), ctx_d);
		relu_input_grad_d.print();
		assert (relu_input_grad_d == relu_input_grad_reference.cuda());
	}

	{
		// Test Sigmoid
		// For a much stronger unit test, please refer to src/unittest/op/activations.cc
		printf("\n\nSigmoid tests\n");
		// Input to the sigmoid function
		Tensor sigmoid_input = Tensor::from_vector({
			-1., 2., -3., 4.,
			5., -6., 7., -8.,
			9., -10., 11., -12.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());
		sigmoid_input.print();

		// Gradient of the output of the sigmoid function (it will be used as the input of the backward function)
		Tensor sigmoid_output_grad = Tensor::from_vector({
			2., 3., 4., 5.,
			6., 7., 8., 9.,
			10., 11., 12., 13.
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());
		sigmoid_output_grad.print();

		// Reference output of the sigmoid function
		Tensor sigmoid_output_reference = Tensor::from_vector({
			0.26894143, 0.8807971, 0.04742587, 0.98201376,
			0.9933072, 0.0024726232, 0.9990889, 0.00033535013,
			0.9998766, 4.5397872e-05, 0.9999833, 6.144174e-06
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());

		// Reference gradient of the input of the sigmoid function
		Tensor sigmoid_input_grad_reference = Tensor::from_vector({
			0.39322387, 0.13552997, 0.04517666, 0.01766271,
			0.00246651, 0.00091022, 0.00033535, 0.00012339,
			4.5395808e-05, 1.670142e-05, 6.144174e-06, 2.260324e-06
		}, {3, 4}, dtype_t::FLOAT32, Device::cpu());

		// Run Sigmoid on CPU
		NeuroFrame::OpContext ctx_h;
		Tensor sigmoid_output_h = NeuroFrame::sigmoid_forward_manual(sigmoid_input, ctx_h);
		sigmoid_output_h.print();
		assert (sigmoid_output_h == sigmoid_output_reference);

		// Run Sigmoid (backward) on CPU
		Tensor sigmoid_input_grad_h = NeuroFrame::sigmoid_backward_manual(sigmoid_output_grad, ctx_h);
		sigmoid_input_grad_h.print();
		assert (sigmoid_input_grad_h == sigmoid_input_grad_reference);

		// Run Sigmoid on GPU
		NeuroFrame::OpContext ctx_d;
		Tensor sigmoid_output_d = NeuroFrame::sigmoid_forward_manual(sigmoid_input.cuda(), ctx_d);
		sigmoid_output_d.print();
		assert (sigmoid_output_d == sigmoid_output_reference.cuda());

		// Run Sigmoid (backward) on GPU
		Tensor sigmoid_input_grad_d = NeuroFrame::sigmoid_backward_manual(sigmoid_output_grad.cuda(), ctx_d);
		sigmoid_input_grad_d.print();
		assert (sigmoid_input_grad_d == sigmoid_input_grad_reference.cuda());
	}
}
