# NeuroFrame

A tiny framework for AI training and inference. My homework for the course "programming in AI".

## Build

```bash
git clone https://github.com/interestingLSY/NeuroFrame.git
cd NeuroFrame
cmake -B build
cmake --build build -j8
```

## Code Structure

The code is organized as follows:

- The `basic` folder contains some basic data structures (e.g. `device`, `scalar`) and basic
	functions (e.g. memory management, logging).
- The `tensor` folder contains the `Tensor` class.
- The `op` folder contains operators (e.g. `matmul`, `relu`, `sigmoid`). Every operator
	has a `forward` and a `backward` function, which are high-level device-agnostic
	differentiable functions that wraps around low-level device-specific kernels in `backend`.
- The `backend` folder contains the actual implementation of the operators. The
	`backend` folder is further divided into `cpu` and `cuda` folders, which contain
	the CPU and CUDA implementations of the operators, respectively.
- The `example` folder contains some examples.

## Design Philosophy

Inside the `op` folder, there is a very important function, `perform_op`. Similar
to the "thin waist" (the IP layer) in computer networking, `perform_op` is the
"thin waist" of NeuroFrame. Here is its signature:

```cpp
typedef std::function<std::vector<Tensor>(const std::vector<Tensor>&, OpContext &)> op_forward_func_t;
typedef std::function<std::vector<Tensor>(const std::vector<Tensor>&, OpContext &)> op_backward_func_t;

std::vector<Tensor> perform_op(
	op_forward_func_t forward_op,
	op_backward_func_t backward_op,
	const std::vector<Tensor> &input
);
```

`perform_op` launches some operator with the given forward function, backward
function, and input. After executing the op it may modify the computation graph
for gradient computation. Every call to an operator (e.g. `relu`) will invoke `perform_op`
indirectly. Usually `forward_op` and `backward_op` invokes the underlying 
kernel in `backend` (e.g. `NeuroFrame::Backend::CPU::relu_forward` or
`NeuroFrame::Backend::CUDA::relu_forward`).

The call stack looks like this:

```plain
Used code ->
Op wrapper (like, `relu(x)`, `batched_gemm`, `sigmoid`...) ->
`perform_op` ->
various kernels on various backends (e.g. `relu` on CPU, `gemm` on CUDA ...)
```

## How to Use

NeuroFrams's APIs are very similar to PyTorch's. You can refer to code under the
`example` folder for more details.

### Tensors

Three fields (`first_elem_offset`, `shape`, and `stride`) in the `Tensor` class
controls how the tensor is viewed.

`first_elem_offset` is the offset of the first element of the tensor in
the underlying memory fragment. `shape` is the shape of the tensor.
`stride[i]` refers to the distance (in elements, not bytes) between two
adjacent elements in the i-th dimension.

Different from PyTorch, stride[i] always equals to the product of
shape[i+1], shape[i+2], ..., shape[n-1], where n is the number of
dimensions of the tensor. In other words, the tensor is always
"continuous" in memory. I think this is a good design because it makes
the implementation of kernels much easier.

#### Tensor Creation

```cpp
// Do not fill
Tensor::Tensor(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
// Fill with zeros (the same as above, just with a different name)
static Tensor Tensor::zeros(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
// Fill with uniform distribution between `low` and `high`
static Tensor Tensor::randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device, Scalar low, Scalar high);
// Fill with uniform distribution between -1 and +1
static Tensor Tensor::randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device);
```

#### Migration between Devices

```cpp
// Migration between devices. .to(Device::cpu()) brings the tensor to CPU memory,
// .to(Device::cuda(device_index)) brings the tensor to GPU memory.
Tensor Tensor::to(Device target) const;
// The same as `to(Device::cpu())`
Tensor Tensor::cpu() const;
// The same as `to(Device::cuda(device_index))`
Tensor Tensor::cuda(int device_index = 0) const;
```

#### Visit an Element

```cpp
// Return the start address. Useful when writing kernels
void* Tensor::data_ptr() const;
// Get the address of one element
void* Tensor::get_elem_addr(const std::vector<int64_t> &pos) const;
// Get the content (in NeuroFrame::Tensor with shape == []) of one element
Tensor Tensor::get_elem(const std::vector<int64_t> &pos) const;
// Return the element as a scalar. Only applicable to scalar tensors (tensors with shape == [])
Scalar Tensor::as_scalar() const;
```

One can use something like `tensor.get_elem().as_scalar().as_double()` to obtain
one value inside the tensor.

#### Other Operations

```cpp
// Return the number of elements in this tensor
int64_t numel() const;
// Print the tensor
void print(int64_t max_display_per_dim = 16) const;
```

### Operators

Operators are something like `relu`, `sigmoid`, `matmul`, `batched_matmul`, etc.
Every invocation of an operator will perform the corresponding forward pass, and
modify the computation graph for backward pass if necessary.

Our operators' signatures are very similar to PyTorch's. If you do not know how
to use an operator, just refer to `src/op/<op_name>.h` where `<op_name>` is the name of the
operator, or you can refer to PyTorch's documentation (ಡωಡ).

In the usual case, users only need to invoke the function that has the same name
as the operator itself, e.g. if you want to apply ReLU on a tensor `x`, just
write `relu(x)`, and NeuroFrame will do the rest (including dispatching to devices,
calling kernels, and modifying the computation graph) for you. Under some 
circumstances (like NeuroFrame's unit tests), you may want to manually invoke
the underlying forward and backward functions without disrupting the computation
graph. In this case, you can use 