#include "tensor.h"
using namespace pybind11::literals;	// For "_a" suffix

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "src/tensor/tensor.h"

using NeuroFrame::Tensor;

void init_tensor(pybind11::module& m) {
	pybind11::class_<Tensor>(m, "Tensor")
		.def("__eq__", &Tensor::operator==)
		.def("__ne__", &Tensor::operator!=)

		.def_readwrite("device", &Tensor::device)
		.def_readwrite("dtype", &Tensor::dtype)

		.def_readwrite("first_elem_offset", &Tensor::first_elem_offset)
		.def_readwrite("shape", &Tensor::shape)
		.def_readwrite("stride", &Tensor::stride)

		.def("numel", &Tensor::numel)
		.def("data_ptr", &Tensor::data_ptr, pybind11::return_value_policy::reference)
		.def("dim", &Tensor::dim)
		// .def("reshape", &Tensor::reshape)	// Prevent the user from using this function since this function won't touch the computation graph

		.def("get_elem_addr", &Tensor::get_elem_addr, pybind11::return_value_policy::reference)
		.def("get_elem", &Tensor::get_elem)
		.def("as_scalar", &Tensor::as_scalar)

		.def("to", &Tensor::to, "target_device"_a)
		.def("cpu", &Tensor::cpu)
		.def("cuda", &Tensor::cuda, "device_index"_a = 0)

		.def("to_string", &Tensor::to_string, "max_display_per_dim"_a = 16, "in_compat_style"_a = false)
		.def("__str__", &Tensor::to_string, "max_display_per_dim"_a = 16, "in_compat_style"_a = false)
		.def("__repr__", &Tensor::repr)
		.def("print", &Tensor::print, "max_display_per_dim"_a = 16, "in_compat_style"_a = false)

		.def("empty", [](Tensor& instance, const std::vector<int64_t>& shape, NeuroFrame::dtype_t dtype, const NeuroFrame::Device& device) {
			new (&instance) Tensor(shape, dtype, device);
		})
		.def_static("zeros", &Tensor::zeros, "shape"_a, "dtype"_a, "device"_a)
		.def_static("randu", static_cast<Tensor(*)(const std::vector<int64_t>&, NeuroFrame::dtype_t, NeuroFrame::Device, NeuroFrame::Scalar, NeuroFrame::Scalar)>(&Tensor::randu), "shape"_a, "dtype"_a, "device"_a, "low"_a = NeuroFrame::Scalar(-1.0f), "high"_a = NeuroFrame::Scalar(1.0f))
		.def_static("randint", &Tensor::randint, "shape"_a, "dtype"_a, "device"_a, "low"_a, "high"_a)
		.def_static("from_vector", &Tensor::from_vector, "data"_a, "shape"_a, "dtype"_a, "device"_a)

		.def("__add__", &Tensor::operator+)
		.def("__sub__", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::operator-))
		.def("__neg__", static_cast<Tensor(Tensor::*)() const>(&Tensor::operator-))

		// The following functions are extensions to the original NeuroFrame::Tensor
		.def("__init__", [](Tensor& instance, const std::vector<int64_t> &data, const std::vector<int64_t>& shape, NeuroFrame::dtype_t dtype, const NeuroFrame::Device& device) {
			std::vector<NeuroFrame::Scalar> scalars = std::vector<NeuroFrame::Scalar>(data.begin(), data.end());
			new (&instance) Tensor(Tensor::from_vector(scalars, shape, dtype, device));
		})
		.def("__init__", [](Tensor& instance, const std::vector<double> &data, const std::vector<int64_t>& shape, NeuroFrame::dtype_t dtype, const NeuroFrame::Device& device) {
			std::vector<NeuroFrame::Scalar> scalars = std::vector<NeuroFrame::Scalar>(data.begin(), data.end());
			new (&instance) Tensor(Tensor::from_vector(scalars, shape, dtype, device));
		})

		// Construct from numpy array
		.def("__init__", [](Tensor &instance, pybind11::array_t<double> &array, NeuroFrame::dtype_t dtype, const NeuroFrame::Device& device) {
			// Retrieve the shape
			std::vector<int64_t> shape;
			for (int i = 0; i < array.ndim(); ++i) {
				shape.push_back(array.shape(i));
			}

			// Dump data to vector
			int64_t numel = array.size();
			std::vector<NeuroFrame::Scalar> data;
			pybind11::array_t<double> array_1d = array.reshape({numel});
			auto accessor = array_1d.unchecked<1>();
			for (int i = 0; i < array.size(); ++i) {
				data.push_back(NeuroFrame::Scalar(accessor(i)));
			}

			// Construct
			new (&instance) Tensor(Tensor::from_vector(data, shape, dtype, device));
		}, "array"_a, "dtype"_a = NeuroFrame::dtype_t::FLOAT32, "device"_a = NeuroFrame::Device::cpu())

		// Convert to numpy array
		// NOTE. It returns a pybind11::array_t<double> no matter what the dtype of the tensor is.
		.def("numpy", [](Tensor &instance) {
			// Dump data to vector
			int64_t numel = instance.numel();
			Tensor instance_1d = instance.reshape({numel});
			std::vector<double> data;
			for (int i = 0; i < numel; ++i) {
				data.push_back(instance_1d.get_elem({i}).as_scalar().as_double());
			}

			// Construct the array
			pybind11::array_t<double> array = pybind11::array_t<double>(numel);
			auto accessor = array.mutable_unchecked<1>();
			for (int i = 0; i < numel; ++i) {
				accessor(i) = data[i];
			}
			array = array.reshape(instance.shape);
			return array;
		});
}
