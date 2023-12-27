#include "basic.h"
using namespace pybind11::literals;	// For "_a" suffix

#include <pybind11/stl.h>

#include "src/basic/device.h"
#include "src/basic/inference_mode.h"
#include "src/basic/random.h"
#include "src/basic/scalar.h"

void init_basic(pybind11::module& m) {
	m.def("set_random_seed", &NeuroFrame::set_random_seed, "seed"_a);
	
	pybind11::class_<NeuroFrame::Device>(m, "Device")
		.def("switch_to", &NeuroFrame::Device::switch_to)
		.def("__str__", &NeuroFrame::Device::to_string)
		.def("__repr__", &NeuroFrame::Device::repr)
		.def("__eq__", &NeuroFrame::Device::operator==)
		.def("__ne__", &NeuroFrame::Device::operator!=)
		.def("is_cpu", &NeuroFrame::Device::is_cpu)
		.def("is_cuda", &NeuroFrame::Device::is_cuda)
		.def_static("cpu", &NeuroFrame::Device::cpu)
		.def_static("cuda", &NeuroFrame::Device::cuda, "device_index"_a = 0)
		.def("get_hardware_name", &NeuroFrame::Device::get_hardware_name)
		.def_static("get_available_devices", &NeuroFrame::Device::get_available_devices)
		.def_static("get_default_device", &NeuroFrame::Device::get_default_device)
		.def_static("set_default_device", &NeuroFrame::Device::set_default_device, "device"_a);
	
	pybind11::enum_<NeuroFrame::dtype_t>(m, "dtype")
		.value("float16", NeuroFrame::dtype_t::FLOAT16)
		.value("float32", NeuroFrame::dtype_t::FLOAT32)
		.value("float64", NeuroFrame::dtype_t::FLOAT64)
		.value("int8", NeuroFrame::dtype_t::INT8)
		.value("int16", NeuroFrame::dtype_t::INT16)
		.value("int32", NeuroFrame::dtype_t::INT32)
		.value("int64", NeuroFrame::dtype_t::INT64)
		.export_values();	// This exports the enum values into the parent scope, so that we can use neuroframe.float32 instead of neuroframe.dtype.float32
	
	pybind11::class_<NeuroFrame::InferenceModeGuard>(m, "InferenceModeGuard")
		.def(pybind11::init<>())
		.def("__enter__", [](NeuroFrame::InferenceModeGuard& instance) {
			instance.__enter__();
		})
		.def("__exit__", [](NeuroFrame::InferenceModeGuard& instance, pybind11::object exc_type, pybind11::object exc_value, pybind11::object traceback) {
			instance.__exit__();
			return false;
		});
	m.def("is_inference_mode", &NeuroFrame::is_inference_mode);
	m.def("inference_mode", []() {
		return NeuroFrame::InferenceModeGuard();
	});
	
	pybind11::class_<NeuroFrame::Scalar>(m, "Scalar")
		.def("__init__", [](NeuroFrame::Scalar& instance, double f) {
			// Default to FLOAT32 when passing a float
			new (&instance) NeuroFrame::Scalar(f, NeuroFrame::dtype_t::FLOAT32);
		})
		.def("__init__", [](NeuroFrame::Scalar& instance, int64_t i) {
			// Default to INT64 when passing an int
			new (&instance) NeuroFrame::Scalar(i, NeuroFrame::dtype_t::INT64);
		})
		.def("__init__", [](NeuroFrame::Scalar& instance, double f, NeuroFrame::dtype_t dtype) {
			new (&instance) NeuroFrame::Scalar(f, dtype);
		})
		.def("__init__", [](NeuroFrame::Scalar& instance, int64_t i, NeuroFrame::dtype_t dtype) {
			new (&instance) NeuroFrame::Scalar(i, dtype);
		})
		.def("__str__", &NeuroFrame::Scalar::to_string)
		.def("__repr__", &NeuroFrame::Scalar::repr)
		.def("__eq__", &NeuroFrame::Scalar::operator==)
		.def("__ne__", &NeuroFrame::Scalar::operator!=)
		.def("as_double", &NeuroFrame::Scalar::as_double)
		.def("as_int64", &NeuroFrame::Scalar::as_int64)
		.def("is_int_family", &NeuroFrame::Scalar::is_int_family)
		.def("is_float_family", &NeuroFrame::Scalar::is_float_family);		
}