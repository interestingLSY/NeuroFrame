#include "tensor.h"

#include <functional>

#include "src/basic/random.h"
#include "src/op/tensor_eq.h"
#include "src/op/tensor_unary_op.h"
#include "src/op/tensor_binary_op.h"
#include "src/cgraph/cgraph_node.h"

namespace NeuroFrame {

Tensor::Tensor(const MemFrag &frag, const Device &dev, const dtype_t &dtype, const int64_t &first_elem_offset, const std::vector<int64_t> &shape):
	mem_frag(frag),
	device(dev),
	dtype(dtype),
	first_elem_offset(first_elem_offset),
	shape(shape),
	cgraph_node(std::make_shared<CGraph::CGraphNode>(CGraph::CGraphNode()))
{
	// Calculate stride
	stride.resize(shape.size());
	if (!shape.empty()) {
		stride[shape.size() - 1] = 1;
		for (int i = shape.size() - 2; i >= 0; i--) {
			stride[i] = stride[i+1] * shape[i+1];
		}
	}
}

int64_t Tensor::get_elem_offset(const std::vector<int64_t> &pos) const {
	if (pos.size() != shape.size()) {
		LOG_FATAL("The number of dimensions of the position vector does not match the number of dimensions of the tensor");
	}
	int64_t offset = first_elem_offset;
	for (int i = 0; i < (int)pos.size(); i++) {
		offset += pos[i] * stride[i];
	}
	return offset;
}

Tensor& Tensor::operator=(const Scalar &other) {
	if (shape.empty()) {
		// Scalar tensor
		Device device = this->device;
		if (device.type != device_type_t::CPU) {
			static char buf[MAX_DTYPE_SIZE];
			other.save_to(buf, dtype);
			NeuroFrame::memcpy(
				this->get_elem_addr({}),
				this->device,
				buf,
				Device::cpu(),
				get_dtype_size(dtype)
			);
		} else {
			other.save_to(this->get_elem_addr({}), dtype);
		}
		return *this;
	} else {
		LOG_FATAL("Cannot assign a scalar to a non-scalar tensor");
	}
}

int64_t Tensor::numel() const {
	return get_product_over_vector(this->shape);
}

void* Tensor::data_ptr() const {
	return (char*)mem_frag.ptr + first_elem_offset * get_dtype_size(dtype);
}

int64_t Tensor::dim() const {
	return shape.size();
}

Tensor Tensor::reshape(const std::vector<int64_t> &new_shape) const {
	if (get_product_over_vector(new_shape) != numel()) {
		LOG_FATAL("The number of elements in the new shape does not match the number of elements in the tensor");
	}
	return Tensor(mem_frag, device, dtype, first_elem_offset, new_shape);
}

Tensor Tensor::copy() const {
	Tensor ret(shape, dtype, device);
	NeuroFrame::memcpy(
		(char*)ret.mem_frag.ptr,
		ret.device,
		(char*)this->mem_frag.ptr + first_elem_offset * get_dtype_size(dtype),
		this->device,
		this->numel() * get_dtype_size(dtype)
	);
	return ret;
}

void* Tensor::get_elem_addr(const std::vector <int64_t> &pos) const {
	int64_t offset = get_elem_offset(pos);
	return (char*)mem_frag.ptr + offset * get_dtype_size(dtype);
}

Tensor Tensor::get_elem(const std::vector<int64_t> &pos) const {
	int64_t offset = get_elem_offset(pos);
	return Tensor(mem_frag, device, dtype, offset, {});
}

Scalar Tensor::as_scalar() const {
	if (device.type != device_type_t::CPU) {
		Tensor t = this->to(Device::cpu());
		return t.as_scalar();
	}
	if (shape.empty()) {
		return Scalar(this->get_elem_addr({}), dtype);
	} else {
		LOG_FATAL("Cannot convert a non-scalar tensor to a scalar");
	}
}

Tensor Tensor::to(Device target) const {
	if (device == target) {
		// No need to migrate
		return *this;
	} else {
		if (shape.size() == 0) {
			// Scala tensor
			Tensor ret({}, dtype, target);
			NeuroFrame::memcpy(
				ret.mem_frag.ptr,
				ret.device,
				this->get_elem_addr({}),
				this->device,
				get_dtype_size(dtype)
			);
			return ret;
		} else {
			Tensor ret(shape, dtype, target);
			NeuroFrame::memcpy(
				(char*)ret.mem_frag.ptr,
				ret.device,
				(char*)this->mem_frag.ptr + first_elem_offset * get_dtype_size(dtype),
				this->device,
				numel() * get_dtype_size(dtype)
			);
			return ret;
		}
	}
}

Tensor Tensor::cpu() const {
	return this->to(Device::cpu());
}

Tensor Tensor::cuda(int device_index) const {
	return this->to(Device::cuda(device_index));
}


std::string Tensor::to_string(int64_t max_display_per_dim /* default: 16 */, bool in_compat_style /* default: false*/) const {
	std::string result = "Tensor(";
	if (shape.empty()) {
		if (this->device.type != device_type_t::CPU) {
			Tensor t = this->to(Device::cpu());
			// Scalar tensor
			Scalar s = t.as_scalar();
			result += s.to_string();
			result += " (scalar)";
		} else {
			// Scalar tensor
			Scalar s = this->as_scalar();
			result += s.to_string();
			result += " (scalar)";
		}
	} else {
		// Non-scalar tensor
		std::function<void(int, const std::vector<int64_t> &)> print_helper = [&](int cur_dim, const std::vector<int64_t> &pos) {
			if (cur_dim == (int)shape.size()) {
				// We have reached the last dimension
				Scalar s = this->get_elem(pos).as_scalar();
				result += s.to_string();
			} else {
				// We have not reached the last dimension
				result += "[";
				for (int64_t i = 0; i < shape[cur_dim]; ++i) {
					if (i != 0) {
						result += ", ";
					}
					if (cur_dim != (int)shape.size() - 1 && !in_compat_style) {
						result += "\n";
						for (int j = 0; j < cur_dim + 1; ++j) {
							result += "  ";
						}
					}
					std::vector<int64_t> new_pos = pos;
					new_pos.push_back(i);
					print_helper(cur_dim + 1, new_pos);
					if (i == max_display_per_dim - 1 && i != shape[cur_dim] - 1) {
						result += ", ...";
						break;
					}
				}
				result += "]";
			}
		};
		print_helper(0, {});
		result += ", shape=[";
		for (int i = 0; i < (int)shape.size(); ++i) {
			result += std::to_string(shape[i]);
			if (i != (int)shape.size() - 1) {
				result += ", ";
			}
		}
		result += "], stride=[";
		for (int i = 0; i < (int)stride.size(); ++i) {
			result += std::to_string(stride[i]);
			if (i != (int)stride.size() - 1) {
				result += ", ";
			}
		}
		result += "]";
	}
	result += ", device=";
	result += device.to_string();
	result += ", dtype=";
	result += dtype2string(dtype);
	result += ")";
	return result;
}

std::string Tensor::repr() const {
	return "<Tensor " + this->to_string() + ">";
}

void Tensor::print(int64_t max_display_per_dim /* default: 16 */, bool in_compat_style /* default: false*/) const {
	printf("%s", this->to_string(max_display_per_dim, in_compat_style).c_str());
}

Tensor::Tensor(const std::vector<int64_t> &shape, dtype_t dtype, Device device):
	mem_frag(device, get_product_over_vector(shape) * get_dtype_size(dtype)),
	device(device),
	dtype(dtype),
	first_elem_offset(0),
	shape(shape),
	cgraph_node(std::make_shared<CGraph::CGraphNode>(CGraph::CGraphNode()))
{
	// Calculate stride
	stride.resize(shape.size());
	if (!shape.empty()) {
		stride[shape.size() - 1] = 1;
		for (int i = shape.size() - 2; i >= 0; i--) {
			stride[i] = stride[i+1] * shape[i+1];
		}
	}
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, dtype_t dtype, Device device) {
	Tensor ret(shape, dtype, device);
	// Now `ret` must be continuous, so we use a single memset
	NeuroFrame::memset(ret.mem_frag.ptr, ret.device, ret.numel() * get_dtype_size(dtype), 0);
	return ret;
}

Tensor Tensor::fill(Scalar x, const std::vector<int64_t> &shape, dtype_t dtype, Device device) {
	Tensor ret_h(shape, dtype, Device::cpu());
	for (int64_t i = 0; i < ret_h.numel(); ++i) {
		x.save_to((char*)ret_h.mem_frag.ptr + i * get_dtype_size(dtype), dtype);
	}
	return ret_h.to(device);
}

Tensor Tensor::randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device,
	Scalar low, Scalar high) {
	Tensor ret_cpu(shape, dtype, Device::cpu());
	double low_d = low.as_double();
	double high_d = high.as_double();
	int64_t numel = ret_cpu.numel();
	// #pragma omp parallel for schedule(static)	Disabled since it may introduce non-determinism
	for (int64_t i = 0; i < numel; ++i) {
		double x = std::uniform_real_distribution<double>(low_d, high_d)(mt19937_64_engine);
		Scalar s(x, dtype);
		s.save_to((char*)ret_cpu.mem_frag.ptr + i * get_dtype_size(dtype), dtype);
	}
	Tensor ret = ret_cpu.to(device);
	return ret;
}

Tensor Tensor::randu(const std::vector<int64_t> &shape, dtype_t dtype, Device device) {
	return randu(shape, dtype, device, Scalar(-1.0f, dtype), Scalar(1.0f, dtype));
}

Tensor Tensor::randint(const std::vector<int64_t> &shape, dtype_t dtype, Device device,
	Scalar low, Scalar high) {
	Tensor ret_cpu(shape, dtype, Device::cpu());
	int64_t low_d = low.as_int64();
	int64_t high_d = high.as_int64();
	int64_t numel = ret_cpu.numel();
	// #pragma omp parallel for schedule(static) Disabled since it may introduce non-determinism
	for (int64_t i = 0; i < numel; ++i) {
		int64_t x = std::uniform_int_distribution<int64_t>(low_d, high_d)(mt19937_64_engine);
		Scalar s(x, dtype);
		s.save_to((char*)ret_cpu.mem_frag.ptr + i * get_dtype_size(dtype), dtype);
	}
	Tensor ret = ret_cpu.to(device);
	return ret;
}

Tensor Tensor::from_vector(const std::vector<Scalar> &data, const std::vector<int64_t> &shape, dtype_t dtype, Device device) {
	Tensor ret_h(shape, dtype, Device::cpu());
	int64_t numel = ret_h.numel();
	if (numel != (int64_t)data.size()) {
		LOG_FATAL("The number of elements in the data vector does not match the number of elements in the tensor");
	}
	for (int64_t i = 0; i < numel; ++i) {
		Scalar s = data[i].to_dtype(dtype);
		s.save_to((char*)ret_h.mem_frag.ptr + i * get_dtype_size(dtype), dtype);
	}
	return ret_h.to(device);
}

bool Tensor::operator==(const Tensor &other) const {
	return tensor_eq(*this, other);
}

bool Tensor::operator!=(const Tensor &other) const {
	return !tensor_eq(*this, other);
}

Tensor Tensor::operator+(const Tensor &other) const {
	return tensor_add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
	return tensor_sub(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
	return tensor_mul(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
	return tensor_div(*this, other);
}

Tensor Tensor::operator-() const {
	return tensor_negate(*this);
}

std::ostream& operator<<(std::ostream &os, const Tensor &tensor) {
	os << tensor.to_string();
	return os;
}

}