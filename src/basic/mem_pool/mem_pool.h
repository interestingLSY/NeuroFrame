#pragma once

#include "src/basic/device.h"
#include "abstract_mem_pool.h"

namespace NeuroFrame {

AbstractMemPool* get_mem_pool(const Device &device);

}
