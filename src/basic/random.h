#pragma once

#include <random>

namespace NeuroFrame {

extern std::mt19937_64 mt19937_64_engine;

void set_random_seed(uint64_t seed);

}
