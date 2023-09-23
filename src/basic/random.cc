#include "random.h"

namespace NeuroFrame {

std::mt19937_64 mt19937_64_engine(0);

void set_random_seed(uint64_t seed) {
	mt19937_64_engine.seed(seed);
}

}