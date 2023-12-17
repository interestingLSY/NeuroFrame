#include "inference_mode.h"

#include <cstdio>

namespace NeuroFrame {

int inference_mode_entity_count = 0;

bool is_inference_mode() {
	return inference_mode_entity_count > 0;
}

InferenceModeGuard::InferenceModeGuard() {}

InferenceModeGuard::~InferenceModeGuard() {}

void InferenceModeGuard::__enter__() {
	inference_mode_entity_count++;
}

void InferenceModeGuard::__exit__() {
	inference_mode_entity_count--;
}

}
