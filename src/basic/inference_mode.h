// inference_mode.h - Manage the inference mode
#pragma once

namespace NeuroFrame {

// The number of inference mode entities
extern int inference_mode_entity_count;

// Return true if the inference mode is enabled (i.e. inference_mode_entity_count > 0)
bool is_inference_mode();

class InferenceModeGuard {
public:
	InferenceModeGuard();
	~InferenceModeGuard();
	void __enter__();
	void __exit__();
};

}
