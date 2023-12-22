#include "optim.h"

namespace NeuroFrame {

Optimizer::Optimizer() {}
Optimizer::~Optimizer() {}

void Optimizer::add_to_focus_list(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	for (const auto &focused_node : focused_nodes) {
		if (focused_node.cgraph_node == node) {
			LOG_FATAL("The tensor is already in the optimizer's focus list");
		}
	}
	focused_nodes.push_back(tensor);
}

void Optimizer::remove_from_focus_list(const Tensor &tensor) {
	std::shared_ptr<CGraph::CGraphNode> node = tensor.cgraph_node;
	for (auto it = focused_nodes.begin(); it != focused_nodes.end(); ++it) {
		if (it->cgraph_node == node) {
			focused_nodes.erase(it);
			return;
		}
	}
	LOG_FATAL("Failed to remove a focused tensor: The tensor is not in the optimizer's focus list");
}

}