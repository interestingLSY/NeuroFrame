#include "cgraph_node.h"

namespace NeuroFrame::CGraph {

CGraphNode::CGraphNode()
{
	this->reset();
}

CGraphNode::~CGraphNode() {
}

bool CGraphNode::is_all_egress_edges_ready() const {
	return num_ready_egress_edges == (int64_t)egress_edges.size();
}

void CGraphNode::reset() {
	grad = std::nullopt;
	ingress_edge = nullptr;
	egress_edges.clear();
	num_ready_egress_edges = 0;
}

}
