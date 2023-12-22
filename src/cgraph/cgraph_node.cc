#include "cgraph_node.h"

namespace NeuroFrame::CGraph {

CGraphNode::CGraphNode()
{
	this->clear_cgraph_elements();
}

CGraphNode::~CGraphNode() {
}

bool CGraphNode::is_all_egress_edges_ready() const {
	return num_ready_egress_edges == (int64_t)egress_edges.size();
}

void CGraphNode::clear_cgraph_elements() {
	grad = std::nullopt;
	ingress_edge = nullptr;
	egress_edges.clear();
	num_ready_egress_edges = 0;
}

void CGraphNode::reset_optim_state() {
	optim_state.reset();
}

void CGraphNode::reset_all()  {
	this->clear_cgraph_elements();
	this->reset_optim_state();
}

}
