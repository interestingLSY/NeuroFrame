#pragma once

#include <optional>

#include "src/tensor/tensor.h"
#include "src/optim/optim_state.h"
#include "cgraph_edge.h"

namespace NeuroFrame::CGraph {

struct CGraphEdge;

// CGraphNode: A node in the compute graph, (usually) a tensor
// This class contains information about the gradient & optimizer state of a tensor
// Every tensor has a field, std::shared_ptr<GradInfo> grad_info, that points to the GradInfo object
// Multiple tensors can share the same GradInfo object as long as they are the
// same tensor (i.e. they are created/assigned from the same tensor)
struct CGraphNode {
	// Whether it is focused by the optimizer
	bool is_focused;

	// The gradient of the tensor that the CGraphNode represents
	std::optional<Tensor> grad;

	// The optimizer state of the tensor that the CGraphNode represents
	// Can be null if the tensor is not focused by any optimizer
	std::shared_ptr<OptimStateBase> optim_state;

	// Ingress edge (op that generates the tensor)
	// and egress edges (ops that use the tensor)
	std::shared_ptr<CGraphEdge> ingress_edge;
	std::vector<std::shared_ptr<CGraphEdge>> egress_edges;

	// Tags related to topology sort
	int64_t num_ready_egress_edges;	// Number of ready egress edges
	bool is_all_egress_edges_ready() const;	// Check if all egress edges are ready (i.e. num_ready_egress_edges == egress_edges.size())

	CGraphNode();
	~CGraphNode();

	void clear_cgraph_elements();	// Reset all fields related to compute graph calculation (exclude optimizer states)
	void reset_optim_state();		// Reset the optimizer state of the CGraphNode to its initial state
	void reset_all();				// Reset the CGraphNode to its initial state
};

}