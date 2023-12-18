#pragma once

#include <optional>

#include "src/tensor/tensor.h"
#include "cgraph_edge.h"

namespace NeuroFrame::CGraph {

struct CGraphEdge;

// CGraphNode: A node in the compute graph, (usually) a tensor
// This class contains information about the gradient & optimizer state of a tensor
// Every tensor has a field, std::shared_ptr<GradInfo> grad_info, that points to the GradInfo object
// Multiple tensors can share the same GradInfo object as long as they are the
// same tensor (i.e. they are created/assigned from the same tensor)
struct CGraphNode {
	std::optional<Tensor> grad;	// The gradient of the tensor that the CGraphNode represents

	// Ingress edge (op that generates the tensor)
	// and egress edges (ops that use the tensor)
	std::shared_ptr<CGraphEdge> ingress_edge;
	std::vector<std::shared_ptr<CGraphEdge>> egress_edges;

	// Tags related to topology sort
	int64_t num_ready_egress_edges;	// Number of ready egress edges
	bool is_all_egress_edges_ready() const;	// Check if all egress edges are ready (i.e. num_ready_egress_edges == egress_edges.size())

	CGraphNode();
	~CGraphNode();

	void reset();	// Reset the CGraphNode to its initial state
};

}