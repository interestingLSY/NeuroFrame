#pragma once

#include "src/op/op.h"
#include "cgraph_node.h"

namespace NeuroFrame::CGraph {

struct CGraphNode;

// CGraphEdge: An edge in the compute graph
// In the compute graph, "edge" is an operation that involves multiple (>= 1) 
// nodes as input and multiple (>= 1) nodes as output
// It also contains the backward function of the operation, as well as the
// context of the operation (OpContext)
struct CGraphEdge {
	// Basic info
	std::vector<std::shared_ptr<CGraphNode>> input_nodes;
	std::vector<std::shared_ptr<CGraphNode>> output_nodes;
	OpContext ctx;
	op_backward_func_t backward_func;
	
	// Tags related to topology sort
	int64_t num_ready_output;			// Number of ready output nodes
	bool is_all_output_ready() const;	// Check if all output nodes are ready (i.e. num_ready_output == output_nodes.size())

	static int64_t instance_count;	// Number of instances of CGraphEdge

	CGraphEdge(
		std::vector<std::shared_ptr<CGraphNode>> input_nodes,
		std::vector<std::shared_ptr<CGraphNode>> output_nodes,
		OpContext ctx,
		op_backward_func_t backward_func
	);

	~CGraphEdge();
};

}