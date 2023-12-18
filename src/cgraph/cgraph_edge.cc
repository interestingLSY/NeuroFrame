#include "cgraph_edge.h"

namespace NeuroFrame::CGraph {
	
int64_t CGraphEdge::instance_count = 0;

CGraphEdge::CGraphEdge(
	std::vector<std::shared_ptr<CGraphNode>> input_nodes,
	std::vector<std::shared_ptr<CGraphNode>> output_nodes,
	OpContext ctx,
	op_backward_func_t backward_func
) : input_nodes(input_nodes),
	output_nodes(output_nodes),
	ctx(ctx),
	backward_func(backward_func)
{
	num_ready_output = 0;
	instance_count += 1;
}

CGraphEdge::~CGraphEdge() {
	instance_count -= 1;
}

bool CGraphEdge::is_all_output_ready() const {
	return num_ready_output == (int64_t)output_nodes.size();
}

}