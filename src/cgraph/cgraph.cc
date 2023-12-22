#include "cgraph.h"

#include <cassert>
#include <functional>
#include <deque>

#include "src/op/op.h"
#include "src/op/tensor_binary_op.h"
#include "src/utils/utils.h"

#include "cgraph_edge.h"
#include "cgraph_node.h"

namespace NeuroFrame::CGraph {

// associated_nodes: A list of all nodes in the compute graph
// We note down all associated nodes so that we can clear them later
static std::vector<std::shared_ptr<CGraphNode>> associated_nodes;


// on_new_calculation: Called by perform_op when a new calculation is issued
// This function will construct a new edge in the compute graph
void on_new_calculation(
	std::vector<Tensor> &inputs,
	std::vector<Tensor> &outputs,
	const OpContext &ctx,
	op_backward_func_t backward_func
) {
	// printf("Fw. Input = %s, output = %s\n", vec_to_string(inputs).c_str(), vec_to_string(outputs).c_str()); 

	// Collect all input nodes & output nodes
	std::vector<std::shared_ptr<CGraphNode>> input_nodes;
	for (Tensor& input : inputs) {
		assert(input.cgraph_node);
		input_nodes.push_back(input.cgraph_node);
		associated_nodes.push_back(input.cgraph_node);
	}
	std::vector<std::shared_ptr<CGraphNode>> output_nodes;
	for (Tensor& output : outputs) {
		assert(output.cgraph_node);
		output_nodes.push_back(output.cgraph_node);
		associated_nodes.push_back(output.cgraph_node);
	}

	// Construct the edge
	std::shared_ptr<CGraphEdge> edge = std::make_shared<CGraphEdge>(CGraphEdge (
		input_nodes,
		output_nodes,
		ctx,
		backward_func
	));

	// Assign the edge to the input nodes
	for (std::shared_ptr<CGraphNode> &input_node : input_nodes) {
		input_node->egress_edges.push_back(edge);
	}

	// Assign the edge to the output nodes
	for (std::shared_ptr<CGraphNode> &output_node : output_nodes) {
		output_node->ingress_edge = edge;
	}
}


// perform_topology_sort: Perform topological sort on the compute graph, and
// invoke the given callbacks when a node/edge is ready
// 
// The `on_node_ready` function is called as long as the node is ready (i.e.
// all egress edges are ready). This means that its gradient is ready
// 
// The `on_edge_ready` function is called as long as the edge is ready (i.e.
// all output nodes are ready)
//
// This function is very flexible as it allows the user to specify what to do
// when a node/edge is ready. You may use it for backward propagation, or
// just returning the result of the topological sort
static void perform_topology_sort(
	Tensor &src,
	Tensor &src_grad,
	std::function<void(std::shared_ptr<CGraphNode>)> on_node_ready,
	std::function<void(std::shared_ptr<CGraphEdge>)> on_edge_ready
) {
	std::deque<std::shared_ptr<CGraphNode>> ready_nodes;
	src.cgraph_node->grad = std::make_optional(src_grad);
	ready_nodes.push_back(src.cgraph_node);
	on_node_ready(src.cgraph_node);

	while (!ready_nodes.empty()) {
		std::shared_ptr<CGraphNode> node = ready_nodes.front();
		ready_nodes.pop_front();

		if (node->ingress_edge) {
			node->ingress_edge->num_ready_output += 1;
			if (node->ingress_edge->is_all_output_ready()) {
				on_edge_ready(node->ingress_edge);
				for (std::shared_ptr<CGraphNode> &edge_input_node : node->ingress_edge->input_nodes) {
					edge_input_node->num_ready_egress_edges += 1;
					if (edge_input_node->is_all_egress_edges_ready()) {
						ready_nodes.push_back(edge_input_node);
						on_node_ready(edge_input_node);
					}
				}
			}
		}
	}
}


// perform_backward: Perform backward propagation, starting from the given tensor
std::vector<Tensor> perform_backward(Tensor &src, Tensor &src_grad, bool log_down_all_grads) {
	std::vector<Tensor> grads;
	perform_topology_sort(
		src,
		src_grad,
		[&grads, log_down_all_grads](std::shared_ptr<CGraphNode> node) {
			if (log_down_all_grads) {
				grads.push_back(node->grad.value());
			}
		},
		[&grads, log_down_all_grads](std::shared_ptr<CGraphEdge> edge) {
			std::vector<Tensor> output_grads;
			for (std::shared_ptr<CGraphNode> &output_node : edge->output_nodes) {
				output_grads.push_back(output_node->grad.value());
			}
			// printf("Bw. Output grads = %s", vec_to_string(output_grads).c_str()); fflush(stdout);
			std::vector<Tensor> input_grads = edge->backward_func(
				output_grads,
				edge->ctx
			);
			// printf(", input grads = %s\n", vec_to_string(input_grads).c_str());
			assert(input_grads.size() == edge->input_nodes.size());
			for (size_t i = 0; i < input_grads.size(); ++i) {
				if (!edge->input_nodes[i]->grad.has_value()) {
					edge->input_nodes[i]->grad = std::make_optional(input_grads[i].copy());
				} else {
					OpContext temp_ctx;
					edge->input_nodes[i]->grad.value() = tensor_add_forward_manual(
						edge->input_nodes[i]->grad.value(),
						input_grads[i],
						temp_ctx
					);
				}
			}
		}
	);
	return grads;
}


Tensor get_computed_grad(const Tensor &src) {
	if (!src.cgraph_node->grad.has_value()) {
		LOG_FATAL("The gradient of the given tensor is not computed");
	}
	return src.cgraph_node->grad.value();
}


// clear_graph: Clear the compute graph
void clear_graph() {
	for (std::shared_ptr<CGraphNode> &node : associated_nodes) {
		node->clear_cgraph_elements();
	}
	associated_nodes.clear();
}

}