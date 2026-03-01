/**
 * @file graph.cu
 * @brief Computation graph — topological sort and backward dispatch.
 */

#include "minitensor/autograd.hpp"
#include <unordered_set>
#include <stdexcept>

namespace minitensor {
namespace autograd {

// ─── Variable helpers ─────────────────────────────────────────────────────────
void Variable::zero_grad() {
    grad = Tensor::zeros(data.shape(), data.dtype(), data.device());
}

void Variable::accumulate_grad(const Tensor& g) {
    // Simple addition — for production, use a CUDA add kernel here
    int64_t n = data.numel();
    if (data.device() == Device::CPU) {
        float* gp = grad.data_ptr<float>();
        const float* gv = g.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) gp[i] += gv[i];
    } else {
        // GPU accumulation via thrust or custom kernel: simplified here
        // In production, call a launch_add kernel
        (void)g;
    }
}

// ─── Topological sort ─────────────────────────────────────────────────────────
/* static */
void ComputeGraph::topological_sort(
    const std::shared_ptr<Node>& node,
    std::vector<std::shared_ptr<Node>>& order,
    std::unordered_set<Node*>& visited)
{
    if (!node || visited.count(node.get())) return;
    visited.insert(node.get());
    for (auto& inp : node->inputs)
        topological_sort(inp, order, visited);
    order.push_back(node);
}

// ─── Backward ─────────────────────────────────────────────────────────────────
/* static */
void ComputeGraph::backward(Variable& root, const Tensor* grad_root) {
    if (!root.grad_fn && !root.is_leaf)
        throw std::runtime_error("backward: root variable has no grad_fn");

    // Initialize root gradient
    if (grad_root) {
        root.grad = *grad_root;
    } else {
        // Scalar loss: initial grad = 1
        root.grad = Tensor::ones(root.data.shape(),
                                 root.data.dtype(), root.data.device());
    }

    if (!root.grad_fn) return;  // leaf — nothing to propagate

    // Topological order
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;
    topological_sort(root.grad_fn, order, visited);

    // Map node → accumulated gradient tensor
    std::unordered_map<Node*, Tensor> node_grads;
    node_grads[root.grad_fn.get()] = root.grad;

    // Reverse topological order — backward pass
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        auto& node = *it;
        auto  gmap_it = node_grads.find(node.get());
        if (gmap_it == node_grads.end()) continue;

        Tensor& incoming_grad = gmap_it->second;

        if (!node->backward_fn) continue;

        std::vector<Tensor> input_grads = node->backward_fn(incoming_grad);

        for (size_t i = 0; i < node->inputs.size() && i < input_grads.size(); ++i) {
            auto& inp = node->inputs[i];
            if (!inp) continue;
            auto& ig = input_grads[i];
            auto  existing = node_grads.find(inp.get());
            if (existing == node_grads.end()) {
                node_grads[inp.get()] = ig;
            } else {
                // Accumulate gradients (needed for multi-use nodes, e.g. weight sharing)
                float* g_ptr  = existing->second.data_ptr<float>();
                const float* add_ptr = ig.data_ptr<float>();
                for (int64_t k = 0; k < ig.numel(); ++k) g_ptr[k] += add_ptr[k];
            }
        }
    }
}

// ─── make_variable_from_op ────────────────────────────────────────────────────
Variable make_variable_from_op(
    Tensor output,
    std::string op_name,
    BackwardFn backward_fn,
    std::vector<std::shared_ptr<Node>> input_nodes)
{
    Variable v(std::move(output), true);
    v.is_leaf  = false;
    v.grad_fn  = std::make_shared<Node>(std::move(op_name), std::move(backward_fn));
    v.grad_fn->inputs = std::move(input_nodes);
    return v;
}

} // namespace autograd
} // namespace minitensor
