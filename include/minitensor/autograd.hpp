/**
 * @file autograd.hpp
 * @brief Tape-based automatic differentiation graph.
 *
 * Each `Variable` wraps a `Tensor` and carries a pointer to the `Node`
 * that produced it.  Calling `backward(root)` on the output Variable
 * performs a reverse-mode pass to accumulate gradients into leaf variables.
 *
 * Usage:
 * @code
 *   Variable x({tensor_data, requires_grad=true});
 *   Variable w({weight_data, requires_grad=true});
 *
 *   Variable y = linear_forward(x, w, b);  // registers nodes
 *   Variable loss = mse_loss(y, target);
 *
 *   backward(loss);   // populates x.grad, w.grad, b.grad
 * @endcode
 */
#pragma once

#include "minitensor/tensor.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace minitensor {
namespace autograd {

// ─── Forward declarations ─────────────────────────────────────────────────────
struct Node;
class Variable;

// ─── BackwardFn ──────────────────────────────────────────────────────────────
/**
 * @brief A backward function associated with a computation node.
 *
 * Takes the incoming gradient (grad_output) and returns a vector of
 * gradients w.r.t. each input of the forward operation.
 */
using BackwardFn = std::function<std::vector<Tensor>(const Tensor& grad_output)>;

// ─── Node ────────────────────────────────────────────────────────────────────
/**
 * @brief A node in the dynamic computation graph.
 *
 * Created automatically by differentiable operations when any input
 * has requires_grad=true.
 */
struct Node {
    std::string  name;                     ///< Debug name (e.g. "MatMul", "ReLU")
    BackwardFn   backward_fn;              ///< Backward function for this op
    std::vector<std::shared_ptr<Node>> inputs; ///< Input nodes (for graph traversal)
    std::vector<std::shared_ptr<Tensor>> input_grads; ///< Accumulated input grads

    explicit Node(std::string name, BackwardFn fn)
        : name(std::move(name)), backward_fn(std::move(fn)) {}
};

// ─── Variable ────────────────────────────────────────────────────────────────
/**
 * @brief Differentiable tensor with autograd support.
 *
 * A Variable is a thin wrapper around Tensor that tracks the computation
 * graph for automatic differentiation.
 */
class Variable {
public:
    Tensor data;                          ///< Underlying tensor
    Tensor grad;                          ///< Gradient (populated after backward)
    bool   requires_grad = false;         ///< Whether to track gradients

    std::shared_ptr<Node> grad_fn;        ///< Node that produced this variable
    bool is_leaf = true;                  ///< True if created by user (not by an op)

    // ── Constructors ─────────────────────────────────────────────────────
    Variable() = default;

    explicit Variable(Tensor data, bool requires_grad = false)
        : data(std::move(data)),
          grad(Tensor::zeros(this->data.shape(), this->data.dtype(), this->data.device())),
          requires_grad(requires_grad),
          is_leaf(true)
    {}

    // ── Grad helpers ─────────────────────────────────────────────────────
    void zero_grad();
    void accumulate_grad(const Tensor& g);

    // ── Utility ───────────────────────────────────────────────────────────
    const std::vector<int64_t>& shape()  const { return data.shape(); }
    Device device()                       const { return data.device(); }
    int64_t numel()                        const { return data.numel(); }
};

// ─── ComputeGraph ────────────────────────────────────────────────────────────
/**
 * @brief Static utility that performs backpropagation through a Variable graph.
 *
 * Performs topological sort from root and calls each node's backward_fn
 * in reverse order, accumulating gradients into leaf Variable::grad tensors.
 */
class ComputeGraph {
public:
    /**
     * @brief Runs backward pass from @p root with initial gradient.
     * @param root      The scalar loss variable
     * @param grad_root Initial gradient (defaults to ones for scalar loss)
     */
    static void backward(Variable& root,
                         const Tensor* grad_root = nullptr);

private:
    static void topological_sort(
        const std::shared_ptr<Node>& node,
        std::vector<std::shared_ptr<Node>>& order,
        std::unordered_set<Node*>& visited);
};

// ─── Convenience free function ────────────────────────────────────────────────
/**
 * @brief Triggers backward pass on a scalar Variable.
 * Equivalent to ComputeGraph::backward(root).
 */
inline void backward(Variable& root) {
    ComputeGraph::backward(root);
}

// ─── Op registration helpers (used by layer implementations) ─────────────────
/**
 * @brief Creates a new Variable that is the output of a differentiable op,
 *        with the given backward function linking it to its inputs.
 */
Variable make_variable_from_op(
    Tensor output,
    std::string op_name,
    BackwardFn backward_fn,
    std::vector<std::shared_ptr<Node>> input_nodes);

} // namespace autograd
} // namespace minitensor
