const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;
const Function = @import("function.zig").Function;
const Allocator = std.mem.Allocator;

pub fn AddBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,

        input_a: *Variable(T),
        input_b: *Variable(T),

        // The node owns the gradient for the variable it created!
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);

            // We use our OWN stored gradient
            const grad = self.output_grad;

            if (self.input_a.requires_grad) {
                // We ask the parent for its gradient pointer (handles leaf/non-leaf logic)
                const a_grad = try self.input_a.getGrad();
                try a_grad.addInPlace(grad);
            }

            if (self.input_b.requires_grad) {
                const b_grad = try self.input_b.getGrad();
                try b_grad.addInPlace(grad);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), list: *std.ArrayList(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input_a.creator) |c| try list.append(c);
            if (self.input_b.creator) |c| try list.append(c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            // We own this tensor, so we free it
            self.output_grad.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn add(allocator: Allocator, a: anytype, b: anytype) !@TypeOf(a.*) {
    const T = @TypeOf(a.data.data[0]);
    const VarT = Variable(T);
    const OpT = AddBackward(T);

    // 1. Forward
    const data = try a.data.add(b.data, allocator);

    // 2. Build Graph if needed
    const needs_grad = a.requires_grad or b.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(OpT);

        // Initialize the gradient tensor for this node (zeroed)
        const grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0); // Important!

        op.* = .{
            .base = .{
                .backward_fn = OpT.backward,
                .collect_parents_fn = OpT.collectParents,
                .get_grad_fn = OpT.getGrad,
                .deinit_fn = OpT.deinit,
            },
            .allocator = allocator,
            .input_a = a,
            .input_b = b,
            .output_grad = grad,
        };
        creator = &op.base;
    }

    // 3. Output
    // We do NOT store 'out's address in the op.
    var out = VarT.init(allocator, data, needs_grad);
    out.creator = creator;
    return out;
}

pub fn MatMulBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,

        input_a: *Variable(T),
        input_b: *Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad; // G (dL/dC)

            // 1. Calculate dL/dA = G @ B^T
            if (self.input_a.requires_grad) {
                // Create transpose permutation for B: swap last two dims
                const ndim = self.input_b.data.shape.len;
                var dims = try self.allocator.alloc(usize, ndim);
                defer self.allocator.free(dims);

                // Initialize [0, 1, 2...]
                for (0..ndim) |i| dims[i] = i;
                // Swap last two for transpose
                std.mem.swap(usize, &dims[ndim - 1], &dims[ndim - 2]);

                // Create view B^T
                var b_T = try self.input_b.data.permute(dims);
                defer b_T.deinit(); // Decrement ref count on view

                // Allocate temp gradient for A
                var d_a = try TensorT.init(self.allocator, self.input_a.data.shape);
                defer d_a.deinit();

                // Perform MatMul: G @ B^T
                try grad.matmul(b_T, &d_a);

                // Accumulate
                const a_grad = try self.input_a.getGrad();
                try a_grad.addInPlace(d_a);
            }

            // 2. Calculate dL/dB = A^T @ G
            if (self.input_b.requires_grad) {
                const ndim = self.input_a.data.shape.len;
                var dims = try self.allocator.alloc(usize, ndim);
                defer self.allocator.free(dims);
                for (0..ndim) |i| dims[i] = i;
                std.mem.swap(usize, &dims[ndim - 1], &dims[ndim - 2]);

                var a_T = try self.input_a.data.permute(dims);
                defer a_T.deinit();

                var d_b = try TensorT.init(self.allocator, self.input_b.data.shape);
                defer d_b.deinit();

                // Perform MatMul: A^T @ G
                try a_T.matmul(grad, &d_b);

                const b_grad = try self.input_b.getGrad();
                try b_grad.addInPlace(d_b);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), list: *std.ArrayList(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input_a.creator) |c| try list.append(c);
            if (self.input_b.creator) |c| try list.append(c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.allocator.destroy(self);
        }
    };
}

/// Autograd-aware Matrix Multiplication
pub fn matmul(allocator: Allocator, a: anytype, b: anytype) !@TypeOf(a.*) {
    const T = @TypeOf(a.data.data[0]);
    const VarT = Variable(T);
    const OpT = MatMulBackward(T);

    // 1. Compute Forward: C = A @ B
    // We need to calculate output shape first to allocate result
    // (Or let tensor.matmul handle allocation? Your tensor api takes a dest pointer)

    // Quick shape calc logic (assuming 2D for simplicity, or copying logic from tensor.zig)
    const a_ndim = a.data.shape.len;
    const b_ndim = b.data.shape.len;
    // ... basic validation omitted, relying on tensor.matmul to fail if wrong ...

    // Create output shape: A[...:-1] + B[-1]
    var out_shape = try allocator.alloc(usize, a_ndim); // simplified rank
    defer allocator.free(out_shape);
    @memcpy(out_shape, a.data.shape);
    out_shape[a_ndim - 1] = b.data.shape[b_ndim - 1];

    var data = try Tensor(T).init(allocator, out_shape);
    try a.data.matmul(b.data, &data);

    // 2. Build Graph
    const needs_grad = a.requires_grad or b.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(OpT);
        const grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = OpT.backward,
                .collect_parents_fn = OpT.collectParents,
                .get_grad_fn = OpT.getGrad,
                .deinit_fn = OpT.deinit,
            },
            .allocator = allocator,
            .input_a = a,
            .input_b = b,
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = VarT.init(allocator, data, needs_grad);
    out.creator = creator;
    return out;
}

/// Backward Node for ReLU: y = max(0, x)
/// Gradient: dL/dx = dL/dy * (1 if x > 0 else 0)
pub fn ReLUBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: *Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad; // dL/dy

            if (self.input.requires_grad) {
                // mask = (input > 0) ? 1 : 0
                // d_input = grad * mask

                // We allocate a temporary tensor for the gradient contribution
                var d_input = try TensorT.init(self.allocator, self.input.data.shape);
                defer d_input.deinit();

                // Optimized loop: d_input[i] = (input[i] > 0) ? grad[i] : 0
                // Assuming contiguous for simplicity, but using flat iterators works generally
                // if we had a flat iterator. For now, we assume contiguous or compatible strides.
                // A robust implementation would use a broadcast kernel.

                // Use a direct loop over data slice for performance (assuming contiguous)
                // If not contiguous, we should use iterators.
                const count = self.input.data.data.len;
                for (0..count) |i| {
                    const val = self.input.data.data[i];
                    const g = grad.data[i];
                    d_input.data[i] = if (val > 0) g else 0;
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), list: *std.ArrayList(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.creator) |c| try list.append(c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.allocator.destroy(self);
        }
    };
}

/// Autograd-aware ReLU
pub fn relu(allocator: Allocator, a: anytype) !@TypeOf(a.*) {
    const T = @TypeOf(a.data.data[0]);
    const VarT = Variable(T);
    const OpT = ReLUBackward(T);

    // 1. Forward: y = clipped(x, 0, inf)
    // We use the Tensor method 'clipped'
    const data = try a.data.clipped(allocator, 0, std.math.inf(T));

    // 2. Build Graph
    const needs_grad = a.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(OpT);
        const grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = OpT.backward,
                .collect_parents_fn = OpT.collectParents,
                .get_grad_fn = OpT.getGrad,
                .deinit_fn = OpT.deinit,
            },
            .allocator = allocator,
            .input = a,
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = VarT.init(allocator, data, needs_grad);
    out.creator = creator;
    return out;
}
