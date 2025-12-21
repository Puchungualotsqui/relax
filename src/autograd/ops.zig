const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;
const Function = @import("function.zig").Function;
const Allocator = std.mem.Allocator;

/// Backward Node for Addition: c = a + b
/// Gradients: grad_a += grad_c, grad_b += grad_c
pub fn AddBackward(comptime T: type) type {
    return struct {
        // Inheritance: Embed the interface
        base: Function(T),

        allocator: Allocator,
        // Pointers to parents to update their gradients
        input_a: *Variable(T),
        input_b: *Variable(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T), grad_output: TensorT) !void {
            // Downcast from interface to concrete struct
            const self: *Self = @fieldParentPtr("base", ptr);

            // 1. Update A's Gradient
            if (self.input_a.requires_grad) {
                // Ensure grad tensor is allocated
                try self.input_a.zeroGrad();

                // Accumulate: grad_a += grad_output * 1.0
                // Note: In a full framework, we would handle broadcasting here
                // (summing axes if shape_a != shape_output).
                // For now, we assume shapes match.
                try self.input_a.grad.?.addInPlace(grad_output);
            }

            // 2. Update B's Gradient
            if (self.input_b.requires_grad) {
                try self.input_b.zeroGrad();
                try self.input_b.grad.?.addInPlace(grad_output);
            }
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.allocator.destroy(self);
        }
    };
}

/// The User-Facing Forward Function
pub fn add(allocator: Allocator, a: anytype, b: anytype) !@TypeOf(a.*) {
    const T = @TypeOf(a.data.data[0]);
    const VarT = Variable(T);

    // 1. Compute Forward Data (using your existing Tensor engine)
    const data = try a.data.add(b.data, allocator);

    // 2. Check if we need to build the graph
    const needs_grad = a.requires_grad or b.requires_grad;

    var creator: ?*Function(T) = null;

    if (needs_grad) {
        // 3. Allocate and Setup Backward Node
        const op = try allocator.create(AddBackward(T));
        op.* = .{
            .base = .{
                .backward_fn = AddBackward(T).backward,
                .deinit_fn = AddBackward(T).deinit,
            },
            .allocator = allocator,
            .input_a = a,
            .input_b = b,
        };
        creator = &op.base;
    }

    // 4. Return Result
    var out = VarT.init(allocator, data, needs_grad);
    out.creator = creator;
    return out;
}
