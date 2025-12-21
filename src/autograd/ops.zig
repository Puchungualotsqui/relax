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
