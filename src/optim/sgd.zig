const std = @import("std");
const Variable = @import("../../autograd/variable.zig").Variable;
const Allocator = std.mem.Allocator;

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();
        const VarT = Variable(T);

        params: std.ArrayList(VarT),
        learning_rate: T,
        allocator: Allocator,

        /// Initialize Optimizer.
        /// 'params' is the list returned by model.parameters().
        /// The optimizer takes ownership of this list (it's a list of ref-counted variables).
        pub fn init(allocator: Allocator, params: std.ArrayList(VarT), lr: T) Self {
            return Self{
                .params = params,
                .learning_rate = lr,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            // release our references to the parameters
            for (self.params.items) |p| {
                p.deinit();
            }
            self.params.deinit();
        }

        /// Clears gradients for all parameters.
        /// Call this before loss.backward().
        pub fn zeroGrad(self: Self) !void {
            for (self.params.items) |p| {
                try p.zeroGrad();
            }
        }

        /// Performs a single optimization step (Weight Update).
        pub fn step(self: Self) !void {
            for (self.params.items) |p| {
                // Skip parameters that don't need gradients
                if (!p.ptr.requires_grad) continue;

                // Get the gradient (might be null if backward wasn't called)
                if (p.ptr.grad) |grad| {
                    const param_data = p.ptr.data.data;
                    const grad_data = grad.data;
                    const lr = self.learning_rate;

                    // Update Rule: p = p - lr * grad
                    // We do this loop manually for performance and simplicity
                    for (0..param_data.len) |i| {
                        param_data[i] -= lr * grad_data[i];
                    }
                }
            }
        }
    };
}
