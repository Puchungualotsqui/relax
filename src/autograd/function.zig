const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

/// The Interface for a Backward Operation (Graph Node)
pub fn Function(comptime T: type) type {
    return struct {
        const Self = @This();
        const Allocator = std.mem.Allocator;

        // V-Table (Function Pointers)

        /// The actual logic: calculate input_grads given output_grad
        backward_fn: *const fn (self: *Self, grad_output: Tensor(T)) anyerror!void,

        /// Cleanup logic: free any captured tensors or parents list
        deinit_fn: *const fn (self: *Self) void,

        // Public Wrappers

        pub fn backward(self: *Self, grad_output: Tensor(T)) !void {
            return self.backward_fn(self, grad_output);
        }

        pub fn deinit(self: *Self) void {
            self.deinit_fn(self);
        }
    };
}
