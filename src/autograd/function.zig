const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

pub fn Function(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Calculate input_grads using the internal output_grad
        backward_fn: *const fn (self: *Self) anyerror!void,

        /// Graph Traversal
        collect_parents_fn: *const fn (self: *Self, list: *std.ArrayList(*Self)) anyerror!void,

        /// Accessor: Returns a pointer to the gradient tensor accumulated at this node
        get_grad_fn: *const fn (self: *Self) *Tensor(T),

        deinit_fn: *const fn (self: *Self) void,

        pub fn backward(self: *Self) !void {
            return self.backward_fn(self);
        }

        pub fn collectParents(self: *Self, list: *std.ArrayList(*Self)) !void {
            return self.collect_parents_fn(self, list);
        }

        pub fn getGrad(self: *Self) *Tensor(T) {
            return self.get_grad_fn(self);
        }

        pub fn deinit(self: *Self) void {
            self.deinit_fn(self);
        }
    };
}
