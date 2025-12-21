const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

pub fn Function(comptime T: type) type {
    return struct {
        const Self = @This();

        backward_fn: *const fn (self: *Self) anyerror!void,

        // UPDATED: Now takes Allocator and ArrayListUnmanaged
        collect_parents_fn: *const fn (self: *Self, allocator: Allocator, list: *std.ArrayListUnmanaged(*Self)) anyerror!void,

        get_grad_fn: *const fn (self: *Self) *Tensor(T),
        deinit_fn: *const fn (self: *Self) void,

        pub fn backward(self: *Self) !void {
            return self.backward_fn(self);
        }

        // Wrapper passes allocator through
        pub fn collectParents(self: *Self, allocator: Allocator, list: *std.ArrayListUnmanaged(*Self)) !void {
            return self.collect_parents_fn(self, allocator, list);
        }

        pub fn getGrad(self: *Self) *Tensor(T) {
            return self.get_grad_fn(self);
        }

        pub fn deinit(self: *Self) void {
            self.deinit_fn(self);
        }
    };
}
