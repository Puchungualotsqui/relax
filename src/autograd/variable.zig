const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Function = @import("function.zig").Function;
const engine = @import("engine.zig");
const Allocator = std.mem.Allocator;

pub fn Variable(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);
        const FuncT = Function(T);

        // The actual data lives on the heap, shared by all copies of this Variable
        pub const State = struct {
            data: TensorT,
            grad: ?TensorT = null,
            creator: ?*FuncT = null,
            requires_grad: bool,
            ref_count: usize,
            allocator: Allocator,
        };

        ptr: *State,

        /// Creates a new Variable. Takes ownership of 'data'.
        pub fn init(allocator: Allocator, data: TensorT, requires_grad: bool) !Self {
            const state = try allocator.create(State);
            state.* = .{
                .data = data,
                .grad = null,
                .creator = null,
                .requires_grad = requires_grad,
                .ref_count = 1,
                .allocator = allocator,
            };
            return Self{ .ptr = state };
        }

        /// Creates a new reference to the same variable (increments ref_count).
        /// Use this when storing the variable in a Graph Node or Optimizer list.
        pub fn clone(self: Self) Self {
            self.ptr.ref_count += 1;
            return Self{ .ptr = self.ptr };
        }

        /// Decrements ref_count. Frees memory if count reaches 0.
        pub fn deinit(self: Self) void {
            self.ptr.ref_count -= 1;
            if (self.ptr.ref_count == 0) {
                self.ptr.data.deinit();
                if (self.ptr.grad) |g| g.deinit();
                // Recursively free the creator (Graph cleanup)
                if (self.ptr.creator) |c| c.deinit();
                self.ptr.allocator.destroy(self.ptr);
            }
        }

        // --- Accessors (proxies to ptr) ---

        pub fn data_ptr(self: Self) TensorT {
            return self.ptr.data;
        }

        pub fn getGrad(self: Self) !*TensorT {
            if (self.ptr.creator) |c| {
                return c.getGrad();
            } else {
                if (self.ptr.grad == null) {
                    self.ptr.grad = try TensorT.init(self.ptr.allocator, self.ptr.data.shape);
                    self.ptr.grad.?.fill(0);
                }
                return &self.ptr.grad.?;
            }
        }

        pub fn zeroGrad(self: Self) !void {
            const g = try self.getGrad();
            g.fill(0);
        }

        pub fn backward(self: Self) !void {
            if (self.ptr.creator == null) return;
            const g = try self.getGrad();
            g.fill(1.0);

            var graph = try engine.topologicalSort(self.ptr.allocator, self.ptr.creator.?);
            // UPDATED: Pass allocator to deinit
            defer graph.deinit(self.ptr.allocator);

            for (graph.items) |node| {
                try node.backward();
            }
        }
    };
}
