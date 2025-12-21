const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Function = @import("function.zig").Function;
const engine = @import("engine.zig"); // Import the topological sort engine
const Allocator = std.mem.Allocator;

pub fn Variable(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);
        const FuncT = Function(T);

        data: TensorT,
        grad: ?TensorT = null,
        requires_grad: bool,

        /// The operation that created this variable.
        /// If null, this is a Leaf variable (Input or Weight).
        creator: ?*FuncT = null,

        allocator: Allocator,

        pub fn init(allocator: Allocator, data: TensorT, requires_grad: bool) Self {
            return Self{
                .data = data,
                .grad = null,
                .requires_grad = requires_grad,
                .creator = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.data.deinit();
            // If we are a leaf, we own our gradient.
            if (self.grad) |g| g.deinit();

            // If we have a creator, we own it (it's a heap-allocated node).
            // Its deinit() will handle freeing the 'output_grad' it holds.
            if (self.creator) |c| c.deinit();
        }

        /// Smart Accessor: Finds the gradient tensor wherever it lives.
        /// - If this is a computed variable, the gradient lives in 'creator'.
        /// - If this is a leaf variable, the gradient lives in 'self.grad'.
        pub fn getGrad(self: *Self) !*TensorT {
            if (self.creator) |c| {
                // Return the gradient stored in the Node
                return c.getGrad();
            } else {
                // We are a Leaf. Lazily allocate gradient if missing.
                if (self.grad == null) {
                    self.grad = try TensorT.init(self.allocator, self.data.shape);
                    self.grad.?.fill(0);
                }
                return &self.grad.?;
            }
        }

        /// Zeros the gradient (useful for Optimizers)
        pub fn zeroGrad(self: *Self) !void {
            const g = try self.getGrad();
            g.fill(0);
        }

        /// THE BIG RED BUTTON: Triggers Backpropagation
        pub fn backward(self: *Self) !void {
            if (self.creator == null) return; // Cannot backward on a leaf/constant

            // 1. Seed the gradient at the end of the chain (Loss) to 1.0
            const g = try self.getGrad();
            g.fill(1.0);

            // 2. Topological Sort to determine execution order
            var graph = try engine.topologicalSort(self.allocator, self.creator.?);
            defer graph.deinit();

            // 3. Execute Backward Pass
            // The list is already reversed (Child -> Parent) by your engine logic.
            for (graph.items) |node| {
                try node.backward();
            }
        }
    };
}
