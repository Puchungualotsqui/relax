const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

pub fn Variable(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);

        /// The value of the node (Forward pass data)
        data: TensorT,

        /// The gradient of the node (Backward pass data)
        /// Initially null. Allocated only if backward() is called or manually set.
        grad: ?TensorT = null,

        /// If true, we will calculate gradients for this node.
        /// (e.g., Weights = true, Input Data = false)
        requires_grad: bool,

        allocator: Allocator,

        /// Wraps an existing tensor into a Variable.
        /// Takes ownership of the 'data' tensor.
        pub fn init(allocator: Allocator, data: TensorT, requires_grad: bool) Self {
            return Self{
                .data = data,
                .grad = null,
                .requires_grad = requires_grad,
                .allocator = allocator,
            };
        }

        /// Frees the data and the gradient (if it exists).
        pub fn deinit(self: Self) void {
            self.data.deinit();
            if (self.grad) |g| g.deinit();
        }

        /// Helper to access the underlying tensor shape
        pub fn shape(self: Self) []const usize {
            return self.data.shape;
        }

        /// Helper to populate the gradient with zeros (useful for optimizers)
        pub fn zeroGrad(self: *Self) !void {
            if (self.grad) |*g| {
                // If grad exists, just fill with 0
                g.fill(0);
            } else {
                // If grad doesn't exist, allocate it
                self.grad = try TensorT.init(self.allocator, self.data.shape);
                self.grad.?.fill(0);
            }
        }
    };
}
