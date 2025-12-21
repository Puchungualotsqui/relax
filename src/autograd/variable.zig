const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
// Import the new interface
const Function = @import("function.zig").Function;
const Allocator = std.mem.Allocator;

pub fn Variable(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);
        const FuncT = Function(T); // Concrete Function type for T

        data: TensorT,
        grad: ?TensorT = null,
        requires_grad: bool,

        /// The operation that created this variable.
        /// If null, this is a Leaf variable (Input or Weight).
        /// If set, this variable is the result of an operation.
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
            if (self.grad) |g| g.deinit();
            // We own the creator, so we must free it.
            // In a real graph, this requires careful Topological cleanup,
            // but for atomic testing, this suffices.
            if (self.creator) |c| c.deinit();
        }

        pub fn zeroGrad(self: *Self) !void {
            if (self.grad) |*g| {
                g.fill(0);
            } else {
                self.grad = try TensorT.init(self.allocator, self.data.shape);
                self.grad.?.fill(0);
            }
        }
    };
}
