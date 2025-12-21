const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("../autograd/variable.zig").Variable;
const Allocator = std.mem.Allocator;

pub fn Checkpoint(comptime T: type) type {
    return struct {
        const Self = @This();
        const TensorT = Tensor(T);
        const VarT = Variable(T);

        // List of raw data clones
        data: std.ArrayListUnmanaged(TensorT),
        allocator: Allocator,

        pub fn capture(allocator: Allocator, params: std.ArrayListUnmanaged(VarT)) !Self {
            var data_list = try std.ArrayListUnmanaged(TensorT).initCapacity(allocator, params.items.len);
            errdefer {
                for (data_list.items) |t| t.deinit();
                data_list.deinit(allocator);
            }

            for (params.items) |p| {
                try data_list.append(allocator, try p.ptr.data.clone());
            }

            return Self{ .data = data_list, .allocator = allocator };
        }

        pub fn restore(self: Self, params: std.ArrayListUnmanaged(VarT)) !void {
            if (self.data.items.len != params.items.len) return error.ParameterMismatch;
            for (0..params.items.len) |i| {
                // Copy raw data back into the parameter's tensor
                @memcpy(params.items[i].ptr.data.data, self.data.items[i].data);
            }
        }

        pub fn deinit(mut_self: *Self) void {
            for (mut_self.data.items) |t| t.deinit();
            mut_self.data.deinit(mut_self.allocator);
        }
    };
}
