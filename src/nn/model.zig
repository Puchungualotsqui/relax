const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Dense = @import("layers/dense.zig").Dense;
const Allocator = std.mem.Allocator;

/// The "Wrapper" for any layer type.
/// As you build more layers (Dropout, Conv2D), add them here.
pub fn Layer(comptime T: type) type {
    return union(enum) {
        dense: Dense(T),
        // dropout: Dropout(T),
        // conv2d: Conv2D(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn deinit(self: Self) void {
            switch (self) {
                .dense => |l| l.deinit(),
            }
        }

        pub fn forward(self: Self, input: TensorT) !TensorT {
            switch (self) {
                .dense => |l| return l.forward(input),
            }
        }
    };
}

/// The Sequential Model Container
pub fn Sequential(comptime T: type) type {
    return struct {
        const Self = @This();
        const LayerT = Layer(T);
        const TensorT = Tensor(T);

        allocator: std.mem.Allocator,
        // Switch to Unmanaged to match the behavior your compiler is enforcing
        layers: std.ArrayListUnmanaged(LayerT),

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                // Unmanaged uses empty init
                .layers = .{},
            };
        }

        pub fn deinit(mut_self: *Self) void {
            for (mut_self.layers.items) |layer| {
                layer.deinit();
            }
            // Pass the allocator to deinit
            mut_self.layers.deinit(mut_self.allocator);
        }

        pub fn add(mut_self: *Self, layer: LayerT) !void {
            // Pass the allocator to append
            try mut_self.layers.append(mut_self.allocator, layer);
        }

        pub fn forward(self: Self, input: TensorT) !TensorT {
            if (self.layers.items.len == 0) return try input.clone();

            // First layer pass
            var current = try self.layers.items[0].forward(input);

            // Subsequent layers
            var i: usize = 1;
            while (i < self.layers.items.len) : (i += 1) {
                const next = try self.layers.items[i].forward(current);
                current.deinit(); // Free intermediate tensor memory
                current = next;
            }

            return current;
        }
    };
}
