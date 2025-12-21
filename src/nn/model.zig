const std = @import("std");
const Allocator = std.mem.Allocator;
const Variable = @import("../autograd/variable.zig").Variable; // Added Import
const Dense = @import("layers/dense.zig").Dense;
const Dropout = @import("layers/dropout.zig").Dropout;

/// The "Wrapper" for any layer type.
pub fn Layer(comptime T: type) type {
    return union(enum) {
        dense: Dense(T),
        dropout: Dropout(T),

        const Self = @This();
        const VarT = Variable(T); // Updated from TensorT to VarT

        pub fn deinit(self: Self) void {
            switch (self) {
                .dense => |l| l.deinit(),
                .dropout => |l| l.deinit(), // Handle deinit
            }
        }

        // Updated signature to take Variable(T)
        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            switch (self) {
                // Pass the flag down to the concrete layers
                .dense => |l| return l.forward(input, is_training),
                .dropout => |l| return l.forward(input, is_training),
            }
        }

        // Propagate parameter collection
        pub fn parameters(self: Self, list: *std.ArrayList(VarT)) !void {
            switch (self) {
                .dense => |l| try l.parameters(list),
                .dropout => |l| try l.parameters(list), // Dropout will just return (no-op)
            }
        }
    };
}

/// The Sequential Model Container
pub fn Sequential(comptime T: type) type {
    return struct {
        const Self = @This();
        const LayerT = Layer(T);
        const VarT = Variable(T); // Updated to VarT

        allocator: std.mem.Allocator,
        layers: std.ArrayListUnmanaged(LayerT),

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                .layers = .{},
            };
        }

        pub fn deinit(mut_self: *Self) void {
            for (mut_self.layers.items) |layer| {
                layer.deinit();
            }
            mut_self.layers.deinit(mut_self.allocator);
        }

        pub fn add(mut_self: *Self, layer: LayerT) !void {
            try mut_self.layers.append(mut_self.allocator, layer);
        }

        // Collects all parameters from all layers
        // Returns an ArrayList containing clones (refs) to every weight/bias in the model
        pub fn parameters(self: Self) !std.ArrayList(VarT) {
            var params = std.ArrayList(VarT).init(self.allocator);
            errdefer params.deinit();

            for (self.layers.items) |layer| {
                try layer.parameters(&params);
            }
            return params;
        }

        // Updated signature to take Variable(T)
        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            if (self.layers.items.len == 0) return input.clone();

            // Pass the flag to the first layer
            var current = try self.layers.items[0].forward(input, is_training);

            var i: usize = 1;
            while (i < self.layers.items.len) : (i += 1) {
                // Pass the flag to subsequent layers
                const next = try self.layers.items[i].forward(current, is_training);
                current.deinit();
                current = next;
            }

            return current;
        }
    };
}
