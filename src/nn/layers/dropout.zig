const std = @import("std");
const Variable = @import("../../autograd/variable.zig").Variable;
const ops = @import("../../autograd/ops.zig");
const initz = @import("../initializers.zig");
const Allocator = std.mem.Allocator;

pub fn Dropout(comptime T: type) type {
    return struct {
        const Self = @This();
        const VarT = Variable(T);

        rate: T,
        rng: *initz.RandomSource, // Need randomness for the mask
        allocator: Allocator,

        pub fn init(allocator: Allocator, rate: T, rng: *initz.RandomSource) Self {
            return Self{
                .rate = rate,
                .rng = rng,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            _ = self; // Nothing to free
        }

        // Dropout has no trainable parameters
        pub fn parameters(self: Self, list: *std.ArrayList(VarT)) !void {
            _ = self;
            _ = list;
        }

        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            if (!is_training or self.rate == 0.0) {
                // Identity during inference
                return input.clone();
            }
            // Use autograd op during training
            const r = self.rng.random;
            return try ops.dropout(self.allocator, input, self.rate, r);
        }
    };
}
