const std = @import("std");
const Tensor = @import("../../tensor.zig").Tensor;
const initz = @import("../initializers.zig");
const acts = @import("../activations.zig"); // Import the file containing the Union
const Allocator = std.mem.Allocator;

pub fn Dense(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const TensorT = Tensor(T);
        pub const ActT = acts.Activation(T); // The Union Type

        pub const Config = struct {
            in_features: usize,
            out_features: usize,
            // LOOK: No complex function pointer signature! Just the Union.
            activation: ActT = .none,
            weight_init: *const fn (anytype, *initz.RandomSource) anyerror!void = initz.heNormal,
            bias_init: *const fn (anytype) void = initz.zeros,
        };

        weights: TensorT,
        bias: TensorT,
        activation: ActT, // Store the enum
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: Config, rng: *initz.RandomSource) !Self {
            var weights = try TensorT.init(allocator, &[_]usize{ config.in_features, config.out_features });
            var bias = try TensorT.init(allocator, &[_]usize{config.out_features});

            try config.weight_init(&weights, rng);
            config.bias_init(&bias);

            return Self{
                .weights = weights,
                .bias = bias,
                .activation = config.activation,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.weights.deinit();
            self.bias.deinit();
        }

        pub fn forward(self: Self, input: TensorT) !TensorT {
            if (input.shape.len < 2 or input.shape[input.shape.len - 1] != self.weights.shape[0]) {
                return error.IncompatibleShapes;
            }

            var out_shape = try self.allocator.dupe(usize, input.shape);
            defer self.allocator.free(out_shape);
            out_shape[out_shape.len - 1] = self.weights.shape[1];

            var linear_out = try TensorT.init(self.allocator, out_shape);

            // Fused MatMul
            try input.linear(self.weights, self.bias, &linear_out);

            // Apply Activation via the Union's dispatcher
            // If it's .none, we just return linear_out (optimization to avoid copy)
            if (self.activation == .none) return linear_out;

            // Otherwise apply and free the intermediate
            defer linear_out.deinit();
            return try self.activation.forward(self.allocator, linear_out);
        }
    };
}
