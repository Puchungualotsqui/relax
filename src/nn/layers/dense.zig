const std = @import("std");
const Tensor = @import("../../tensor.zig").Tensor;
const Variable = @import("../../autograd/variable.zig").Variable;
const autograd_ops = @import("../../autograd/ops.zig");
const initz = @import("../initializers.zig");
const acts = @import("../activations.zig");
const Allocator = std.mem.Allocator;

pub fn Dense(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const TensorT = Tensor(T);
        pub const VarT = Variable(T);
        pub const ActT = acts.Activation(T);

        pub const Config = struct {
            in_features: usize,
            out_features: usize,
            activation: ActT = .none,
            weight_init: *const fn (anytype, *initz.RandomSource) anyerror!void = initz.heNormal,
            bias_init: *const fn (anytype) void = initz.zeros,
            // Optimization: whether this layer should track gradients
            requires_grad: bool = true,
        };

        // Parameters are now Variables
        weights: VarT,
        bias: VarT,
        activation: ActT,
        allocator: Allocator,

        pub fn init(allocator: Allocator, config: Config, rng: *initz.RandomSource) !Self {
            // 1. Initialize Raw Tensors
            var w_data = try TensorT.init(allocator, &[_]usize{ config.in_features, config.out_features });
            errdefer w_data.deinit();
            var b_data = try TensorT.init(allocator, &[_]usize{config.out_features});
            errdefer b_data.deinit();

            // 2. Apply Initializers
            try config.weight_init(&w_data, rng);
            config.bias_init(&b_data);

            // 3. Wrap into Variables (Ownership moves to Variables)
            const weights = try VarT.init(allocator, w_data, config.requires_grad);
            errdefer weights.deinit();
            const bias = try VarT.init(allocator, b_data, config.requires_grad);
            errdefer bias.deinit();

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

        /// Returns references to the trainable parameters
        pub fn parameters(self: Self, allocator: Allocator, list: *std.ArrayListUnmanaged(VarT)) !void {
            try list.append(allocator, self.weights.clone());
            try list.append(allocator, self.bias.clone());
        }

        /// Differentiable Forward Pass
        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            _ = is_training;

            // 1. Linear part
            var mm = try autograd_ops.matmul(self.allocator, input, self.weights);
            defer mm.deinit();

            var linear_out = try autograd_ops.add(self.allocator, mm, self.bias);

            defer linear_out.deinit();
            return try self.activation.forward(self.allocator, linear_out);
        }
    };
}
