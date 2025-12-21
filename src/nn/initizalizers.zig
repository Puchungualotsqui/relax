const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

pub const InitializerError = error{
    InvalidShape,
};

/// Basic Wrapper for Zig's PRNG
pub const RandomSource = struct {
    prng: std.rand.DefaultPrng,
    random: std.rand.Random,

    pub fn init(seed: u64) RandomSource {
        var prng = std.rand.DefaultPrng.init(seed);
        return .{
            .prng = prng,
            .random = prng.random(),
        };
    }
};

/// Fills a tensor with values from a uniform distribution [min, max)
pub fn uniform(tensor: anytype, rng: *RandomSource, min: f32, max: f32) void {
    for (tensor.data) |*val| {
        val.* = rng.random.float(f32) * (max - min) + min;
    }
}

/// Fills a tensor with values from a normal distribution (mean=0, std=1)
pub fn normal(tensor: anytype, rng: *RandomSource, mean: f32, std_dev: f32) void {
    for (tensor.data) |*val| {
        val.* = (rng.random.floatNorm(f32) * std_dev) + mean;
    }
}

/// Xavier (Glorot) Uniform Initialization
/// Best for Tanh/Sigmoid
pub fn xavier(tensor: anytype, rng: *RandomSource) !void {
    if (tensor.shape.len < 2) return InitializerError.InvalidShape;

    const fan_in = @as(f32, @floatFromInt(tensor.shape[tensor.shape.len - 2]));
    const fan_out = @as(f32, @floatFromInt(tensor.shape[tensor.shape.len - 1]));

    const limit = std.math.sqrt(6.0 / (fan_in + fan_out));
    uniform(tensor, rng, -limit, limit);
}

/// He (Kaiming) Normal Initialization
/// Best for ReLU
pub fn heNormal(tensor: anytype, rng: *RandomSource) !void {
    if (tensor.shape.len < 2) return InitializerError.InvalidShape;

    const fan_in = @as(f32, @floatFromInt(tensor.shape[tensor.shape.len - 2]));
    const std_dev = std.math.sqrt(2.0 / fan_in);

    normal(tensor, rng, 0.0, std_dev);
}

/// Zero initialization (Standard for biases)
pub fn zeros(tensor: anytype) void {
    tensor.fill(0);
}
