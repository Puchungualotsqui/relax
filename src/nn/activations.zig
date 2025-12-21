const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const base = @import("../ops/base.zig");

/// ReLU: max(0, x)
pub fn relu(allocator: std.mem.Allocator, x: anytype) !@TypeOf(x) {
    // ReLU is essentially clipping the lower bound to 0
    return try x.clipped(allocator, 0, std.math.inf(@TypeOf(x.data[0])));
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(allocator: std.mem.Allocator, x: anytype) !@TypeOf(x) {
    const T = @TypeOf(x.data[0]);
    var out = try @TypeOf(x).init(allocator, x.shape);
    errdefer out.deinit();

    const closure = struct {
        fn apply(d: *T, s: T) void {
            d.* = 1.0 / (1.0 + std.math.exp(-s));
        }
    }.apply;

    try base.mapOp(&out, x, closure);
    return out;
}

/// Stable Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
pub fn softmax(allocator: std.mem.Allocator, x: anytype, axis: usize) !@TypeOf(x) {
    const T = @TypeOf(x.data[0]);

    // 1. LogSumExp is the "gold standard" for stable softmax denominators
    // We calculate: exp(x - LSE(x))
    const lse = try x.logSumExp(allocator, axis);
    defer lse.deinit();

    var out = try @TypeOf(x).init(allocator, x.shape);
    errdefer out.deinit();

    const closure = struct {
        fn apply(d: *T, s: T, l: T) void {
            d.* = std.math.exp(s - l);
        }
    }.apply;

    // Use broadcastOp2 to subtract the LSE reduction from the original tensor
    try base.broadcastOp2(&out, x, lse, closure);
    return out;
}

/// User-Defined Custom Activation Helper
/// Allows a user to pass any function f(x) -> x
pub fn custom(allocator: std.mem.Allocator, x: anytype, comptime func: anytype) !@TypeOf(x) {
    const T = @TypeOf(x.data[0]);
    var out = try @TypeOf(x).init(allocator, x.shape);
    errdefer out.deinit();

    const wrapper = struct {
        fn apply(d: *T, s: T) void {
            d.* = func(s);
        }
    }.apply;

    try base.mapOp(&out, x, wrapper);
    return out;
}
