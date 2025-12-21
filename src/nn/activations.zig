const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("../autograd/variable.zig").Variable;
const ops = @import("../autograd/ops.zig");
const base = @import("../ops/base.zig");
const Allocator = std.mem.Allocator;

/// The "Menu" of available activations for a specific type T.
pub fn Activation(comptime T: type) type {
    return union(enum) {
        none,
        relu,
        sigmoid,
        softmax: usize, // Payload: axis

        const Self = @This();
        // 2. Define VarT alias
        const VarT = Variable(T);

        // 3. Update Signature: input: VarT, return: VarT
        pub fn forward(self: Self, allocator: Allocator, input: VarT) !VarT {
            switch (self) {
                // For .none, we clone the variable reference (cheap) to keep ownership consistent
                .none => return input.clone(),

                // Delegate to autograd ops
                .relu => return ops.relu(allocator, input),
                .sigmoid => return ops.sigmoid(allocator, input),

                // ops.softmax currently ignores axis, but we accept it for future proofing
                .softmax => |_| return ops.softmax(allocator, input),
            }
        }
    };
}

// --- Your existing kernels remain below ---

pub fn relu(allocator: Allocator, x: anytype) !@TypeOf(x) {
    return try x.clipped(allocator, 0, std.math.inf(@TypeOf(x.data[0])));
}

pub fn sigmoid(allocator: Allocator, x: anytype) !@TypeOf(x) {
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

pub fn softmax(allocator: Allocator, x: anytype, axis: usize) !@TypeOf(x) {
    const T = @TypeOf(x.data[0]);

    // 1. Calculate Max for stability (safe against overflow)
    var max_vals = try x.max(allocator, axis);
    defer max_vals.deinit();

    // 2. Calculate Exp(x - max)
    // We need to broadcast sub first, then exp.
    // Optimization: We can fuse this. But for now, let's keep it composed.
    var shifted = try Tensor(T).init(allocator, x.shape);
    defer shifted.deinit();

    const sub_closure = struct {
        fn apply(d: *T, s: T, m: T) void {
            d.* = s - m;
        }
    }.apply;
    try base.broadcastOp2(&shifted, x, max_vals, sub_closure);
    try shifted.expInPlace(); // shifted is now exp(x-max)

    // 3. Calculate Sum of Exps
    var sum_exps = try shifted.sum(allocator, axis);
    defer sum_exps.deinit();

    // 4. Divide: shifted / sum_exps
    var out = try Tensor(T).init(allocator, x.shape);
    // Reuse broadcastOp2 for division
    const div_closure = struct {
        fn apply(d: *T, n: T, den: T) void {
            d.* = n / den;
        }
    }.apply;
    try base.broadcastOp2(&out, shifted, sum_exps, div_closure);

    return out;
}
