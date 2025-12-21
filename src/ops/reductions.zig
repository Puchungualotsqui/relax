const std = @import("std");
const base = @import("base.zig");

pub fn mean(dest: anytype, src: anytype, axis: usize) void {
    const T = @TypeOf(src.data[0]);
    const closures = struct {
        fn add(acc: *T, val: T) void {
            acc.* += val;
        }
    }.add;

    // 1. Perform Sum
    base.reduce(dest, src, axis, @as(T, 0), closures);

    // 2. Divide by count (Element-wise)
    const count = @as(T, @floatFromInt(src.shape[axis]));
    for (dest.data) |*d| {
        d.* /= count;
    }
}

pub fn variance(dest: anytype, src: anytype, mean_tensor: anytype, axis: usize) void {
    const T = @TypeOf(src.data[0]);
    const s_ndim = src.shape.len;
    dest.fill(0);

    // OPTIMIZATION: Last-axis contiguous path
    if (axis == s_ndim - 1 and src.strides[s_ndim - 1] == 1) {
        const inner_len = src.shape[s_ndim - 1];
        const outer_elements = src.data.len / inner_len;
        const count = @as(T, @floatFromInt(inner_len));

        for (0..outer_elements) |i| {
            const src_row = src.data[i * inner_len ..][0..inner_len];
            const mu = mean_tensor.data[i];
            var ssd: T = 0; // Sum of Squared Differences

            // SIMD Autovectorization target
            for (src_row) |val| {
                const diff = val - mu;
                ssd += diff * diff;
            }
            dest.data[i] = ssd / count;
        }
        return;
    }

    // Generic Fallback (Rolling Pointers)
    var indices = [_]usize{0} ** 16;
    var src_off: usize = 0;
    for (0..src.data.len) |_| {
        // Map to dest/mean offset
        var d_off: usize = 0;
        var d_dim: usize = 0;
        inline for (0..16) |j| {
            if (j >= s_ndim) break;
            if (j != axis) {
                d_off += indices[j] * dest.strides[d_dim];
                d_dim += 1;
            }
        }

        const diff = src.data[src_off] - mean_tensor.data[d_off];
        dest.data[d_off] += (diff * diff);

        // Standard Rolling Update
        var j = s_ndim;
        while (j > 0) {
            j -= 1;
            indices[j] += 1;
            if (indices[j] < src.shape[j]) {
                src_off += src.strides[j];
                break;
            } else {
                src_off -= (src.shape[j] - 1) * src.strides[j];
                indices[j] = 0;
            }
        }
    }

    const count = @as(T, @floatFromInt(src.shape[axis]));
    for (dest.data) |*d| d.* /= count;
}

pub fn logSumExp(dest: anytype, src: anytype, axis: usize) !void {
    const T = @TypeOf(src.data[0]);
    const s_ndim = src.shape.len;

    // 1. Find Max values along the axis for stability
    // Use your existing reduce logic
    const max_closures = struct {
        fn apply(acc: *T, val: T) void {
            if (val > acc.*) acc.* = val;
        }
    };
    base.reduce(dest, src, axis, std.math.floatMin(T), max_closures.apply);

    // 2. We need a temporary buffer to hold the sum of exps
    // To keep it low-level, we use the dest buffer to store max,
    // and another temporary buffer for the sum.
    const allocator = dest.allocator;
    var sum_exp = try allocator.alloc(T, dest.data.len);
    defer allocator.free(sum_exp);
    @memset(sum_exp, 0);

    // 3. Fused Exp-Subtract-Sum
    var indices = [_]usize{0} ** 16;
    var src_off: usize = 0;
    for (0..src.data.len) |_| {
        var d_off: usize = 0;
        var d_dim: usize = 0;
        inline for (0..16) |j| {
            if (j >= s_ndim) break;
            if (j != axis) {
                d_off += indices[j] * dest.strides[d_dim];
                d_dim += 1;
            }
        }

        const val = src.data[src_off];
        const m = dest.data[d_off];
        sum_exp[d_off] += std.math.exp(val - m);

        // Standard Rolling Update
        var j = s_ndim;
        while (j > 0) {
            j -= 1;
            indices[j] += 1;
            if (indices[j] < src.shape[j]) {
                src_off += src.strides[j];
                break;
            } else {
                src_off -= (src.shape[j] - 1) * src.strides[j];
                indices[j] = 0;
            }
        }
    }

    // 4. Final step: dest = max + log(sum_exp)
    for (dest.data, 0..) |*m, i| {
        m.* += @log(sum_exp[i]);
    }
}
