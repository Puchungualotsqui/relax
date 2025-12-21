const std = @import("std");
const base = @import("base.zig");

/// Fills the destination tensor with a specific scalar value.
pub fn fill(dest: anytype, value: anytype) void {
    // OPTIMIZATION: Linear memset for contiguous tensors
    if (dest.isContiguous()) {
        @memset(dest.data, value);
        return;
    }

    // Fallback: Stride-aware filling using your recursive or iterative logic
    const ndim = dest.shape.len;
    if (ndim == 0) {
        dest.data[0] = value;
        return;
    }

    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.data.len / inner_len;
    const stride = dest.strides[ndim - 1];

    var indices = [_]usize{0} ** 16;
    var cur_dest: usize = 0;

    for (0..outer_elements) |block_idx| {
        // Inner loop
        if (stride == 1) {
            @memset(dest.data[cur_dest..][0..inner_len], value);
        } else {
            for (0..inner_len) |k| {
                dest.data[cur_dest + k * stride] = value;
            }
        }

        // Rolling Index Update
        if (ndim > 1 and block_idx < outer_elements - 1) {
            var j = ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) {
                    cur_dest += dest.strides[j];
                    break;
                } else {
                    indices[j] = 0;
                    cur_dest -= (dest.shape[j] - 1) * dest.strides[j];
                }
            }
        }
    }
}

pub fn exp(dest: anytype, src: anytype) !void {
    const closures = struct {
        fn apply(d: anytype, s: anytype) void {
            d.* = std.math.exp(s);
        }
    };
    try base.mapOp(dest, src, closures.apply);
}

pub fn log(dest: anytype, src: anytype) !void {
    const closures = struct {
        fn apply(d: anytype, s: anytype) void {
            d.* = std.math.log(@TypeOf(s), s);
        }
    };
    try base.mapOp(dest, src, closures.apply);
}

pub fn sqrt(dest: anytype, src: anytype) !void {
    const closures = struct {
        fn apply(d: anytype, s: anytype) void {
            d.* = std.math.sqrt(s);
        }
    };
    try base.mapOp(dest, src, closures.apply);
}

pub fn clip(dest: anytype, src: anytype, min_val: anytype, max_val: anytype) !void {
    const ndim = dest.shape.len;

    // 1. Handle 0D (Scalar) Case
    if (ndim == 0) {
        dest.data[0] = @max(min_val, @min(max_val, src.data[0]));
        return;
    }

    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.data.len / inner_len;
    const can_simd = (src.strides[ndim - 1] == 1) and (dest.strides[ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var cur_src: usize = 0;
    var cur_dest: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd) {
            // OPTIMIZATION: Slice access for autovectorization
            const d_slice = dest.data[cur_dest..][0..inner_len];
            const s_slice = src.data[cur_src..][0..inner_len];

            // LLVM will turn this into SIMD instructions (e.g., VMAXPS, VMINPS)
            for (d_slice, s_slice) |*d, s| {
                d.* = @max(min_val, @min(max_val, s));
            }
        } else {
            // Stride-aware fallback
            const s_s = src.strides[ndim - 1];
            const d_s = dest.strides[ndim - 1];
            for (0..inner_len) |k| {
                const s_val = src.data[cur_src + k * s_s];
                dest.data[cur_dest + k * d_s] = @max(min_val, @min(max_val, s_val));
            }
        }

        // Rolling Index Update
        if (ndim > 1 and block_idx < outer_elements - 1) {
            var j = ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) {
                    cur_dest += dest.strides[j];
                    cur_src += src.strides[j];
                    break;
                } else {
                    indices[j] = 0;
                    cur_dest -= (dest.shape[j] - 1) * dest.strides[j];
                    cur_src -= (src.shape[j] - 1) * src.strides[j];
                }
            }
        }
    }
}

pub fn scalarOp(dest: anytype, src: anytype, scalar: anytype, comptime op: anytype) void {
    const ndim = dest.shape.len;

    // 0D Scalar Case
    if (ndim == 0) {
        dest.data[0] = op(src.data[0], scalar);
        return;
    }

    // --- CONTIGUOUS FAST PATH ---
    // If shapes match and are contiguous, use a flat loop.
    if (dest.isContiguous() and src.isContiguous() and std.mem.eql(usize, dest.shape, src.shape)) {
        for (dest.data, src.data) |*d, s| {
            d.* = op(s, scalar);
        }
        return;
    }

    // --- STRIDED FALLBACK ---
    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.data.len / inner_len;
    const can_simd_inner = (src.strides[ndim - 1] == 1) and (dest.strides[ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var cur_src: usize = 0;
    var cur_dest: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd_inner) {
            const d_slice = dest.data[cur_dest..][0..inner_len];
            const s_slice = src.data[cur_src..][0..inner_len];
            for (d_slice, s_slice) |*d, s| {
                d.* = op(s, scalar);
            }
        } else {
            const s_s = src.strides[ndim - 1];
            const d_s = dest.strides[ndim - 1];
            for (0..inner_len) |k| {
                dest.data[cur_dest + k * d_s] = op(src.data[cur_src + k * s_s], scalar);
            }
        }

        // Rolling Update
        if (ndim > 1 and block_idx < outer_elements - 1) {
            var j = ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) {
                    cur_dest += dest.strides[j];
                    cur_src += src.strides[j];
                    break;
                } else {
                    indices[j] = 0;
                    cur_dest -= (dest.shape[j] - 1) * dest.strides[j];
                    cur_src -= (src.shape[j] - 1) * src.strides[j];
                }
            }
        }
    }
}

// Specialized Wrappers
pub fn addScalar(dest: anytype, src: anytype, val: anytype) void {
    const closures = struct {
        fn apply(s: anytype, v: anytype) @TypeOf(s) {
            return s + v;
        }
    }.apply;
    scalarOp(dest, src, val, closures);
}

pub fn subScalar(dest: anytype, src: anytype, val: anytype) void {
    const closures = struct {
        fn apply(s: anytype, v: anytype) @TypeOf(s) {
            return s - v;
        }
    }.apply;
    scalarOp(dest, src, val, closures);
}

pub fn mulScalar(dest: anytype, src: anytype, val: anytype) void {
    const closures = struct {
        fn apply(s: anytype, v: anytype) @TypeOf(s) {
            return s * v;
        }
    }.apply;
    scalarOp(dest, src, val, closures);
}

pub fn divScalar(dest: anytype, src: anytype, val: anytype) void {
    const closures = struct {
        fn apply(s: anytype, v: anytype) @TypeOf(s) {
            return s / v;
        }
    }.apply;
    scalarOp(dest, src, val, closures);
}

pub fn powScalar(dest: anytype, src: anytype, val: anytype) void {
    const T = @TypeOf(src.data[0]);
    const closures = struct {
        fn apply(s: T, v: T) T {
            return std.math.pow(T, s, v);
        }
    }.apply;
    scalarOp(dest, src, val, closures);
}
