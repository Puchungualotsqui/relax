const std = @import("std");
const TensorError = @import("../errors.zig").TensorError;
const Tensor = @import("../tensor.zig").Tensor;

pub fn broadcastOp(dest: anytype, src: anytype, comptime op_scalar: anytype) !void {
    const d_ndim = dest.shape.len;
    const s_ndim = src.shape.len;

    // 1. Validation: Ensure src isn't "bigger" in rank than dest for in-place ops
    if (s_ndim > d_ndim) return error.IncompatibleShapes;

    // 2. Validation: Check dimension-by-dimension compatibility
    inline for (0..16) |i| {
        if (i >= s_ndim) break;
        const d_idx = d_ndim - 1 - i;
        const s_idx = s_ndim - 1 - i;
        if (src.shape[s_idx] != 1 and src.shape[s_idx] != dest.shape[d_idx]) {
            return error.IncompatibleShapes;
        }
    }

    const inner_len = dest.shape[d_ndim - 1];
    const outer_elements = dest.storage.data.len / inner_len;

    // Check if we can SIMD the last dimension
    // Only possible if src isn't broadcasting (len > 1) and both are contiguous
    const src_inner_is_broadcast = if (s_ndim > 0) src.shape[s_ndim - 1] == 1 else true;
    const can_simd_inner = !src_inner_is_broadcast and
        (src.strides[s_ndim - 1] == 1) and
        (dest.strides[d_ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;

    for (0..outer_elements) |_| {
        var src_base_offset: usize = 0;
        var dest_base_offset: usize = 0;

        // Calculate base offsets for the outer dimensions
        inline for (0..15) |i| {
            if (i >= d_ndim - 1) break;

            const dest_stride = dest.strides[i];
            dest_base_offset += indices[i] * dest_stride;

            // Map dest index to src index
            const s_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(s_ndim));
            const src_idx_signed = @as(isize, @intCast(i)) - s_diff;

            if (src_idx_signed >= 0) {
                const s_idx = @as(usize, @intCast(src_idx_signed));
                // If shape is 1, stride is effectively 0 (broadcasting)
                const src_stride = if (src.shape[s_idx] == 1) 0 else src.strides[s_idx];
                src_base_offset += indices[i] * src_stride;
            }
        }

        if (can_simd_inner) {
            const dest_slice = dest.storage.data[dest_base_offset..][0..inner_len];
            const src_slice = src.storage.data[src_base_offset..][0..inner_len];
            for (dest_slice, src_slice) |*d, s| op_scalar(d, s);
        } else {
            // Scalar path: handle inner-most dimension broadcasting
            const src_stride = if (!src_inner_is_broadcast and s_ndim > 0) src.strides[s_ndim - 1] else 0;
            const dest_stride = dest.strides[d_ndim - 1];

            for (0..inner_len) |k| {
                op_scalar(&dest.storage.data[dest_base_offset + k * dest_stride], src.storage.data[src_base_offset + k * src_stride]);
            }
        }

        // Increment indices
        if (d_ndim > 1) {
            var j = d_ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) break;
                indices[j] = 0;
            }
        }
    }
}

/// Performs element-wise addition: self = self + other
pub fn add(self: anytype, other: anytype) TensorError!void {
    const T = @TypeOf(self.storage.data[0]);
    if (self.storage.data.len != other.storage.data.len) return error.IncompatibleShapes;

    const simd_len = std.simd.suggestVectorLength(T) orelse 8;
    const Vec = @Vector(simd_len, T);

    var i: usize = 0;
    while (i + simd_len <= self.storage.data.len) : (i += simd_len) {
        const v1: Vec = self.storage.data[i..][0..simd_len].*;
        const v2: Vec = other.storage.data[i..][0..simd_len].*;
        self.storage.data[i..][0..simd_len].* = v1 + v2;
    }

    while (i < self.storage.data.len) : (i += 1) {
        self.storage.data[i] += other.storage.data[i];
    }
}
