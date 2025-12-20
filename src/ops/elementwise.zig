const std = @import("std");
const TensorError = @import("../errors.zig").TensorError;
const Tensor = @import("../tensor.zig").Tensor;

pub fn broadcastOp(dest: anytype, src: anytype, comptime op_scalar: anytype) void {
    const ndim = dest.shape.len;

    // If the last dimension of both is contiguous and has the same length,
    // we can use SIMD on the inner-most loop.
    const can_simd_inner = (src.shape[ndim - 1] == dest.shape[ndim - 1]) and
                            (src.strides[ndim - 1] == 1) and
                            (dest.strides[ndim - 1] == 1);

    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.storage.data.len / inner_len;

    var indices = [_]usize{0} ** 16;

    for (0..outer_elements) |_| {
        // Calculate base offsets for the current outer "row"
        var src_base_offset: usize = 0;
        var dest_base_offset: usize = 0;

        inline for (0..15) |i| { // ndim - 1
            if (i >= ndim - 1) break;
            const src_stride = if (src.shape[i] == 1) 0 else src.strides[i];
            src_base_offset += indices[i] * src_stride;
            dest_base_offset += indices[i] * dest.strides[i];
        }

        if (can_simd_inner) {
            // OPTIMIZATION: SIMD / Fast path for contiguous row
            const dest_slice = dest.storage.data[dest_base_offset..][0..inner_len];
            const src_slice = src.storage.data[src_base_offset..][0..inner_len];

            // We use a simple loop here, but Zig's Autovectorizer
            // will turn this into SIMD instructions because strides are 1
            for (dest_slice, src_slice) |*d, s| {
                op_scalar(d, s);
            }
        } else {
            // Scalar fallback for the inner dimension (e.g., if broadcasting or transposed)
            const src_stride = if (src.shape[ndim - 1] == 1) 0 else src.strides[ndim - 1];
            for (0..inner_len) |k| {
                op_scalar(
                    &dest.storage.data[dest_base_offset + k * dest.strides[ndim - 1]],
                    src.storage.data[src_base_offset + k * src_stride]
                );
            }
        }

        // Increment outer indices
        if (ndim > 1) {
            var j = ndim - 1;
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
