const std = @import("std");
const TensorError = @import("../errors.zig").TensorError;
const Tensor = @import("../tensor.zig").Tensor;

pub fn broadcastOp(dest: anytype, src: anytype, comptime op_scalar: anytype) !void {
    const d_ndim = dest.shape.len;
    const s_ndim = src.shape.len;

    // --- Validation ---
    if (s_ndim > d_ndim) return error.IncompatibleShapes;
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

    const src_inner_is_broadcast = if (s_ndim > 0) src.shape[s_ndim - 1] == 1 else true;
    const can_simd_inner = !src_inner_is_broadcast and
        (src.strides[s_ndim - 1] == 1) and
        (dest.strides[d_ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var curr_src_base: usize = 0;
    var curr_dest_base: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd_inner) {
            const d_ptr = dest.storage.data[curr_dest_base..][0..inner_len];
            const s_ptr = src.storage.data[curr_src_base..][0..inner_len];

            for (d_ptr, s_ptr) |*d, s| {
                op_scalar(d, s);
            }
        } else {
            // Scalar fallback (used for broadcasting or non-contiguous/transposed tensors)
            const s_stride = if (!src_inner_is_broadcast and s_ndim > 0) src.strides[s_ndim - 1] else 0;
            const d_stride = dest.strides[d_ndim - 1];

            for (0..inner_len) |k| {
                op_scalar(
                    &dest.storage.data[curr_dest_base + k * d_stride],
                    src.storage.data[curr_src_base + k * s_stride],
                );
            }
        }

        // --- Rolling Index Update ---
        if (d_ndim > 1 and block_idx < outer_elements - 1) {
            var j = d_ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;

                const s_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(s_ndim));
                const s_idx_signed = @as(isize, @intCast(j)) - s_diff;

                if (indices[j] < dest.shape[j]) {
                    curr_dest_base += dest.strides[j];
                    if (s_idx_signed >= 0) {
                        const s_idx = @as(usize, @intCast(s_idx_signed));
                        if (src.shape[s_idx] != 1) curr_src_base += src.strides[s_idx];
                    }
                    break;
                } else {
                    indices[j] = 0;
                    curr_dest_base -= (dest.shape[j] - 1) * dest.strides[j];
                    if (s_idx_signed >= 0) {
                        const s_idx = @as(usize, @intCast(s_idx_signed));
                        if (src.shape[s_idx] != 1) {
                            curr_src_base -= (src.shape[s_idx] - 1) * src.strides[s_idx];
                        }
                    }
                }
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

pub fn reduce(dest: anytype, src: anytype, axis: usize, comptime init_val: anytype, comptime op_scalar: anytype) void {
    const s_ndim = src.shape.len;
    dest.fill(init_val);

    // OPTIMIZATION: If we are reducing the last dimension and it's contiguous
    if (axis == s_ndim - 1 and src.strides[s_ndim - 1] == 1) {
        const inner_len = src.shape[s_ndim - 1];
        const outer_elements = src.storage.data.len / inner_len;

        for (0..outer_elements) |i| {
            const src_row = src.storage.data[i * inner_len ..][0..inner_len];
            // The destination for a last-axis reduction maps 1-to-1 with the outer loop index
            const d_ptr = &dest.storage.data[i];

            for (src_row) |val| {
                op_scalar(d_ptr, val);
            }
        }
        return;
    }

    // Generic Fallback (The Rolling Pointer logic you already have)
    var indices = [_]usize{0} ** 16;
    var curr_src_offset: usize = 0;
    var curr_dest_offset: usize = 0;

    for (0..src.storage.data.len) |_| {
        op_scalar(&dest.storage.data[curr_dest_offset], src.storage.data[curr_src_offset]);

        var j = s_ndim;
        while (j > 0) {
            j -= 1;
            indices[j] += 1;
            if (indices[j] < src.shape[j]) {
                curr_src_offset += src.strides[j];
                if (j != axis) {
                    var d_dim: usize = 0;
                    // Find matching dim in dest
                    inline for (0..16) |prev| {
                        if (prev >= j) break;
                        if (prev != axis) d_dim += 1;
                    }
                    curr_dest_offset += dest.strides[d_dim];
                }
                break;
            } else {
                indices[j] = 0;
                curr_src_offset -= (src.shape[j] - 1) * src.strides[j];
                if (j != axis) {
                    var d_dim: usize = 0;
                    inline for (0..16) |prev| {
                        if (prev >= j) break;
                        if (prev != axis) d_dim += 1;
                    }
                    curr_dest_offset -= (src.shape[j] - 1) * dest.strides[d_dim];
                }
            }
        }
    }
}

pub fn concat(dest: anytype, inputs: anytype, axis: usize) !void {
    if (inputs.len == 0) return;

    const ndim = dest.shape.len;

    // 1. Pre-calculate the block size (inner dimensions) for the tensors.
    // All tensors must have the same inner dimensions except for the 'axis' itself.
    var outer_elements: usize = 1;
    for (0..axis) |i| {
        outer_elements *= dest.shape[i];
    }

    // We track the current read-offset for each input tensor
    // Using a stack-allocated array for speed (supports up to 32 inputs)
    var src_cursors = [_]usize{0} ** 32;
    if (inputs.len > 32) return error.TooManyInputs;

    // Pre-calculate block sizes for each input to avoid redundant loops
    var src_block_sizes = [_]usize{0} ** 32;
    for (inputs, 0..) |src, i| {
        var block: usize = 1;
        for (axis..ndim) |dim_idx| {
            block *= src.shape[dim_idx];
        }
        src_block_sizes[i] = block;
    }

    var dest_offset: usize = 0;

    // 2. The Main Copy Loop
    for (0..outer_elements) |_| {
        for (inputs, 0..) |src, i| {
            const block_size = src_block_sizes[i];

            // Safety check for non-contiguous inputs
            // In a pro library, we'd cloned here or used a strided copy
            if (!src.isContiguous()) return error.NotContiguous;

            const src_slice = src.storage.data[src_cursors[i]..][0..block_size];
            const dest_slice = dest.storage.data[dest_offset..][0..block_size];

            @memcpy(dest_slice, src_slice);

            // Advance cursors
            dest_offset += block_size;
            src_cursors[i] += block_size;
        }
    }
}
