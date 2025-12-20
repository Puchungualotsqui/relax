const std = @import("std");

pub fn matmul(dest: anytype, a: anytype, b: anytype) !void {
    const T = @TypeOf(dest.storage.data[0]);
    const d_ndim = dest.shape.len;
    const a_ndim = a.shape.len;
    const b_ndim = b.shape.len;

    // 1. Basic Rank/Shape Validation
    if (a_ndim < 2 or b_ndim < 2) return error.IncompatibleShapes;

    const M = a.shape[a_ndim - 2];
    const K = a.shape[a_ndim - 1];
    const K_b = b.shape[b_ndim - 2];
    const N = b.shape[b_ndim - 1];

    if (K != K_b) return error.IncompatibleShapes;

    // 2. Batch Logic
    // Total number of 2D matrix multiplications to perform
    const batch_dims = d_ndim - 2;
    var total_batches: usize = 1;
    for (0..batch_dims) |i| total_batches *= dest.shape[i];

    var batch_indices = [_]usize{0} ** 16;

    for (0..total_batches) |_| {
        // Calculate the base offsets for this specific batch
        var a_offset: usize = 0;
        var b_offset: usize = 0;
        var d_offset: usize = 0;

        inline for (0..14) |i| { // Up to 14 batch dimensions (14 + 2 = 16)
            if (i >= batch_dims) break;

            // Offset for A (Broadcasting-aware)
            const a_idx = @as(isize, @intCast(i)) + @as(isize, @intCast(a_ndim)) - @as(isize, @intCast(d_ndim));
            if (a_idx >= 0) {
                const s_idx = @as(usize, @intCast(a_idx));
                a_offset += batch_indices[i] * (if (a.shape[s_idx] == 1) 0 else a.strides[s_idx]);
            }

            // Offset for B (Broadcasting-aware)
            const b_idx = @as(isize, @intCast(i)) + @as(isize, @intCast(b_ndim)) - @as(isize, @intCast(d_ndim));
            if (b_idx >= 0) {
                const s_idx = @as(usize, @intCast(b_idx));
                b_offset += batch_indices[i] * (if (b.shape[s_idx] == 1) 0 else b.strides[s_idx]);
            }

            d_offset += batch_indices[i] * dest.strides[i];
        }

        // 3. Call Tiled Core for this Batch
        try tiledMatMulCore(T, dest.storage.data[d_offset..], a.storage.data[a_offset..], b.storage.data[b_offset..], M, K, N, a.strides[a_ndim - 2 ..][0..2].*, b.strides[b_ndim - 2 ..][0..2].*, dest.strides[d_ndim - 2 ..][0..2].*);

        // Increment Batch Indices
        if (batch_dims > 0) {
            var j = batch_dims;
            while (j > 0) {
                j -= 1;
                batch_indices[j] += 1;
                if (batch_indices[j] < dest.shape[j]) break;
                batch_indices[j] = 0;
            }
        }
    }
}

/// The high-performance tiled core acting on raw slices with provided strides
fn tiledMatMulCore(comptime T: type, dest: []T, a: []const T, b: []const T, M: usize, K: usize, N: usize, a_strides: [2]usize, b_strides: [2]usize, d_strides: [2]usize) !void {
    const BLOCK_SIZE = 64;
    const simd_len = std.simd.suggestVectorLength(T) orelse 8;
    const Vec = @Vector(simd_len, T);

    var ii: usize = 0;
    while (ii < M) : (ii += BLOCK_SIZE) {
        const i_end = @min(ii + BLOCK_SIZE, M);
        var kk: usize = 0;
        while (kk < K) : (kk += BLOCK_SIZE) {
            const k_end = @min(kk + BLOCK_SIZE, K);
            var jj: usize = 0;
            while (jj < N) : (jj += BLOCK_SIZE) {
                const j_end = @min(jj + BLOCK_SIZE, N);

                // Micro-kernel
                for (ii..i_end) |i| {
                    for (kk..k_end) |k| {
                        const a_val = a[i * a_strides[0] + k * a_strides[1]];
                        const a_splat: Vec = @splat(a_val);

                        var j = jj;
                        while (j + simd_len <= j_end) : (j += simd_len) {
                            const b_idx = k * b_strides[0] + j * b_strides[1];
                            const d_idx = i * d_strides[0] + j * d_strides[1];

                            const b_vec: Vec = b[b_idx..][0..simd_len].*;
                            const d_vec: Vec = dest[d_idx..][0..simd_len].*;

                            const res = @mulAdd(Vec, a_splat, b_vec, d_vec);
                            @memcpy(dest[d_idx..][0..simd_len], &@as([simd_len]T, res));
                        }
                        // Scalar tail
                        while (j < j_end) : (j += 1) {
                            dest[i * d_strides[0] + j * d_strides[1]] += a_val * b[k * b_strides[0] + j * b_strides[1]];
                        }
                    }
                }
            }
        }
    }
}
