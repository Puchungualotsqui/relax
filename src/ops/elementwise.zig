const std = @import("std");
const TensorError = @import("../errors.zig").TensorError;
const Tensor = @import("../tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

pub fn broadcastOp(dest: anytype, src: anytype, comptime op_scalar: anytype) !void {
    if (dest.shape.len == 0) {
        op_scalar(&dest.data[0], src.data[0]);
        return;
    }

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
    const outer_elements = dest.data.len / inner_len;

    const src_inner_is_broadcast = if (s_ndim > 0) src.shape[s_ndim - 1] == 1 else true;
    const can_simd_inner = !src_inner_is_broadcast and
        (src.strides[s_ndim - 1] == 1) and
        (dest.strides[d_ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var curr_src_base: usize = 0;
    var curr_dest_base: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd_inner) {
            const d_ptr = dest.data[curr_dest_base..][0..inner_len];
            const s_ptr = src.data[curr_src_base..][0..inner_len];

            for (d_ptr, s_ptr) |*d, s| {
                op_scalar(d, s);
            }
        } else {
            // Scalar fallback (used for broadcasting or non-contiguous/transposed tensors)
            const s_stride = if (!src_inner_is_broadcast and s_ndim > 0) src.strides[s_ndim - 1] else 0;
            const d_stride = dest.strides[d_ndim - 1];

            for (0..inner_len) |k| {
                op_scalar(
                    &dest.data[curr_dest_base + k * d_stride],
                    src.data[curr_src_base + k * s_stride],
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

pub fn broadcastOp2(dest: anytype, src_a: anytype, src_b: anytype, comptime op_scalar: anytype) !void {
    if (dest.shape.len == 0) {
        // If dest is 0D, src_a and src_b must be 0D or 1-element tensors
        op_scalar(&dest.data[0], src_a.data[0], src_b.data[0]);
        return;
    }

    const d_ndim = dest.shape.len;
    const sa_ndim = src_a.shape.len;
    const sb_ndim = src_b.shape.len;

    // --- Validation (Simplified for brevity, same logic as before for both srcs) ---
    if (sa_ndim > d_ndim or sb_ndim > d_ndim) return error.IncompatibleShapes;

    const inner_len = dest.shape[d_ndim - 1];
    const outer_elements = dest.data.len / inner_len;

    const sa_is_bcast = if (sa_ndim > 0) src_a.shape[sa_ndim - 1] == 1 else true;
    const sb_is_bcast = if (sb_ndim > 0) src_b.shape[sb_ndim - 1] == 1 else true;

    // SIMD only if both sources and dest are contiguous and not broadcasting in the inner dim
    const can_simd = !sa_is_bcast and !sb_is_bcast and
        (src_a.strides[sa_ndim - 1] == 1) and
        (src_b.strides[sb_ndim - 1] == 1) and
        (dest.strides[d_ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var cur_a: usize = 0;
    var cur_b: usize = 0;
    var cur_d: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd) {
            const d_ptr = dest.data[cur_d..][0..inner_len];
            const a_ptr = src_a.data[cur_a..][0..inner_len];
            const b_ptr = src_b.data[cur_b..][0..inner_len];

            for (d_ptr, a_ptr, b_ptr) |*d, a, b| {
                op_scalar(d, a, b);
            }
        } else {
            const s_a = if (!sa_is_bcast and sa_ndim > 0) src_a.strides[sa_ndim - 1] else 0;
            const s_b = if (!sb_is_bcast and sb_ndim > 0) src_b.strides[sb_ndim - 1] else 0;
            const s_d = dest.strides[d_ndim - 1];

            for (0..inner_len) |k| {
                op_scalar(&dest.data[cur_d + k * s_d], src_a.data[cur_a + k * s_a], src_b.data[cur_b + k * s_b]);
            }
        }

        // Rolling Index Update (Update 3 pointers)
        if (d_ndim > 1 and block_idx < outer_elements - 1) {
            var j = d_ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) {
                    cur_d += dest.strides[j];
                    // Update A
                    const sa_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(sa_ndim));
                    const sa_idx = @as(isize, @intCast(j)) - sa_diff;
                    if (sa_idx >= 0 and src_a.shape[@intCast(sa_idx)] != 1) cur_a += src_a.strides[@intCast(sa_idx)];
                    // Update B
                    const sb_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(sb_ndim));
                    const sb_idx = @as(isize, @intCast(j)) - sb_diff;
                    if (sb_idx >= 0 and src_b.shape[@intCast(sb_idx)] != 1) cur_b += src_b.strides[@intCast(sb_idx)];
                    break;
                } else {
                    indices[j] = 0;
                    cur_d -= (dest.shape[j] - 1) * dest.strides[j];
                    // Reset A
                    const sa_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(sa_ndim));
                    const sa_idx = @as(isize, @intCast(j)) - sa_diff;
                    if (sa_idx >= 0 and src_a.shape[@intCast(sa_idx)] != 1) cur_a -= (src_a.shape[@intCast(sa_idx)] - 1) * src_a.strides[@intCast(sa_idx)];
                    // Reset B
                    const sb_diff = @as(isize, @intCast(d_ndim)) - @as(isize, @intCast(sb_ndim));
                    const sb_idx = @as(isize, @intCast(j)) - sb_diff;
                    if (sb_idx >= 0 and src_b.shape[@intCast(sb_idx)] != 1) cur_b -= (src_b.shape[@intCast(sb_idx)] - 1) * src_b.strides[@intCast(sb_idx)];
                }
            }
        }
    }
}

pub fn calculateBroadcastShape(allocator: Allocator, shape_a: []const usize, shape_b: []const usize) ![]usize {
    const rank_a = shape_a.len;
    const rank_b = shape_b.len;
    const out_rank = @max(rank_a, rank_b);

    const out_shape = try allocator.alloc(usize, out_rank);
    errdefer allocator.free(out_shape);

    var i: usize = 0;
    while (i < out_rank) : (i += 1) {
        // Indices from the right
        const a_idx = if (i < rank_a) rank_a - 1 - i else null;
        const b_idx = if (i < rank_b) rank_b - 1 - i else null;

        const dim_a = if (a_idx) |idx| shape_a[idx] else 1;
        const dim_b = if (b_idx) |idx| shape_b[idx] else 1;

        if (dim_a != dim_b and dim_a != 1 and dim_b != 1) {
            return TensorError.IncompatibleShapes;
        }

        out_shape[out_rank - 1 - i] = @max(dim_a, dim_b);
    }

    return out_shape;
}

/// Performs element-wise addition: self = self + other
pub fn add(self: anytype, other: anytype) !void {
    const closures = struct {
        fn apply(d: *@TypeOf(self.data[0]), s: @TypeOf(self.data[0])) void {
            d.* += s;
        }
    };

    try broadcastOp(self, other, closures.apply);
}

pub fn mean(dest: anytype, src: anytype, axis: usize) void {
    const T = @TypeOf(src.data[0]);
    const closures = struct {
        fn add(acc: *T, val: T) void { acc.* += val; }
    }.add;

    // 1. Perform Sum
    reduce(dest, src, axis, @as(T, 0), closures);

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

pub fn mapOp(dest: anytype, src: anytype, comptime op_scalar: anytype) !void {
    const ndim = dest.shape.len;

    if (ndim == 0) {
        op_scalar(&dest.data[0], src.data[0]);
        return;
    }

    if (src.shape.len != ndim) return error.IncompatibleShapes;

    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.data.len / inner_len;

    // SIMD only if both are contiguous in the last dimension
    const can_simd = (src.strides[ndim - 1] == 1) and (dest.strides[ndim - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var cur_src: usize = 0;
    var cur_dest: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd) {
            const d_ptr = dest.data[cur_dest..][0..inner_len];
            const s_ptr = src.data[cur_src..][0..inner_len];
            for (d_ptr, s_ptr) |*d, s| op_scalar(d, s);
        } else {
            const s_s = src.strides[ndim - 1];
            const d_s = dest.strides[ndim - 1];
            for (0..inner_len) |k| {
                op_scalar(&dest.data[cur_dest + k * d_s], src.data[cur_src + k * s_s]);
            }
        }

        // Rolling Index Update (Unary version)
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

pub fn exp(dest: anytype, src: anytype) !void {
    const closures = struct {
        fn apply(d: anytype, s: anytype) void { d.* = std.math.exp(s); }
    };
    try mapOp(dest, src, closures.apply);
}

pub fn reduce(dest: anytype, src: anytype, axis: usize, comptime init_val: anytype, comptime op_scalar: anytype) void {
    const s_ndim = src.shape.len;
    dest.fill(init_val);

    // OPTIMIZATION: If we are reducing the last dimension and it's contiguous
    if (axis == s_ndim - 1 and src.strides[s_ndim - 1] == 1) {
        const inner_len = src.shape[s_ndim - 1];
        const outer_elements = src.data.len / inner_len;

        for (0..outer_elements) |i| {
            const src_row = src.data[i * inner_len ..][0..inner_len];
            // The destination for a last-axis reduction maps 1-to-1 with the outer loop index
            const d_ptr = &dest.data[i];

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

    for (0..src.data.len) |_| {
        op_scalar(&dest.data[curr_dest_offset], src.data[curr_src_offset]);

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

            const src_slice = src.data[src_cursors[i]..][0..block_size];
            const dest_slice = dest.data[dest_offset..][0..block_size];

            @memcpy(dest_slice, src_slice);

            // Advance cursors
            dest_offset += block_size;
            src_cursors[i] += block_size;
        }
    }
}

pub fn where(dest: anytype, mask: anytype, src_a: anytype, src_b: anytype) !void {
    const ndim = dest.shape.len;

    // 0D (Scalar) Case
    if (ndim == 0) {
        dest.data[0] = if (mask.data[0] != 0) src_a.data[0] else src_b.data[0];
        return;
    }

    const inner_len = dest.shape[ndim - 1];
    const outer_elements = dest.data.len / inner_len;

    // SIMD is only safe if ALL involved buffers are contiguous in the inner dimension
    // and sources are not broadcasting in the inner dimension.
    const sa_bcast = (src_a.shape.len > 0) and (src_a.shape[src_a.shape.len - 1] == 1);
    const sb_bcast = (src_b.shape.len > 0) and (src_b.shape[src_b.shape.len - 1] == 1);

    const can_simd = (dest.strides[ndim - 1] == 1) and
        (mask.strides[mask.shape.len - 1] == 1) and
        (!sa_bcast and src_a.strides[src_a.shape.len - 1] == 1) and
        (!sb_bcast and src_b.strides[src_b.shape.len - 1] == 1);

    var indices = [_]usize{0} ** 16;
    var cur_d: usize = 0;
    var cur_m: usize = 0;
    var cur_a: usize = 0;
    var cur_b: usize = 0;

    for (0..outer_elements) |block_idx| {
        if (can_simd) {
            const d_slice = dest.data[cur_d..][0..inner_len];
            const m_slice = mask.data[cur_m..][0..inner_len];
            const a_slice = src_a.data[cur_a..][0..inner_len];
            const b_slice = src_b.data[cur_b..][0..inner_len];

            for (d_slice, m_slice, a_slice, b_slice) |*d, m, a, b| {
                d.* = if (m != 0) a else b;
            }
        } else {
            // Scalar fallback with full broadcasting stride support
            const sd = dest.strides[ndim - 1];
            const sm = mask.strides[mask.shape.len - 1];
            const sa = if (!sa_bcast) src_a.strides[src_a.shape.len - 1] else 0;
            const sb = if (!sb_bcast) src_b.strides[src_b.shape.len - 1] else 0;

            for (0..inner_len) |k| {
                const m_val = mask.data[cur_m + k * sm];
                dest.data[cur_d + k * sd] = if (m_val != 0) src_a.data[cur_a + k * sa] else src_b.data[cur_b + k * sb];
            }
        }

        // Rolling Index Update for 4 pointers
        if (ndim > 1 and block_idx < outer_elements - 1) {
            var j = ndim - 1;
            while (j > 0) {
                j -= 1;
                indices[j] += 1;
                if (indices[j] < dest.shape[j]) {
                    cur_d += dest.strides[j];
                    cur_m += mask.strides[j];

                    // A and B require rank-offset checks for broadcasting
                    const sa_diff = @as(isize, @intCast(ndim)) - @as(isize, @intCast(src_a.shape.len));
                    const sa_idx = @as(isize, @intCast(j)) - sa_diff;
                    if (sa_idx >= 0 and src_a.shape[@intCast(sa_idx)] != 1) cur_a += src_a.strides[@intCast(sa_idx)];

                    const sb_diff = @as(isize, @intCast(ndim)) - @as(isize, @intCast(src_b.shape.len));
                    const sb_idx = @as(isize, @intCast(j)) - sb_diff;
                    if (sb_idx >= 0 and src_b.shape[@intCast(sb_idx)] != 1) cur_b += src_b.strides[@intCast(sb_idx)];
                    break;
                } else {
                    indices[j] = 0;
                    cur_d -= (dest.shape[j] - 1) * dest.strides[j];
                    cur_m -= (mask.shape[j] - 1) * mask.strides[j];

                    const sa_diff = @as(isize, @intCast(ndim)) - @as(isize, @intCast(src_a.shape.len));
                    const sa_idx = @as(isize, @intCast(j)) - sa_diff;
                    if (sa_idx >= 0 and src_a.shape[@intCast(sa_idx)] != 1)
                        cur_a -= (src_a.shape[@intCast(sa_idx)] - 1) * src_a.strides[@intCast(sa_idx)];

                    const sb_diff = @as(isize, @intCast(ndim)) - @as(isize, @intCast(src_b.shape.len));
                    const sb_idx = @as(isize, @intCast(j)) - sb_diff;
                    if (sb_idx >= 0 and src_b.shape[@intCast(sb_idx)] != 1)
                        cur_b -= (src_b.shape[@intCast(sb_idx)] - 1) * src_b.strides[@intCast(sb_idx)];
                }
            }
        }
    }
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
