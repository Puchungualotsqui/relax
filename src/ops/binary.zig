const base = @import("base.zig");

/// Performs element-wise addition: self = self + other
pub fn add(self: anytype, other: anytype) !void {
    const closures = struct {
        fn apply(d: *@TypeOf(self.data[0]), s: @TypeOf(self.data[0])) void {
            d.* += s;
        }
    };

    try base.broadcastOp(self, other, closures.apply);
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
