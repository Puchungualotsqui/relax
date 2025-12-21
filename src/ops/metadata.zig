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
