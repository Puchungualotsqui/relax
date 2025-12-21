const std = @import("std");
pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;

test {
    // This will run all tests in tensor.zig and ops/elementwise.zig as well
    std.testing.refAllDecls(@This());
}

test "broadcasting: add vector to matrix" {
    const allocator = std.testing.allocator;

    // Matrix (2x3)
    // [1, 2, 3]
    // [4, 5, 6]
    var mat = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer mat.deinit();

    // Vector (1x3) - Will be broadcasted across rows
    // [10, 20, 30]
    var vec = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 3 }, &[_]f32{ 10, 20, 30 });
    defer vec.deinit();

    try mat.addInPlace(vec);

    try std.testing.expectEqual(@as(f32, 11.0), mat.at(&[_]usize{ 0, 0 })); // 1 + 10
    try std.testing.expectEqual(@as(f32, 22.0), mat.at(&[_]usize{ 0, 1 })); // 2 + 20
    try std.testing.expectEqual(@as(f32, 36.0), mat.at(&[_]usize{ 1, 2 })); // 6 + 30
}

test "transpose and add (non-contiguous check)" {
    const allocator = std.testing.allocator;

    // Create 2x2: [1, 2]
    //             [3, 4]
    var t1 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1, 2, 3, 4 });
    defer t1.deinit();

    // Create another 2x2: [10, 20]
    //                     [30, 40]
    var t2 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 10, 20, 30, 40 });
    defer t2.deinit();

    // Transpose t2 in-place: [10, 30]
    //                       [20, 40]
    t2.transpose();

    // This triggers the "Silent Clone" because t2 is now non-contiguous
    try t1.addInPlace(t2);

    // Expected t1: [1+10, 2+30] -> [11, 32]
    //              [3+20, 4+40] -> [23, 44]
    try std.testing.expectEqual(@as(f32, 11.0), t1.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 32.0), t1.at(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 23.0), t1.at(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 44.0), t1.at(&[_]usize{ 1, 1 }));
}

test "reshape in-place" {
    const allocator = std.testing.allocator;

    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 4 }, &[_]f32{ 1, 2, 3, 4 });
    defer t.deinit();

    try t.reshape(&[_]usize{ 2, 2 });

    try std.testing.expectEqual(@as(usize, 2), t.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), t.shape[1]);
    try std.testing.expectEqual(@as(f32, 3.0), t.at(&[_]usize{ 1, 0 }));
}

test "broadcasting different ranks" {
    const allocator = std.testing.allocator;
    var mat = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1, 1, 1, 1 });
    defer mat.deinit();

    var vec = try Tensor(f32).fromSlice(allocator, &[_]usize{1}, &[_]f32{10});
    defer vec.deinit();

    // Must use add() and catch new tensor because result is a new view/allocation
    var res = try mat.add(vec, allocator);
    defer res.deinit();

    try std.testing.expectEqual(@as(f32, 11.0), res.at(&[_]usize{ 0, 0 }));
}

test "batched matrix multiplication" {
    const allocator = std.testing.allocator;

    // Shape [2, 2, 2] -> 2 matrices of 2x2
    const shape = &[_]usize{ 2, 2, 2 };
    var a = try Tensor(f32).fromSlice(allocator, shape, &[_]f32{
        1, 0, 0, 1, // Matrix 0 (Identity)
        2, 0, 0, 2, // Matrix 1 (2 * Identity)
    });
    defer a.deinit();

    var b = try Tensor(f32).fromSlice(allocator, shape, &[_]f32{
        4, 5, 6, 7, // Matrix 0
        1, 1, 1, 1, // Matrix 1
    });
    defer b.deinit();

    var res = try Tensor(f32).init(allocator, shape);
    defer res.deinit();
    res.fill(0);

    try a.matmul(b, &res);

    // Batch 0: Identity * [4,5,6,7] = [4,5,6,7]
    try std.testing.expectEqual(@as(f32, 4.0), res.at(&[_]usize{ 0, 0, 0 }));
    // Batch 1: 2*Identity * [1,1,1,1] = [2,2,2,2]
    try std.testing.expectEqual(@as(f32, 2.0), res.at(&[_]usize{ 1, 1, 1 }));
}

test "tensor sum reduction" {
    const allocator = std.testing.allocator;

    // 2x3 Matrix
    // [1, 2, 3]
    // [4, 5, 6]
    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer t.deinit();

    // Reduce along axis 0 (rows) -> Result should be [5, 7, 9]
    var res = try t.sum(allocator, 0);
    defer res.deinit();

    try std.testing.expectEqual(@as(usize, 1), res.shape.len);
    try std.testing.expectEqual(@as(f32, 5.0), res.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 7.0), res.at(&[_]usize{1}));
    try std.testing.expectEqual(@as(f32, 9.0), res.at(&[_]usize{2}));
}

test "tensor concatenation" {
    const allocator = std.testing.allocator;

    // t1: [1, 2]
    var t1 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 1, 2 });
    defer t1.deinit();

    // t2: [3, 4]
    var t2 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 3, 4 });
    defer t2.deinit();

    const inputs = &[_]Tensor(f32){ t1, t2 };

    // Concatenate along axis 0 -> Result shape [2, 2]: [[1, 2], [3, 4]]
    var res = try Tensor(f32).concatenate(allocator, inputs, 0);
    defer res.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), res.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 3.0), res.at(&[_]usize{ 1, 0 }));
}

test "tensor slicing" {
    const allocator = std.testing.allocator;

    // 4x4 Matrix:
    //  0   1   2   3
    //  4   5   6   7
    //  8   9  10  11
    // 12  13  14  15
    var t = try Tensor(f32).init(allocator, &[_]usize{ 4, 4 });
    defer t.deinit();

    // Fill with 0..15
    for (0..16) |i| t.data[i] = @as(f32, @floatFromInt(i));

    // Slice rows 1 to 3 (exclusive) -> Rows 1 and 2
    //  4   5   6   7
    //  8   9  10  11
    var view = try t.slice(0, 1, 3);
    defer view.deinit(); // Decrements ref_count, doesn't free storage yet

    try std.testing.expectEqual(@as(usize, 2), view.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), view.shape[1]);

    // Check data
    // view.at(0, 0) should be t.at(1, 0) -> 4
    try std.testing.expectEqual(@as(f32, 4.0), view.at(&[_]usize{ 0, 0 }));
    // view.at(1, 3) should be t.at(2, 3) -> 11
    try std.testing.expectEqual(@as(f32, 11.0), view.at(&[_]usize{ 1, 3 }));

    // Modify View -> Should affect Original (Shared Storage)
    view.data[0] = 999.0;
    try std.testing.expectEqual(@as(f32, 999.0), t.at(&[_]usize{ 1, 0 }));
}

test "comparison result values" {
    const allocator = std.testing.allocator;

    var a = try Tensor(f32).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 1, 5, 2, 8 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 4, 4, 4, 4 });
    defer b.deinit();

    var mask = try a.gt(b, allocator);
    defer mask.deinit();

    // Mask should be [0, 1, 0, 1] (u8)
    try std.testing.expectEqual(@as(u8, 0), mask.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(u8, 1), mask.at(&[_]usize{1}));
    try std.testing.expectEqual(@as(u8, 0), mask.at(&[_]usize{2}));
    try std.testing.expectEqual(@as(u8, 1), mask.at(&[_]usize{3}));
}

test "argmax reduction" {
    const allocator = std.testing.allocator;

    // 2x3 Matrix
    // [[1, 10, 3],
    //  [4,  5, 6]]
    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 10, 3, 4, 5, 6 });
    defer t.deinit();

    // Argmax along axis 1 (across columns for each row)
    // Row 0: max is 10 at index 1
    // Row 1: max is 6 at index 2
    var indices = try t.argmax(allocator, 1);
    defer indices.deinit();

    try std.testing.expectEqual(@as(usize, 1), indices.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(usize, 2), indices.at(&[_]usize{1}));
}

test "mean and variance" {
    const allocator = std.testing.allocator;

    // 1D Tensor [1, 2, 3, 4, 5, 6] -> Mean 3.5
    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{6}, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer t.deinit();

    const mu = try t.mean(allocator, 0);
    defer mu.deinit();
    try std.testing.expectEqual(@as(f32, 3.5), mu.data[0]);

    const res = try t.var_std(allocator, 0);
    defer res.v.deinit();
    defer res.s.deinit();

    // Variance of 1..6 is ~2.9167
    try std.testing.expect(std.math.approxEqAbs(f32, 2.9167, res.v.data[0], 0.001));
}

test "tensor where selection (method API)" {
    const allocator = std.testing.allocator;

    // 1. Setup Mask [1, 0, 1, 0]
    var mask = try Tensor(u8).fromSlice(allocator, &[_]usize{4}, &[_]u8{ 1, 0, 1, 0 });
    defer mask.deinit();

    // 2. Setup A and B
    var a = try Tensor(f32).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 10, 10, 10, 10 });
    defer a.deinit();
    var b = try Tensor(f32).fromSlice(allocator, &[_]usize{4}, &[_]f32{ 20, 20, 20, 20 });
    defer b.deinit();

    // 3. Use the method API: mask.select(a, b, allocator)
    // This internally calls ops.where and handles allocation
    var res = try mask.select(a, b, allocator);
    defer res.deinit();

    // Expected: [10, 20, 10, 20]
    try std.testing.expectEqual(@as(f32, 10.0), res.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 20.0), res.at(&[_]usize{1}));
    try std.testing.expectEqual(@as(f32, 10.0), res.at(&[_]usize{2}));
    try std.testing.expectEqual(@as(f32, 20.0), res.at(&[_]usize{3}));
}

test "where with broadcasting (method API)" {
    const allocator = std.testing.allocator;

    // Mask (2, 2)
    var mask = try Tensor(u8).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]u8{ 1, 0, 0, 1 });
    defer mask.deinit();

    // A is a scalar-like (1,) [99]
    var a = try Tensor(f32).fromSlice(allocator, &[_]usize{1}, &[_]f32{99.0});
    defer a.deinit();

    // B is a matching (2, 2) [1, 2, 3, 4]
    var b = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1, 2, 3, 4 });
    defer b.deinit();

    // Use the method API
    var res = try mask.select(a, b, allocator);
    defer res.deinit();

    // Expected: [[99, 2], [3, 99]]
    try std.testing.expectEqual(@as(f32, 99.0), res.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2.0), res.at(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 3.0), res.at(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 99.0), res.at(&[_]usize{ 1, 1 }));
}

test "tensor clipping (method API)" {
    const allocator = std.testing.allocator;

    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{4}, &[_]f32{ -10.0, 0.5, 10.0, 2.0 });
    defer t.deinit();

    // Test In-place clip
    try t.clipInPlace(0.0, 1.0);

    try std.testing.expectEqual(@as(f32, 0.0), t.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 0.5), t.at(&[_]usize{1}));
    try std.testing.expectEqual(@as(f32, 1.0), t.at(&[_]usize{2}));

    // Test Allocating clip
    var t2 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ -5.0, 5.0 });
    defer t2.deinit();

    var res = try t2.clipped(allocator, -1.0, 1.0);
    defer res.deinit();

    try std.testing.expectEqual(@as(f32, -1.0), res.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 1.0), res.at(&[_]usize{1}));
}

test "logSumExp stability and correctness" {
    const allocator = std.testing.allocator;

    // Input: [1.0, 2.0, 3.0]
    // Standard Math: log(exp(1) + exp(2) + exp(3))
    // = log(2.718 + 7.389 + 20.085) = log(30.192) ≈ 3.4076
    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{3}, &[_]f32{ 1.0, 2.0, 3.0 });
    defer t.deinit();

    var res = try t.logSumExp(allocator, 0);
    defer res.deinit();

    try std.testing.expect(std.math.approxEqAbs(f32, 3.4076, res.data[0], 0.001));
}

test "scalar operations" {
    const allocator = std.testing.allocator;

    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{2, 2}, &[_]f32{ 1, 2, 3, 4 });
    defer t.deinit();

    // Test In-place Add
    t.addScalar(10.0);
    try std.testing.expectEqual(@as(f32, 11.0), t.at(&[_]usize{ 0, 0 }));

    // Test In-place Mul
    t.mulScalar(2.0);
    try std.testing.expectEqual(@as(f32, 22.0), t.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 28.0), t.at(&[_]usize{ 1, 1 }));

    // Test In-place Pow (2^2, 3^2, etc on original-ish values)
    var t2 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 2.0, 3.0 });
    defer t2.deinit();
    t2.powScalar(2.0);
    try std.testing.expectEqual(@as(f32, 4.0), t2.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 9.0), t2.at(&[_]usize{1}));
}

test "metadata views: squeeze, unsqueeze, permute" {
    const allocator = std.testing.allocator;

    // Start with (2, 3)
    var t = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer t.deinit();

    // 1. Unsqueeze -> (2, 1, 3)
    var t_un = try t.unsqueeze(1);
    defer t_un.deinit();
    try std.testing.expectEqual(@as(usize, 1), t_un.shape[1]);
    try std.testing.expectEqual(@as(f32, 4.0), t_un.at(&[_]usize{ 1, 0, 0 }));

    // 2. Squeeze -> Back to (2, 3)
    var t_sq = try t_un.squeeze(1);
    defer t_sq.deinit();
    try std.testing.expectEqual(@as(usize, 2), t_sq.shape.len);

    // 3. Permute (swap 0 and 1) -> (3, 2)
    var t_per = try t.permute(&[_]usize{ 1, 0 });
    defer t_per.deinit();
    try std.testing.expectEqual(@as(usize, 3), t_per.shape[0]);
    // Logical (1, 0) in (3, 2) is physical (0, 1) in (2, 3) which is '2'
    try std.testing.expectEqual(@as(f32, 2.0), t_per.at(&[_]usize{ 1, 0 }));
}

test "fused matmul with bias (linear layer)" {
    const allocator = std.testing.allocator;

    // X: (2, 2)
    var x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1, 2, 3, 4 });
    defer x.deinit();

    // W: (2, 2)
    var w = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 10, 20, 30, 40 });
    defer w.deinit();

    // B: (2,)
    var b = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 5, 10 });
    defer b.deinit();

    // Destination: (2, 2)
    var res = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer res.deinit();

    // Using the method API on the tensor itself
    try x.linear(w, b, &res);

    try std.testing.expectEqual(@as(f32, 75.0), res.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 110.0), res.at(&[_]usize{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 155.0), res.at(&[_]usize{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 230.0), res.at(&[_]usize{ 1, 1 }));
}
