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

    try mat.add(vec);

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
    try t1.add(t2);

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

    // 2x2 Matrix
    var mat = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 2 }, &[_]f32{ 1, 1, 1, 1 });
    defer mat.deinit();

    // 1D Vector [10] -> Should act like [10, 10] (broadcasted) or adding 10 to everything
    var vec = try Tensor(f32).fromSlice(allocator, &[_]usize{1}, &[_]f32{10});
    defer vec.deinit();

    try mat.add(vec);

    try std.testing.expectEqual(@as(f32, 11.0), mat.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 11.0), mat.at(&[_]usize{ 1, 1 }));
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
