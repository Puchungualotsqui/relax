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
