const std = @import("std");
const allocator = std.testing.allocator;
const VarF32 = @import("../autograd/variable.zig").Variable(f32);
const TensorF32 = @import("../tensor.zig").Tensor(f32);
const ops = @import("../autograd/ops.zig");

// 1. Basic Wrapper Test
test "Autograd: Variable wrapper and ownership" {
    // Create raw tensor
    const t = try TensorF32.fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 2.0 });

    // Wrap in Variable
    var v = try VarF32.init(allocator, t, true);
    defer v.deinit();

    // Verify Data Access (via .ptr.data)
    try std.testing.expectEqual(@as(f32, 1.0), v.ptr.data.at(&[_]usize{0}));

    // Verify Grad is initially null
    try std.testing.expect(v.ptr.grad == null);

    // Initialize Grad
    try v.zeroGrad();
    try std.testing.expect(v.ptr.grad != null);
    try std.testing.expectEqual(@as(f32, 0.0), v.ptr.grad.?.at(&[_]usize{0}));
}

// 2. Addition Backward Test
test "Autograd: Add Backward (c = a + b)" {
    // a = 10
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{10.0});
    var a = try VarF32.init(allocator, t_a, true);
    defer a.deinit();

    // b = 20
    const t_b = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{20.0});
    var b = try VarF32.init(allocator, t_b, true);
    defer b.deinit();

    // c = a + b = 30
    var c = try ops.add(allocator, a, b);
    defer c.deinit();

    // Verify Forward
    try std.testing.expectEqual(@as(f32, 30.0), c.ptr.data.at(&[_]usize{0}));
    try std.testing.expect(c.ptr.creator != null);

    // Backward
    try c.backward();

    // Verify Gradients
    // dL/da = 1.0
    const ga = try a.getGrad();
    try std.testing.expectEqual(@as(f32, 1.0), ga.at(&[_]usize{0}));

    // dL/db = 1.0
    const gb = try b.getGrad();
    try std.testing.expectEqual(@as(f32, 1.0), gb.at(&[_]usize{0}));
}

// 3. Gradient Accumulation Test (Crucial!)
// If we use 'a' twice: y = a + a, then dy/da should be 2.0
test "Autograd: Gradient Accumulation (y = a + a)" {
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{10.0});
    var a = try VarF32.init(allocator, t_a, true);
    defer a.deinit();

    // y = a + a
    var y = try ops.add(allocator, a, a); // We clone 'a' inside ops.add automatically
    defer y.deinit();

    try y.backward();

    const ga = try a.getGrad();
    // Should be 1.0 (from first usage) + 1.0 (from second usage) = 2.0
    try std.testing.expectEqual(@as(f32, 2.0), ga.at(&[_]usize{0}));
}

// 4. Matrix Multiplication Backward
test "Autograd: MatMul Backward" {
    // A (2x3)
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var a = try VarF32.init(allocator, t_a, true);
    defer a.deinit();

    // B (3x2)
    const t_b = try TensorF32.fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f32{ 7, 8, 9, 10, 11, 12 });
    var b = try VarF32.init(allocator, t_b, true);
    defer b.deinit();

    // C = A @ B (2x2)
    var c = try ops.matmul(allocator, a, b);
    defer c.deinit();

    try c.backward();

    // Verify dL/dA = dL/dC @ B^T
    // With dL/dC = 1s, result for A[0,0] should be 15.0
    const ga = try a.getGrad();
    try std.testing.expectEqual(@as(f32, 15.0), ga.at(&[_]usize{ 0, 0 }));

    // Verify dL/dB = A^T @ dL/dC
    // Result for B[0,0] should be 5.0
    const gb = try b.getGrad();
    try std.testing.expectEqual(@as(f32, 5.0), gb.at(&[_]usize{ 0, 0 }));
}

// 5. ReLU Backward Test
test "Autograd: ReLU Backward" {
    // Input: [-10, 10]
    const t_x = try TensorF32.fromSlice(allocator, &[_]usize{2}, &[_]f32{ -10.0, 10.0 });
    var x = try VarF32.init(allocator, t_x, true);
    defer x.deinit();

    var y = try ops.relu(allocator, x);
    defer y.deinit();

    // Forward: max(0, -10) = 0, max(0, 10) = 10
    try std.testing.expectEqual(@as(f32, 0.0), y.ptr.data.at(&[_]usize{0}));
    try std.testing.expectEqual(@as(f32, 10.0), y.ptr.data.at(&[_]usize{1}));

    try y.backward();

    // Backward: Gradient passes through positive, killed at negative
    const gx = try x.getGrad();

    // Grad for -10 should be 0
    try std.testing.expectEqual(@as(f32, 0.0), gx.at(&[_]usize{0}));
    // Grad for 10 should be 1.0 (passed through)
    try std.testing.expectEqual(@as(f32, 1.0), gx.at(&[_]usize{1}));
}
