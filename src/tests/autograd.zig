const std = @import("std");
const allocator = std.testing.allocator;
const VarF32 = @import("../autograd/variable.zig").Variable(f32);
const TensorF32 = @import("../tensor.zig").Tensor(f32);

test "Autograd: Variable wrapper" {
    // 1. Create a raw tensor
    const t = try TensorF32.fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 2.0 });
    // Note: We do NOT defer t.deinit() here because the Variable will take ownership.

    // 2. Wrap it in a Variable
    var v = VarF32.init(allocator, t, true);
    defer v.deinit(); // This will free 't' inside 'v'

    // 3. Verify Data
    try std.testing.expectEqual(@as(f32, 1.0), v.data.at(&[_]usize{0}));

    // 4. Verify Grad is initially null
    try std.testing.expect(v.grad == null);

    // 5. Initialize Grad
    try v.zeroGrad();
    try std.testing.expect(v.grad != null);
    try std.testing.expectEqual(@as(f32, 0.0), v.grad.?.at(&[_]usize{0}));
}
