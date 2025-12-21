const std = @import("std");
const allocator = std.testing.allocator;
const VarF32 = @import("../autograd/variable.zig").Variable(f32);
const TensorF32 = @import("../tensor.zig").Tensor(f32);
const FuncF32 = @import("../autograd/function.zig").Function(f32);

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

test "Autograd: Graph Connection" {
    // 1. Define a Mock Operation
    // It captures an integer 'id' just to prove we can store state.
    const MockOp = struct {
        base: FuncF32, // Inheritance by embedding
        allocator: std.mem.Allocator,
        id: usize,
        was_run: *bool,

        fn backward(ptr: *FuncF32, grad_output: TensorF32) !void {
            // Downcast: Get the MockOp pointer from the base pointer
            const self: *@This() = @fieldParentPtr("base", ptr);
            self.was_run.* = true;
            // In a real op, we would use grad_output here
            _ = grad_output;
        }

        fn deinit(ptr: *FuncF32) void {
            const self: *@This() = @fieldParentPtr("base", ptr);
            self.allocator.destroy(self);
        }
    };

    // 2. Setup Variable
    const t = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{10.0});
    var v = VarF32.init(allocator, t, true);
    defer v.deinit();

    // 3. Create and Attach the Mock Creator
    var run_flag = false;
    const op = try allocator.create(MockOp);
    op.* = .{
        .base = .{
            .backward_fn = MockOp.backward,
            .deinit_fn = MockOp.deinit,
        },
        .allocator = allocator,
        .id = 123,
        .was_run = &run_flag,
    };

    // Attach to variable
    v.creator = &op.base;

    // 4. Trigger the graph
    // We simulate a backward pass
    var dummy_grad = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{1.0});
    defer dummy_grad.deinit();

    if (v.creator) |c| {
        try c.backward(dummy_grad);
    }

    // 5. Verify
    try std.testing.expect(run_flag == true);
}
