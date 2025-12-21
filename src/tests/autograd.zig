const std = @import("std");
const allocator = std.testing.allocator;
const VarF32 = @import("../autograd/variable.zig").Variable(f32);
const TensorF32 = @import("../tensor.zig").Tensor(f32);
const FuncF32 = @import("../autograd/function.zig").Function(f32);
const ops = @import("../autograd/ops.zig");

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

test "Autograd: Add Operation" {
    // 1. Create Inputs (a=10, b=20)
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{10.0});
    var a = VarF32.init(allocator, t_a, true);
    defer a.deinit();

    const t_b = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{20.0});
    var b = VarF32.init(allocator, t_b, true);
    defer b.deinit();

    // 2. Perform Operation (c = a + b)
    var c = try ops.add(allocator, &a, &b);
    defer c.deinit();

    // Verify Forward
    try std.testing.expectEqual(@as(f32, 30.0), c.data.at(&[_]usize{0}));
    try std.testing.expect(c.creator != null); // It should have a creator

    // 3. Trigger Backward manually
    // We simulate the gradient coming from "loss" being 1.0
    var grad_c = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{1.0});
    defer grad_c.deinit();

    if (c.creator) |op| {
        try op.backward(grad_c);
    }

    // 4. Verify Gradients on Parents
    // dL/da = 1.0 * 1.0 = 1.0
    try std.testing.expect(a.grad != null);
    try std.testing.expectEqual(@as(f32, 1.0), a.grad.?.at(&[_]usize{0}));

    // dL/db = 1.0 * 1.0 = 1.0
    try std.testing.expect(b.grad != null);
    try std.testing.expectEqual(@as(f32, 1.0), b.grad.?.at(&[_]usize{0}));
}

test "Autograd: Full Backward Pass (c = a + b)" {
    // 1. Create Leafs (a=10, b=20)
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{10.0});
    var a = VarF32.init(allocator, t_a, true);
    defer a.deinit();

    const t_b = try TensorF32.fromSlice(allocator, &[_]usize{1}, &[_]f32{20.0});
    var b = VarF32.init(allocator, t_b, true);
    defer b.deinit();

    // 2. Forward (c = a + b)
    var c = try ops.add(allocator, &a, &b);
    defer c.deinit();

    // 3. Backward
    // This should:
    // - Set c.grad to 1.0
    // - Sort the graph (c -> a, b)
    // - Run AddBackward: adds c.grad to a.grad and b.grad
    try c.backward();

    // 4. Verify Gradients
    // dL/da = 1.0
    const ga = try a.getGrad();
    try std.testing.expectEqual(@as(f32, 1.0), ga.at(&[_]usize{0}));

    // dL/db = 1.0
    const gb = try b.getGrad();
    try std.testing.expectEqual(@as(f32, 1.0), gb.at(&[_]usize{0}));
}

test "Autograd: MatMul Backward" {
    // 1. Define Matrices
    // A (2x3)
    const t_a = try TensorF32.fromSlice(allocator, &[_]usize{ 2, 3 }, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var a = VarF32.init(allocator, t_a, true);
    defer a.deinit();

    // B (3x2)
    const t_b = try TensorF32.fromSlice(allocator, &[_]usize{ 3, 2 }, &[_]f32{ 7, 8, 9, 10, 11, 12 });
    var b = VarF32.init(allocator, t_b, true);
    defer b.deinit();

    // 2. Forward: C = A @ B
    // Expected C (2x2):
    // [1*7+2*9+3*11,  1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11,  4*8+5*10+6*12] = [139, 154]
    var c = try ops.matmul(allocator, &a, &b);
    defer c.deinit();

    // Verify Forward
    try std.testing.expectEqual(@as(f32, 58.0), c.data.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 154.0), c.data.at(&[_]usize{ 1, 1 }));

    // 3. Backward
    // We simulate a Loss = Sum(C).
    // Therefore, dL/dC is a matrix of 1.0s.
    // We need to set c.grad manually before calling backward,
    // OR just call c.backward() which defaults to setting grad to 1.0 (perfect for scalar outputs,
    // but for matrix outputs it implicitly treats it as sum reduction).

    try c.backward();

    // 4. Verify Gradients
    // dL/dA = dL/dC @ B^T
    //       = [[1, 1], [1, 1]] @ [[7, 9, 11], [8, 10, 12]]
    //       = [[15, 19, 23], [15, 19, 23]]
    const ga = try a.getGrad();
    try std.testing.expectEqual(@as(f32, 15.0), ga.at(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 23.0), ga.at(&[_]usize{ 0, 2 }));
    try std.testing.expectEqual(@as(f32, 15.0), ga.at(&[_]usize{ 1, 0 }));

    // dL/dB = A^T @ dL/dC
    //       = [[1, 4], [2, 5], [3, 6]] @ [[1, 1], [1, 1]]
    //       = [[5, 5], [7, 7], [9, 9]]
    const gb = try b.getGrad();
    try std.testing.expectEqual(@as(f32, 5.0), gb.at(&[_]usize{ 0, 0 })); // Row 0
    try std.testing.expectEqual(@as(f32, 7.0), gb.at(&[_]usize{ 1, 0 })); // Row 1
    try std.testing.expectEqual(@as(f32, 9.0), gb.at(&[_]usize{ 2, 1 })); // Row 2, Col 1
}
