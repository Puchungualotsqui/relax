const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("../autograd/variable.zig").Variable; // Added Import
const dense_mod = @import("../nn/layers/dense.zig");
const model_mod = @import("../nn/model.zig");
const initz = @import("../nn/initializers.zig");
const acts = @import("../nn/activations.zig");
const Allocator = std.mem.Allocator;
const allocator = std.testing.allocator;
const loss = @import("../nn/loss.zig");

test "Dense layer usage (Clean API)" {
    // 1. Setup
    var rng = initz.RandomSource.init(42);
    const DenseF32 = @import("../nn/layers/dense.zig").Dense(f32);

    // 2. Define Layer
    const config = DenseF32.Config{
        .in_features = 2,
        .out_features = 3,
        .activation = .relu,
    };

    var layer = try DenseF32.init(allocator, config, &rng);
    defer layer.deinit();

    // 3. Wrap Input in Variable
    // Create raw tensor
    const t_input = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 0.5, -0.5 });
    // Wrap in Variable (requires_grad=false for inference/input)
    // Note: We don't defer t_input.deinit() because Variable takes ownership.
    var input = try Variable(f32).init(allocator, t_input, false);
    defer input.deinit();

    // 4. Run Forward (Returns a Variable)
    var output = try layer.forward(input);
    defer output.deinit();

    // 5. Verify Output (Access data via ptr.data)
    try std.testing.expectEqual(output.ptr.data.shape[1], 3);
}

test "Sequential model end-to-end" {
    // 1. Setup
    var rng = initz.RandomSource.init(1234);
    const DenseF32 = dense_mod.Dense(f32);
    const SequentialF32 = model_mod.Sequential(f32);

    // 2. Initialize Model
    var model = SequentialF32.init(allocator);
    defer model.deinit();

    // 3. Add Layers
    try model.add(.{ .dense = try DenseF32.init(allocator, .{ .in_features = 10, .out_features = 32, .activation = .relu }, &rng) });

    try model.add(.{
        .dense = try DenseF32.init(allocator, .{
            .in_features = 32,
            .out_features = 1,
            .activation = .sigmoid,
        }, &rng),
    });

    // 4. Create Input Variable
    const t_input = try Tensor(f32).init(allocator, &[_]usize{ 4, 10 });
    initz.uniform(t_input, &rng, -1.0, 1.0);

    var input = try Variable(f32).init(allocator, t_input, false);
    defer input.deinit();

    // 5. Forward Pass
    var output = try model.forward(input);
    defer output.deinit();

    // 6. Verify Output
    try std.testing.expectEqual(output.ptr.data.shape[0], 4);
    try std.testing.expectEqual(output.ptr.data.shape[1], 1);

    // Check constraints via ptr.data
    for (output.ptr.data.data) |val| {
        try std.testing.expect(val >= 0.0 and val <= 1.0);
    }
}

test "Loss Functions" {
    // Note: Loss functions currently accept raw Tensors in your implementation.
    // If you haven't updated loss.zig to take Variables yet, this test remains as is.
    // If you updated loss.zig, you'd need to wrap these in Variables too.

    // 1. Test MSE
    var p1 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 0.0, 0.0 });
    defer p1.deinit();
    var t1 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 1.0 });
    defer t1.deinit();

    var mse_val = try loss.mse(allocator, p1, t1);
    defer mse_val.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), mse_val.data[0]);

    // 2. Test Categorical Cross Entropy
    var logits = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 10.0, 0.0 });
    defer logits.deinit();
    var targets = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 1.0, 0.0 });
    defer targets.deinit();

    var ce_val = try loss.crossEntropy(allocator, logits, targets);
    defer ce_val.deinit();

    try std.testing.expect(ce_val.data[0] < 0.01);
}
