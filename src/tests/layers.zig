const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
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
    // NO WRAPPERS! Just .activation = .relu
    const config = DenseF32.Config{
        .in_features = 2,
        .out_features = 3,
        .activation = .relu,
    };

    var layer = try DenseF32.init(allocator, config, &rng);
    defer layer.deinit();

    // 3. Run
    var input = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 0.5, -0.5 });
    defer input.deinit();

    var output = try layer.forward(input);
    defer output.deinit();

    try std.testing.expectEqual(output.shape[1], 3);
}

test "Sequential model end-to-end" {
    // 1. Setup
    var rng = initz.RandomSource.init(1234);
    const DenseF32 = dense_mod.Dense(f32);
    const SequentialF32 = model_mod.Sequential(f32);

    // 2. Initialize Model
    var model = SequentialF32.init(allocator);
    defer model.deinit();

    // 3. Add Layers (Input: 10 -> Hidden: 32 -> Output: 1)
    try model.add(.{ .dense = try DenseF32.init(allocator, .{ .in_features = 10, .out_features = 32, .activation = .relu }, &rng) });

    try model.add(.{
        .dense = try DenseF32.init(allocator, .{
            .in_features = 32,
            .out_features = 1,
            .activation = .sigmoid, // Binary classification output
        }, &rng),
    });

    // 4. Create Dummy Batch (Batch Size = 4)
    // Input shape: (4, 10)
    var input = try Tensor(f32).init(allocator, &[_]usize{ 4, 10 });
    defer input.deinit();
    initz.uniform(input, &rng, -1.0, 1.0); // Random inputs

    // 5. Forward Pass
    var output = try model.forward(input);
    defer output.deinit();

    // 6. Verify Output Shape (4, 1)
    try std.testing.expectEqual(output.shape[0], 4);
    try std.testing.expectEqual(output.shape[1], 1);

    // Check constraints (Sigmoid output must be between 0 and 1)
    for (output.data) |val| {
        try std.testing.expect(val >= 0.0 and val <= 1.0);
    }
}

test "Loss Functions" {
    // 1. Test MSE
    // Preds: [0, 0], Targets: [1, 1] -> Diff: [-1, -1] -> Sq: [1, 1] -> Mean: 1.0
    var p1 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 0.0, 0.0 });
    defer p1.deinit();
    var t1 = try Tensor(f32).fromSlice(allocator, &[_]usize{2}, &[_]f32{ 1.0, 1.0 });
    defer t1.deinit();

    var mse_val = try loss.mse(allocator, p1, t1);
    defer mse_val.deinit();

    // Result should be 1.0
    try std.testing.expectEqual(@as(f32, 1.0), mse_val.data[0]);

    // 2. Test Categorical Cross Entropy (Stable)
    // Logits: [10.0, 0.0] (High confidence class 0)
    // Target: [1.0, 0.0] (Class 0 is true)
    // Softmax([10, 0]) approx [1.0, 0.0]. Log(1.0) is 0. Loss should be near 0.
    var logits = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 10.0, 0.0 });
    defer logits.deinit();
    var targets = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 1.0, 0.0 });
    defer targets.deinit();

    var ce_val = try loss.crossEntropy(allocator, logits, targets);
    defer ce_val.deinit();

    // Loss should be very small (~0.000045)
    try std.testing.expect(ce_val.data[0] < 0.01);
}
