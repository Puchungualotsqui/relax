const std = @import("std");
const allocator = std.testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const Sequential = @import("../nn/model.zig").Sequential(f32);
const Dense = @import("../nn/layers/dense.zig").Dense(f32);
const Dropout = @import("../nn/layers/dropout.zig").Dropout(f32);
const initz = @import("../nn/initializers.zig");

test "Keras-style API Integration with Validation and Early Stopping" {
    var rng = initz.RandomSource.init(42);

    // 1. Build Model
    var model = Sequential.init(allocator);
    defer model.deinit();

    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 1, .out_features = 8, .activation = .relu }, &rng) });
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 8, .out_features = 1 }, &rng) });

    // 2. Compile (Using Adam)
    try model.compile(.{
        .optimizer = .adam,
        .lr = 0.02,
        .loss = .mse,
    });

    // 3. Create Training Data (y = 0.5x)
    var x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer x.deinit();
    var y = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 0.5, 1.0, 1.5, 2.0 });
    defer y.deinit();

    // 4. Create Validation Data
    var vx = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 1 }, &[_]f32{ 5.0, 6.0 });
    defer vx.deinit();
    var vy = try Tensor(f32).fromSlice(allocator, &[_]usize{ 2, 1 }, &[_]f32{ 2.5, 3.0 });
    defer vy.deinit();

    // 5. Fit with new Config
    // This will track val_loss, save the best weights, and stop if patience is reached.
    try model.fit(x, y, .{
        .epochs = 1000,
        .val_data = .{ .x = vx, .y = vy },
        .patience = 50, // Stop if no improvement for 50 epochs
        .verbose = true,
    });

    // 6. Predict
    var test_x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f32{7.0});
    defer test_x.deinit();

    var pred = try model.predict(test_x);
    defer pred.deinit();

    // Expected: 3.5 (since 7.0 * 0.5 = 3.5)
    const val = pred.data[0];
    std.debug.print("Prediction for 7.0: {d:.4}\n", .{val});

    // The best model was restored automatically, so accuracy should be high
    try std.testing.expect(std.math.approxEqAbs(f32, val, 3.5, 0.5));
}

test "Classification: Softmax + CrossEntropy (Sum > 1.0)" {
    var rng = initz.RandomSource.init(1234);

    // 1. Build Model (2 inputs -> 8 hidden -> 2 classes)
    var model = Sequential.init(allocator);
    defer model.deinit();

    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 2, .out_features = 8, .activation = .relu }, &rng) });
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 8, .out_features = 2, .activation = .{ .softmax = 1 } }, &rng) });

    // 2. Compile
    try model.compile(.{
        .optimizer = .adam,
        .lr = 0.05, // High LR for fast convergence on simple toy data
        .loss = .cross_entropy,
    });

    // 3. Create Training Data
    // Rule: Class 1 if (x1 + x2) > 1.0, else Class 0
    // Inputs: (4, 2)
    var x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f32{
        0.1, 0.2, // Sum=0.3 -> Class 0
        0.8, 0.9, // Sum=1.7 -> Class 1
        0.4, 0.4, // Sum=0.8 -> Class 0
        0.9, 0.5, // Sum=1.4 -> Class 1
    });
    defer x.deinit();

    // Targets: One-Hot Encoded (4, 2)
    // Class 0 = [1, 0], Class 1 = [0, 1]
    var y = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f32{
        1.0, 0.0, // Class 0
        0.0, 1.0, // Class 1
        1.0, 0.0, // Class 0
        0.0, 1.0, // Class 1
    });
    defer y.deinit();

    // 4. Fit
    try model.fit(x, y, .{
        .epochs = 200, // Enough for Adam to solve this
        .verbose = false, // Keep output clean
    });

    // 5. Predict / Verify
    // Test Case 1: (0.2, 0.1) -> Sum 0.3 -> Expect Class 0 ~[1, 0]
    var test_1 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 0.2, 0.1 });
    defer test_1.deinit();
    var p1 = try model.predict(test_1);
    defer p1.deinit();

    std.debug.print("Pred (0.2, 0.1): Class 0={d:.3}, Class 1={d:.3}\n", .{ p1.data[0], p1.data[1] });

    // Check that Class 0 probability is dominant
    try std.testing.expect(p1.data[0] > 0.8);
    try std.testing.expect(p1.data[1] < 0.2);

    // Test Case 2: (0.9, 0.9) -> Sum 1.8 -> Expect Class 1 ~[0, 1]
    var test_2 = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 0.9, 0.9 });
    defer test_2.deinit();
    var p2 = try model.predict(test_2);
    defer p2.deinit();

    std.debug.print("Pred (0.9, 0.9): Class 0={d:.3}, Class 1={d:.3}\n", .{ p2.data[0], p2.data[1] });

    // Check that Class 1 probability is dominant
    try std.testing.expect(p2.data[1] > 0.8);
    try std.testing.expect(p2.data[0] < 0.2);
}
