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
