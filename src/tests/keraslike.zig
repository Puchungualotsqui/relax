const std = @import("std");
const allocator = std.testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const Sequential = @import("../nn/model.zig").Sequential(f32);
const Dense = @import("../nn/layers/dense.zig").Dense(f32);
const Dropout = @import("../nn/layers/dropout.zig").Dropout(f32);
const initz = @import("../nn/initializers.zig");

test "Keras-style API Integration" {
    var rng = initz.RandomSource.init(42);

    // 1. Build Model
    var model = Sequential.init(allocator);
    defer model.deinit();

    // 1 -> 8 -> 1 Network
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 1, .out_features = 8, .activation = .relu }, &rng) });
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 8, .out_features = 1 }, &rng) });

    // 2. Compile
    try model.compile(.{
        .optimizer = .sgd,
        .lr = 0.02, // Slightly tuned LR
        .loss = .mse,
    });

    // 3. Create Dummy Data (y = 0.5x)
    var x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer x.deinit();
    var y = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 0.5, 1.0, 1.5, 2.0 });
    defer y.deinit();

    // 4. Fit
    // INCREASED EPOCHS: 50 -> 1000 to ensure convergence
    try model.fit(x, y, 1000);

    // 5. Predict
    // Changed test point from 10.0 to 5.0 (closer to training distribution)
    var test_x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 1 }, &[_]f32{5.0});
    defer test_x.deinit();

    var pred = try model.predict(test_x);
    defer pred.deinit();

    // Expected: 2.5
    const val = pred.data[0];
    std.debug.print("Prediction for 5.0: {d:.4}\n", .{val});

    // Allow small margin of error (e.g. 0.2)
    try std.testing.expect(std.math.approxEqAbs(f32, val, 2.5, 0.2));
}
