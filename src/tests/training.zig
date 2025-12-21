const std = @import("std");
const allocator = std.testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("../autograd/variable.zig").Variable;
const Sequential = @import("../nn/model.zig").Sequential(f32);
const Dense = @import("../nn/layers/dense.zig").Dense(f32);
const SGD = @import("../nn/optimizers/sgd.zig").SGD(f32);
const ops = @import("../autograd/ops.zig");
const initz = @import("../nn/initializers.zig");

test "Integration: Training a Network (y = 2x)" {
    var rng = initz.RandomSource.init(42);

    // 1. Define Model: 1 Input -> Hidden(4) -> 1 Output
    // We use no activation on output (Linear regression)
    var model = Sequential.init(allocator);
    defer model.deinit();

    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 1, .out_features = 4, .activation = .relu }, &rng) });
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 4, .out_features = 1, .activation = .none }, &rng) });

    // 2. Setup Optimizer
    const params = try model.parameters();
    // Learning Rate 0.01
    var optimizer = SGD.init(allocator, params, 0.01);
    defer optimizer.deinit();

    // 3. Create Dataset (y = 2x)
    // Input: [[1], [2], [3], [4]]
    const t_x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    // Target: [[2], [4], [6], [8]]
    const t_y = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{ 2.0, 4.0, 6.0, 8.0 });

    // Wrap in Variables
    var x = try Variable(f32).init(allocator, t_x, false);
    defer x.deinit();
    var y = try Variable(f32).init(allocator, t_y, false);
    defer y.deinit();

    // 4. Training Loop
    const epochs = 100;
    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;

    for (0..epochs) |epoch| {
        // A. Zero Gradients
        try optimizer.zeroGrad();

        // B. Forward Pass
        var preds = try model.forward(x);
        // Note: preds is part of the graph. We defer its deinit.
        defer preds.deinit();

        // C. Compute Loss (Autograd-aware)
        var loss_var = try ops.mse_loss(allocator, preds, y);
        defer loss_var.deinit();

        if (epoch == 0) initial_loss = loss_var.ptr.data.data[0];
        final_loss = loss_var.ptr.data.data[0];

        // D. Backward Pass
        try loss_var.backward();

        // E. Update Weights
        try optimizer.step();
    }

    std.debug.print("\nTraining Results:\nInitial Loss: {d:.4}\nFinal Loss:   {d:.4}\n", .{ initial_loss, final_loss });

    // 5. Verification
    try std.testing.expect(final_loss < initial_loss);
    try std.testing.expect(final_loss < 0.1); // Should converge close to 0
}
