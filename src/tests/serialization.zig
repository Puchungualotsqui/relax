const std = @import("std");
const allocator = std.testing.allocator;
const Tensor = @import("../tensor.zig").Tensor;
const Sequential = @import("../nn/model.zig").Sequential(f32);
const Dense = @import("../nn/layers/dense.zig").Dense(f32);
const initz = @import("../nn/initializers.zig");

test "Serialization: Save and Load" {
    const path = "test_model.bin";

    // Cleanup previous runs
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var rng = initz.RandomSource.init(42);

    // 1. Train a model briefly
    var model = Sequential.init(allocator);
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 2, .out_features = 2, .activation = .none }, &rng) });

    // Manually set weights to known values to simulate training
    // We can access params via parameters()
    {
        const params = try model.parameters();
        defer {
            for (params.items) |p| p.deinit();
            var p_list = params;
            p_list.deinit(allocator);
        }
        // Set Weight to 0.5, Bias to 0.1
        params.items[0].ptr.data.fill(0.5);
        params.items[1].ptr.data.fill(0.1);
    }

    // 2. Predict before saving
    var input = try Tensor(f32).fromSlice(allocator, &[_]usize{ 1, 2 }, &[_]f32{ 1.0, 2.0 });
    var pred1 = try model.predict(input);
    const p1_val = pred1.data[0];
    pred1.deinit();

    // 3. Save
    try model.save(path);
    model.deinit(); // Destroy model

    // 4. Create NEW model with same architecture (random weights)
    var model2 = Sequential.init(allocator);
    defer model2.deinit();
    try model2.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 2, .out_features = 2, .activation = .none }, &rng) });

    // 5. Load
    try model2.load(path);

    // 6. Predict with loaded model
    var pred2 = try model2.predict(input);
    defer pred2.deinit();

    input.deinit(); // Now free input

    const p2_val = pred2.data[0];

    std.debug.print("Original: {d}, Loaded: {d}\n", .{ p1_val, p2_val });

    try std.testing.expectEqual(p1_val, p2_val);
}
