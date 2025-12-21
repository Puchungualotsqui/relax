const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const initz = @import("../nn/initializers.zig");
const acts = @import("../nn/activations.zig");
const Allocator = std.mem.Allocator;
const allocator = std.testing.allocator;

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
