const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const base = @import("../ops/base.zig");
const Allocator = std.mem.Allocator;

/// Mean Squared Error: mean((preds - targets)^2)
/// Used for Regression.
pub fn mse(allocator: Allocator, preds: anytype, targets: anytype) !@TypeOf(preds) {
    // 1. Diff = preds - targets
    var diff = try preds.sub(targets, allocator);
    defer diff.deinit();

    // 2. Square = Diff * Diff (in-place)
    // We can use powScalar(2) or just mul(diff). Let's use mulInPlace for speed.
    // Actually, we need to multiply diff by itself.
    var squared = try diff.mul(diff, allocator);
    defer squared.deinit();

    // 3. Mean over ALL dimensions (Result is a 0D scalar tensor)
    // For now, our .mean() reduces one axis. To get a global mean,
    // we can flatten first, or just sum all and divide by count.

    // Efficient approach: Flatten view -> Mean(axis=0)
    var flat = try squared.flatten();
    defer flat.deinit();

    return try flat.mean(allocator, 0);
}

/// Categorical Cross Entropy (from Logits)
/// Loss = -Sum(Targets * LogSoftmax(Preds)) / BatchSize
/// This is numerically stable because it combines Log and Softmax.
pub fn crossEntropy(allocator: Allocator, logits: anytype, targets: anytype) !@TypeOf(logits) {
    // 1. LogSoftmax = Logits - LogSumExp(Logits)
    // We assume the last dimension (axis=ndim-1) is the class dimension.
    const axis = logits.shape.len - 1;

    var lse = try logits.logSumExp(allocator, axis);
    defer lse.deinit();

    // Broadcast sub: logits - lse
    // Note: lse has rank N-1, we need to broadcast it back to N.
    // Our broadcast engine handles this if shapes align (e.g. (Batch, 1) vs (Batch, Class))
    // But logSumExp removes the dim. We need to unsqueeze it back.
    var lse_bcast = try lse.unsqueeze(axis);
    defer lse_bcast.deinit();

    var log_probs = try logits.sub(lse_bcast, allocator);
    defer log_probs.deinit();

    // 2. Target * LogProbs
    var terms = try targets.mul(log_probs, allocator);
    defer terms.deinit();

    // 3. Negative Sum over classes (axis 1), then Mean over batch (axis 0)
    // Standard reduction: Sum the class probabilities
    var class_sum = try terms.sum(allocator, axis);
    defer class_sum.deinit();

    // Now we have (BatchSize,). Calculate mean over batch.
    var batch_mean = try class_sum.mean(allocator, 0);

    // Negate the result (Loss is negative log likelihood)
    batch_mean.mulScalar(-1.0);

    return batch_mean;
}

/// Binary Cross Entropy (from Logits)
/// For binary classification (Sigmoid).
/// Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
/// where x = logits, z = targets.
pub fn binaryCrossEntropy(allocator: Allocator, logits: anytype, targets: anytype) !@TypeOf(logits) {
    const T = @TypeOf(logits.data[0]);
    var out = try @TypeOf(logits).init(allocator, logits.shape);
    errdefer out.deinit();

    const closure = struct {
        fn apply(d: *T, x: T, z: T) void {
            // Stable BCEWithLogits formula
            const max_val = if (x > 0) x else 0;
            const abs_x = @abs(x);
            d.* = max_val - (x * z) + std.math.log(T, 1.0 + std.math.exp(-abs_x));
        }
    }.apply;

    try base.broadcastOp2(&out, logits, targets, closure);

    // Reduce to scalar (Mean)
    var flat = try out.flatten();
    defer flat.deinit();
    defer out.deinit();

    return try flat.mean(allocator, 0);
}
