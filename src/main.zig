const std = @import("std");

pub const Tensor = @import("tensor.zig").Tensor;
const linalg = @import("ops/linalg.zig");
const base = @import("ops/base.zig");
const metadata = @import("ops/metadata.zig");
const reductions = @import("ops/reductions.zig");
const binary = @import("ops/binary.zig");
const unary = @import("ops/unary.zig");
pub const nn = struct {
    pub const layers = @import("nn/layers/dense.zig");
    pub const activations = @import("nn/activations.zig");
    pub const initializers = @import("nn/initializers.zig");
};

test {
    _ = @import("tests/tensor.zig");

    _ = @import("tests/layers.zig");

    _ = @import("tests/autograd.zig");

    _ = @import("tests/keraslike.zig");

    _ = @import("tests/serialization.zig");
}
