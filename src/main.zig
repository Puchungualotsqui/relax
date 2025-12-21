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
    // 1. Reference the Tensor Math tests
    _ = @import("tests/tensor.zig");

    // 2. Reference the Layer tests
    _ = @import("tests/layers.zig");

    _ = @import("tests/autograd.zig");
}
