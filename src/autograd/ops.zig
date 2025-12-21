const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;
const Function = @import("function.zig").Function;
const Allocator = std.mem.Allocator;

// --- Addition Operation (With Broadcasting for Bias) ---

pub fn AddBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input_a: Variable(T),
        input_b: Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            // Helper to apply gradient, reducing if necessary (Rank 2 -> Rank 1)
            const apply_grad = struct {
                fn run(target_var: Variable(T), source_grad: Tensor(T)) !void {
                    const target_grad = try target_var.getGrad();

                    // Case 1: Exact Match
                    if (std.mem.eql(usize, target_grad.shape, source_grad.shape)) {
                        try target_grad.addInPlace(source_grad);
                        return;
                    }

                    // Case 2: Bias Reduction (2D -> 1D)
                    // Gradient is (Batch, Features), Target is (Features)
                    if (source_grad.shape.len == 2 and target_grad.shape.len == 1) {
                        if (source_grad.shape[1] != target_grad.shape[0]) return error.IncompatibleShapes;

                        // Sum over batch dimension (axis 0)
                        const batch_size = source_grad.shape[0];
                        const features = source_grad.shape[1];
                        const s_data = source_grad.data;
                        const t_data = target_grad.data;

                        var i: usize = 0;
                        while (i < batch_size * features) {
                            for (0..features) |f| {
                                t_data[f] += s_data[i];
                                i += 1;
                            }
                        }
                        return;
                    }

                    return error.BroadcastBackwardNotImplemented;
                }
            }.run;

            if (self.input_a.ptr.requires_grad) {
                try apply_grad(self.input_a, grad);
            }
            if (self.input_b.ptr.requires_grad) {
                try apply_grad(self.input_b, grad);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input_a.ptr.creator) |c| try list.append(allocator, c);
            if (self.input_b.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input_a.deinit();
            self.input_b.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn add(allocator: Allocator, a: anytype, b: anytype) !@TypeOf(a) {
    const T = @TypeOf(a.ptr.data.data[0]);

    const data = try a.ptr.data.add(b.ptr.data, allocator);
    const needs_grad = a.ptr.requires_grad or b.ptr.requires_grad;

    var creator: ?*Function(T) = null;
    if (needs_grad) {
        const op = try allocator.create(AddBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = AddBackward(T).backward,
                .collect_parents_fn = AddBackward(T).collectParents,
                .get_grad_fn = AddBackward(T).getGrad,
                .deinit_fn = AddBackward(T).deinit,
            },
            .allocator = allocator,
            .input_a = a.clone(),
            .input_b = b.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- Matrix Multiplication Operation ---
// (Keep MatMulBackward and matmul exactly as they were in your previous code)
pub fn MatMulBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input_a: Variable(T),
        input_b: Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            if (self.input_a.ptr.requires_grad) {
                const ndim = self.input_b.ptr.data.shape.len;
                var dims = try self.allocator.alloc(usize, ndim);
                defer self.allocator.free(dims);
                for (0..ndim) |i| dims[i] = i;
                std.mem.swap(usize, &dims[ndim - 1], &dims[ndim - 2]);

                var b_T = try self.input_b.ptr.data.permute(dims);
                defer b_T.deinit();

                var d_a = try TensorT.init(self.allocator, self.input_a.ptr.data.shape);
                defer d_a.deinit();
                try grad.matmul(b_T, &d_a);

                const a_grad = try self.input_a.getGrad();
                try a_grad.addInPlace(d_a);
            }

            if (self.input_b.ptr.requires_grad) {
                const ndim = self.input_a.ptr.data.shape.len;
                var dims = try self.allocator.alloc(usize, ndim);
                defer self.allocator.free(dims);
                for (0..ndim) |i| dims[i] = i;
                std.mem.swap(usize, &dims[ndim - 1], &dims[ndim - 2]);

                var a_T = try self.input_a.ptr.data.permute(dims);
                defer a_T.deinit();

                var d_b = try TensorT.init(self.allocator, self.input_b.ptr.data.shape);
                defer d_b.deinit();
                try a_T.matmul(grad, &d_b);

                const b_grad = try self.input_b.getGrad();
                try b_grad.addInPlace(d_b);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input_a.ptr.creator) |c| try list.append(allocator, c);
            if (self.input_b.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input_a.deinit();
            self.input_b.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn matmul(allocator: Allocator, a: anytype, b: anytype) !@TypeOf(a) {
    const T = @TypeOf(a.ptr.data.data[0]);

    const a_ndim = a.ptr.data.shape.len;
    var out_shape = try allocator.alloc(usize, a_ndim);
    defer allocator.free(out_shape);
    @memcpy(out_shape, a.ptr.data.shape);
    out_shape[a_ndim - 1] = b.ptr.data.shape[b.ptr.data.shape.len - 1];

    var data = try Tensor(T).init(allocator, out_shape);
    try a.ptr.data.matmul(b.ptr.data, &data);

    const needs_grad = a.ptr.requires_grad or b.ptr.requires_grad;
    var creator: ?*Function(T) = null;
    if (needs_grad) {
        const op = try allocator.create(MatMulBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = MatMulBackward(T).backward,
                .collect_parents_fn = MatMulBackward(T).collectParents,
                .get_grad_fn = MatMulBackward(T).getGrad,
                .deinit_fn = MatMulBackward(T).deinit,
            },
            .allocator = allocator,
            .input_a = a.clone(),
            .input_b = b.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- ReLU Operation ---
// (Keep ReLUBackward/relu as is)
pub fn ReLUBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                for (0..self.input.ptr.data.data.len) |i| {
                    d_input.data[i] = if (self.input.ptr.data.data[i] > 0) grad.data[i] else 0;
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn relu(allocator: Allocator, a: anytype) !@TypeOf(a) {
    const T = @TypeOf(a.ptr.data.data[0]);
    const data = try a.ptr.data.clipped(allocator, 0, std.math.inf(T));

    const needs_grad = a.ptr.requires_grad;
    var creator: ?*Function(T) = null;
    if (needs_grad) {
        const op = try allocator.create(ReLUBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = ReLUBackward(T).backward,
                .collect_parents_fn = ReLUBackward(T).collectParents,
                .get_grad_fn = ReLUBackward(T).getGrad,
                .deinit_fn = ReLUBackward(T).deinit,
            },
            .allocator = allocator,
            .input = a.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- Sigmoid Operation ---
// (Keep Sigmoid as is)
pub fn SigmoidBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                const x_data = self.input.ptr.data.data;
                const g_data = grad.data;
                const d_data = d_input.data;

                for (0..x_data.len) |i| {
                    const s = 1.0 / (1.0 + std.math.exp(-x_data[i]));
                    d_data[i] = g_data[i] * s * (1.0 - s);
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn sigmoid(allocator: Allocator, a: anytype) !@TypeOf(a) {
    const T = @TypeOf(a.ptr.data.data[0]);

    var data = try Tensor(T).init(allocator, a.ptr.data.shape);
    const in_data = a.ptr.data.data;
    for (0..in_data.len) |i| {
        data.data[i] = 1.0 / (1.0 + std.math.exp(-in_data[i]));
    }

    const needs_grad = a.ptr.requires_grad;
    var creator: ?*Function(T) = null;
    if (needs_grad) {
        const op = try allocator.create(SigmoidBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = SigmoidBackward(T).backward,
                .collect_parents_fn = SigmoidBackward(T).collectParents,
                .get_grad_fn = SigmoidBackward(T).getGrad,
                .deinit_fn = SigmoidBackward(T).deinit,
            },
            .allocator = allocator,
            .input = a.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- MSE Operation ---
// (Keep MSE as is)
pub fn MSEBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T),
        target: Variable(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad_scale = self.output_grad.data[0];

            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                const pred = self.input.ptr.data.data;
                const targ = self.target.ptr.data.data;
                const N: T = @floatFromInt(pred.len);
                const factor = (2.0 / N) * grad_scale;

                for (0..pred.len) |i| {
                    d_input.data[i] = factor * (pred[i] - targ[i]);
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
            if (self.target.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input.deinit();
            self.target.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn mse_loss(allocator: Allocator, input: anytype, target: anytype) !@TypeOf(input) {
    const T = @TypeOf(input.ptr.data.data[0]);

    var loss_val: T = 0.0;
    const pred = input.ptr.data.data;
    const targ = target.ptr.data.data;
    for (0..pred.len) |i| {
        const diff = pred[i] - targ[i];
        loss_val += diff * diff;
    }
    loss_val /= @floatFromInt(pred.len);

    var data = try Tensor(T).init(allocator, &[_]usize{1});
    data.data[0] = loss_val;

    const needs_grad = input.ptr.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(MSEBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = MSEBackward(T).backward,
                .collect_parents_fn = MSEBackward(T).collectParents,
                .get_grad_fn = MSEBackward(T).getGrad,
                .deinit_fn = MSEBackward(T).deinit,
            },
            .allocator = allocator,
            .input = input.clone(),
            .target = target.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- Dropout Operation ---
// FIXED: Uses self.input (no broadcasting needed)

pub fn DropoutBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T),
        mask: Tensor(T),
        scale: T,
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            // CORRECT: Check self.input, NOT self.input_a
            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                // Element-wise multiply: grad * mask * scale
                const count = grad.data.len;
                for (0..count) |i| {
                    d_input.data[i] = grad.data[i] * self.mask.data[i] * self.scale;
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.mask.deinit();
            self.input.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn dropout(allocator: Allocator, input: anytype, probability: anytype, rng: std.Random) !@TypeOf(input) {
    const T = @TypeOf(input.ptr.data.data[0]);

    const keep_prob = 1.0 - probability;
    const scale = 1.0 / keep_prob;

    var mask = try Tensor(T).init(allocator, input.ptr.data.shape);
    const mask_data = mask.data;

    for (0..mask_data.len) |i| {
        mask_data[i] = if (rng.float(T) < keep_prob) 1.0 else 0.0;
    }

    const out_data = try Tensor(T).init(allocator, input.ptr.data.shape);
    const input_data = input.ptr.data.data;
    const out_slice = out_data.data;

    for (0..out_slice.len) |i| {
        out_slice[i] = input_data[i] * mask_data[i] * scale;
    }

    const needs_grad = input.ptr.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(DropoutBackward(T));
        var grad = try Tensor(T).init(allocator, out_data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = DropoutBackward(T).backward,
                .collect_parents_fn = DropoutBackward(T).collectParents,
                .get_grad_fn = DropoutBackward(T).getGrad,
                .deinit_fn = DropoutBackward(T).deinit,
            },
            .allocator = allocator,
            .input = input.clone(),
            .mask = mask,
            .scale = scale,
            .output_grad = grad,
        };
        creator = &op.base;
    } else {
        mask.deinit();
    }

    var out = try Variable(T).init(allocator, out_data, needs_grad);
    out.ptr.creator = creator;
    return out;
}

// --- Softmax Operation ---
pub fn SoftmaxBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T),
        // CHANGE 1: Store Tensor, not Variable, to break the cycle
        output: Tensor(T),
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad = self.output_grad;

            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                // CHANGE 2: Access data directly from the stored Tensor
                const y_data = self.output.data;
                const g_data = grad.data;
                const d_data = d_input.data;

                const batch_size = self.input.ptr.data.shape[0];
                const classes = self.input.ptr.data.shape[1];

                var offset: usize = 0;
                for (0..batch_size) |_| {
                    var sum_yg: T = 0;
                    for (0..classes) |c| {
                        sum_yg += y_data[offset + c] * g_data[offset + c];
                    }

                    for (0..classes) |c| {
                        const idx = offset + c;
                        d_data[idx] = y_data[idx] * (g_data[idx] - sum_yg);
                    }
                    offset += classes;
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input.deinit();
            self.output.deinit(); // Frees the Tensor ref
            self.allocator.destroy(self);
        }
    };
}

pub fn softmax(allocator: Allocator, input: anytype) !@TypeOf(input) {
    const T = @TypeOf(input.ptr.data.data[0]);
    if (input.ptr.data.shape.len != 2) return error.SoftmaxOnlySupports2D;

    const batch_size = input.ptr.data.shape[0];
    const classes = input.ptr.data.shape[1];
    const in_data = input.ptr.data.data;

    const out_data = try Tensor(T).init(allocator, input.ptr.data.shape);
    const y_data = out_data.data;

    var offset: usize = 0;
    for (0..batch_size) |_| {
        var max_val = in_data[offset];
        for (1..classes) |c| {
            if (in_data[offset + c] > max_val) max_val = in_data[offset + c];
        }

        var sum_exp: T = 0;
        for (0..classes) |c| {
            const val = std.math.exp(in_data[offset + c] - max_val);
            y_data[offset + c] = val;
            sum_exp += val;
        }

        for (0..classes) |c| {
            y_data[offset + c] /= sum_exp;
        }
        offset += classes;
    }

    const needs_grad = input.ptr.requires_grad;

    // We construct the result Variable
    var out = try Variable(T).init(allocator, out_data, needs_grad);

    var creator: ?*Function(T) = null;
    if (needs_grad) {
        const op = try allocator.create(SoftmaxBackward(T));
        var grad = try Tensor(T).init(allocator, out_data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = SoftmaxBackward(T).backward,
                .collect_parents_fn = SoftmaxBackward(T).collectParents,
                .get_grad_fn = SoftmaxBackward(T).getGrad,
                .deinit_fn = SoftmaxBackward(T).deinit,
            },
            .allocator = allocator,
            .input = input.clone(),
            // CHANGE 3: Store clone of the Tensor, not the Variable
            .output = try out.ptr.data.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    out.ptr.creator = creator;
    return out;
}

// --- Categorical Cross Entropy Loss ---
// Loss = - Sum(Target * log(Prediction))

pub fn CrossEntropyBackward(comptime T: type) type {
    return struct {
        base: Function(T),
        allocator: Allocator,
        input: Variable(T), // Predictions (Probabilities)
        target: Variable(T), // One-Hot Targets
        output_grad: Tensor(T),

        const Self = @This();
        const TensorT = Tensor(T);

        pub fn backward(ptr: *Function(T)) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            const grad_scale = self.output_grad.data[0];

            if (self.input.ptr.requires_grad) {
                var d_input = try TensorT.init(self.allocator, self.input.ptr.data.shape);
                defer d_input.deinit();

                const pred = self.input.ptr.data.data;
                const targ = self.target.ptr.data.data;
                const N: T = @floatFromInt(self.input.ptr.data.shape[0]); // Batch size
                const epsilon: T = 1e-7; // Avoid division by zero

                // dL/dp = -t / p
                // We also divide by Batch Size (N) for mean reduction
                for (0..pred.len) |i| {
                    const p = if (pred[i] < epsilon) epsilon else pred[i];
                    d_input.data[i] = (-targ[i] / p) * (grad_scale / N);
                }

                const input_grad = try self.input.getGrad();
                try input_grad.addInPlace(d_input);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            if (self.input.ptr.creator) |c| try list.append(allocator, c);
            if (self.target.ptr.creator) |c| try list.append(allocator, c);
        }

        pub fn deinit(ptr: *Function(T)) void {
            const self: *Self = @fieldParentPtr("base", ptr);
            self.output_grad.deinit();
            self.input.deinit();
            self.target.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn cross_entropy_loss(allocator: Allocator, input: anytype, target: anytype) !@TypeOf(input) {
    const T = @TypeOf(input.ptr.data.data[0]);

    // 1. Forward: -Sum(t * log(p)) / N
    const pred = input.ptr.data.data;
    const targ = target.ptr.data.data;
    const epsilon: T = 1e-7;

    var loss_sum: T = 0;
    for (0..pred.len) |i| {
        // Clamp p to epsilon to avoid log(0)
        const p = if (pred[i] < epsilon) epsilon else pred[i];
        loss_sum -= targ[i] * std.math.log(T, std.math.e, p);
    }

    const batch_size = input.ptr.data.shape[0];
    const loss_val = loss_sum / @as(T, @floatFromInt(batch_size));

    var data = try Tensor(T).init(allocator, &[_]usize{1});
    data.data[0] = loss_val;

    // 2. Build Graph
    const needs_grad = input.ptr.requires_grad;
    var creator: ?*Function(T) = null;

    if (needs_grad) {
        const op = try allocator.create(CrossEntropyBackward(T));
        var grad = try Tensor(T).init(allocator, data.shape);
        grad.fill(0);

        op.* = .{
            .base = .{
                .backward_fn = CrossEntropyBackward(T).backward,
                .collect_parents_fn = CrossEntropyBackward(T).collectParents,
                .get_grad_fn = CrossEntropyBackward(T).getGrad,
                .deinit_fn = CrossEntropyBackward(T).deinit,
            },
            .allocator = allocator,
            .input = input.clone(),
            .target = target.clone(),
            .output_grad = grad,
        };
        creator = &op.base;
    }

    var out = try Variable(T).init(allocator, data, needs_grad);
    out.ptr.creator = creator;
    return out;
}
