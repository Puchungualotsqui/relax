const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;
const Function = @import("function.zig").Function;
const Allocator = std.mem.Allocator;

// --- Addition Operation ---

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
            // We need a pointer to the gradient to pass to addInPlace if we wanted to be safe,
            // but here we just read from self.output_grad.
            // The issue in your code was likely in the BUILDER function, not here.
            const grad = self.output_grad;

            if (self.input_a.ptr.requires_grad) {
                const a_grad = try self.input_a.getGrad();
                try a_grad.addInPlace(grad);
            }
            if (self.input_b.ptr.requires_grad) {
                const b_grad = try self.input_b.getGrad();
                try b_grad.addInPlace(grad);
            }
        }

        pub fn getGrad(ptr: *Function(T)) *TensorT {
            const self: *Self = @fieldParentPtr("base", ptr);
            return &self.output_grad;
        }

        pub fn collectParents(ptr: *Function(T), allocator: Allocator, list: *std.ArrayListUnmanaged(*Function(T))) !void {
            const self: *Self = @fieldParentPtr("base", ptr);
            // Append now requires allocator
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

        // FIX: strict var for mutation
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

        // FIX: strict var for mutation
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

        // FIX: strict var for mutation
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

                // dL/dx = grad * sigmoid(x) * (1 - sigmoid(x))
                // We recompute sigmoid(x) here to avoid storing the output variable (breaking cycles)
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

    // 1. Forward
    var data = try Tensor(T).init(allocator, a.ptr.data.shape);
    const in_data = a.ptr.data.data;
    for (0..in_data.len) |i| {
        data.data[i] = 1.0 / (1.0 + std.math.exp(-in_data[i]));
    }

    // 2. Build Graph
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
