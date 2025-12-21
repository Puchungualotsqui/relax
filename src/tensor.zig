const std = @import("std");
const linalg = @import("ops/linalg.zig");
const base = @import("ops/base.zig");
const metadata = @import("ops/metadata.zig");
const reductions = @import("ops/reductions.zig");
const binary = @import("ops/binary.zig");
const unary = @import("ops/unary.zig");
const Allocator = std.mem.Allocator;
const TensorError = @import("errors.zig").TensorError;

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();
        const alignment_val = 64;
        const tensor_alignment = std.mem.Alignment.fromByteUnits(alignment_val);

        // Move Storage inside the struct
        pub const Storage = struct {
            data: []align(alignment_val) T,
            ref_count: usize,
            allocator: Allocator,

            fn deinit(self: *Storage) void {
                self.ref_count -= 1;
                if (self.ref_count == 0) {
                    self.allocator.free(self.data);
                    self.allocator.destroy(self);
                }
            }
        };

        data: []T,
        storage: *Storage,
        shape: []usize,
        strides: []usize,
        allocator: Allocator,
        view_start: usize = 0,

        pub fn init(allocator: Allocator, shape: []const usize) !Self {
            const size = calculateSize(shape);

            const storage = try allocator.create(Storage);
            errdefer allocator.destroy(storage);

            const raw_data = try allocator.alignedAlloc(T, tensor_alignment, size);
            errdefer allocator.free(raw_data);

            storage.* = .{
                .data = raw_data,
                .ref_count = 1,
                .allocator = allocator,
            };

            const shape_copy = try allocator.dupe(usize, shape);
            errdefer allocator.free(shape_copy);

            const strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);

            var self = Self{
                .data = raw_data,
                .storage = storage,
                .shape = shape_copy,
                .strides = strides,
                .allocator = allocator,
            };
            self.refreshStrides();
            return self;
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
            self.storage.deinit();
        }

        fn refreshStrides(self: Self) void {
            var acc: usize = 1;
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                self.strides[i] = acc;
                acc *= self.shape[i];
            }
        }

        fn calculateSize(shape: []const usize) usize {
            var total: usize = 1;
            for (shape) |dim| total *= dim;
            return total;
        }

        // Helper to check if memory is contiguous (needed for SIMD)
        pub fn isContiguous(self: Self) bool {
            var expected_stride: usize = 1;
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                if (self.strides[i] != expected_stride) return false;
                expected_stride *= self.shape[i];
            }
            return true;
        }

        pub fn fromSlice(allocator: Allocator, shape: []const usize, values: []const T) !Self {
            var self = try Self.init(allocator, shape);
            if (values.len != calculateSize(shape)) {
                self.deinit();
                return error.IncompatibleShapes;
            }
            @memcpy(self.data, values);
            return self;
        }

        // We also need to add back the internal copyTo helper used by arithmetic operations
        fn copyTo(self: Self, dest: *Self) !void {
            if (calculateSize(self.shape) != calculateSize(dest.shape)) return error.IncompatibleShapes;

            // Simple assignment op closure
            const closure = struct {
                fn apply(d: *T, s: T) void {
                    d.* = s;
                }
            }.apply;

            try base.broadcastOp(dest, self, closure);
        }

        pub fn clone(self: Self) !Self {
            var new_tensor = try Self.init(self.allocator, self.shape);

            // If self is contiguous, we can use fast @memcpy
            if (self.isContiguous()) {
                const size = calculateSize(self.shape);
                @memcpy(new_tensor.data, self.data[0..size]);
            } else {
                // Use our internal broadcasting copy for non-contiguous views
                try self.copyTo(&new_tensor);
            }
            return new_tensor;
        }

        /// Returns the value at the specified multi-dimensional indices.
        pub fn at(self: Self, indices: []const usize) T {
            // Safety check: indices length should match rank
            if (indices.len != self.shape.len) {
                // In a scalar case (rank 0), indices.len is 0, which is correct.
                if (self.shape.len == 0) return self.data[0];
            }

            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                offset += idx * self.strides[i];
            }
            return self.data[offset];
        }

        /// Sets the value at the specified multi-dimensional indices.
        pub fn set(self: *Self, indices: []const usize, value: T) void {
            if (self.shape.len == 0) {
                self.data[0] = value;
                return;
            }

            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                offset += idx * self.strides[i];
            }
            self.data[offset] = value;
        }

        pub fn transpose(self: *Self) void {
            if (self.shape.len < 2) return;
            const n = self.shape.len;
            std.mem.swap(usize, &self.shape[n - 1], &self.shape[n - 2]);
            std.mem.swap(usize, &self.strides[n - 1], &self.strides[n - 2]);
        }

        /// Static method to join multiple tensors along an axis.
        pub fn concatenate(allocator: Allocator, tensors: []const Self, axis: usize) !Self {
            if (tensors.len == 0) return error.EmptyInput;
            const ndim = tensors[0].shape.len;
            if (axis >= ndim) return error.InvalidAxis;

            var new_shape = try allocator.dupe(usize, tensors[0].shape);
            defer allocator.free(new_shape);

            var concat_dim_size: usize = 0;
            for (tensors) |t| {
                if (t.shape.len != ndim) return error.IncompatibleShapes;
                concat_dim_size += t.shape[axis];
            }
            new_shape[axis] = concat_dim_size;

            var dest = try Self.init(allocator, new_shape);
            try metadata.concat(&dest, tensors, axis);
            return dest;
        }

        pub fn slice(self: Self, axis: usize, start: usize, end: usize) !Self {
            if (axis >= self.shape.len) return error.InvalidAxis;
            if (start >= end or end > self.shape[axis]) return error.InvalidRange;

            // Calculate the new pointer offset based on the stride of the sliced axis
            const offset = start * self.strides[axis];

            // Duplicate metadata because the slice will have a different shape/stride locally
            const new_shape = try self.allocator.dupe(usize, self.shape);
            const new_strides = try self.allocator.dupe(usize, self.strides);

            // Update the size of the specific dimension we are slicing
            new_shape[axis] = end - start;

            // Increment reference count as this view shares the same Storage
            self.storage.ref_count += 1;

            return Self{
                .data = self.data[offset..], // Pointer arithmetic happens here
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        // --- ARITHMETIC (Allocating) ---

        pub fn add(self: Self, other: anytype, allocator: Allocator) !Self {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Self.init(allocator, out_shape);
            // Initialize dest with self then add other, or use broadcastOp2 logic
            try self.copyTo(&dest);
            try binary.add(&dest, other);
            return dest;
        }

        pub fn sub(self: Self, other: anytype, allocator: Allocator) !Self {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Self.init(allocator, out_shape);
            try self.copyTo(&dest);
            try binary.sub(&dest, other);
            return dest;
        }

        pub fn mul(self: Self, other: anytype, allocator: Allocator) !Self {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Self.init(allocator, out_shape);
            try self.copyTo(&dest);
            try binary.mul(&dest, other);
            return dest;
        }

        pub fn div(self: Self, other: anytype, allocator: Allocator) !Self {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Self.init(allocator, out_shape);
            try self.copyTo(&dest);
            try binary.div(&dest, other);
            return dest;
        }

        // --- ARITHMETIC (In-place) ---

        pub fn addInPlace(self: *Self, other: anytype) !void {
            try binary.add(self, other);
        }

        pub fn subInPlace(self: *Self, other: anytype) !void {
            try binary.sub(self, other);
        }

        pub fn mulInPlace(self: *Self, other: anytype) !void {
            try binary.mul(self, other);
        }

        pub fn divInPlace(self: *Self, other: anytype) !void {
            try binary.div(self, other);
        }

        // --- COMPARISON (Returns Tensor(u8)) ---

        pub fn eq(self: Self, other: anytype, allocator: Allocator) !Tensor(u8) {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Tensor(u8).init(allocator, out_shape);
            try binary.equal(&dest, self, other);
            return dest;
        }

        pub fn gt(self: Self, other: anytype, allocator: Allocator) !Tensor(u8) {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Tensor(u8).init(allocator, out_shape);
            try binary.greaterThan(&dest, self, other);
            return dest;
        }

        pub fn lt(self: Self, other: anytype, allocator: Allocator) !Tensor(u8) {
            const out_shape = try base.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);
            var dest = try Tensor(u8).init(allocator, out_shape);
            try binary.lessThan(&dest, self, other);
            return dest;
        }

        // --- SELECTION ---

        /// Method style: mask.select(a, b)
        /// Note: This is meant to be called on a Tensor(u8)
        pub fn select(self: anytype, a: anytype, b: anytype, allocator: Allocator) !@TypeOf(a) {
            var dest = try @TypeOf(a).init(allocator, self.shape);
            try binary.where(&dest, self, a, b);
            return dest;
        }

        // --- UNARY MATH (Allocating) ---

        pub fn fill(self: *Self, value: T) void {
            unary.fill(self, value);
        }

        pub fn exp(self: Self, allocator: Allocator) !Self {
            var dest = try Self.init(allocator, self.shape);
            try unary.exp(&dest, self);
            return dest;
        }

        pub fn log(self: Self, allocator: Allocator) !Self {
            var dest = try Self.init(allocator, self.shape);
            try unary.log(&dest, self);
            return dest;
        }

        pub fn sqrt(self: Self, allocator: Allocator) !Self {
            var dest = try Self.init(allocator, self.shape);
            try unary.sqrt(&dest, self);
            return dest;
        }

        pub fn clipped(self: Self, allocator: Allocator, min_val: T, max_val: T) !Self {
            var dest = try Self.init(allocator, self.shape);
            try unary.clip(&dest, self, min_val, max_val);
            return dest;
        }

        // --- UNARY MATH (In-place) ---

        pub fn expInPlace(self: *Self) !void {
            try unary.exp(self, self.*);
        }

        pub fn logInPlace(self: *Self) !void {
            try unary.log(self, self.*);
        }

        pub fn sqrtInPlace(self: *Self) !void {
            try unary.sqrt(self, self.*);
        }

        pub fn clipInPlace(self: *Self, min_val: T, max_val: T) !void {
            try unary.clip(self, self.*, min_val, max_val);
        }

        // --- SCALAR MATH ---

        pub fn addScalar(self: *Self, val: T) void {
            unary.addScalar(self, self.*, val);
        }

        pub fn subScalar(self: *Self, val: T) void {
            unary.subScalar(self, self.*, val);
        }

        pub fn mulScalar(self: *Self, val: T) void {
            unary.mulScalar(self, self.*, val);
        }

        pub fn divScalar(self: *Self, val: T) void {
            unary.divScalar(self, self.*, val);
        }

        pub fn powScalar(self: *Self, val: T) void {
            unary.powScalar(self, self.*, val);
        }

        // --- INTERNAL HELPER ---
        fn createReductionShape(self: Self, allocator: Allocator, axis: usize) ![]usize {
            if (axis >= self.shape.len) return error.InvalidAxis;
            var new_shape = try allocator.alloc(usize, self.shape.len - 1);
            var d: usize = 0;
            for (self.shape, 0..) |s, i| {
                if (i == axis) continue;
                new_shape[d] = s;
                d += 1;
            }
            return new_shape;
        }

        // --- REDUCTIONS ---

        pub fn sum(self: Self, allocator: Allocator, axis: usize) !Self {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);
            var dest = try Self.init(allocator, new_shape);
            reductions.sum(&dest, self, axis);
            return dest;
        }

        pub fn max(self: Self, allocator: Allocator, axis: usize) !Self {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);
            var dest = try Self.init(allocator, new_shape);
            reductions.max(&dest, self, axis);
            return dest;
        }

        pub fn min(self: Self, allocator: Allocator, axis: usize) !Self {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);
            var dest = try Self.init(allocator, new_shape);
            reductions.min(&dest, self, axis);
            return dest;
        }

        pub fn mean(self: Self, allocator: Allocator, axis: usize) !Self {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);
            var dest = try Self.init(allocator, new_shape);
            reductions.mean(&dest, self, axis);
            return dest;
        }

        pub fn variance(self: Self, allocator: Allocator, axis: usize) !Self {
            const mu = try self.mean(allocator, axis);
            defer mu.deinit();
            var dest = try Self.init(allocator, mu.shape);
            reductions.variance(&dest, self, mu, axis);
            return dest;
        }

        pub fn stdDev(self: Self, allocator: Allocator, axis: usize) !Self {
            // 1. Calculate variance. Ownership of 'v' begins here.
            var v = try self.variance(allocator, axis);

            // 2. If sqrtInPlace fails, we MUST free 'v'.
            errdefer v.deinit();

            // 3. Perform the square root.
            try v.sqrtInPlace();

            // 4. Return 'v'. No defer will run on success.
            return v;
        }

        pub fn logSumExp(self: Self, allocator: Allocator, axis: usize) !Self {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);
            var dest = try Self.init(allocator, new_shape);
            try reductions.logSumExp(&dest, self, axis);
            return dest;
        }

        pub fn argmax(self: Self, allocator: Allocator, axis: usize) !Tensor(usize) {
            const new_shape = try self.createReductionShape(allocator, axis);
            defer allocator.free(new_shape);

            // Note: argmax returns indices, so it produces a Tensor(usize)
            var dest = try Tensor(usize).init(allocator, new_shape);
            var max_vals = try Self.init(allocator, new_shape);
            defer max_vals.deinit();

            reductions.argmax(&dest, &max_vals, self, axis);
            return dest;
        }

        // --- SHAPE MANIPULATION ($O(1)$ Views) ---

        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            if (calculateSize(new_shape) != calculateSize(self.shape)) return error.IncompatibleShapes;

            const new_shape_copy = try self.allocator.dupe(usize, new_shape);
            const new_strides = try self.allocator.alloc(usize, new_shape.len);

            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            self.shape = new_shape_copy;
            self.strides = new_strides;
            self.refreshStrides();
        }

        pub fn flatten(self: Self) !Self {
            if (!self.isContiguous()) {
                var contig = try self.clone();
                defer contig.deinit();
                return try contig.flatten();
            }

            const new_shape = try self.allocator.alloc(usize, 1);
            const new_strides = try self.allocator.alloc(usize, 1);
            try metadata.flatten(new_shape, new_strides, self.shape, true);

            self.storage.ref_count += 1;
            return Self{
                .data = self.data,
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        pub fn unsqueeze(self: Self, axis: usize) !Self {
            if (axis > self.shape.len) return error.InvalidAxis;
            const new_ndim = self.shape.len + 1;
            const new_shape = try self.allocator.alloc(usize, new_ndim);
            const new_strides = try self.allocator.alloc(usize, new_ndim);

            var old_i: usize = 0;
            for (0..new_ndim) |new_i| {
                if (new_i == axis) {
                    new_shape[new_i] = 1;
                    new_strides[new_i] = if (old_i < self.strides.len) self.strides[old_i] else 1;
                } else {
                    new_shape[new_i] = self.shape[old_i];
                    new_strides[new_i] = self.strides[old_i];
                    old_i += 1;
                }
            }

            self.storage.ref_count += 1;
            return Self{
                .data = self.data,
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        pub fn squeeze(self: Self, axis: usize) !Self {
            if (axis >= self.shape.len) return error.InvalidAxis;
            if (self.shape[axis] != 1) return error.DimensionNotSqueezable;

            const new_ndim = self.shape.len - 1;
            const new_shape = try self.allocator.alloc(usize, new_ndim);
            const new_strides = try self.allocator.alloc(usize, new_ndim);

            var new_i: usize = 0;
            for (self.shape, 0..) |dim, old_i| {
                if (old_i == axis) continue;
                new_shape[new_i] = dim;
                new_strides[new_i] = self.strides[old_i];
                new_i += 1;
            }

            self.storage.ref_count += 1;
            return Self{
                .data = self.data,
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        pub fn permute(self: Self, dims: []const usize) !Self {
            try metadata.validatePermutation(self.shape.len, dims);

            const new_shape = try self.allocator.alloc(usize, self.shape.len);
            const new_strides = try self.allocator.alloc(usize, self.shape.len);

            for (dims, 0..) |old_axis, new_axis| {
                new_shape[new_axis] = self.shape[old_axis];
                new_strides[new_axis] = self.strides[old_axis];
            }

            self.storage.ref_count += 1;
            return Self{
                .data = self.data,
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        // --- JOINING (Allocating) ---

        pub fn concat(allocator: Allocator, tensors: []const Self, axis: usize) !Self {
            if (tensors.len == 0) return error.EmptyInput;
            const ndim = tensors[0].shape.len;
            if (axis >= ndim) return error.InvalidAxis;

            var new_shape = try allocator.dupe(usize, tensors[0].shape);
            defer allocator.free(new_shape);

            var concat_dim_size: usize = 0;
            for (tensors) |t| {
                if (t.shape.len != ndim) return error.IncompatibleShapes;
                concat_dim_size += t.shape[axis];
            }
            new_shape[axis] = concat_dim_size;

            var dest = try Self.init(allocator, new_shape);
            try metadata.concat(&dest, tensors, axis);
            return dest;
        }

        /// Standard Matrix Multiplication: dest = self * other
        pub fn matmul(self: Self, other: Self, dest: *Self) !void {
            // Pass null for the bias parameter
            return linalg.matmul(dest, self, other, null);
        }

        /// Fused Linear Layer: dest = (self * weights) + bias
        /// This is much faster than doing matmul() then add()
        pub fn linear(self: Self, weights: Self, bias: Self, dest: *Self) !void {
            return linalg.matmul(dest, self, weights, bias);
        }
    };
}
