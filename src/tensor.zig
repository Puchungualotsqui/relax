const std = @import("std");
const TensorError = @import("errors.zig").TensorError;
const ops = @import("ops/elementwise.zig");
const linalg = @import("ops/linalg.zig");
const Allocator = std.mem.Allocator;

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        // Move Storage inside the struct
        pub const Storage = struct {
            data: []T,
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

            const raw_data = try allocator.alloc(T, size);
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

        pub fn gt(self: Self, other: anytype, allocator: Allocator) !Tensor(u8) {
            const out_shape = try ops.calculateBroadcastShape(allocator, self.shape, other.shape);
            defer allocator.free(out_shape);

            var result = try Tensor(u8).init(allocator, out_shape);

            try greaterThan(&result, self, other);

            return result;
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

        // Creates a fresh, contiguous copy of the tensor
        pub fn clone(self: Self) !Self {
            var new_tensor = try Self.init(self.allocator, self.shape);

            // If self is contiguous, we can use fast @memcpy
            if (self.isContiguous()) {
                const size = calculateSize(self.shape);
                @memcpy(new_tensor.data, self.data[0..size]);
            } else {
                try self.copyTo(&new_tensor);
            }
            return new_tensor;
        }

        // Internal helper to copy data based on strides
        fn copyTo(self: Self, dest: *Self) !void {
            if (calculateSize(self.shape) != calculateSize(dest.shape)) return error.IncompatibleShapes;

            // A simple copy is just a broadcastOp where the 'op' is assignment
            const closure = struct {
                fn apply(d: *T, s: T) void {
                    d.* = s;
                }
            }.apply;

            // Use our new logic
            ops.broadcastOp(dest, self, closure);
        }

        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            if (calculateSize(new_shape) != calculateSize(self.shape)) return error.IncompatibleShapes;

            // Create new metadata
            const new_shape_copy = try self.allocator.dupe(usize, new_shape);
            const new_strides = try self.allocator.alloc(usize, new_shape.len);

            // Free OLD metadata
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);

            // Assign new metadata
            self.shape = new_shape_copy;
            self.strides = new_strides;

            // Recalculate strides based on the NEW shape
            var acc: usize = 1;
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                self.strides[i] = acc;
                acc *= self.shape[i];
            }
        }

        pub fn transpose(self: *Self) void {
            std.mem.reverse(usize, self.shape);
            std.mem.reverse(usize, self.strides);
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

        /// Converts multi-dimensional indices to a flat data offset
        pub fn calculateOffset(self: Self, indices: []const usize) usize {
            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                offset += idx * self.strides[i];
            }
            return offset;
        }

        pub fn at(self: Self, indices: []const usize) T {
            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                offset += idx * self.strides[i];
            }
            return self.data[offset];
        }

        pub fn fill(self: *Self, value: T) void {
            if (self.isContiguous()) {
                const size = calculateSize(self.shape);
                @memset(self.data[0..size], value);
            } else {
                self.fillStrided(self.data, 0, 0, value);
            }
        }

        // Helper for non-contiguous filling
        fn fillStrided(self: *Self, buffer: []T, dim: usize, offset: usize, value: T) void {
            if (dim == self.shape.len - 1) {
                const stride = self.strides[dim];
                var ptr = offset;
                for (0..self.shape[dim]) |_| {
                    buffer[ptr] = value;
                    ptr += stride;
                }
            } else {
                const stride = self.strides[dim];
                var ptr = offset;
                for (0..self.shape[dim]) |_| {
                    self.fillStrided(buffer, dim + 1, ptr, value);
                    ptr += stride;
                }
            }
        }

        pub fn slice(self: *Self, axis: usize, start: usize, end: usize) !Self {
            if (axis >= self.shape.len) return error.InvalidAxis;
            if (start >= end or end > self.shape[axis]) return error.InvalidRange;

            const offset = start * self.strides[axis];

            const new_shape = try self.allocator.dupe(usize, self.shape);
            const new_strides = try self.allocator.dupe(usize, self.strides);

            new_shape[axis] = end - start;

            self.storage.ref_count += 1;

            return Self{
                .data = self.data[offset..],
                .storage = self.storage,
                .shape = new_shape,
                .strides = new_strides,
                .allocator = self.allocator,
            };
        }

        pub fn concatenate(allocator: std.mem.Allocator, tensors: []const Self, axis: usize) !Self {
            if (tensors.len == 0) return error.EmptyInput;
            const ndim = tensors[0].shape.len;

            var new_shape = try allocator.alloc(usize, ndim);
            defer allocator.free(new_shape);
            @memcpy(new_shape, tensors[0].shape);

            var concat_dim_size: usize = 0;
            for (tensors) |t| {
                if (t.shape.len != ndim) return error.IncompatibleShapes;
                concat_dim_size += t.shape[axis];
            }
            new_shape[axis] = concat_dim_size;

            var dest = try Self.init(allocator, new_shape);

            try ops.concat(&dest, tensors, axis);

            return dest;
        }

        pub fn add(self: *Self, other: Self) TensorError!void {
            // 1. Check if shapes are exactly the same for SIMD path
            if (std.mem.eql(usize, self.shape, other.shape) and self.isContiguous() and other.isContiguous()) {
                return ops.add(self, other);
            }

            // 2. Otherwise, use Broadcasting path (Slower but flexible)
            const closures = struct {
                fn add(d: *T, s: T) void {
                    d.* += s;
                }
            };

            try ops.broadcastOp(self, other, closures.add);
        }

        pub fn matmul(self: Self, other: Self, dest: *Self) !void {
            return linalg.matmul(dest, self, other);
        }

        pub fn sum(self: Self, allocator: std.mem.Allocator, axis: usize) !Self {
            if (axis >= self.shape.len) return error.InvalidAxis;

            // Create the smaller shape
            var new_shape = try allocator.alloc(usize, self.shape.len - 1);
            defer allocator.free(new_shape);
            var d: usize = 0;
            for (self.shape, 0..) |s, i| {
                if (i == axis) continue;
                new_shape[d] = s;
                d += 1;
            }

            var dest = try Self.init(allocator, new_shape);

            // Reduction closure
            // We use 'T' directly here because it is visible in this scope
            const closures = struct {
                fn add(acc: *T, val: T) void {
                    acc.* += val;
                }
            };

            ops.reduce(&dest, self, axis, @as(T, 0), closures.add);
            return dest;
        }

        pub fn equal(dest: anytype, a: anytype, b: anytype) !void {
            const closures = struct {
                fn apply(d: *u8, val_a: anytype, val_b: anytype) void {
                    d.* = if (val_a == val_b) 1 else 0;
                }
            };
            try ops.broadcastOp2(dest, a, b, closures.apply);
        }

        pub fn greaterThan(dest: anytype, a: anytype, b: anytype) !void {
            const closures = struct {
                fn apply(d: *u8, val_a: anytype, val_b: anytype) void {
                    d.* = if (val_a > val_b) 1 else 0;
                }
            };
            try ops.broadcastOp2(dest, a, b, closures.apply);
        }
    };
}
