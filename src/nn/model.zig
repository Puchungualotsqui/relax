const std = @import("std");
const Variable = @import("../autograd/variable.zig").Variable;
const Tensor = @import("../tensor.zig").Tensor;
const Dense = @import("layers/dense.zig").Dense;
const Dropout = @import("layers/dropout.zig").Dropout;
const Optimizer = @import("../optim/optimizers.zig").Optimizer;
const SGD = @import("../optim/sgd.zig").SGD;
const Adam = @import("../optim/adam.zig").Adam;
const ops = @import("../autograd/ops.zig");

const Allocator = std.mem.Allocator;

/// The "Wrapper" for any layer type.
pub fn Layer(comptime T: type) type {
    return union(enum) {
        dense: Dense(T),
        dropout: Dropout(T),

        const Self = @This();
        const VarT = Variable(T);

        pub fn deinit(self: Self) void {
            switch (self) {
                .dense => |l| l.deinit(),
                .dropout => |l| l.deinit(),
            }
        }

        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            switch (self) {
                .dense => |l| return l.forward(input, is_training),
                .dropout => |l| return l.forward(input, is_training),
            }
        }

        pub fn parameters(self: Self, allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(VarT)) !void {
            switch (self) {
                .dense => |l| try l.parameters(allocator, list),
                .dropout => |l| try l.parameters(allocator, list),
            }
        }
    };
}

/// The Sequential Model Container
pub fn Sequential(comptime T: type) type {
    return struct {
        const Self = @This();
        const LayerT = Layer(T);
        const VarT = Variable(T);
        const OptT = Optimizer(T);

        allocator: std.mem.Allocator,
        layers: std.ArrayListUnmanaged(LayerT),

        // --- Training State ---
        optimizer: ?OptT = null,
        // Function pointer for loss: fn(allocator, prediction, target) -> LossVariable
        loss_fn: ?*const fn (Allocator, VarT, VarT) anyerror!VarT = null,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{
                .allocator = allocator,
                .layers = .{},
            };
        }

        pub fn deinit(mut_self: *Self) void {
            for (mut_self.layers.items) |layer| {
                layer.deinit();
            }
            mut_self.layers.deinit(mut_self.allocator);

            if (mut_self.optimizer) |*opt| {
                opt.deinit();
            }
        }

        pub fn add(mut_self: *Self, layer: LayerT) !void {
            try mut_self.layers.append(mut_self.allocator, layer);
        }

        pub fn parameters(self: Self) !std.ArrayListUnmanaged(VarT) {
            // Unmanaged init is just empty struct literal
            var params = std.ArrayListUnmanaged(VarT){};
            errdefer params.deinit(self.allocator);

            for (self.layers.items) |layer| {
                // Pass allocator
                try layer.parameters(self.allocator, &params);
            }
            return params;
        }

        pub fn forward(self: Self, input: VarT, is_training: bool) !VarT {
            if (self.layers.items.len == 0) return input.clone();

            var current = try self.layers.items[0].forward(input, is_training);

            var i: usize = 1;
            while (i < self.layers.items.len) : (i += 1) {
                const next = try self.layers.items[i].forward(current, is_training);
                current.deinit();
                current = next;
            }

            return current;
        }

        // ============================================================
        // Keras-like High Level API
        // ============================================================

        fn ce_wrapper(a: Allocator, p: VarT, t: VarT) anyerror!VarT {
            return ops.cross_entropy_loss(a, p, t);
        }

        pub const CompileConfig = struct {
            optimizer: enum { sgd, adam },
            lr: T = 0.01,
            loss: enum { mse, cross_entropy },
        };

        // Helper to concretize generic ops
        fn mse_wrapper(a: Allocator, p: VarT, t: VarT) anyerror!VarT {
            return ops.mse_loss(a, p, t);
        }

        /// Configures the model for training.
        pub fn compile(self: *Self, config: CompileConfig) !void {
            // Clean up existing optimizer
            if (self.optimizer) |*opt| {
                opt.deinit();
            }

            const params = try self.parameters();

            switch (config.optimizer) {
                .sgd => {
                    self.optimizer = .{ .sgd = SGD(T).init(self.allocator, params, config.lr) };
                },
                .adam => {
                    // Adam init can fail (allocations), so we use try
                    self.optimizer = .{ .adam = try Adam(T).init(self.allocator, params, config.lr) };
                },
            }

            switch (config.loss) {
                .mse => self.loss_fn = mse_wrapper,
                .cross_entropy => self.loss_fn = ce_wrapper,
            }
        }

        pub const FitConfig = struct {
            epochs: usize,
            batch_size: usize = 32,
            val_data: ?struct { x: Tensor(T), y: Tensor(T) } = null,
            verbose: bool = true,
            patience: ?usize = null,
        };

        /// Trains the model for a fixed number of epochs.
        /// Takes raw Tensors (x, y), creates copies for the graph, and runs the loop.
        pub fn fit(self: *Self, x: Tensor(T), y: Tensor(T), config: FitConfig) !void {
            if (self.optimizer == null or self.loss_fn == null) return error.ModelNotCompiled;

            const num_samples = x.shape[0];
            if (x.shape[0] != y.shape[0]) return error.BatchSizeMismatch;

            var val_vars: ?struct { x: VarT, y: VarT } = null;
            if (config.val_data) |vd| {
                val_vars = .{
                    .x = try VarT.init(self.allocator, try vd.x.clone(), false),
                    .y = try VarT.init(self.allocator, try vd.y.clone(), false),
                };
            }
            defer if (val_vars) |vv| {
                vv.x.deinit();
                vv.y.deinit();
            };

            const CheckpointT = @import("checkpoint.zig").Checkpoint(T);
            var best_checkpoint: ?CheckpointT = null;
            defer if (best_checkpoint) |*cp| cp.deinit();

            var best_val_loss = std.math.inf(T);
            var patience_counter: usize = 0;
            const params = try self.parameters();
            defer {
                for (params.items) |p| p.deinit();
                var p_list = params;
                p_list.deinit(self.allocator);
            }

            var opt_ptr = &self.optimizer.?;

            std.debug.print("Training on {d} samples...\n", .{x.shape[0]});

            for (0..config.epochs) |epoch| {
                var epoch_loss: T = 0;
                var batch_count: usize = 0;

                // --- MINI-BATCH LOOP ---
                var start: usize = 0;
                while (start < num_samples) : (start += config.batch_size) {
                    const end = @min(start + config.batch_size, num_samples);
                    // 1. Slice Raw Tensors
                    const x_batch_t = try x.sliceDimension(0, start, end, self.allocator);
                    const y_batch_t = try y.sliceDimension(0, start, end, self.allocator);

                    // 2. Wrap in Variables
                    var x_batch = try VarT.init(self.allocator, x_batch_t, false);
                    var y_batch = try VarT.init(self.allocator, y_batch_t, false);
                    defer {
                        x_batch.deinit();
                        y_batch.deinit();
                    }

                    // 3. Step
                    try opt_ptr.zeroGrad();
                    var preds = try self.forward(x_batch, true);
                    defer preds.deinit();

                    var loss_var = try self.loss_fn.?(self.allocator, preds, y_batch);
                    defer loss_var.deinit();

                    try loss_var.backward();
                    try opt_ptr.step();

                    epoch_loss += loss_var.ptr.data.data[0];
                    batch_count += 1;
                }

                // Calculate Average Loss for the Epoch
                const avg_train_loss = epoch_loss / @as(T, @floatFromInt(batch_count));

                // --- VALIDATION STEP ---
                var val_loss_val: ?T = null;
                if (val_vars) |vv| {
                    var v_preds = try self.forward(vv.x, false);
                    defer v_preds.deinit();

                    var v_loss = try self.loss_fn.?(self.allocator, v_preds, vv.y);
                    defer v_loss.deinit();
                    val_loss_val = v_loss.ptr.data.data[0];

                    if (val_loss_val.? < best_val_loss) {
                        best_val_loss = val_loss_val.?;
                        patience_counter = 0;
                        if (best_checkpoint) |*cp| cp.deinit();
                        best_checkpoint = try CheckpointT.capture(self.allocator, params);
                    } else {
                        patience_counter += 1;
                    }
                }

                // --- LOGGING ---
                // Corrected: Uses avg_train_loss instead of 'loss'
                if (config.verbose and (epoch % 10 == 0 or epoch == config.epochs - 1)) {
                    if (val_loss_val) |vl| {
                        std.debug.print("Epoch {d}/{d} - loss: {d:.4} - val_loss: {d:.4}\n", .{ epoch + 1, config.epochs, avg_train_loss, vl });
                    } else {
                        std.debug.print("Epoch {d}/{d} - loss: {d:.4}\n", .{ epoch + 1, config.epochs, avg_train_loss });
                    }
                }

                // --- EARLY STOPPING CHECK ---
                if (config.patience) |p| {
                    if (patience_counter >= p) {
                        if (config.verbose) std.debug.print("\nEarly stopping triggered at epoch {d}\n", .{epoch + 1});
                        break;
                    }
                }
            }

            if (best_checkpoint) |cp| {
                try cp.restore(params);
                if (config.verbose) std.debug.print("Restored best model with val_loss: {d:.4}\n", .{best_val_loss});
            }
        }

        /// Generates predictions for the input samples.
        /// Returns a raw Tensor (not a Variable). Caller owns the result.
        pub fn predict(self: Self, x: Tensor(T)) !Tensor(T) {
            // Wrap input
            var x_var = try VarT.init(self.allocator, try x.clone(), false);
            defer x_var.deinit();

            // Forward (is_training = false)
            var preds = try self.forward(x_var, false);
            defer preds.deinit();

            // Return a clone of the raw data (detaching from graph)
            return try preds.ptr.data.clone();
        }

        fn writeU64(file: std.fs.File, val: u64) !void {
            var buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &buf, val, .little);
            try file.writeAll(&buf);
        }

        fn readU64(file: std.fs.File) !u64 {
            var buf: [8]u8 = undefined;
            const n = try file.readAll(&buf);
            if (n != 8) return error.EndOfStream;
            return std.mem.readInt(u64, &buf, .little);
        }

        pub fn save(self: *Self, path: []const u8) !void {
            const file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            const params = try self.parameters();
            defer {
                for (params.items) |p| p.deinit();
                var p_list = params;
                p_list.deinit(self.allocator);
            }

            // 1. Write Header
            try file.writeAll("RELAX01");
            try writeU64(file, params.items.len);

            // 2. Write Tensors
            for (params.items) |p| {
                const tensor = p.ptr.data;

                try writeU64(file, tensor.shape.len);

                for (tensor.shape) |s| {
                    try writeU64(file, s);
                }

                try file.writeAll(std.mem.sliceAsBytes(tensor.data));
            }
        }

        pub fn load(self: *Self, path: []const u8) !void {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            const params = try self.parameters();
            defer {
                for (params.items) |p| p.deinit();
                var p_list = params;
                p_list.deinit(self.allocator);
            }

            // 1. Read Header
            var magic_buf: [7]u8 = undefined;
            const magic_n = try file.readAll(&magic_buf);
            if (magic_n != 7) return error.InvalidFileFormat;
            if (!std.mem.eql(u8, &magic_buf, "RELAX01")) return error.InvalidFileFormat;

            const count = try readU64(file);
            if (count != params.items.len) return error.ParameterCountMismatch;

            // 2. Read Tensors
            for (params.items) |p| {
                const target_tensor = p.ptr.data;

                const rank = try readU64(file);
                if (rank != target_tensor.shape.len) return error.ShapeMismatch;

                for (target_tensor.shape) |expected_dim| {
                    const saved_dim = try readU64(file);
                    if (saved_dim != expected_dim) return error.ShapeMismatch;
                }

                // Read Data
                const byte_slice = std.mem.sliceAsBytes(target_tensor.data);
                const n = try file.readAll(byte_slice);
                if (n != byte_slice.len) return error.EndOfStream;
            }
        }
    };
}
