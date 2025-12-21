const std = @import("std");
const Variable = @import("../autograd/variable.zig").Variable;
const Tensor = @import("../tensor.zig").Tensor;
const Dense = @import("layers/dense.zig").Dense;
const Dropout = @import("layers/dropout.zig").Dropout;
const Optimizer = @import("../optim/optimizers.zig").Optimizer;
const SGD = @import("../optim/sgd.zig").SGD;
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

        pub const CompileConfig = struct {
            optimizer: enum { sgd },
            lr: T = 0.01,
            loss: enum { mse },
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
            }

            switch (config.loss) {
                .mse => self.loss_fn = mse_wrapper,
            }
        }

        /// Trains the model for a fixed number of epochs.
        /// Takes raw Tensors (x, y), creates copies for the graph, and runs the loop.
        pub fn fit(self: *Self, x: Tensor(T), y: Tensor(T), epochs: usize) !void {
            if (self.optimizer == null or self.loss_fn == null) {
                return error.ModelNotCompiled;
            }

            if (x.shape[0] != y.shape[0]) return error.BatchSizeMismatch;

            // ... (setup x_var, y_var) ...
            var x_var = try VarT.init(self.allocator, try x.clone(), false);
            defer x_var.deinit();
            var y_var = try VarT.init(self.allocator, try y.clone(), false);
            defer y_var.deinit();

            // Use pointer capture for optimizer calls
            var opt_ptr = &self.optimizer.?;

            std.debug.print("Training on {d} samples...\n", .{x.shape[0]});

            for (0..epochs) |epoch| {
                // A. Zero Grad (Call on pointer)
                try opt_ptr.zeroGrad();

                // ... (Forward, Loss, Backward) ...
                var preds = try self.forward(x_var, true);
                defer preds.deinit();

                var loss = try self.loss_fn.?(self.allocator, preds, y_var);
                defer loss.deinit();

                try loss.backward();

                // E. Step (Call on pointer)
                try opt_ptr.step();

                if (epoch % 10 == 0 or epoch == epochs - 1) {
                    const val = loss.ptr.data.data[0];
                    std.debug.print("Epoch {d}/{d} \t Loss: {d:.4}\n", .{ epoch + 1, epochs, val });
                }
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
    };
}
