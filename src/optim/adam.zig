const std = @import("std");
const Variable = @import("../autograd/variable.zig").Variable;
const Tensor = @import("../tensor.zig").Tensor;
const Allocator = std.mem.Allocator;

pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();
        const VarT = Variable(T);
        const TensorT = Tensor(T);

        params: std.ArrayListUnmanaged(VarT),
        // Adam State: Must match 1-to-1 with params
        m: std.ArrayListUnmanaged(TensorT), // 1st moment vector
        v: std.ArrayListUnmanaged(TensorT), // 2nd moment vector

        // Hyperparameters
        lr: T,
        beta1: T = 0.9,
        beta2: T = 0.999,
        epsilon: T = 1e-8,

        t: usize = 0, // Time step
        allocator: Allocator,

        pub fn init(allocator: Allocator, params: std.ArrayListUnmanaged(VarT), lr: T) !Self {
            // Allocate state buffers
            var m_list = try std.ArrayListUnmanaged(TensorT).initCapacity(allocator, params.items.len);
            var v_list = try std.ArrayListUnmanaged(TensorT).initCapacity(allocator, params.items.len);

            // Initialize states to Zeros
            for (params.items) |p| {
                const shape = p.ptr.data.shape;
                var m_tensor = try TensorT.init(allocator, shape);
                m_tensor.fill(0);

                var v_tensor = try TensorT.init(allocator, shape);
                v_tensor.fill(0);

                m_list.appendAssumeCapacity(m_tensor);
                v_list.appendAssumeCapacity(v_tensor);
            }

            return Self{
                .params = params,
                .m = m_list,
                .v = v_list,
                .lr = lr,
                .allocator = allocator,
            };
        }

        pub fn deinit(mut_self: *Self) void {
            // Free params
            for (mut_self.params.items) |p| p.deinit();
            mut_self.params.deinit(mut_self.allocator);

            // Free states
            for (mut_self.m.items) |t| t.deinit();
            mut_self.m.deinit(mut_self.allocator);

            for (mut_self.v.items) |t| t.deinit();
            mut_self.v.deinit(mut_self.allocator);
        }

        pub fn zeroGrad(self: *Self) !void {
            for (self.params.items) |p| {
                try p.zeroGrad();
            }
        }

        pub fn step(self: *Self) !void {
            self.t += 1;
            const t_float: T = @floatFromInt(self.t);

            // Bias corrections
            // correction1 = 1 / (1 - beta1^t)
            const correction1 = 1.0 / (1.0 - std.math.pow(T, self.beta1, t_float));
            // correction2 = 1 / (1 - beta2^t)
            const correction2 = 1.0 / (1.0 - std.math.pow(T, self.beta2, t_float));

            for (0..self.params.items.len) |i| {
                const p = self.params.items[i];
                if (!p.ptr.requires_grad or p.ptr.grad == null) continue;

                const grad = p.ptr.grad.?.data;
                const param_data = p.ptr.data.data;

                const m_data = self.m.items[i].data;
                const v_data = self.v.items[i].data;

                // Loop over every element in the parameter tensor
                for (0..param_data.len) |j| {
                    const g = grad[j];

                    // 1. Update biased first moment estimate
                    // m_t = beta1 * m_{t-1} + (1 - beta1) * g
                    m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;

                    // 2. Update biased second raw moment estimate
                    // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
                    v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * (g * g);

                    // 3. Compute bias-corrected moments
                    const m_hat = m_data[j] * correction1;
                    const v_hat = v_data[j] * correction2;

                    // 4. Update parameters
                    // theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
                    param_data[j] -= self.lr * m_hat / (std.math.sqrt(v_hat) + self.epsilon);
                }
            }
        }
    };
}
