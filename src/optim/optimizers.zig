const std = @import("std");
const Allocator = std.mem.Allocator;
const SGD = @import("sgd.zig").SGD;
const Adam = @import("adam.zig").Adam;

pub fn Optimizer(comptime T: type) type {
    return union(enum) {
        sgd: SGD(T),
        adam: Adam(T),
        // Add adam: Adam(T), etc. here later

        const Self = @This();

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                .sgd => |*o| o.deinit(),
                .adam => |*o| o.deinit(),
            }
        }

        pub fn step(self: *Self) !void {
            switch (self.*) {
                .sgd => |*o| try o.step(),
                .adam => |*o| try o.step(),
            }
        }

        pub fn zeroGrad(self: *Self) !void {
            switch (self.*) {
                .sgd => |*o| try o.zeroGrad(),
                .adam => |*o| try o.zeroGrad(),
            }
        }
    };
}
