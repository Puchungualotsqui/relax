const std = @import("std");
const SGD = @import("sgd.zig").SGD;
const Allocator = std.mem.Allocator;

pub fn Optimizer(comptime T: type) type {
    return union(enum) {
        sgd: SGD(T),
        // Add adam: Adam(T), etc. here later

        const Self = @This();

        pub fn deinit(self: *Self) void {
            switch (self.*) {
                .sgd => |*o| o.deinit(),
            }
        }

        pub fn step(self: *Self) !void {
            switch (self.*) {
                .sgd => |*o| try o.step(),
            }
        }

        pub fn zeroGrad(self: *Self) !void {
            switch (self.*) {
                .sgd => |*o| try o.zeroGrad(),
            }
        }
    };
}
