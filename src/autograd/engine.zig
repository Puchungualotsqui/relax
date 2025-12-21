const std = @import("std");

/// Performs a Topological Sort.
/// Returns an Unmanaged ArrayList. Caller must deinit it using the same allocator.
pub fn topologicalSort(allocator: std.mem.Allocator, root: anytype) !std.ArrayListUnmanaged(@TypeOf(root)) {
    const FuncT = @TypeOf(root); // Fix comptime type resolution

    // 1. Setup Data Structures
    var visited = std.AutoHashMap(FuncT, void).init(allocator);
    defer visited.deinit();

    // Switch to Unmanaged
    var order = std.ArrayListUnmanaged(FuncT){};
    errdefer order.deinit(allocator);

    const Helper = struct {
        visited: *std.AutoHashMap(FuncT, void),
        order: *std.ArrayListUnmanaged(FuncT),
        allocator: std.mem.Allocator,

        fn build(self: @This(), node: FuncT) !void {
            if (self.visited.contains(node)) return;
            try self.visited.put(node, {});

            // Parents list (Unmanaged)
            var parents = std.ArrayListUnmanaged(FuncT){};
            defer parents.deinit(self.allocator);

            // Pass allocator to collectParents
            try node.collectParents(self.allocator, &parents);

            for (parents.items) |parent| {
                try self.build(parent);
            }

            // Pass allocator to append
            try self.order.append(self.allocator, node);
        }
    };

    var helper = Helper{ .visited = &visited, .order = &order, .allocator = allocator };
    try helper.build(root);

    std.mem.reverse(FuncT, order.items);
    return order;
}
