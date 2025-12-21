const std = @import("std");
const Function = @import("function.zig").Function;

/// Performs a Topological Sort on the graph starting from `root`.
/// Returns a list of Functions in the order they should be executed (Backward).
/// Caller owns the returned ArrayList.
pub fn topologicalSort(allocator: std.mem.Allocator, root: anytype) !std.ArrayList(root) {
    const FuncT = root; // Should be *Function(T)

    // 1. Setup Data Structures
    var visited = std.AutoHashMap(FuncT, void).init(allocator);
    defer visited.deinit();

    var order = std.ArrayList(FuncT).init(allocator);
    // If the sort fails, we clean up; otherwise caller owns 'order'.
    errdefer order.deinit();

    // 2. Recursive Build Function
    // We define a struct to hold context for recursion
    const Helper = struct {
        visited: *std.AutoHashMap(FuncT, void),
        order: *std.ArrayList(FuncT),

        fn build(self: @This(), node: FuncT) !void {
            // If already visited, skip (Handles diamonds and cycles)
            if (self.visited.contains(node)) return;

            // Mark as visited
            try self.visited.put(node, {});

            // Collect parents (dependencies)
            var parents = std.ArrayList(FuncT).init(self.visited.allocator);
            defer parents.deinit();
            try node.collectParents(&parents);

            // Visit all parents first
            for (parents.items) |parent| {
                try self.build(parent);
            }

            // After visiting parents, add self to list
            try self.order.append(node);
        }
    };

    var helper = Helper{ .visited = &visited, .order = &order };
    try helper.build(root);

    // 3. The Result is currently [Inputs ... Output] (Post-Order).
    // For Backprop, we want to execute the Output (Loss) FIRST, then push grads to inputs.
    // So we reverse the list.
    std.mem.reverse(FuncT, order.items);

    return order;
}
