# Relax

Relax is a lightweight, high-performance Deep Learning library written from scratch in Zig. It features a dynamic reverse-mode automatic differentiation engine (Autograd), N-dimensional tensors with broadcasting, and a high-level, Keras-style API for building and training neural networks.

## Features

* **Core Engine**
    * N-Dimensional Tensors with generic type support (f32, f64).
    * Automatic memory management for computation graphs.
    * Reverse-mode Autograd (Define-by-Run).
    * Broadcasting and Slicing support.

* **Neural Networks**
    * **Sequential API:** Easy model composition.
    * **Layers:** Dense (Fully Connected), Dropout.
    * **Activations:** ReLU, Sigmoid, Tanh, Softmax.
    * **Loss Functions:** Mean Squared Error (MSE), Cross Entropy.
    * **Optimizers:** SGD (Stochastic Gradient Descent), Adam.
    * **Initializers:** Glorot (Xavier) Uniform, He Normal.

* **Utilities**
    * Binary Serialization (Save and Load models).
    * Training loop with batching and callbacks.

## Installation

Relax is designed to be included as a module in your Zig project.

1.  Clone the repository:
    ```bash
    git clone https://github.com/Puchungualotsqui/relax
    ```

2.  Add it to your `build.zig.zon` (if using Zig package manager) or simply reference the source in your `build.zig`.

## Quick Start

Here is a complete example of training a simple neural network to solve the XOR problem.

```
const std = @import("std");
const relax = @import("relax");
const Tensor = relax.Tensor;
const Sequential = relax.Sequential(f32);
const Dense = relax.layers.Dense(f32);
const initz = relax.initializers;

pub fn main() !void {
    // 1. Setup Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 2. Initialize Random Number Generator
    var rng = initz.RandomSource.init(42);

    // 3. Build the Model
    var model = Sequential.init(allocator);
    defer model.deinit();

    // Add layers: 2 Inputs -> 8 Hidden (ReLU) -> 1 Output (Linear)
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 2, .out_features = 8, .activation = .relu }, &rng) });
    try model.add(.{ .dense = try Dense.init(allocator, .{ .in_features = 8, .out_features = 1, .activation = .none }, &rng) });

    // 4. Compile the Model
    try model.compile(.{
        .optimizer = .adam,
        .lr = 0.01,Serialization
        
        Relax supports a compact binary for
        .loss = .mse,
    });

    // 5. Prepare Data (XOR Problem)
    // Inputs: (4 samples, 2 features)
    var x = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 2 }, &[_]f32{
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    });
    defer x.deinit();

    // Targets: (4 samples, 1 output)
    var y = try Tensor(f32).fromSlice(allocator, &[_]usize{ 4, 1 }, &[_]f32{
        0.0,
        1.0,
        1.0,
        0.0,
    });
    defer y.deinit();

    // 6. Train
    std.debug.print("Training...\n", .{});
    try model.fit(x, y, .{
        .epochs = 1000,
        .verbose = false,
    });

    // 7. Predict / Verify
    var pred = try model.predict(x);
    defer pred.deinit();

    std.debug.print("Predictions:\n", .{});
    for (0..4) |i| {
        std.debug.print("Input: {d:.1}, {d:.1} -> Output: {d:.4}\n", .{
            x.data[i*2], x.data[i*2+1], pred.data[i]
        });
    }

    // 8. Save the Model
    try model.save("xor_model.bin");
}
```

## Serialization
Relax supports a compact binary format for saving model parameters.
### Saving:
``` 
try model.save("my_model.bin");
```
### Loading:
```
// Initialize a model with the SAME architecture first
var model = Sequential.init(allocator);
try model.add(...) 

// Load weights
try model.load("my_model.bin");
```
## Architecture
### Tensor
The Tensor(T) struct is the fundamental data block. It manages a generic slice of data and shape information. It supports strided views (slicing) and broadcasting, allowing operations between shapes like (32, 10) and (1, 10) without data duplication.
### Autograd
The Variable(T) wrapper tracks the history of operations performed on a Tensor. When .backward() is called on a scalar variable (like Loss), gradients are propagated backwards through the graph using the chain rule. This engine handles arbitrary directed acyclic graphs (DAGs).
## License
MIT License.
