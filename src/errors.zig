pub const TensorError = error{
    IncompatibleShapes,
    OutOfMemory,
    NotContiguous,
};
