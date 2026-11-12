# tinycnn

A tiny neural network library, written from scratch in Rust. Convolutions, backpropagation, batching and multithreading are all hand-written; the only dependencies are `ndarray` (plus `ndarray-rand` and `rand`) for arrays and random weights, `mnist` for loading the dataset, and `crossbeam` for threads.

I wrote it in September 2022, by hand, a few months before ChatGPT existed. Back then I stopped once the loss was going down. Years later I finally counted how many digits it gets right: **97.7% on MNIST**, with just 6 convolution filters.

**Read the full story, and draw your own digits for it to read in the browser: [tmon.ch/blog/tinycnn](https://tmon.ch/blog/tinycnn/)**

> **This is a snapshot, on purpose.** The code stays exactly as I wrote it in 2022, bugs and all. I know what I'd fix, and the blog post lists most of it, but I'm deliberately leaving it this way instead of polishing it into something it never was.

## Build networks by chaining layers

```rust
let mut net = TinyNetwork::new(vec![1, 28, 28])   // 1 channel, 28×28 pixels
    .conv(2, 7).sigmoid()                         // 2 filters, 7×7
    .conv(4, 5).sigmoid()                         // 4 filters, 5×5
    .reshape(vec![4 * 18 * 18, 1])
    .dense(100).sigmoid()
    .dense(10).sigmoid();                         // one output per digit

let loss = net.train(input_batch, expected_batch, 20); // 20 threads
```

Every layer works out its input shape from the layer before it, so you only ever say what a layer *outputs*. If you forget to flatten before a dense layer, it tells you:

```text
DenseLayer must have a Nx1 inputshape, try adding a ReshapeLayer
```

It isn't limited to CNNs either. The same builder makes a plain network for XOR:

```rust
let mut xor = TinyNetwork::new(vec![2, 1])
    .dense(3).sigmoid()
    .dense(1).sigmoid();
```

## Features

- **Layers:** convolution (any number of filters, any number of input channels), dense, sigmoid and reshape. Each one implements one shared `Layer` trait with a forward pass, a backward pass and weight updates, so adding a new layer type means writing one struct.
- **Backpropagation through convolutions:** filter gradients by cross-correlating the input with the output gradient, and input gradients by a full convolution with the 180°-flipped filter.
- **Mini-batch training,** with every sample in a batch run on its own thread via `crossbeam`.
- **Binary cross-entropy loss.**
- **MNIST included:** a small wrapper turns the raw files in `data/` into normalized `ndarray` arrays, already split into training, validation and test sets.

## Results

Measured for the blog post, in a separate copy that adds an accuracy check after each pass. The Rust code in this repo is unchanged since 2022.

| Passes over the 60,000 training images | Validation (9,000) | Test (1,000) |
| --- | --- | --- |
| 1 | 92.2% | 92.2% |
| 5 | 96.4% | 97.0% |
| 15 | **97.7%** | **97.7%** |

15 passes take under 5 minutes on a laptop CPU.

## Run it

```bash
cargo run --release
```

`main.rs` runs three examples in order: a small convolution printed to the terminal, the XOR network, then the MNIST network, printing the loss after every batch.

Heads-up: the MNIST loop doesn't stop on its own, so it panics once it runs past the last training image. That, and a few other things I'd do differently now, are in the [blog post](https://tmon.ch/blog/tinycnn/).

## Layout

```text
src/
├── main.rs             examples: convolution, XOR, MNIST
├── settings.rs         dataset sizes and learning rate
├── mnist_wrapper.rs    MNIST files → ndarray
└── layers/
    ├── mod.rs          TinyNetwork: builder, training, loss
    ├── conv_layer.rs
    ├── dense_layer.rs
    ├── sigmoid_layer.rs
    ├── reshape_layer.rs
    └── input_layer.rs
```
