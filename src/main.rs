use rulinalg::matrix::{Matrix, BaseMatrixMut};
use mnist_stuff::*;
use conv_layer::*;

mod settings;
mod mnist_stuff;
mod conv_layer;
fn main() {
    let kerneldata = vec![1.0, 0.0, -1.0,
                          1.0, 0.0, -1.0,
                          1.0, 0.0, -1.0];

    let kernel = Matrix::new(3, 3, kerneldata);
    
    println!("{}", kernel);

    let mnist = TinyMNIST::initialize();
    let imagesample: Vec<TinyPair> = mnist.training.get_n_random_pairs(50);

    let img = imagesample[1].img.clone().apply(&|p| p.round());
    println!("{}", img);

    let convolved_img = convolve(&img, &kernel).apply(&|p| p.round());
    println!("{}", convolved_img);

    println!("{:?}", imagesample[1].lbl);
}