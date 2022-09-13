use ndarray::prelude::*;
mod settings;
mod mnist_wrapper;
mod conv_layer;
fn main() {
    let kernel = array![[1., 0., -1.],
                        [1., 0., -1.],
                        [1., 2., -1.]];

    println!("{:#1.0}", conv_layer::fullconvolve(kernel.view(), kernel.view()));

    let mnist = mnist_wrapper::TinyMNIST::new();
    let (images, labels) = mnist.training;

    let threechannelimg = images.slice(s![0..3_usize, .., ..]);
    println!("{:#1.0}\nlabels: {:?}", threechannelimg, labels.slice(s![0..3_usize, ..]));

    let mut cl = conv_layer::ConvolutionalLayer::new([3, 28, 28], 1, 3);
    let featuremaps = cl.forward_pass(threechannelimg.view()).map(|x| if x.round() == 0. {0} else {1});
    println!("{:#1.0}", featuremaps);
}