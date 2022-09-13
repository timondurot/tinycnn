mod layers;
use layers::*;

mod settings;
mod mnist_wrapper;
use ndarray::prelude::*;


fn main() {
    //simple full convolution example
    let kernel = array![[1., 0., -1.],
                        [1., 0., -1.],
                        [1., 2., -1.]];

    println!("{:#1.0}", conv_layer::fullconvolve(kernel.view(), kernel.view()));

    //simple xor example
    let xorin = vec![ array![[0.],[0.]] , array![[0.],[1.]], array![[1.],[0.]], array![[1.],[1.]] ];
    let xorout = vec![ array![[0.]], array![[1.]], array![[1.]], array![[0.]] ];

    let mut my_xor_network = TinyNetwork::new()
                            .dense(2,3)
                            .sigmoid()
                            .dense(3, 1)
                            .sigmoid();

    for z in 0..10000 {
        let input = xorin[z%4].clone().into_dyn();
        let output = my_xor_network.feedforward(input);

        let error = bce(xorout[z%4].clone().into_dyn(), output.clone());

        if z % 100 == 0 {println!("error: {}", error);}
        
        my_xor_network.feedbackward(bce_derivative(xorout[z%4].clone().into_dyn(), output));
    }

    //deeply intricate cnn mnist example
    
    let mut my_mnist_network = TinyNetwork::new()
                            .conv([1, 28, 28], 2, 11).sigmoid()
                            .conv([2, 18, 18], 3, 5).sigmoid()
                            .reshape(&[3, 14, 14], &[3 * 14 * 14, 1])
                            .dense(3 * 14 * 14, 50).sigmoid()
                            .dense(50, 10).sigmoid();

    let mnist = mnist_wrapper::TinyMNIST::new();
    let (images, labels) = mnist.training;

    let mut error = 0.0;

    for z in 0..100000 {
        //what a mess...
        let input = images.slice(s![z..z+1, .., ..]).to_owned().into_dyn();
        let output = my_mnist_network.feedforward(input);

        let label_one_hot = labels.slice(s![z..z+1, ..]).t().to_owned().into_dyn().map(|x| *x as f32);

        error += bce(label_one_hot.clone(), output.clone());
        if z % 25 == 0 {println!("error: {}", error/25.0); error = 0.0;}
        
        my_mnist_network.feedbackward(bce_derivative(label_one_hot, output));
    }
}

//move into own layer
fn bce(real: ArrayD<f32>, pred: ArrayD<f32>) -> f32 {
    let a = &real * &pred.map(|x| x.ln()) + (-&real + 1.) * (-&pred + 1.).map(|x| x.ln());
    (-a).mean().unwrap()
}

fn bce_derivative(real: ArrayD<f32>, pred: ArrayD<f32>) -> ArrayD<f32> {
    let a =(-&real + 1.) / (-&pred + 1.) - &real / &pred;
    &a / (real.len() as f32)
}