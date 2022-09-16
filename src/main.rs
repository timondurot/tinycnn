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
    let xorin = vec![ array![[0.],[0.]].into_dyn(),
                      array![[0.],[1.]].into_dyn(), 
                      array![[1.],[0.]].into_dyn(), 
                      array![[1.],[1.]].into_dyn()
                    ];
    let xorout = vec![ array![[0.]].into_dyn(),
                       array![[1.]].into_dyn(), 
                       array![[1.]].into_dyn(), 
                       array![[0.]].into_dyn() 
                    ];

    let mut my_xor_network = TinyNetwork::new(vec![2, 1])
                            .dense(3).sigmoid()
                            .dense(1).sigmoid();

    for _ in 0..10000 {
        let error = my_xor_network.train(xorin.clone(), xorout.clone(), 4);
        println!("error: {}", error);
    }

    //deeply intricate cnn mnist example
    
    let mut my_mnist_network = TinyNetwork::new(vec![1, 28, 28])
                            .conv(2, 7).sigmoid()
                            .conv(4, 5).sigmoid()
                            .reshape(vec![4 * 18 * 18, 1])
                            .dense(100).sigmoid()
                            .dense(10).sigmoid();

    let mnist = mnist_wrapper::TinyMNIST::new();
    let (images, labels) = mnist.training;

    let mut cur = 0;

    loop {
        let mut inputbatch: Vec<ArrayD<f32>> = vec![];
        let mut realoutputbatch: Vec<ArrayD<f32>> = vec![];
    
        for z in cur..(cur+20) {
            inputbatch.push(images.slice(s![z..z+1 as usize, .., ..]).to_owned().into_dyn());
            realoutputbatch.push(labels.slice(s![z..z+1, ..]).t().to_owned().into_dyn().map(|x| *x as f32));
        }
    
        let error = my_mnist_network.train(inputbatch, realoutputbatch, 20);
        println!("error: {}", error);
        cur += 20;
    }
}

