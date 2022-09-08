use crate::mnist_stuff::TinyPair;

mod settings;
mod mnist_stuff;
mod conv_layer;
fn main() {
    println!("{}", conv_layer::gen_rand_matrix(2,3));

    let mnist = mnist_stuff::TinyMNIST::initialize();
    
    let imagesample: Vec<TinyPair> = mnist.training.get_n_random_pairs(50);
    println!("{}", imagesample[1].img);
    println!("{:?}", imagesample[1].lbl);
}