mod settings;
mod mnist_stuff;
mod conv_layer;
fn main() {
    println!("{}", conv_layer::gen_rand_matrix(2,3));

    let mnist = mnist_stuff::initialize_mnist();
    let tm = mnist_stuff::tinify_mnist(mnist);
    
    let (img, lbl) = tm.training.get_img_lbl_pair(1);
    println!("{img}");
    println!("{lbl:?}");
}