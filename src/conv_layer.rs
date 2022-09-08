use rand::Rng;
use rulinalg::matrix::{Matrix};

struct KernelVec {
    //as many subkernels as the input has channels
    subkernels: Vec<Matrix<f32>>,
    //the bias matrix has the same shape as the output of the convolutional layer
    bias: Matrix<f32>
}

impl KernelVec {
    pub fn new(channels: usize, kernelsidelength: usize, input_width: usize, input_height: usize) -> Self{
        Self {
            subkernels: (0..channels).map(|_| gen_rand_matrix(kernelsidelength, kernelsidelength)).collect(),
            bias: gen_rand_matrix(input_height - kernelsidelength + 1, input_width - kernelsidelength + 1)
        }
    }
}

pub struct ConvolutionalLayer {
    //4d kernels: one dimension for each kernelvec, which in turn contains a subkernel (2d matrix) for every channel + 1 bias matrix
    kernels: Vec<KernelVec>,
}

//#TODO: add generic types to support passing in other layers, passing in channels and input dimensions seems bloated
impl ConvolutionalLayer {
    pub fn new(channels: usize, kernelcount: usize, kernelsidelength: usize, input_width: usize, input_height: usize) -> Self {
        Self {
            kernels: (0..kernelcount).map(|_| KernelVec::new(channels, kernelsidelength, input_width, input_height)).collect()
        }
    }

    //takes input image/featuremap with some amount of channels and outputs the convolution for each kernelvec
    //pub fn forward_pass(&self, input: &Vec<Matrix<f32>>) -> Vec<Matrix<f32>> {
    pub fn forward_pass(&self, input: &Vec<Matrix<f32>>) {
        println!("{}", input[0]);
    }
}

//rows -> height, columns -> width
pub fn gen_rand_matrix(rows: usize, cols: usize) -> Matrix<f32> {
    //next line generates a vec of size width*height with random floats from -1 to 1
    let data: Vec<f32> = (0..rows*cols).map(|_| rand::thread_rng().gen_range(-1.0..1.0)).collect();
    //returns 2d matrix
    Matrix::new(rows, cols, data)
}