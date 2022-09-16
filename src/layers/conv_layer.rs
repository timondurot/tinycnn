use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray::{prelude::*, Zip};
use crate::settings;
use super::Layer;

pub struct ConvolutionalLayer {
    //depth (or channels, think rgb), height, width
    inputshape: Vec<usize>,
    //depth=kernelcount (each kernel takes all input channels and outputs a featuremap), height, width
    outputshape: Vec<usize>,
    //how many kernels there are (each kernel has a subkernel for each input channel)
    kernelcount: usize,
    //how many subkernels there are per kernel (=input channels)
    subkernelcount: usize,
    //4d kernels: kernels[kernel_index][subkernel_index][row][col]
    kernels: Array4<f32>,
    //3d biases: matches outputshape
    biases: ArrayD<f32>
}

impl ConvolutionalLayer {
    pub fn new(inputshape: Vec<usize>, kernelcount: usize, kernelsidelength: usize) -> Self {
        let input_depth = inputshape[0];
        let input_height = inputshape[1];
        let input_width = inputshape[2];
        let kernelsshape = [kernelcount, input_depth, kernelsidelength, kernelsidelength];
        let outputshape = vec![kernelcount, input_height - kernelsidelength + 1, input_width - kernelsidelength + 1];

        Self {
            inputshape,
            kernelcount,
            outputshape: outputshape.clone(),
            subkernelcount: input_depth,
            biases: ArrayD::random(outputshape, Uniform::new(-1., 1.)),
            kernels: Array4::random(kernelsshape, Uniform::new(-1., 1.))
        }
    }
}

impl Layer for ConvolutionalLayer {
    fn get_output_shape(&self) -> Vec<usize> { self.outputshape.clone() }
    //accepts 3d array containing images
    //outputs 3d array containing featuremaps
    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        assert_eq!(input.shape(), self.inputshape);
        //initialize all featuremaps with their respective biases
        let mut featuremaps = self.biases.clone();
        for kernel_index in 0..self.kernelcount{
            //access the featuremap at depth=kernel_index
            let mut fm = featuremaps.slice_mut(s![kernel_index, .., ..]);
            //add channelwise (1 channel = 1 subkernel) correlations to fm
            for subkernel_idx in 0..self.subkernelcount{
                let image = input.slice(s![subkernel_idx, .., ..]);
                let kernel = self.kernels.slice(s![kernel_index, subkernel_idx, .., ..]);
                fm += &validcorrelate(image, kernel);
            }
        }
        featuremaps.into_dyn()
    }

    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>){
        let mut kernels_deriv = Array::<f32, _>::zeros(self.kernels.raw_dim());
        let mut input_deriv = Array::zeros(self.inputshape.clone());

        for kernel_index in 0..self.kernelcount{
            for subkernel_idx in 0..self.subkernelcount{
                //update kernels_deriv
                let mut subkernel_deriv = kernels_deriv.slice_mut(s![kernel_index, subkernel_idx, .., ..]);
                let image = previnput.slice(s![subkernel_idx, .., ..]);
                let o_deriv = output_derivatives.slice(s![kernel_index, .., ..]);
                subkernel_deriv.assign(&validcorrelate(image, o_deriv));
                //update input_deriv
                let mut channel_deriv = input_deriv.slice_mut(s![subkernel_idx, .., ..]);
                let subkernel = self.kernels.slice(s![kernel_index, subkernel_idx, .., ..]);
                channel_deriv += &fullconvolve(o_deriv, subkernel);
            }
        }

        let deltas = vec![
            (settings::LEARNING_RATE * kernels_deriv).into_dyn(),
            (settings::LEARNING_RATE * output_derivatives)
        ];

        (input_deriv.into_dyn(), deltas)
    }

    fn apply_deltas(&mut self, input: Vec<ArrayD<f32>>){
        self.kernels -= &input[0];
        self.biases -= &input[1];
    }
}

pub fn validcorrelate(image: ArrayView2<f32>, kernel: ArrayView2<f32>) -> Array2<f32> {
    Zip::from(image.windows(kernel.raw_dim())).map_collect(|window| {
		(&window * &kernel).sum()
	})
}

pub fn fullconvolve(image: ArrayView2<f32>, kernel: ArrayView2<f32>) -> Array2<f32> {
    //pad on kernelsidelength - 1 on every side of image
    let padamount = kernel.shape()[0] - 1;
    let image_rows = image.shape()[0];
    let image_cols = image.shape()[1];
    let mut padded = Array2::zeros((image_rows + padamount * 2, image_cols + padamount * 2));
    padded.slice_mut(s![padamount..(image_rows+padamount), padamount..(image_cols+padamount)]).assign(&image);
    //flip the kernel
    let mut flipped = kernel.clone();
    flipped.invert_axis(Axis(0));
    validcorrelate(padded.view(), flipped.slice(s![..,..;-1]).view())
}