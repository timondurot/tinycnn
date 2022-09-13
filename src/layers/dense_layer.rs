use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray::prelude::*;
use crate::settings;

pub struct DenseLayer {
    //is set when forward_pass is called
    previnput: Array2<f32>,
    // weights
    weights: Array2<f32>,
    //1d biases: matches outputshape
    biases: Array2<f32>
}

impl DenseLayer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            previnput: Array2::zeros((1, 1)),
            weights: Array2::random((outputs, inputs), Uniform::new(-1., 1.)),
            biases: Array2::random((outputs, 1), Uniform::new(-1., 1.))
        }
    }

    pub fn forward_pass(&mut self, input: Array2<f32>) -> ArrayD<f32> {
        self.previnput = input.clone();
        (self.weights.dot(&input) + &self.biases).into_dyn()
    }

    pub fn backward_pass(&mut self, output_deriv: Array2<f32>) -> ArrayD<f32>{
        let weights_deriv: Array2<f32> = output_deriv.dot(&self.previnput.t());

        self.weights -= &(settings::LEARNING_RATE * weights_deriv);
        self.biases -= &(settings::LEARNING_RATE * &output_deriv);

        self.weights.t().dot(&output_deriv).into_dyn()
    }
}
