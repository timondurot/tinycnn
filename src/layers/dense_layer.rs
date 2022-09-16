use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray::prelude::*;
use crate::settings;
use super::Layer;

pub struct DenseLayer {
    //weights
    weights: Array2<f32>,
    //2d biases: matches outputshape
    biases: Array2<f32>,
    outputshape: Vec<usize>
}

impl DenseLayer {
    pub fn new(inputshape: [usize; 2], neurons: usize) -> Self {
        let outputshape = [neurons, 1];
        let weightsshape = [neurons, inputshape[0]];

        Self {
            outputshape: Vec::from(outputshape),
            biases: Array2::random((neurons, 1), Uniform::new(-1., 1.)),
            weights: Array2::random(weightsshape, Uniform::new(-1., 1.))
        }
    }
}

impl Layer for DenseLayer {
    fn get_output_shape(&self) -> Vec<usize> { self.outputshape.clone() }

    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let input = input.clone().into_dimensionality::<Ix2>().unwrap();

        (self.weights.dot(&input) + &self.biases).into_dyn()
    }

    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>){
        let output_derivatives = output_derivatives.clone().into_dimensionality::<Ix2>().unwrap();
        let previnput = previnput.clone().into_dimensionality::<Ix2>().unwrap();
        let weights_deriv: Array2<f32> = output_derivatives.dot(&previnput.t());

        let deltas = vec![
            (settings::LEARNING_RATE * weights_deriv).into_dyn(),
            (settings::LEARNING_RATE * &output_derivatives).into_dyn()
        ];

        (self.weights.t().dot(&output_derivatives).into_dyn(), deltas)
    }

    fn apply_deltas(&mut self, input: Vec<ArrayD<f32>>){
        self.weights -= &input[0];
        self.biases -= &input[1];
    }
}
