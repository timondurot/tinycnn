use ndarray::prelude::*;

use super::Layer;

pub struct ReshapeLayer {
    inputshape: Vec<usize>,
    outputshape: Vec<usize>
}

impl ReshapeLayer {
    pub fn new(inputshape: Vec<usize>, outputshape: Vec<usize>) -> Self {
        Self { inputshape, outputshape }
    }
}

impl Layer for ReshapeLayer {
    fn get_output_shape(&self) -> Vec<usize> { self.outputshape.clone() }

    //TODO: implement unwrap error handling
    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        Array::from_shape_vec(self.outputshape.clone(), input.clone().into_raw_vec()).unwrap()
    }

    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, _previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>){
        (Array::from_shape_vec(self.inputshape.clone(), output_derivatives.clone().into_raw_vec()).unwrap(), vec![])
    }
    fn apply_deltas(&mut self, _input: Vec<ArrayD<f32>>){}
}