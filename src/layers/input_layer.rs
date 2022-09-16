use ndarray::prelude::*;
use super::Layer;

pub struct InputLayer {
    shape: Vec<usize>
}

impl InputLayer {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Layer for InputLayer {
    fn get_output_shape(&self) -> Vec<usize> { self.shape.clone() }
    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> { input.clone() }
    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, _previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>){
        (output_derivatives.clone(), vec![])
    }
    fn apply_deltas(&mut self, _input: Vec<ArrayD<f32>>){}
}