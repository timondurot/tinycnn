use ndarray::prelude::*;
use super::Layer;

pub struct SigmoidLayer {
    shape: Vec<usize>
}

impl SigmoidLayer {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Layer for SigmoidLayer {
    fn get_output_shape(&self) -> Vec<usize> { self.shape.clone() }

    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        input.map(|x| sigmoid(*x)).into_dyn()
    }

    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>){
        let a: ArrayD<f32> = previnput.map(|x| sigmoid_derivative(*x));
        ((output_derivatives * a).into_dyn(), vec![])
    }
    fn apply_deltas(&mut self, _input: Vec<ArrayD<f32>>){}
}

fn sigmoid(x: f32) -> f32{
    1. / (1. + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32{
    let s = sigmoid(x);
    s * (1. - s)
}

/* make this choosable in new()
fn sigmoid(x: f32) -> f32{
    x.tanh()
}

fn sigmoid_derivative(x: f32) -> f32{
    1. - x.tanh().powi(2)
}*/