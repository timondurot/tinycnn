use ndarray::prelude::*;

pub struct SigmoidLayer {
    //is set when forward_pass is called
    previnput: ArrayD<f32>,
}

impl SigmoidLayer {
    pub fn new() -> Self {
        Self {
            previnput: ArrayD::zeros([1, 1, 1, 1, 1].as_ref())
        }
    }

    pub fn forward_pass(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.previnput = input.clone();
        input.map(|x| sigmoid(*x)).into_dyn()
    }

    pub fn backward_pass(&mut self, output_deriv: ArrayD<f32>) -> ArrayD<f32>{
        let a: ArrayD<f32> = self.previnput.map(|x| sigmoid_derivative(*x));
        (&output_deriv * &a).into_dyn()
    }
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