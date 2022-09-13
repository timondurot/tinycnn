use ndarray::prelude::*;

pub struct ReshapeLayer {
    inputshape: Vec<usize>,
    outputshape: Vec<usize>
}

impl ReshapeLayer {
    pub fn new(inputshape: Vec<usize>, outputshape: Vec<usize>) -> Self {
        Self {
            inputshape, outputshape
        }
    }

    pub fn forward_pass(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        //input.into_shape((self.outputshape.clone())).unwrap().into_dyn()
        Array::from_shape_vec(self.outputshape.clone(), input.into_raw_vec()).unwrap()
    }

    pub fn backward_pass(&mut self, output: ArrayD<f32>) -> ArrayD<f32>{
        //output.into_shape(self.inputshape.clone()).unwrap().into_dyn()
        Array::from_shape_vec(self.inputshape.clone(), output.into_raw_vec()).unwrap()
    }
}