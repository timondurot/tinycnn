use ndarray::prelude::*;

pub mod conv_layer;
pub mod dense_layer;
pub mod reshape_layer;
pub mod sigmoid_layer;

enum Layer {
    Conv(conv_layer::ConvolutionalLayer),
    Dense(dense_layer::DenseLayer),
    ReshapeLayer(reshape_layer::ReshapeLayer),
    SigmoidLayer(sigmoid_layer::SigmoidLayer)
}
pub struct TinyNetwork {
    layers: Vec<Layer>,
}

//todo make this wayy neater
impl TinyNetwork {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn conv(mut self, inputshape: [usize; 3], kernelcount: usize, kernelsidelength: usize) -> Self{
        self.layers.push(Layer::Conv(conv_layer::ConvolutionalLayer::new(inputshape, kernelcount, kernelsidelength)));
        self
    }

    pub fn dense(mut self, inputs: usize, outputs: usize) -> Self{
        self.layers.push(Layer::Dense(dense_layer::DenseLayer::new(inputs, outputs)));
        self
    }

    pub fn reshape(mut self, inputshape: &[usize], outputshape: &[usize]) -> Self{
        self.layers.push(Layer::ReshapeLayer(reshape_layer::ReshapeLayer::new(Vec::from(inputshape), Vec::from(outputshape))));
        self
    }

    pub fn sigmoid(mut self) -> Self{
        self.layers.push(Layer::SigmoidLayer(sigmoid_layer::SigmoidLayer::new()));
        self
    }

    pub fn feedforward(&mut self, i: ArrayD<f32>) -> ArrayD<f32>{
        let mut input = i.clone();
        for layer in &mut self.layers{
            match layer {
                Layer::Dense(a) => { input = a.forward_pass(input.into_dimensionality::<Ix2>().unwrap()); },
                Layer::Conv(a) => { input = a.forward_pass(input.into_dimensionality::<Ix3>().unwrap()); },
                Layer::ReshapeLayer(a) => { input = a.forward_pass(input); },
                Layer::SigmoidLayer(a) => { input = a.forward_pass(input); }
            }
        }
        input
    }

    pub fn feedbackward(&mut self, o: ArrayD<f32>) -> ArrayD<f32>{
        let mut output = o.clone();
        for layer in &mut self.layers.iter_mut().rev(){
            match layer {
                Layer::Dense(a) => { output = a.backward_pass(output.into_dimensionality::<Ix2>().unwrap()); },
                Layer::Conv(a) => { output = a.backward_pass(output.into_dimensionality::<Ix3>().unwrap()); },
                Layer::ReshapeLayer(a) => { output = a.backward_pass(output); },
                Layer::SigmoidLayer(a) => { output = a.backward_pass(output); }
            }
        }
        output
    }
}
