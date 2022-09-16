use crossbeam::thread::ScopedJoinHandle;
use ndarray::prelude::*;

pub mod conv_layer;
pub mod input_layer;
pub mod dense_layer;
pub mod reshape_layer;
pub mod sigmoid_layer;

trait Layer {
    fn get_output_shape(&self) -> Vec<usize>;
    fn forward_pass(&self, input: &ArrayD<f32>) -> ArrayD<f32>;
    fn backward_pass(&self, output_derivatives: &ArrayD<f32>, previnput: &ArrayD<f32>) -> (ArrayD<f32>, Vec<ArrayD<f32>>);
    fn apply_deltas(&mut self, input: Vec<ArrayD<f32>>);
}

pub struct TinyNetwork {
    layers: Vec<Box<dyn Layer>>,
}

unsafe impl Send for TinyNetwork {}
unsafe impl Sync for TinyNetwork {}

impl TinyNetwork {
    //Initialize the network with an InputLayer
    pub fn new(inputshape: Vec<usize>) -> Self {
        Self { layers: vec![Box::new(input_layer::InputLayer::new(inputshape))] }
    }

    fn get_last_output_shape(&self) -> Vec<usize> {
        self.layers.last().unwrap().get_output_shape()
    }

    pub fn conv(mut self, kernelcount: usize, kernelsidelength: usize) -> Self{
        let inputshape = self.get_last_output_shape();
        self.layers.push(Box::new(
            conv_layer::ConvolutionalLayer::new(inputshape, kernelcount, kernelsidelength)
        ));
        self
    }

    pub fn dense(mut self, neurons: usize) -> Self{
        let inputshape = self.get_last_output_shape();
        if !(inputshape.len() == 2 && inputshape[1] == 1) {
            panic!("DenseLayer must have a Nx1 inputshape, try adding a ReshapeLayer");
        }
        let inputshape = [inputshape[0], inputshape[1]];
        self.layers.push(Box::new(
            dense_layer::DenseLayer::new(inputshape, neurons)
        ));
        self
    }

    pub fn reshape(mut self, outputshape: Vec<usize>) -> Self{
        let inputshape = self.get_last_output_shape();
        self.layers.push(Box::new(
            reshape_layer::ReshapeLayer::new(inputshape, outputshape)
        ));
        self
    }

    pub fn sigmoid(mut self) -> Self{
        let inputshape = self.get_last_output_shape();
        self.layers.push(Box::new(
            sigmoid_layer::SigmoidLayer::new(inputshape)
        ));
        self
    }

    fn feedforward(&self, i: &ArrayD<f32>) -> Vec<ArrayD<f32>>{
        let mut inputperlayer = vec![i.clone()];
        for layer in &self.layers{
            inputperlayer.push(layer.forward_pass(inputperlayer.last().unwrap()));
        }
        inputperlayer
    }

    fn feedbackward(&self, error: ArrayD<f32>, inputperlayer: Vec<ArrayD<f32>>) -> Vec<Vec<ArrayD<f32>>>{
        let mut deltasperlayer: Vec<Vec<ArrayD<f32>>> = Vec::new();
        let mut lasterror = error;
        for layer_idx in (0..self.layers.len()).rev(){
            let t = self.layers[layer_idx].backward_pass(&lasterror, &inputperlayer[layer_idx]);
            lasterror = t.0;
            deltasperlayer.push(t.1);
        }
        deltasperlayer
    }

    pub fn train(&mut self, inputbatch: Vec<ArrayD<f32>>, realoutputbatch: Vec<ArrayD<f32>>, threads: usize) -> f32 {
        let mut batch_idx = 0;
        
        let mut deltasperlayerperthread = vec![(0., vec![]); inputbatch.len()];

        crossbeam::thread::scope(|scope| {
            let mut handles: Vec<ScopedJoinHandle<()>> = vec![];
            for deltasperlayer_error_tuple in &mut deltasperlayerperthread {
                let netref = &self;
                let input = &inputbatch[batch_idx];
                let realoutput = &realoutputbatch[batch_idx]; 

                handles.push(scope.spawn(move |_| {
                    train_job(netref, input, realoutput, &mut deltasperlayer_error_tuple.0, &mut deltasperlayer_error_tuple.1);
                }));

                if handles.len() == threads || batch_idx == inputbatch.len() - 1 {
                    while !handles.is_empty(){
                        handles.pop().unwrap().join().ok();
                    }
                }

                batch_idx += 1;
            }
        }).unwrap();

        //println!("{:?}", deltasperlayerperthread);

        let mut errorsum = 0.;
        for (e, _) in &deltasperlayerperthread {
            errorsum += e;
        }

        let mut dpl = deltasperlayerperthread[0].1.clone();
        for (_, t) in deltasperlayerperthread{
            for layer_idx in 0..self.layers.len(){
                for delta_idx in 0..t[layer_idx].len() {
                    dpl[layer_idx][delta_idx] += &t[layer_idx][delta_idx]
                }
            }
        }

        for layer in &mut self.layers{
            layer.apply_deltas(dpl.pop().unwrap());
        }

        errorsum / inputbatch.len() as f32
    }
}

//todo put this somewhere else
fn train_job(net: &TinyNetwork, i: &ArrayD<f32>, o: &ArrayD<f32>, errorsum: &mut f32, deltas: &mut Vec<Vec<ArrayD<f32>>>){
    //input of the 2nd layer = output of the 1st layer
    let inputperlayer = net.feedforward(i);
    *errorsum = bce(&o, inputperlayer.last().unwrap());
    let error_derivatives = bce_derivative(&o, inputperlayer.last().unwrap());
    let dpl = net.feedbackward(error_derivatives, inputperlayer);

    deltas.extend(dpl);
}

//move into own layer
fn bce(real: &ArrayD<f32>, pred: &ArrayD<f32>) -> f32 {
    let a = real * pred.map(|x| x.ln()) + (-real + 1.) * (-pred + 1.).map(|x| x.ln());
    (-a).mean().unwrap()
}

fn bce_derivative(real: &ArrayD<f32>, pred: &ArrayD<f32>) -> ArrayD<f32> {
    let a =(-real + 1.) / (-pred + 1.) - real / pred;
    &a / (real.len() as f32)
}