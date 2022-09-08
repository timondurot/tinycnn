use mnist::{MnistBuilder};
use rand::seq::SliceRandom;
use rulinalg::matrix::{Matrix};
use crate::settings;

#[derive(Clone)]
pub struct TinyPair{
    pub img: Matrix<f32>,
    pub lbl: u8
}
pub struct TinySet{
    pairs: Vec<TinyPair>
}
pub struct TinyMNIST{
    pub training: TinySet,
    pub validation: TinySet,
    pub test: TinySet
}

impl TinySet{
    fn initialize(images: Vec<u8>, labels: Vec<u8>) -> Self{
        let pixels_per_image = 28*28;
        let newimages: Vec<f32> = images.iter().map(|i| *i as f32 / 255.0).collect();
        let mut count = 0;
        let mut pairs: Vec<TinyPair> = Vec::new();
        //return empty vec if the set has length = 0
        if images.len() == 0 { return TinySet { pairs }};
        
        while count < newimages.len() {
            let matrixdata: Vec<f32> = newimages[count..(count + pixels_per_image)].to_vec();
            pairs.push(TinyPair{
                img: Matrix::new(28, 28, matrixdata),
                lbl: labels[count/pixels_per_image]
            });
            count += pixels_per_image;
        }
        TinySet { pairs }
    }

    pub fn get_n_random_pairs(&self, n: usize) -> Vec<TinyPair> {
        self.pairs.choose_multiple(&mut rand::thread_rng(), n).cloned().collect()
    }
}

impl TinyMNIST{
    pub fn initialize() -> Self {
        let mnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(settings::TRAINING_SIZE)
            .validation_set_length(settings::VALIDATION_SIZE)
            .test_set_length(settings::TEST_SIZE)
            .finalize();
        TinyMNIST { 
            training: TinySet::initialize(mnist.trn_img, mnist.trn_lbl),
            validation: TinySet::initialize(mnist.val_img, mnist.val_lbl), 
            test: TinySet::initialize(mnist.tst_img, mnist.tst_lbl) 
        }
    }
}