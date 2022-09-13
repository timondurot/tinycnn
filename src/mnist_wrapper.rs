use mnist::{MnistBuilder};
use ndarray::prelude::*;
use crate::settings;

pub struct TinyMNIST {
    pub training: (Array3<f32>, Array2<u8>),
    pub validation: (Array3<f32>, Array2<u8>),
    pub testing: (Array3<f32>, Array2<u8>)
}

impl TinyMNIST {
    pub fn new() -> Self {
        let mnist = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(settings::TRAINING_SIZE)
            .validation_set_length(settings::VALIDATION_SIZE)
            .test_set_length(settings::TEST_SIZE)
            .finalize();
        
        Self { 
            training: make_set_tuple(mnist.trn_img, mnist.trn_lbl, settings::TRAINING_SIZE),
            validation: make_set_tuple(mnist.val_img, mnist.val_lbl, settings::VALIDATION_SIZE),
            testing: make_set_tuple(mnist.tst_img, mnist.tst_lbl, settings::TEST_SIZE)
        }
    }
}

fn normalize_pixels_and_make_3d(img: Vec<u8>, images_in_set: u32) -> Array3<f32> {
    let normalized: Vec<f32> = img.iter().map(|i| *i as f32 / 255.0).collect();
    Array3::from_shape_vec((images_in_set as usize, 28, 28), normalized).unwrap()
}

fn make_labels_2d(lbl: Vec<u8>, labels_in_set: u32) -> Array2<u8> {
    Array2::from_shape_vec((labels_in_set as usize, 10), lbl).unwrap()
}

fn make_set_tuple(img: Vec<u8>, lbl: Vec<u8>, setsize: u32) -> (Array3<f32>, Array2<u8>) {
    (normalize_pixels_and_make_3d(img, setsize), make_labels_2d(lbl, setsize))
}