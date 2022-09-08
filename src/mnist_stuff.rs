use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, Matrix};
use crate::settings;
pub struct Set{
    img: Matrix<f32>,
    lbl: Matrix<u8>,
}

impl Set{
    //this function uses the select_rows function, which needs a vec containing all the rows indices
    //that's why the row_indexes range thingy is a little funny looking
    pub fn get_img_lbl_pair(&self, idx: usize) -> (Matrix<f32>, Matrix<u8>){
        let row_indexes = (28*idx..28*(idx+1)).collect::<Vec<_>>();
        let fragment_img: Matrix<f32> = self.img.select_rows(&row_indexes);

        let row_indexes = (10*idx..10*(idx+1)).collect::<Vec<_>>();
        let fragment_lbl: Matrix<u8> = self.lbl.select_rows(&row_indexes);

        (fragment_img, fragment_lbl)
    }
}

pub struct TinyMNIST{
    pub training: Set,
    pub validation: Set,
    pub test: Set
}

pub fn initialize_mnist() -> Mnist {
    MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(settings::TRAINING_SIZE)
        .validation_set_length(settings::VALIDATION_SIZE)
        .test_set_length(settings::TEST_SIZE)
        .finalize()
}

fn convert_img_to_matrix(v: Vec<u8>) -> Matrix<f32> {
    //This works because v.len() is SET_SIZE * 28 * 28
    let total_rows = v.len() / 28;
    let unnormalized = Matrix::new(total_rows, 28, v);
    //Divide all pixels by 255
    let result: Matrix<f32> = unnormalized.try_into().unwrap() / 255.0;

    result
}

pub fn tinify_mnist(mnist: Mnist) -> TinyMNIST {
    //TODO: Not sure if this is needed (could just use a vec):
    //Labels are one hot encoded, so we generate a matrix with just one column and lots of rows
    TinyMNIST {
        training: Set { 
            img: convert_img_to_matrix(mnist.trn_img),
            lbl: Matrix::new(mnist.trn_lbl.len(), 1, mnist.trn_lbl)
        },
        validation: Set {
            img: convert_img_to_matrix(mnist.val_img),
            lbl: Matrix::new(mnist.val_lbl.len(), 1, mnist.val_lbl)
        },
        test: Set {
            img: convert_img_to_matrix(mnist.tst_img),
            lbl: Matrix::new(mnist.tst_lbl.len(), 1, mnist.tst_lbl)
        }
    }
}

