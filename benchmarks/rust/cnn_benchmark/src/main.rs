use ndarray::{Array1, Array2, Array4};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::BufReader;
use tch::nn::{Module, Conv2D, Linear, ModuleT, OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor};
use zip::ZipArchive;

fn load_npz(file_path: &str) -> (Array4<f32>, Array1<u8>) {
    let file = File::open(file_path).expect("Failed to open file");
    let mut npz = ZipArchive::new(file).expect("Failed to read .npz file");

    let x_data: Array4<f32> = {
        let x_file = npz.by_name("x.npy").expect("Failed to find x.npy in .npz file");
        let reader = BufReader::new(x_file);
        ReadNpyExt::read_npy(reader).expect("Failed to read x data")
    };

    let y_data: Array1<u8> = {
        let y_file = npz.by_name("y.npy").expect("Failed to find y.npy in .npz file");
        let reader = BufReader::new(y_file);
        match ReadNpyExt::read_npy(reader) {
            Ok(data) => data,
            Err(_) => {
                // If the read as Array1<u8> fails, try reading as Array2<u8> and flatten it
                let y_file = npz.by_name("y.npy").expect("Failed to find y.npy in .npz file");
                let reader = BufReader::new(y_file);
                let array2: Array2<u8> = ReadNpyExt::read_npy(reader).expect("Failed to read y data");
                array2.clone().into_shape(array2.len()).unwrap()
            }
        }
    };

    (x_data, y_data)
}

#[derive(Debug)]
struct Net {
    conv1: Conv2D,
    conv2: Conv2D,
    fc1: Linear,
    fc2: Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs / "conv1", 3, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs / "fc1", 1024, 256, Default::default());
        let fc2 = nn::linear(vs / "fc2", 256, 10, Default::default());
        Net { conv1, conv2, fc1, fc2 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 3, 32, 32])
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

fn main() {
    let train_file = "benchmarks/rust/cnn_benchmark/tests/data/cifar10/cifar10_train.npz";
    let test_file = "benchmarks/rust/cnn_benchmark/tests/data/cifar10/cifar10_test.npz";

    let (x_train, y_train) = load_npz(train_file);
    let (x_test, y_test) = load_npz(test_file);

    println!("Loaded train data: x shape = {:?}, y shape = {:?}", x_train.shape(), y_train.shape());
    println!("Loaded test data: x shape = {:?}, y shape = {:?}", x_test.shape(), y_test.shape());

    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let num_epochs = 10;
    let batch_size = 100;

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;
        for (xs, ys) in x_train
            .axis_chunks_iter(Axis(0), batch_size)
            .zip(y_train.axis_chunks_iter(Axis(0), batch_size))
        {
            let xs = Tensor::from_slice(xs.as_slice().unwrap()).view([batch_size as i64, 3, 32, 32]);
            let ys = Tensor::from_slice(ys.as_slice().unwrap()).to_kind(Kind::Int64);

            let loss = net
                .forward(&xs)
                .cross_entropy_for_logits(&ys);

            opt.backward_step(&loss);
            total_loss += f64::from(loss);
        }
        println!("Epoch: {}, Loss: {:.4}", epoch, total_loss / batch_size as f64);
    }

    let mut correct = 0;
    let mut total = 0;

    for (xs, ys) in x_test
        .axis_chunks_iter(Axis(0), batch_size)
        .zip(y_test.axis_chunks_iter(Axis(0), batch_size))
    {
        let xs = Tensor::from_slice(xs.as_slice().unwrap()).view([batch_size as i64, 3, 32, 32]);
        let ys = Tensor::from_slice(ys.as_slice().unwrap()).to_kind(Kind::Int64);

        let output = net.forward(&xs);
        let pred = output.argmax(1, false);
        correct += pred.eq1(&ys).sum(Kind::Int64).int64_value(&[]);
        total += ys.size1().unwrap();
    }

    println!("Test accuracy: {:.2}%", 100.0 * correct as f64 / total as f64);
}
