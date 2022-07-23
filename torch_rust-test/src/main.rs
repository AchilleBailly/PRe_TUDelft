extern crate anyhow;
extern crate tch;

use anyhow::Result;
use tch::kind::{DOUBLE_CUDA, INT64_CPU};
use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Kind, Tensor};

pub mod net;
use net::Net;

const LEARNING_RATE: f64 = 0.001;
const UNITS: i64 = 256;
const TRACE_LEN: i64 = 99500;
const BATCH_SIZE: i64 = 8;
const EPOCHS: i64 = 2;

pub fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root(), TRACE_LEN, UNITS);

    let input = Tensor::ones(&[BATCH_SIZE, TRACE_LEN, 1], (Kind::Float, device));
    let labels = Tensor::ones(&[BATCH_SIZE], (Kind::Double, Device::Cpu))
        .onehot(UNITS)
        .to_kind(Kind::Int64)
        .to(device);
    println!("input shape: {:?}", input.size());
    println!("labels shape: {:?}", labels.size());
    println!("Network initialized!");

    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;
    for epoch in 0..EPOCHS {
        let out = net.forward_t(&input, true).view([UNITS * BATCH_SIZE]);
        let loss = out.cross_entropy_for_logits(&labels.view([UNITS * BATCH_SIZE]));
        opt.backward_step(&loss);
        println!("epoch: {:4} loss: {:?}%", epoch, loss);
    }
    Ok(())
}
