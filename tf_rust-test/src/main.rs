extern crate rand;
extern crate tensorflow;
extern crate vec;

use std::error::Error;

pub mod dataloader;
use crate::dataloader::Dataloader;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

fn train() -> Result<(), Box<dyn Error>> {
    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!
    let path_to_files = "/media/achille/Externe/Cours2A/PRe/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5";
    let dl: Dataloader = Dataloader::new(&[String::from(path_to_files)], 99500, true, true, 0);

    //Path of the saved model
    let save_dir = "custom_model";

    //Create a graph
    let mut graph = Graph::new();

    //Load saved model as graph
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
        .expect("Can't load saved model");
    let session = &bundle.session;

    // train
    let train_signature = bundle.meta_graph_def().get_signature("train")?;
    let traces_info = train_signature.get_input("traces")?;
    let labels_info = train_signature.get_input("labels")?;
    let loss_info = train_signature.get_output("loss")?;
    let op_traces = graph.operation_by_name_required(&traces_info.name().name)?;
    let op_labels = graph.operation_by_name_required(&labels_info.name().name)?;
    let op_train = graph.operation_by_name_required(&loss_info.name().name)?;

    // Train the model (e.g. for fine tuning).
    for i in 0..2 {
        let (trace, label) = dl.get(i as usize)?;

        let input_tensor: Tensor<f32> =
            Tensor::new(&[1, 99500, 1]).with_values(trace.as_slice())?;
        let label_tensor: Tensor<f32> = Tensor::new(&[1, 256, 1]).with_values(label.as_slice())?;
        let mut output_tensor: Tensor<f32> = Tensor::new(&[1, 256, 1]);
        let mut loss_tensor: Tensor<f32> = Tensor::new(&[1, 1]);

        let mut train_step = SessionRunArgs::new();

        train_step.add_feed(&op_traces, 0, &input_tensor);
        train_step.add_feed(&op_labels, 0, &label_tensor);
        train_step.add_target(&op_train);
        let loss_request = train_step.request_fetch(&op_train, 0);

        session.run(&mut train_step)?;

        loss_tensor = train_step.fetch(loss_request)?;
        println!("Loss value: {:?}", &loss_tensor.to_vec());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    train()?;

    Ok(())
}
