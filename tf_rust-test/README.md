# TF Examples in Rust

This Rust crate implements the network from the "Pay Attention to Raw Traces" paper. You first need to run the `create_model.py` python script. You can then build and run (`cargo run`) the Rust program.
Note that the dataloader has some issues, when loading the ASCAD metadata dataset that contains the key, the value are really weird.

# My opinion
I feel like it is much harder to use the tensorflow bindings (maybe because I also had to learn Rust at the same time), it is the really low level version of Tensorflow where you have to handle the graph. 

This crate may be useful to use a model created and tested in python and port it to rust to be able to train and use it.