# Torch example in Rust

This Rust crate implements the model from the "Pay Attention to Raw Traces" paper. With the issues I had with the dataloader in the TF version, I didn't use it this time, only made some dummy tensors. 

# My opinion

I prefer those bindings to the TF ones. They feel a bit more natural to use, although it is still in a early stage and could use some improvement. Because it uses the Torch C++ API, you have every function that is available in PyTorch. It also feels a bit more like Python to build the network. The CUDA version is also really easy to implement (but I didn't really take a deep look at using CUDA in TF).