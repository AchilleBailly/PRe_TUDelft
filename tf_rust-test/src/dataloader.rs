extern crate hdf5;
extern crate ndarray;
extern crate num;
extern crate std;
extern crate tensorflow;
extern crate vec;

use crate::dataloader::hdf5::H5Type;
use hdf5::types::CompoundType;
use ndarray::Array0;
use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, ArrayView1, Axis, OwnedRepr};
use std::array;
use std::error::Error;
use std::fmt;
use std::ops::Index;
use tensorflow::Tensor;

use hdf5::Dataset;
use hdf5::File;

struct OutOfBounds;

impl fmt::Debug for OutOfBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Index out of bound, file: {}, line {}", file!(), line!())
    }
}

impl fmt::Display for OutOfBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Index out of bound, file: {}, line {}", file!(), line!())
    }
}

impl std::error::Error for OutOfBounds {
    fn description(&self) -> &str {
        ""
    }
}

trait Extrema {
    fn argmin(&self) -> usize;
}

impl<A: num::Num + std::cmp::PartialOrd> Extrema for Array1<A> {
    fn argmin(&self) -> usize {
        let mut min = self.get([0]).unwrap();
        let mut index: usize = 0;

        for (i, value) in self.indexed_iter() {
            if value < min {
                min = Some(value).unwrap();
                index = i;
            }
        }
        return index;
    }
}

pub struct Dataloader {
    files: Vec<File>,
    added_sizes: Vec<usize>,
    len: usize,
    trace_length: usize,
    predict: bool,
    shuffle: bool,
    byte_index: usize,
}

struct Metadata<'a> {
    pt: &'a [u8; 16],
    ct: &'a [u8; 16],
    key: &'a [u8; 16],
    masks: &'a [u8; 16],
}

impl Dataloader {
    pub fn new(
        filenames: &[String],
        trace_length: usize,
        predict: bool,
        shuffle: bool,
        byte_index: usize,
    ) -> Self {
        let mut files: Vec<File> = Vec::new();
        let mut added_sizes: Vec<usize> = Vec::new();
        let mut len: usize = 0;

        for filename in filenames {
            let file = hdf5::File::open(filename).expect("Could not open file.");
            files.push(file)
        }
        println!("Files opened.");

        let first_ds = files[0].dataset("traces").unwrap();
        let shape = first_ds.shape();
        for file in &files {
            let cur_ds = file.dataset("traces").unwrap();
            let cur_shape = cur_ds.shape();
            if cur_shape != shape {
                panic!("Not all traces have the same shape!");
            }

            len += cur_shape[0];
            added_sizes.push(len);
        }
        let added_sizes_len = added_sizes.len();
        added_sizes[added_sizes_len - 1] -= 1;
        len -= 1;

        return Dataloader {
            files,
            added_sizes,
            len,
            trace_length,
            predict,
            shuffle,
            byte_index,
        };
    }

    pub fn get(&self, i: usize) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
        if i > *self.added_sizes.last().ok_or(OutOfBounds)? {
            return Err(Box::new(OutOfBounds));
        }
        let sizes_ar: Array1<usize> = ndarray::arr1(&self.added_sizes);
        let mut file_number = (sizes_ar - i).argmin();
        if self.added_sizes[file_number] <= i {
            file_number += 1;
        }

        let index: i32 = if file_number <= 0 {
            i.try_into().unwrap()
        } else {
            (i - self.added_sizes[file_number]).try_into().unwrap()
        };

        let trace = self.files[file_number]
            .dataset("traces")?
            .read_slice(s![index, ..self.trace_length])?;

        type metadata_type = ([u8; 16], [u8; 16], [u8; 16], [u8; 16]);
        let t: Array0<metadata_type> = self.files[file_number]
            .dataset("metadata")?
            .read_slice(s![index])?;

        // println!("datasets: {:?}", t);
        // println!(
        //     "Type descriptor (file): {:#?}",
        //     self.files[file_number]
        //         .dataset("metadata")?
        //         .dtype()?
        //         .to_descriptor()?
        // );
        // println!(
        //     "Type descriptor (mine): {:#?}",
        //     metadata_type::type_descriptor()
        // );

        // println!("Traces: {:?}", trace);
        let md: Array0<metadata_type> = self.files[file_number]
            .dataset("metadata")?
            .read_slice(s![index])?;
        let label_int = md.first().unwrap().2[self.byte_index] as usize;
        let mut label = vec![0.0; 256];
        label[label_int] = 1.0;
        return Ok((trace.to_vec(), label));
    }

    // pub fn get_multiple(
    //     &self,
    //     indices: &[usize],
    // ) -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    //     let (trace, label) = self.get(indices[0])?;

    //     let mut traces: Array2<f32> = trace.into_dimensionality()?;
    //     let mut labels: Array2<f32> = label.into_dimensionality()?;

    //     for i in &indices[1..] {
    //         let (trace, label) = self.get(*i)?;
    //         traces = ndarray::concatenate(
    //             Axis(0),
    //             &[traces.view(), trace.into_dimensionality()?.view()],
    //         )?;
    //         labels = ndarray::concatenate(
    //             Axis(0),
    //             &[labels.view(), label.into_dimensionality()?.view()],
    //         )?;
    //     }

    //     Ok((traces, labels))
    // }
}
