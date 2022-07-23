extern crate tch;

use tch::nn::{batch_norm1d, conv1d, linear, lstm, BatchNorm, Conv1D, Linear, LSTM, RNN};
use tch::Kind::Float;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct Net {
    trace_length: i64,
    units: i64,
    conv1: Conv1D,
    conv2: Conv1D,
    conv3: Conv1D,
    conv4: Conv1D,
    conv5: Conv1D,
    conv6: Conv1D,
    batchnorm1: BatchNorm,
    batchnorm2: BatchNorm,
    batchnorm3: BatchNorm,
    batchnorm4: BatchNorm,
    batchnorm5: BatchNorm,
    batchnorm6: BatchNorm,
    fw_lstm: LSTM,
    bw_lstm: LSTM,
    fw_dense: Linear,
    bw_dense: Linear,
    fw_rnn_batchnorm: BatchNorm,
    bw_rnn_batchnorm: BatchNorm,
    fw_at_batchnorm: BatchNorm,
    bw_at_batchnorm: BatchNorm,
    out_dense: Linear,
    out_batchnorm: BatchNorm,
}

impl Net {
    pub fn new(vs: &nn::Path, trace_length: i64, units: i64) -> Net {
        let conv1 = conv1d(vs, 1, 4, 26, Default::default());
        let conv2 = conv1d(vs, 4, 8, 3, Default::default());
        let conv3 = conv1d(vs, 8, 16, 3, Default::default());
        let conv4 = conv1d(vs, 16, 32, 3, Default::default());
        let conv5 = conv1d(vs, 32, 64, 3, Default::default());
        let conv6 = conv1d(vs, 64, 128, 3, Default::default());
        let batchnorm1 = batch_norm1d(vs, 4, Default::default());
        let batchnorm2 = batch_norm1d(vs, 8, Default::default());
        let batchnorm3 = batch_norm1d(vs, 16, Default::default());
        let batchnorm4 = batch_norm1d(vs, 32, Default::default());
        let batchnorm5 = batch_norm1d(vs, 64, Default::default());
        let batchnorm6 = batch_norm1d(vs, 128, Default::default());
        let fw_lstm = lstm(vs, 128, units, Default::default());
        let bw_lstm = lstm(vs, 128, units, Default::default());
        let fw_dense = linear(vs, 256, 1, Default::default());
        let bw_dense = linear(vs, 256, 1, Default::default());
        let fw_rnn_batchnorm = batch_norm1d(vs, 256, Default::default());
        let fw_at_batchnorm = batch_norm1d(vs, 1553, Default::default());
        let bw_rnn_batchnorm = batch_norm1d(vs, 256, Default::default());
        let bw_at_batchnorm = batch_norm1d(vs, 1553, Default::default());
        let mut out_dense = linear(vs, 512, 256, Default::default());
        let out_batchnorm = batch_norm1d(vs, units, Default::default());
        Net {
            trace_length,
            units,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            batchnorm1,
            batchnorm2,
            batchnorm3,
            batchnorm4,
            batchnorm5,
            batchnorm6,
            fw_lstm,
            bw_lstm,
            fw_dense,
            bw_dense,
            fw_rnn_batchnorm,
            bw_rnn_batchnorm,
            fw_at_batchnorm,
            bw_at_batchnorm,
            out_dense,
            out_batchnorm,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out_conv = self.batchnorm1.forward_t(
            &xs.view([-1, 1, self.trace_length])
                .zero_pad1d(104, 0)
                .apply(&self.conv1),
            train,
        );
        let out_conv = self.batchnorm2.forward_t(
            &out_conv
                .elu()
                .max_pool1d(&[2], &[2], &[0], &[1], false)
                .apply(&self.conv2),
            train,
        );
        let out_conv = self.batchnorm3.forward_t(
            &out_conv
                .elu()
                .max_pool1d(&[2], &[2], &[0], &[1], false)
                .apply(&self.conv3),
            train,
        );
        let out_conv = self.batchnorm4.forward_t(
            &out_conv
                .elu()
                .max_pool1d(&[2], &[2], &[0], &[1], false)
                .apply(&self.conv4),
            train,
        );
        let out_conv = self.batchnorm5.forward_t(
            &out_conv
                .elu()
                .max_pool1d(&[2], &[2], &[0], &[1], false)
                .apply(&self.conv5),
            train,
        );
        let out_conv = self
            .batchnorm6
            .forward_t(
                &out_conv
                    .elu()
                    .max_pool1d(&[2], &[2], &[0], &[1], false)
                    .apply(&self.conv6),
                train,
            )
            .elu()
            .max_pool1d(&[2], &[2], &[0], &[1], false);

        let fw_lstm_out = self
            .fw_lstm
            .seq(&out_conv.permute(&[0, 2, 1]))
            .0
            .permute(&[0, 2, 1]);
        let fw_lstm_bn = self.fw_rnn_batchnorm.forward_t(&fw_lstm_out, train);
        let fw = self
            .fw_at_batchnorm
            .forward_t(
                &fw_lstm_bn
                    .tanh()
                    .permute(&[0, 2, 1])
                    .apply(&self.fw_dense)
                    .flatten(1, 2),
                train,
            )
            .softmax(1, Float)
            .repeat(&[self.units, 1, 1])
            .permute(&[1, 0, 2])
            .multiply(&fw_lstm_bn)
            .sum_dim_intlist(&[2], false, Float)
            .tanh();

        let bw_lstm_out = self
            .bw_lstm
            .seq(&out_conv.flip(&[2]).permute(&[0, 2, 1]))
            .0
            .flip(&[2])
            .permute(&[0, 2, 1]);

        let bw_lstm_bn = self.bw_rnn_batchnorm.forward_t(&bw_lstm_out, train);
        let bw = self
            .bw_at_batchnorm
            .forward_t(
                &bw_lstm_bn
                    .tanh()
                    .permute(&[0, 2, 1])
                    .apply(&self.bw_dense)
                    .flatten(1, 2),
                train,
            )
            .softmax(1, Float)
            .repeat(&[self.units, 1, 1])
            .permute(&[1, 0, 2])
            .multiply(&fw_lstm_out)
            .sum_dim_intlist(&[2], false, Float)
            .tanh();
        let out = self
            .out_batchnorm
            .forward_t(&Tensor::concat(&[fw, bw], 1).apply(&self.out_dense), train)
            .softmax(1, Float);
        out
    }
}
