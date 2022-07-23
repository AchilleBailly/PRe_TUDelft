from inspect import trace
import keras
import tensorflow as tf
from keras import backend as K
from keras import constraints
from keras.layers import (Activation,
                          BatchNormalization, Concatenate, Conv1D,
                          CuDNNLSTM, LSTM, Dense, Flatten, Input,
                          Lambda, MaxPooling1D,
                          Multiply, Permute, RepeatVector,
                          ZeroPadding1D)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from stub_hdf5_1thread import Assembler, Dataloader
import numpy as np
import os

keras.backend.clear_session()

# choose GPU card
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)


def get_model(trace_length, units, optimizer):

    _input = Input(shape=(trace_length, 1))

    # here for conv encoder
    input_pad1 = ZeroPadding1D((104, 0))(_input)
    # first kernel size =  half clock
    Conv1 = Conv1D(filters=4, kernel_size=26, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(input_pad1)
    Conv1 = BatchNormalization(axis=-1)(Conv1)
    Conv1 = Activation('elu')(Conv1)
    Conv1 = MaxPooling1D(pool_size=2, strides=2)(Conv1)

    Conv2 = Conv1D(filters=8, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(Conv1)
    Conv2 = BatchNormalization(axis=-1)(Conv2)
    Conv2 = Activation('elu')(Conv2)
    Conv2 = MaxPooling1D(pool_size=2, strides=2)(Conv2)

    Conv3 = Conv1D(filters=16, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(Conv2)
    Conv3 = BatchNormalization(axis=-1)(Conv3)
    Conv3 = Activation('elu')(Conv3)
    Conv3 = MaxPooling1D(pool_size=2, strides=2)(Conv3)

    Conv4 = Conv1D(filters=32, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(Conv3)
    Conv4 = BatchNormalization(axis=-1)(Conv4)
    Conv4 = Activation('elu')(Conv4)
    Conv4 = MaxPooling1D(pool_size=2, strides=2)(Conv4)

    Conv5 = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(Conv4)
    Conv5 = BatchNormalization(axis=-1)(Conv5)
    Conv5 = Activation('elu')(Conv5)
    Conv5 = MaxPooling1D(pool_size=2, strides=2)(Conv5)

    Conv6 = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid', activation=None, use_bias=True
                   #                kernel_regularizer=regularizers.l2(1e-3),
                   #                bias_regularizer=regularizers.l2(1e-3)
                   )(Conv5)
    Conv6 = BatchNormalization(axis=-1)(Conv6)
    Conv6 = Activation('elu')(Conv6)
    Conv6 = MaxPooling1D(pool_size=2, strides=2)(Conv6)

    FW_LSTM_out = LSTM(units, return_sequences=True,
                       #                         recurrent_regularizer=regularizers.l2(1e-5),
                       #                         kernel_regularizer=regularizers.l2(1e-5),
                       #                         bias_regularizer=regularizers.l2(1e-5)
                       kernel_constraint=constraints.UnitNorm(axis=0),
                       recurrent_constraint=constraints.UnitNorm(axis=0)
                       )(Conv6)

    BW_LSTM_out = LSTM(units, return_sequences=True, go_backwards=True,
                       #                         recurrent_regularizer=regularizers.l2(1e-5),
                       #                         kernel_regularizer=regularizers.l2(1e-5),
                       #                         bias_regularizer=regularizers.l2(1e-5)
                       kernel_constraint=constraints.UnitNorm(axis=0),
                       recurrent_constraint=constraints.UnitNorm(axis=0)
                       )(Conv6)
    BW_LSTM_out = Lambda(lambda xin: K.reverse(xin, axes=-2))(BW_LSTM_out)

    FW_LSTM_out_BN = BatchNormalization()(FW_LSTM_out)
    FW_LSTM_out_BN_act = Activation('tanh')(FW_LSTM_out_BN)

    FW_attention = Dense(1, use_bias=False)(FW_LSTM_out_BN_act)
    FW_attention = Flatten()(FW_attention)
    FW_attention = BatchNormalization()(FW_attention)
    FW_attention = Activation('softmax', name='FW_attention')(FW_attention)

    FW_attention = RepeatVector(units)(FW_attention)
    FW_attention = Permute([2, 1])(FW_attention)

    FW_sent_representation = Multiply()([FW_LSTM_out_BN, FW_attention])
    FW_sent_representation = Lambda(lambda xin: K.sum(
        xin, axis=-2), output_shape=(units,))(FW_sent_representation)
    FW_sent_representation = Activation('tanh')(FW_sent_representation)

    BW_LSTM_out_BN = BatchNormalization()(BW_LSTM_out)
    BW_LSTM_out_BN_act = Activation('tanh')(BW_LSTM_out_BN)

    BW_attention = Dense(1, use_bias=False)(BW_LSTM_out_BN_act)
    BW_attention = Flatten()(BW_attention)
    BW_attention = BatchNormalization()(BW_attention)
    BW_attention = Activation('softmax', name='BW_attention')(BW_attention)

    BW_attention = RepeatVector(units)(BW_attention)
    BW_attention = Permute([2, 1])(BW_attention)

    BW_sent_representation = Multiply()([BW_LSTM_out_BN, BW_attention])
    BW_sent_representation = Lambda(lambda xin: K.sum(
        xin, axis=-2), output_shape=(units,))(BW_sent_representation)
    BW_sent_representation = Activation('tanh')(BW_sent_representation)

    FB_represent = Concatenate()(
        [FW_sent_representation, BW_sent_representation])

    output_probabilities = Dense(256)(FB_represent)
    output_probabilities = BatchNormalization()(output_probabilities)
    output_probabilities = Activation('softmax', name="output_main")(
        output_probabilities)

    model = Model(inputs=_input, outputs=output_probabilities)

    model.compile(loss=CategoricalCrossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    return model


class MyModel(tf.Module):

    def __init__(self, trace_length, units, optimizer):
        self.model: Model = get_model(trace_length, units, optimizer)
        self._md = {"trace_length": trace_length, 'units': units}

        # Necessary to specify at runtime the input shape with a variable, not as nice as decorator version tho
        self.train = tf.function(self._train, input_signature=[
                                 tf.TensorSpec(
                                     shape=(None, self._md['trace_length'], 1), dtype=tf.float32, name='traces'),
                                 tf.TensorSpec(shape=(None, self._md['units'], 1), dtype=tf.float32, name='labels')])
        self.__call__ = tf.function(self.call, input_signature=[
            tf.TensorSpec(shape=(None, self._md['trace_length'], 1), dtype=tf.float32, name='traces')])

    def call(self, trace):
        res = self.model(trace)
        return res

    @tf.function(input_signature=[])
    def metadata(self):
        return self._md

    @tf.function(input_signature=[])
    def get_trainable_params(self):
        return self.model.trainable_variables

    def _train(self, trace, labels):
        with tf.GradientTape() as tape:
            output = self.model(trace)
            loss = self.model.loss
            if isinstance(loss, str):
                loss = tf.keras.losses.get(loss)
            loss = loss()(output, labels)
            grads = tape.gradient(loss, self.model.trainable_variables)
            _ = self.model.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
        return {'loss': loss}


if __name__ == '__main__':
    trace_length = 99500
    units = 256

    my_Adam = Adam(lr=0.0001, name="train")
    model = MyModel(trace_length, units, my_Adam)

    #model: Model = get_model(trace_length, units, my_Adam)

    # trc, labels = dl[0]
    # print(model.train(trc, labels))

    sig = {'train': model.train, "trainable_variables": model.get_trainable_params,
           'metadata': model.metadata}

    tf.saved_model.save(model, "custom_model", signatures=sig)
