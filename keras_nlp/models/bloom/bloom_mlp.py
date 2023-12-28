# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_nlp.backend import keras
from keras_nlp.backend import ops


class BloomMLP(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def _gelu_function(self, x):
        return (
            x * 0.5 * (1.0 + ops.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
        )

    def build(self, input_shape):
        if input_shape[-1] != self.hidden_dim:
            raise ValueError("hiddens sizes doesn't match")

        self._dense_h_to_4h = keras.layers.Dense(
            4 * self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="dense_h_to_4h",
        )
        self._dense_h_to_4h.build(input_shape)

        self._dense_4h_to_h = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="dense_4h_to_h",
        )
        input_shape_for_dense_4h_to_h = (
            self._dense_h_to_4h.compute_output_shape(input_shape)
        )
        self._dense_4h_to_h.build(input_shape_for_dense_4h_to_h)

        self._dropout = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy, name="dropout"
        )

    def call(self, hidden_states):
        x = self._dense_h_to_4h(hidden_states)
        x = self._gelu_function(x)
        x = self._dense_4h_to_h(x)
        hidden_states = self._dropout(x)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
            }
        )
        return config
