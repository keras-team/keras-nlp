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

"""Masked Language Model (MaskedLM) head."""

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops


@keras_nlp_export("keras_nlp.layers.MaskedLMHead")
class MaskedLMHead(keras.layers.Layer):
    """Masked Language Model (MaskedLM) head.

    This layer takes two inputs:

     - `inputs`: which should be a tensor of encoded tokens with shape
       `(batch_size, sequence_length, encoding_dim)`.
     - `mask_positions`: which should be a tensor of integer positions to
       predict with shape `(batch_size, masks_per_sequence)`.

    The token encodings should usually be the last output of an encoder model,
    and mask positions should be the interger positions you would like to
    predict for the MaskedLM task.

    The layer will first gather the token encodings at the mask positions. These
    gathered tokens will be passed through a dense layer the same size as
    encoding dimension, then transformed to predictions the same size as the
    input vocabulary. This layer will produce a single output with shape
    `(batch_size, masks_per_sequence, vocabulary_size)`, which can be used to
    compute an MaskedLM loss function.

    This layer is often be paired with `keras_nlp.layers.MaskedLMMaskGenerator`,
    which will help prepare inputs for the MaskedLM task.

    Args:
        vocabulary_size: The total size of the vocabulary for predictions.
        embedding_weights: Optional. The weights of the word embedding used
            to transform input token ids. The transpose of this weight matrix
            will be used to project a token embedding vector to a prediction
            over all input words, as described
            [here](https://arxiv.org/abs/1608.05859).
        intermediate_activation: The activation function of inner dense layer.
        activation: The activation function for the outputs of the layer.
            Usually either `None` (return logits), or `"softmax"`
            (return probabilities).
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        name: string. The name of the layer. Defaults to `None`.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    batch_size = 32
    vocab_size = 100
    encoding_size = 32
    seq_length = 50
    mask_length = 10

    # Generate a random encoding.
    encoded_tokens = np.random.normal(
        size=(batch_size, seq_length, encoding_size),
    )
    # Generate random positions and labels
    mask_positions = np.random.randint(
        seq_length, size=(batch_size, mask_length),
    )
    mask_ids = np.random.randint(
        vocab_size, size=(batch_size, mask_length),
    )

    # Predict an output word for each masked input token.
    mask_preds = keras_nlp.layers.MaskedLMHead(
        vocabulary_size=vocab_size,
        activation="softmax",
    )(encoded_tokens, mask_positions=mask_positions)
    # Calculate a loss.
    keras.losses.sparse_categorical_crossentropy(mask_ids, mask_preds)
    ```

    References:
     - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        vocabulary_size=None,
        embedding_weights=None,
        intermediate_activation="relu",
        activation=None,
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.vocabulary_size = vocabulary_size
        self.embedding_weights = embedding_weights
        self.intermediate_activation = keras.activations.get(
            intermediate_activation
        )
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False

        if vocabulary_size is None and embedding_weights is None:
            raise ValueError(
                "One of `vocabulary_size` or `embedding_weights` must be set. "
                "Received: `vocabulary_size=None`, `embedding_weights=None`"
            )

        if embedding_weights is not None:
            shape = embedding_weights.shape
            if vocabulary_size is not None and vocabulary_size != shape[0]:
                raise ValueError(
                    "`vocabulary_size` should match the first dimension of the "
                    "shape of `embedding_weights`. Received: "
                    f"`vocabulary_size={vocabulary_size}`, "
                    f"`embedding_weights.shape={shape}`"
                )
            self.vocabulary_size = shape[0]

    def build(self, inputs_shape, masked_positions_shape=None):
        if self.embedding_weights is not None:
            feature_size = self.embedding_weights.shape[-1]
        else:
            feature_size = inputs_shape[-1]

        self._dense = keras.layers.Dense(
            feature_size,
            activation=self.intermediate_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self._dtype_policy,
        )
        self._layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self._dtype_policy,
        )
        if masked_positions_shape:
            gather_length = masked_positions_shape[1]
            shape = (inputs_shape[0], gather_length, inputs_shape[-1])
            self._dense.build(shape)
            shape = (inputs_shape[0], gather_length, feature_size)
            self._layer_norm.build(shape)
        if self.embedding_weights is None:
            self._kernel = self.add_weight(
                name="output_kernel",
                shape=[feature_size, self.vocabulary_size],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
            )
        self._bias = self.add_weight(
            name="output_bias",
            shape=[self.vocabulary_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
        )

    def call(self, inputs, masked_positions):
        # Gather the encoded tokens at the masked indices.
        masked_positions = ops.expand_dims(masked_positions, axis=-1)
        x = ops.take_along_axis(inputs, masked_positions, axis=1)

        # Apply a trainable linear transformation and a layer norm.
        x = self._dense(x)
        x = self._layer_norm(x)

        # Transform encodings to vocabulary_size predictions.
        if self.embedding_weights is None:
            kernel = self._kernel
        else:
            kernel = ops.cast(self.embedding_weights, self.compute_dtype)
            kernel = ops.transpose(kernel)
        outputs = ops.matmul(x, kernel)
        outputs = outputs + self._bias

        # Apply a final activation.
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "intermediate_activation": keras.activations.serialize(
                    self.intermediate_activation
                ),
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape, masked_positions_shape):
        output_shape = list(masked_positions_shape)
        output_shape[-1] = self.vocabulary_size
        return tuple(output_shape)
