# Copyright 2022 The KerasNLP Authors
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
"""XLM-RoBERTa backbone models."""

import os

from tensorflow import keras

from keras_nlp.models.roberta import roberta_models
from keras_nlp.models.xlm_roberta.xlm_roberta_checkpoints import checkpoints
from keras_nlp.models.xlm_roberta.xlm_roberta_checkpoints import (
    compatible_checkpoints,
)
from keras_nlp.models.xlm_roberta.xlm_roberta_checkpoints import vocabularies


def _handle_pretrained_model_arguments(
    xlm_roberta_variant, weights, vocabulary_size
):
    """Look up pretrained defaults for model arguments.

    This helper will validate the `weights` and `vocabulary_size` arguments, and
    fully resolve them in the case we are loading pretrained weights.
    """
    if (vocabulary_size is None and weights is None) or (
        vocabulary_size and weights
    ):
        raise ValueError(
            "One of `vocabulary_size` or `weights` must be specified "
            "(but not both). "
            f"Received: weights={weights}, "
            f"vocabulary_size={vocabulary_size}"
        )

    if weights:
        arch_checkpoints = compatible_checkpoints(xlm_roberta_variant)
        if weights not in arch_checkpoints:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(arch_checkpoints)}. """
                f"Received: {weights}"
            )
        metadata = checkpoints[weights]
        vocabulary = metadata["vocabulary"]
        vocabulary_size = vocabularies[vocabulary]["vocabulary_size"]

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", weights),
            file_hash=metadata["weights_hash"],
        )

    return weights, vocabulary_size


class XLMRobertaCustom(roberta_models.RobertaCustom):
    """XLM-RoBERTa encoder with a customizable set of hyperparameters.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    This class gives a fully configurable XLM-R model with any number of
    layers, heads, and embedding dimensions. The graph of XLM-R is
    exactly the same as RoBERTa's. For specific XLM-R architectures
    defined in the paper, see, for example, `keras_nlp.models.XLMRobertaBase`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLM-R model
    model = keras_nlp.models.XLMRobertaCustom(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    output = model(input_data)
    ```
    """

    pass


MODEL_DOCSTRING = """XLM-RoBERTa "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    Args:
        weights: string, optional. Name of pretrained model to load weights.
            Should be one of {names}.
        vocabulary_size: int, optional. The size of the token vocabulary.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLMRoberta{type} encoder
    model = keras_nlp.models.XLMRoberta{type}(weights=None, vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.ones((1, 512)),
    }}
    output = model(input_data)
    ```
"""


def XLMRobertaBase(
    weights="xlm_roberta_base", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "XLMRobertaBase", weights, vocabulary_size
    )

    model = XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def XLMRobertaLarge(
    weights="xlm_roberta_large", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "XLMRobertaLarge", weights, vocabulary_size
    )

    model = XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


setattr(
    XLMRobertaBase,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Base", names=", ".join(compatible_checkpoints("XLMRobertaBase"))
    ),
)

setattr(
    XLMRobertaLarge,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Large", names=", ".join(compatible_checkpoints("XLMRobertaLarge"))
    ),
)
