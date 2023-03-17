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
"""ALBERT classification model."""

import copy

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_backbone import albert_kernel_initializer
from keras_nlp.models.albert.albert_preprocessor import AlbertPreprocessor
from keras_nlp.models.albert.albert_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.AlbertClassifier")
class AlbertClassifier(Task):
    """An end-to-end ALBERT model for classification tasks

    This model attaches a classification head to a
    `keras_nlp.model.AlbertBackbone`instance, mapping from the backbone outputs to logit output suitable for
    a classification task. For usage of this model with pre-trained weights, use
    the `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.AlertBackbone` instance.
        num_classes: int. Number of classes to predict.
        dropout: float. The dropout probability value, applied after the dense
            layer.
        preprocessor: A `keras_nlp.models.AlbertPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Use a shorter sequence length.
    preprocessor = keras_nlp.models.AlbertPreprocessor.from_preset(
        "albert_base_en_uncased",
        sequence_length=128,
    )

    #Pretrained classifier.
    classifier = keras_nlp.models.AlbertClassifier.from_preset(
        "albert_base_en_uncased",
        num_classes=4,
        preprocessor=preprocessor,
    )
    # Re-compile (e.g., with a new learning rate).
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    # Create a dataset with preprocessed features in an `(x, y)` format.
    preprocessed_features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "segment_ids": tf.constant(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_nlp.models.AlbertClassifier.from_preset(
        "albert_base_en_uncased",
        num_classes=4,
        preprocessor=None,
    )
    classifier.fit(x=preprocessed_features, y=labels, batch_size=2)
    ```

     Custom backbone and vocabulary.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_nlp.models.AlbertTokenizer(
        vocabulary=vocab,
    )
    preprocessor = keras_nlp.models.AlbertPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.AlbertBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_nlp.models.AlbertClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        dropout=0.1,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        pooled = keras.layers.Dropout(dropout)(pooled)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=albert_kernel_initializer(),
            name="logits",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self._preprocessor = preprocessor
        self.num_classes = num_classes
        self.dropout = dropout

        # Default compilation
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(5e-5),
            metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            }
        )

        return config

    @classproperty
    def backbone_cls(cls):
        return AlbertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return AlbertPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})
