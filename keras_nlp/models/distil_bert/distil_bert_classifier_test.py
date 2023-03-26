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
"""Tests for DistilBERT classification model."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.distil_bert.distil_bert_backbone import DistilBertBackbone
from keras_nlp.models.distil_bert.distil_bert_classifier import (
    DistilBertClassifier,
)
from keras_nlp.models.distil_bert.distil_bert_preprocessor import (
    DistilBertPreprocessor,
)
from keras_nlp.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)


class DistilBertClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        # Setup model

        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "."]
        self.preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        self.backbone = DistilBertBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )
        self.classifier = DistilBertClassifier(
            self.backbone,
            4,
            preprocessor=self.preprocessor,
        )

        self.raw_batch = tf.constant(
            [
                "the quick brown fox.",
                "the slow brown fox.",
                "the smelly brown fox.",
                "the old brown fox.",
            ]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            (self.raw_batch, tf.ones((4,)))
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_classifier(self):
        self.classifier(self.preprocessed_batch)

    def test_classifier_predict(self):
        self.classifier.predict(self.raw_batch)
        self.classifier.preprocessor = None
        self.classifier.predict(self.preprocessed_batch)

    def test_classifier_fit(self):
        self.classifier.fit(self.raw_dataset)
        self.classifier.preprocessor = None
        self.classifier.fit(self.preprocessed_dataset)

    def test_distilbert_classifier_fit_default_compile(self):
        self.classifier.fit(self.raw_dataset)

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.classifier)
        new_classifier = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_classifier.get_config(),
            self.classifier.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saving_model(self, save_format, filename):
        model_output = self.classifier.predict(self.raw_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.classifier.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DistilBertClassifier)

        # Check that output matches.
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
