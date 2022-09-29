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
"""Test for DistilBERT backbone models."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.distilbert.distilbert_models import DistilBertBase
from keras_nlp.models.distilbert.distilbert_models import DistilBertCustom


class DistilBertTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = DistilBertCustom(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.batch_size = 8
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_distilbert(self):
        self.model(self.input_batch)

    def test_variable_sequence_length_call_distilbert(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    def test_valid_call_distilbert_base(self):
        model = DistilBertBase(vocabulary_size=1000, name="encoder")
        input_data = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }
        model(input_data)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_distilbert_base_compile(self, jit_compile):
        model = DistilBertBase(vocabulary_size=1000, name="encoder")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_distilbert_base_compile_batched_ds(self, jit_compile):
        model = DistilBertBase(vocabulary_size=1000, name="encoder")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_dataset)

    def test_distilbert_base_vocab_error(self):
        # Need `vocabulary_size` or `weights`
        with self.assertRaises(ValueError):
            DistilBertBase(name="encoder")

        # Only one of `vocabulary_size` or `weights`
        with self.assertRaises(ValueError):
            DistilBertBase(
                weights="distilbert_base_uncased_en",
                vocabulary_size=1000,
                name="encoder",
            )

        # Not a checkpoint name
        with self.assertRaises(ValueError):
            DistilBertBase(
                weights="distilbert_base_uncased_clowntown",
                name="encoder",
            )

    def test_saving_model(self):
        model_output = self.model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "model")
        self.model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)
