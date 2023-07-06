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
"""Test for BART backbone models."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.tests.test_case import TestCase


class BartBackboneTest(TestCase):
    def setUp(self):
        self.model = BartBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
        )
        self.batch_size = 8
        self.input_batch = {
            "encoder_token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "encoder_padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "decoder_token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "decoder_padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_bart(self):
        self.model(self.input_batch)

        # Check default name passed through
        self.assertRegexpMatches(self.model.name, "bart_backbone")

    def test_variable_sequence_length_call_bart(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "encoder_token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "encoder_padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "decoder_token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "decoder_padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile_batched_ds(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_dataset)

    @pytest.mark.large
    def test_saved_model(self):
        model_output = self.backbone(self.input_batch)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        self.model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BartBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            model_output["encoder_sequence_output"],
            restored_output["encoder_sequence_output"],
        )
        self.assertAllClose(
            model_output["decoder_sequence_output"],
            restored_output["decoder_sequence_output"],
        )


@pytest.mark.tpu
@pytest.mark.usefixtures("tpu_test_class")
class BartBackboneTPUTest(TestCase):
    def setUp(self):
        with self.tpu_strategy.scope():
            self.model = BartBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_heads=2,
                hidden_dim=64,
                intermediate_dim=128,
                max_sequence_length=128,
            )
        self.input_batch = {
            "encoder_token_ids": tf.ones((8, 128), dtype="int32"),
            "encoder_padding_mask": tf.ones((8, 128), dtype="int32"),
            "decoder_token_ids": tf.ones((8, 128), dtype="int32"),
            "decoder_padding_mask": tf.ones((8, 128), dtype="int32"),
        }
        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.model.compile()
        self.model.predict(self.input_dataset)
