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
"""Test for FNet backbone model."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.f_net.f_net_backbone import FNetBackbone


class FNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.backbone = FNetBackbone(
            vocabulary_size=1000,
            num_layers=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=5,
            num_segments=4,
        )
        self.input_batch = {
            "token_ids": tf.ones((2, 5), dtype="int32"),
            "segment_ids": tf.ones((2, 5), dtype="int32"),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_f_net(self):
        self.backbone(self.input_batch)

        # Check default name passed through
        self.assertRegexpMatches(self.backbone.name, "f_net_backbone")

    def test_variable_sequence_length_call_f_net(self):
        for seq_length in (2, 3, 4):
            input_data = {
                "token_ids": tf.ones((2, seq_length), dtype="int32"),
                "segment_ids": tf.ones((2, seq_length), dtype="int32"),
            }
            self.backbone(input_data)

    def test_predict(self):
        self.backbone.predict(self.input_batch)
        self.backbone.predict(self.input_dataset)
    
    def test_serialization(self):
        new_backbone = keras.utils.deserialize_keras_object(
            keras.utils.serialize_keras_object(self.backbone)
        )
        self.assertEqual(new_backbone.get_config(), self.backbone.get_config())
    
    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )

    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        model_output = self.backbone(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.backbone.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, FNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            model_output["pooled_output"], restored_output["pooled_output"]
        )


@pytest.mark.tpu
@pytest.mark.usefixtures("tpu_test_class")
class FNetBackboneTPUTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        with self.tpu_strategy.scope():
            self.backbone = FNetBackbone(
                vocabulary_size=1000,
                num_layers=2,
                hidden_dim=16,
                intermediate_dim=32,
                max_sequence_length=128,
                num_segments=4,
            )
        self.input_batch = {
            "token_ids": tf.ones((8, 128), dtype="int32"),
            "segment_ids": tf.ones((8, 128), dtype="int32"),
        }
        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.backbone.compile()
        self.backbone.predict(self.input_dataset)
