# Copyright 2024 The KerasNLP Authors
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

import os

import keras
import numpy as np
import pytest
from absl.testing import parameterized

from keras_nlp.src.models.resnet_vd.resnet_vd_backbone import ResNetVdBackbone
from keras_nlp.src.tests.test_case import TestCase


class ResNetVdBackboneTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = ResNetVdBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = ResNetVdBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = ResNetVdBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "resnet_vd_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.saving.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, ResNetVdBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            keras.ops.convert_to_numpy(model_output),
            keras.ops.convert_to_numpy(restored_output),
        )

    def test_feature_pyramid_inputs(self):
        model = ResNetVdBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        pyramid_outputs = {
            key: model.get_layer(name).output
            for key, name in model.pyramid_level_inputs.items()
        }
        backbone_model = keras.Model(
            inputs=model.inputs, outputs=pyramid_outputs
        )
        input_size = 256
        inputs = keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        levels = ["P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P2"].shape,
            (None, input_size // 2**2, input_size // 2**2, 256),
        )
        self.assertEquals(
            outputs["P3"].shape,
            (None, input_size // 2**3, input_size // 2**3, 512),
        )
        self.assertEquals(
            outputs["P4"].shape,
            (None, input_size // 2**4, input_size // 2**4, 1024),
        )
        self.assertEquals(
            outputs["P5"].shape,
            (None, input_size // 2**5, input_size // 2**5, 2048),
        )

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        # ResNet50 model
        model = ResNetVdBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[3, 4, 6, 3],
            stackwise_strides=[1, 2, 2, 2],
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 2048))
