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

import copy
import math

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import keras
from keras_nlp.src.models import utils
from keras_nlp.src.models.backbones.backbone_presets import backbone_presets
from keras_nlp.src.models.text_detection.diffbin.diffbin_presets import (
    basnet_presets,
)
from keras_nlp.src.models.text_detection.diffbin.diffbin_presets import (
    presets_no_weights,
)
from keras_nlp.src.models.text_detection.diffbin.diffbin_presets import (
    presets_with_weights,
)
from keras_nlp.src.models.task import Task
from keras_nlp.src.utils.python_utils import classproperty


@keras_nlp_export(
    [
        "keras_nlp.models.DiffBin",
        "keras_nlp.models.text_detection.DiffBin",
    ]
)
class DiffBin(Task):
    """
    A Keras model implementing the Differential Binarization
    architecture for text detection.

        ```
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        input_shape=(None, None, 3),
        input_tensor=None,
        include_rescaling=False,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance"
                f" or `keras.Model`. Received instead"
                f" backbone={backbone} (of type {type(backbone)})."
            )

        if backbone.input_shape != (None, None, None, 3):
            raise ValueError(
                "Do not specify 'input_shape' or 'input_tensor' within the"
                " 'DiffBin' backbone. \nPlease provide 'input_shape' or"
                " 'input_tensor' while initializing the 'DiffBin' model."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        x = extract_backbone_features(backbone)(x)
        x = diffbin_fpn_model(x, out_channels=256, kernel_list=[3, 2, 2])
        probability_maps = diffbin_head(x, in_channels=256, name='head_prob')
        threshold_maps = diffbin_head(x, in_channels=256, name='head_thresh')
        binary_maps = step_function(probability_maps, threshold_maps)
        outputs = keras.layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps]
        )

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        
        self.backbone = backbone
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling

    def get_config(self):
        return {
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "input_shape": self.input_shape[1:],
            "input_tensor": keras.saving.serialize_keras_object(
                self.input_tensor
            ),
            "include_rescaling": self.include_rescaling,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            input_shape = (None, None, 3)
            if isinstance(config["backbone"]["config"]["input_shape"], list):
                input_shape = list(input_shape)
            if config["backbone"]["config"]["input_shape"] != input_shape:
                config["input_shape"] = config["backbone"]["config"][
                    "input_shape"
                ]
                config["backbone"]["config"]["input_shape"] = input_shape
            config["backbone"] = keras.layers.deserialize(config["backbone"])

        if "input_tensor" in config and isinstance(
            config["input_tensor"], dict
        ):
            config["input_tensor"] = keras.layers.deserialize(
                config["input_tensor"]
            )

        return super().from_config(config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        filtered_backbone_presets = copy.deepcopy(
            {
                k: v
                for k, v in backbone_presets.items()
                if k in ("resnet50_vd", )
            }
        )

        return copy.deepcopy({**filtered_backbone_presets, **basnet_presets})

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return copy.deepcopy(presets_with_weights)

    @classproperty
    def presets_without_weights(cls):
        """
        Dictionary of preset names and configurations that has no weights.
        """
        return copy.deepcopy(presets_no_weights)

    @classproperty
    def backbone_presets(cls):
        """
        Dictionary of preset names and configurations of compatible backbones.
        """
        filtered_backbone_presets = copy.deepcopy(
            {
                k: v
                for k, v in backbone_presets.items()
                if k in ("resnet50_vd", )
            }
        )
        filtered_presets = copy.deepcopy(filtered_backbone_presets)
        return filtered_presets

def extract_backbone_features(resnet_backbone):
    levels = ["P2", "P3", "P4", "P5"]
    layer_names = [resnet_backbone.pyramid_level_inputs[level] for level in levels]
    items = zip(levels, layer_names)
    outputs = {key: resnet_backbone.get_layer(name).output for key, name in items}
    backbone = keras.Model(resnet_backbone.inputs, outputs=outputs, name='backbone')
    return backbone

def diffbin_fpn_model(inputs, out_channels):
    c2 = inputs["P2"]
    c3 = inputs["P3"]
    c4 = inputs["P4"]
    c5 = inputs["P5"]
    in2 = keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, name='neck_in2')(c2)
    in3 = keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, name='neck_in3')(c3)
    in4 = keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, name='neck_in4')(c4)
    in5 = keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, name='neck_in5')(c5)
    out4 = keras.layers.Add(name='add1')([keras.layers.UpSampling2D()(in5), in4])
    out3 = keras.layers.Add(name='add2')([keras.layers.UpSampling2D()(out4), in3])
    out2 = keras.layers.Add(name='add3')([keras.layers.UpSampling2D()(out3), in2])
    p5 = keras.layers.Conv2D(
        out_channels // 4, kernel_size=3, padding="same", use_bias=False,
        name='neck_p5'
    )(in5)
    p4 = keras.layers.Conv2D(
        out_channels // 4, kernel_size=3, padding="same", use_bias=False,
        name='neck_p4'
    )(out4)
    p3 = keras.layers.Conv2D(
        out_channels // 4, kernel_size=3, padding="same", use_bias=False,
        name='neck_p3'
    )(out3)
    p2 = keras.layers.Conv2D(
        out_channels // 4, kernel_size=3, padding="same", use_bias=False,
        name='neck_p2'
    )(out2)
    p5 = keras.layers.UpSampling2D((8, 8))(p5)
    p4 = keras.layers.UpSampling2D((4, 4))(p4)
    p3 = keras.layers.UpSampling2D((2, 2))(p3)

    fused = keras.layers.Concatenate(axis=-1)([p5, p4, p3, p2])
    return fused

def step_function(x, y, k=50):
    return 1.0 / (1.0 + keras.ops.exp(-k * (x - y)))

def diffbin_head(inputs, in_channels, kernel_list, name):
    x = keras.layers.Conv2D(
        in_channels // 4,
        kernel_size=kernel_list[0],
        padding="same",
        use_bias=False,
        name=f'{name}_conv0_weights'
    )(inputs)
    x = keras.layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f'{name}_conv0_bn'
    )(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2DTranspose(
        in_channels // 4,
        kernel_size=kernel_list[1],
        strides=2,
        padding="valid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
            maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
        ),
        name=f'{name}_conv1_weights'
    )(x)
    x = keras.layers.BatchNormalization(
        beta_initializer=keras.initializers.Constant(1e-4),
        gamma_initializer=keras.initializers.Constant(1.0),
        name=f'{name}_conv1_bn'
    )(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2DTranspose(
        1,
        kernel_size=kernel_list[2],
        strides=2,
        padding="valid",
        activation="sigmoid",
        bias_initializer=keras.initializers.RandomUniform(
            minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
            maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
        ),
        name=f'{name}_conv2_weights'
    )(x)
    return x
