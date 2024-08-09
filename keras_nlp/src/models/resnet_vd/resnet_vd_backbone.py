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
"""ResNet_vd backbone model.
Reference:
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    (CVPR 2015)
  - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)  # noqa: E501
"""

import copy

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.backbones.resnet_vd.resnet_vd_backbone_presets import (
    backbone_presets,
)
from keras_nlp.src.models.backbones.resnet_vd.resnet_vd_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_nlp.src.utils.python_utils import classproperty

BN_AXIS = 3
BN_EPSILON = 1.001e-5


@keras_nlp_export("keras_nlp.models.ResNetVdBackbone")
class ResNetVdBackbone(Backbone):
    """Instantiates the ResNet_vd architecture.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)

    This variant of ResNet uses a different layout in the model's stem
    and modified pooling operations within shortcut connections of
    residual blocks.

    Args:
        stackwise_filters: list of ints, number of filters for each stack in
            the model.
        stackwise_blocks: list of ints, number of blocks for each stack in the
            model.
        stackwise_strides: list of ints, stride for each stack in the model.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.

    Examples:
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Standard architecture from preset
    ```python
    model = keras_nlp.models.ResNetVdBackbone.from_preset("resnet50_vd")
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = ResNetVdBackbone(
        stackwise_filters=[64, 128, 256, 512],
        stackwise_blocks=[2, 2, 2, 2],
        stackwise_strides=[1, 2, 2, 2],
        include_rescaling=False
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_rescaling,
        input_shape=(None, None, 3),
        block_type="block",
        **kwargs,
    ):
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        x = keras.layers.Conv2D(
            32, 3, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)

        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)

        x = keras.layers.Conv2D(
            32, 3, strides=1, use_bias=False, padding="same", name="conv2_conv"
        )(x)

        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv2_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv2_relu")(x)

        x = keras.layers.Conv2D(
            64, 3, strides=1, use_bias=False, padding="same", name="conv3_conv"
        )(x)

        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv3_bn"
        )(x)
        x = keras.layers.Activation("relu", name="conv3_relu")(x)

        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)

        num_stacks = len(stackwise_filters)

        pyramid_level_inputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                block_type=block_type,
                first_shortcut=(block_type == "block" or stack_index > 0),
                name=f"vd_stack_{stack_index}",
            )
            pyramid_level_inputs[f"P{stack_index + 2}"] = (
                x._keras_history.operation.name
            )

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.block_type = block_type

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_filters": self.stackwise_filters,
                "stackwise_blocks": self.stackwise_blocks,
                "stackwise_strides": self.stackwise_strides,
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "block_type": self.block_type,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)


def apply_basic_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A basic residual block (vd style).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"vd_basic_block_{keras.backend.get_uid('vd_basic_block_')}"

    if conv_shortcut:
        if stride > 1:
            shortcut = keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=stride, padding="same"
            )(x)
        else:
            shortcut = x
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            strides=1,
            use_bias=False,
            name=name + "_0_conv",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=stride,
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)

    x = keras.layers.Add(name=name + "_add")([shortcut, x])
    x = keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def apply_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A residual block (vd style).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"vd_block_{keras.backend.get_uid('vd_block')}"

    if conv_shortcut:
        if stride > 1:
            shortcut = keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=stride, padding="same"
            )(x)
        else:
            shortcut = x
        shortcut = keras.layers.Conv2D(
            4 * filters,
            1,
            strides=1,
            use_bias=False,
            name=name + "_0_conv",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = keras.layers.Conv2D(
        filters, 1, strides=stride, use_bias=False, name=name + "_1_conv"
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = keras.layers.Conv2D(
        4 * filters, 1, use_bias=False, name=name + "_3_conv"
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_3_bn"
    )(x)

    x = keras.layers.Add(name=name + "_add")([shortcut, x])
    x = keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def apply_stack(
    x,
    filters,
    blocks,
    stride=2,
    name=None,
    block_type="block",
    first_shortcut=True,
):
    """A set of stacked residual blocks.

    Args:
        x: input tensor.
        filters: int, filters of the layer in a block.
        blocks: int, blocks in the stacked blocks.
        stride: int, stride of the first layer in the first block, defaults to
            2.
        name: string, optional prefix for the layer names used in the block.
        block_type: string, one of "basic_block" or "block". The block type to
              stack. Use "basic_block" for ResNet18 and ResNet34.
        first_shortcut: bool. Use convolution shortcut if `True` (default),
              otherwise uses identity or pooling shortcut, based on stride.

    Returns:
        Output tensor for the stacked blocks.
    """

    if name is None:
        name = "vd_stack"

    if block_type == "basic_block":
        block_fn = apply_basic_block
    elif block_type == "block":
        block_fn = apply_block
    else:
        raise ValueError(
            """`block_type` must be either "basic_block" or "block". """
            f"Received block_type={block_type}."
        )

    x = block_fn(
        x,
        filters,
        stride=stride,
        name=name + "_block1",
        conv_shortcut=first_shortcut,
    )
    for i in range(2, blocks + 1):
        x = block_fn(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x
