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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbones.resnet_vd.resnet_vd_backbone import (
    ResNetBackbone,
)
from keras_nlp.src.models.backbones.resnet_vd.resnet_vd_backbone_presets import (
    backbone_presets,
)
from keras_nlp.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """ResNetBackbone (Vd) model with {num_layers} layers.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)

    This variant of ResNet uses a different layout in the model's stem
    and within shortcut connections of residual blocks.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = ResNet{num_layers}VdBackbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_nlp_export("keras_nlp.models.ResNet18VdBackbone")
class ResNet18VdBackbone(ResNetVdBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetVdBackbone.from_preset("resnet18_vd", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_nlp_export("keras_nlp.models.ResNet34VdBackbone")
class ResNet34VdBackbone(ResNetVdBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetVdBackbone.from_preset("resnet34_vd", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_nlp_export("keras_nlp.models.ResNet50VdBackbone")
class ResNet50VdBackbone(ResNetVdBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetVdBackbone.from_preset("resnet50_vd", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_nlp_export("keras_nlp.models.ResNet101VdBackbone")
class ResNet101VdBackbone(ResNetVdBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetVdBackbone.from_preset("resnet101_vd", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_nlp_export("keras_nlp.models.ResNet152VdBackbone")
class ResNet152VdBackbone(ResNetVdBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ResNetVdBackbone.from_preset("resnet152_vd", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


setattr(ResNet18VdBackbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=18))
setattr(ResNet34VdBackbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=34))
setattr(ResNet50VdBackbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=50))
setattr(ResNet101VdBackbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=101))
setattr(ResNet152VdBackbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=152))
