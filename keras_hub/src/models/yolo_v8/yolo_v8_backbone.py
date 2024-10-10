import keras
from keras import ops
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Rescaling

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_conv_bn
from keras_hub.src.models.yolo_v8.yolo_v8_layers import apply_csp_block


def get_tensor_input_name(tensor):
    return tensor._keras_history.operation.name


def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor


def apply_spatial_pyramid_pooling_fast(
    inputs, pool_size=5, activation="swish", name="spp_fast"
):
    channel_axis = -1
    input_channels = inputs.shape[channel_axis]
    hidden_channels = int(input_channels // 2)

    x = apply_conv_bn(
        inputs,
        hidden_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    pool_1 = MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool1"
    )(x)
    pool_2 = MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool2"
    )(pool_1)
    pool_3 = MaxPooling2D(
        pool_size=pool_size, strides=1, padding="same", name=f"{name}_pool3"
    )(pool_2)

    out = ops.concatenate([x, pool_1, pool_2, pool_3], axis=channel_axis)
    out = apply_conv_bn(
        out,
        input_channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out


@keras_hub_export("keras_hub.models.YOLOV8Backbone")
class YOLOV8Backbone(Backbone):
    """Implements the YOLOV8 backbone for object detection.

    This backbone is a variant of the `CSPDarkNetBackbone` architecture.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_channels: A list of int. The number of channels for each dark
            level in the model.
        stackwise_depth: A list of int. The depth for each dark level in the
            model.
        include_rescaling: bool. Rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        activation: str. The activation functions to use in the backbone to
            use in the CSPDarkNet blocks. Defaults to "swish".
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Returns:
        A `keras.Model` instance.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_hub.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_hub.models.YOLOV8Backbone(
        stackwise_channels=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        stackwise_channels,
        stackwise_depth,
        include_rescaling,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = Rescaling(1 / 255.0)(x)

        """ Stem """
        stem_width = stackwise_channels[0]
        x = apply_conv_bn(
            x,
            stem_width // 2,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_1",
        )
        x = apply_conv_bn(
            x,
            stem_width,
            kernel_size=3,
            strides=2,
            activation=activation,
            name="stem_2",
        )

        """ blocks """
        pyramid_level_inputs = {"P1": get_tensor_input_name(x)}
        for stack_id, (channel, depth) in enumerate(
            zip(stackwise_channels, stackwise_depth)
        ):
            stack_name = f"stack{stack_id + 1}"
            if stack_id >= 1:
                x = apply_conv_bn(
                    x,
                    channel,
                    kernel_size=3,
                    strides=2,
                    activation=activation,
                    name=f"{stack_name}_downsample",
                )
            x = apply_csp_block(
                x,
                depth=depth,
                expansion=0.5,
                activation=activation,
                name=f"{stack_name}_c2f",
            )

            if stack_id == len(stackwise_depth) - 1:
                x = apply_spatial_pyramid_pooling_fast(
                    x,
                    pool_size=5,
                    activation=activation,
                    name=f"{stack_name}_spp_fast",
                )
            pyramid_level_inputs[f"P{stack_id + 2}"] = get_tensor_input_name(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.include_rescaling = include_rescaling
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "stackwise_channels": self.stackwise_channels,
                "stackwise_depth": self.stackwise_depth,
                "activation": self.activation,
            }
        )
        return config
