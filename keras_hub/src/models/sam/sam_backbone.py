# Copyright 2024 The KerasHub Authors
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

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


@keras_hub_export("keras_hub.models.SAMBackbone")
class SAMBackbone(Backbone):
    """A backbone for the Segment Anything Model (SAM).

    Args:
        image_encoder: `keras_hub.models.ViTDetBackbone`. A feature extractor for
            the input images.
        prompt_encoder: `keras_hub.layers.SAMPromptEncoder`. A Keras layer to
            compute embeddings for points, box, and mask prompt.
        mask_decoder: `keras_hub.layers.SAMMaskDecoder`. A Keras layer to
            generate segmentation masks given the embeddings generated by the
            backbone and the prompt encoder.
        dtype: The dtype of the layer weights.

    Example:
    ```python
    image_size=128
    batch_size=2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels": np.ones((batch_size, 1), dtype="float32"),
        "boxes": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks": np.zeros(
            (batch_size, 0, image_size, image_size, 1)
        ),
    }
    image_encoder = keras_hub.models.ViTDetBackbone(
        hidden_size=16,
        num_layers=16,
        intermediate_dim=16 * 4,
        num_heads=16,
        global_attention_layer_indices=[2, 5, 8, 11],
        patch_size=16,
        num_output_channels=8,
        window_size=2,
        image_shape=(image_size, image_size, 3),
    )
    prompt_encoder = keras_hub.layers.SAMPromptEncoder(
        hidden_size=8,
        image_embedding_size=(8, 8),
        input_image_size=(
            image_size,
            image_size,
        ),
        mask_in_channels=16,
    )
    mask_decoder = keras_hub.layers.SAMMaskDecoder(
        num_layers=2,
        hidden_size=8,
        intermediate_dim=32,
        num_heads=8,
        embedding_dim=8,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=8,
    )
    backbone = keras_hub.models.SAMBackbone(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        image_shape=(image_size, image_size, 3),
    )
    backbone(input_data)
    ```
    """

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # === Functional model
        image_input = self.image_encoder.input

        inputs = {
            "images": image_input,
            "points": keras.Input(shape=[None, 2], name="points"),
            "labels": keras.Input(shape=[None], name="labels"),
            "boxes": keras.Input(shape=[None, 2, 2], name="boxes"),
            "masks": keras.Input(shape=[None, None, None, 1], name="masks"),
        }
        image_embeddings = self.image_encoder.output
        prompt_embeddings = self.prompt_encoder(inputs)
        outputs = {
            "image_embeddings": image_embeddings,
        }
        outputs.update(prompt_embeddings)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.layers.serialize(self.image_encoder),
                "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
                "mask_decoder": keras.layers.serialize(self.mask_decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "image_encoder": keras.layers.deserialize(
                    config["image_encoder"]
                ),
                "prompt_encoder": keras.layers.deserialize(
                    config["prompt_encoder"]
                ),
                "mask_decoder": keras.layers.deserialize(
                    config["mask_decoder"]
                ),
            }
        )

        return super().from_config(config)
