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
from keras import layers
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone
from keras_nlp.src.models.stable_diffusion_3.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
)
from keras_nlp.src.models.stable_diffusion_3.mmdit import MMDiT
from keras_nlp.src.models.stable_diffusion_3.vae_image_decoder import (
    VAEImageDecoder,
)
from keras_nlp.src.utils.keras_utils import standardize_data_format


class Projection(layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)

        self.dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="dense",
        )

    def build(self, inputs_shape, token_ids_shape):
        inputs_shape = list(inputs_shape)
        self.dense.build([None, inputs_shape[-1]])
        self.dense._kernel.assign(
            ops.transpose(ops.eye(self.hidden_dim), (1, 0))
        )

    def call(self, inputs, token_ids):
        indices = ops.expand_dims(
            ops.cast(ops.argmax(token_ids, axis=-1), "int32"), axis=-1
        )
        pooled_output = ops.take_along_axis(inputs, indices[:, :, None], axis=1)
        pooled_output = ops.squeeze(pooled_output, axis=1)
        projection_output = self.dense(pooled_output)
        return projection_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras_nlp_export("keras_nlp.models.StableDiffusion3Backbone")
class StableDiffusion3Backbone(Backbone):
    def __init__(
        self,
        mmdit_patch_size,
        mmdit_num_heads,
        mmdit_hidden_dim,
        mmdit_depth,
        mmdit_position_size,
        vae_stackwise_num_filters,
        vae_stackwise_num_blocks,
        clip_l,
        clip_g,
        t5=None,
        latent_channels=16,
        output_channels=3,
        num_train_timesteps=1000,
        shift=1.0,
        height=None,
        width=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        height = int(height or 1024)
        width = int(width or 1024)
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "`height` and `width` must be divisible by 8. "
                f"Received: height={height}, width={width}"
            )
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        latent_shape = (height // 8, width // 8, latent_channels)

        # === Layers ===
        self.clip_l = clip_l
        self.clip_l_projection = Projection(
            clip_l.hidden_dim, dtype=dtype, name="clip_l_projection"
        )
        self.clip_l_projection.build([None, clip_l.hidden_dim], None)
        self.clip_g = clip_g
        self.clip_g_projection = Projection(
            clip_g.hidden_dim, dtype=dtype, name="clip_g_projection"
        )
        self.clip_g_projection.build([None, clip_g.hidden_dim], None)
        self.t5 = t5
        self.mmdit = MMDiT(
            mmdit_patch_size,
            mmdit_num_heads,
            mmdit_hidden_dim,
            mmdit_depth,
            mmdit_position_size,
            latent_shape=latent_shape,
            data_format=data_format,
            dtype=dtype,
            name="mmdit",
        )
        self.vae = VAEImageDecoder(
            vae_stackwise_num_filters,
            vae_stackwise_num_blocks,
            output_channels,
            latent_shape=latent_shape,
            data_format=data_format,
            dtype=dtype,
            name="vae",
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

        # === Functional Model ===
        # TODO: Can we define the model here?
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.mmdit_patch_size = mmdit_patch_size
        self.mmdit_num_heads = mmdit_num_heads
        self.mmdit_hidden_dim = mmdit_hidden_dim
        self.mmdit_depth = mmdit_depth
        self.mmdit_position_size = mmdit_position_size
        self.vae_stackwise_num_filters = vae_stackwise_num_filters
        self.vae_stackwise_num_blocks = vae_stackwise_num_blocks
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        # We don't add `height` and `width` to config to make the backbone more
        # flexible.

    @property
    def latent_shape(self):
        return (None,) + tuple(self.mmdit.latent_shape)

    @property
    def clip_hidden_dim(self):
        return self.clip_l.hidden_dim + self.clip_g.hidden_dim

    @property
    def t5_hidden_dim(self):
        return 4096 if self.t5 is None else self.t5.hidden_dim

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mmdit_patch_size": self.mmdit_patch_size,
                "mmdit_num_heads": self.mmdit_num_heads,
                "mmdit_hidden_dim": self.mmdit_hidden_dim,
                "mmdit_depth": self.mmdit_depth,
                "mmdit_position_size": self.mmdit_position_size,
                "vae_stackwise_num_filters": self.vae_stackwise_num_filters,
                "vae_stackwise_num_blocks": self.vae_stackwise_num_blocks,
                "clip_l": layers.serialize(self.clip_l),
                "clip_g": layers.serialize(self.clip_g),
                "t5": layers.serialize(self.t5),
                "latent_channels": self.latent_channels,
                "output_channels": self.output_channels,
                "num_train_timesteps": self.num_train_timesteps,
                "shift": self.shift,
            }
        )
        return config
