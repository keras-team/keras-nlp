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
from keras import ops
from keras import random

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.stable_diffusion_v3.clip_preprocessor import (
    CLIPPreprocessor,
)
from keras_nlp.src.models.stable_diffusion_v3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_nlp.src.models.stable_diffusion_v3.t5_preprocessor import (
    T5Preprocessor,
)
from keras_nlp.src.models.text_to_image import TextToImage


@keras_nlp_export("keras_nlp.models.StableDiffusion3TextToImage")
class StableDiffusion3TextToImage(TextToImage):
    backbone_cls = StableDiffusion3Backbone
    clip_l_preprocessor_cls = CLIPPreprocessor
    clip_g_preprocessor_cls = CLIPPreprocessor
    t5_preprocessor_cls = T5Preprocessor

    def __init__(
        self,
        backbone,
        clip_l_preprocessor,
        clip_g_preprocessor,
        t5_preprocessor=None,
        **kwargs,
    ):
        if not isinstance(backbone, StableDiffusion3Backbone):
            raise ValueError
        if hasattr(backbone, "t5_text_encoder") and t5_preprocessor is None:
            raise ValueError(
                "`t5_preprocessor` must be provided if "
                "`backbone.t5_text_encoder` is provided."
            )
        # === Layers ===
        self.backbone = backbone
        self.clip_l_preprocessor = clip_l_preprocessor
        self.clip_g_preprocessor = clip_g_preprocessor
        self.t5_preprocessor = t5_preprocessor

        # === Functional Model ===
        # TODO: Can we define the model here?
        super().__init__(**kwargs)

        self.latent_shape = backbone.latent_shape

    def preprocess(self, x):
        token_ids = {}
        token_ids["clip_l"] = self.clip_l_preprocessor(x)["token_ids"]
        token_ids["clip_g"] = self.clip_g_preprocessor(x)["token_ids"]
        if self.t5_preprocessor is not None:
            token_ids["t5"] = self.t5_preprocessor(x)["token_ids"]
        return token_ids

    def encode_step(self, token_ids, negative_token_ids):
        clip_hidden_dim = (
            self.backbone.clip_l_hidden_dim + self.backbone.clip_g_hidden_dim
        )
        clip_sequence_length = self.backbone.clip_l_sequence_length
        t5_hidden_dim = self.backbone.t5_hidden_dim

        def encode(token_ids):
            clip_l_outputs = self.backbone.clip_l_text_encoder(
                token_ids["clip_l"]
            )
            clip_g_outputs = self.backbone.clip_g_text_encoder(
                token_ids["clip_g"]
            )
            pooled_embeddings = ops.concatenate(
                [
                    clip_l_outputs["encoder_projection_output"],
                    clip_g_outputs["encoder_projection_output"],
                ],
                axis=-1,
            )
            embeddings = ops.concatenate(
                [
                    clip_l_outputs["encoder_intermediate_output"],
                    clip_g_outputs["encoder_intermediate_output"],
                ],
                axis=-1,
            )
            embeddings = ops.pad(
                embeddings,
                [[0, 0], [0, 0], [0, t5_hidden_dim - clip_hidden_dim]],
            )
            if hasattr(self.backbone, "t5_text_encoder"):
                t5_outputs = self.backbone.t5_text_encoder(token_ids["t5"])
                embeddings = ops.concatenate([embeddings, t5_outputs], axis=-2)
            else:
                embeddings = ops.pad(
                    embeddings, [[0, 0], [0, clip_sequence_length], [0, 0]]
                )
            return embeddings, pooled_embeddings

        positive_embeddings, positive_pooled_embeddings = encode(token_ids)
        negative_embeddings, negative_pooled_embeddings = encode(
            negative_token_ids
        )

        # Concatenation for classifier-free guidance.
        embeddings = ops.concatenate(
            [positive_embeddings, negative_embeddings], axis=0
        )
        pooled_embeddings = ops.concatenate(
            [positive_pooled_embeddings, negative_pooled_embeddings], axis=0
        )
        return embeddings, pooled_embeddings

    def denoise_step(
        self,
        latents,
        embeddings,
        steps,
        num_steps,
        classifier_free_guidance_scale,
    ):
        contexts, pooled_projections = embeddings
        sigma = self.backbone.noise_scheduler.get_sigma(steps, num_steps)
        sigma_next = self.backbone.noise_scheduler.get_sigma(
            steps + 1, num_steps
        )

        # Sigma to timestep.
        timestep = self.backbone.noise_scheduler.sigma_to_timestep(sigma)
        timestep = ops.broadcast_to(timestep, ops.shape(latents)[:1])

        # Diffusion.
        predicted_noise = self.backbone.mmdit_diffuser(
            {
                "latent": ops.concatenate([latents, latents], axis=0),
                "context": contexts,
                "pooled_projection": pooled_projections,
                "timestep": ops.concatenate([timestep, timestep], axis=0),
            },
            training=False,
        )
        predicted_noise = ops.cast(predicted_noise, "float32")

        # Classifier-free guidance.
        classifier_free_guidance_scale = ops.cast(
            classifier_free_guidance_scale, predicted_noise.dtype
        )
        positive_noise, negative_noise = ops.split(predicted_noise, 2, axis=0)
        predicted_noise = negative_noise + classifier_free_guidance_scale * (
            positive_noise - negative_noise
        )

        # Euler step.
        latents = self.backbone.noise_scheduler.step(
            latents, predicted_noise, sigma, sigma_next
        )
        return latents

    def decode_step(self, latents):
        # Latent calibration.
        latents = ops.add(ops.divide(latents, 1.5305), 0.0609)

        # Decoding.
        outputs = self.backbone.vae_image_decoder(latents, training=False)
        return outputs

    def text_to_image(
        self,
        inputs,
        negative_inputs=None,
        num_steps=28,
        classifier_free_guidance_scale=7.0,
        seed=None,
    ):
        # Setup our three main passes.
        # 1. Preprocessing strings to dense integer tensors.
        # 2. Invoke compiled functions on dense tensors.
        # 3. Postprocess dense tensors back to images.
        encode_function = self.make_encode_function()
        denoise_function = self.make_denoise_function()
        decode_function = self.make_decode_function()

        # Normalize and preprocess inputs.
        inputs, input_is_scalar = self._normalize_inputs(inputs)
        negative_inputs, _ = self._normalize_inputs(negative_inputs)
        token_ids = self.preprocess(inputs)
        negative_token_ids = self.preprocess(negative_inputs)

        # Initialize random latents.
        latent_shape = (len(inputs),) + tuple(self.latent_shape)[1:]
        latents = random.normal(latent_shape, dtype="float32", seed=seed)

        # Encode inputs.
        embeddings = encode_function(token_ids, negative_token_ids)

        # Denoise.
        for i in range(num_steps):
            latents = denoise_function(
                latents,
                embeddings,
                ops.convert_to_tensor(i),
                ops.convert_to_tensor(num_steps),
                ops.convert_to_tensor(classifier_free_guidance_scale),
            )

        # Decode.
        outputs = decode_function(latents)
        images = self._normalize_outputs(outputs, input_is_scalar)
        return images