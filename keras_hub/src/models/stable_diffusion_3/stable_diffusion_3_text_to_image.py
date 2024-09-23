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
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_backbone import (
    StableDiffusion3Backbone,
)
from keras_hub.src.models.stable_diffusion_3.stable_diffusion_3_text_to_image_preprocessor import (
    StableDiffusion3TextToImagePreprocessor,
)
from keras_hub.src.models.text_to_image import TextToImage


@keras_hub_export("keras_hub.models.StableDiffusion3TextToImage")
class StableDiffusion3TextToImage(TextToImage):
    """An end-to-end Stable Diffusion 3 model for text-to-image generation.

    This model has a `generate()` method, which generates image based on a
    prompt.

    Args:
        backbone: A `keras_hub.models.StableDiffusion3Backbone` instance.
        preprocessor: A
            `keras_hub.models.StableDiffusion3TextToImagePreprocessor` instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    text_to_image = keras_hub.models.StableDiffusion3TextToImage.from_preset(
        "stable_diffusion_3_medium", height=512, width=512
    )
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    # Generate with batched prompts.
    text_to_image.generate(
        ["cute wallpaper art of a cat", "cute wallpaper art of a dog"]
    )

    # Generate with different `num_steps` and `classifier_free_guidance_scale`.
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        num_steps=50,
        classifier_free_guidance_scale=5.0,
    )
    ```
    """

    backbone_cls = StableDiffusion3Backbone
    preprocessor_cls = StableDiffusion3TextToImagePreprocessor

    def __init__(
        self,
        backbone,
        preprocessor,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # TODO: Can we define the model here?
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Currently, `fit` is not supported for "
            "`StableDiffusion3TextToImage`."
        )

    def generate_step(
        self,
        latents,
        token_ids,
        negative_token_ids,
        num_steps,
        classifier_free_guidance_scale,
    ):
        """A compilable generation function for batched of inputs.

        This function represents the inner, XLA-compilable, generation function
        for batched inputs.

        Args:
            latents: A <float>[batch_size, height, width, channels] tensor
                containing the latents to start generation from. Typically, this
                tensor is sampled from the Gaussian distribution.
            token_ids: A <int>[batch_size, num_tokens] tensor containing the
                tokens based on the input prompts.
            negative_token_ids: A <int>[batch_size, num_tokens] tensor
                 containing the negative tokens based on the input prompts.
            num_steps: int. The number of diffusion steps to take.
            classifier_free_guidance_scale: float. The scale defined in
                [Classifier-Free Diffusion Guidance](
                https://arxiv.org/abs/2207.12598). Higher scale encourages to
                generate images that are closely linked to prompts, usually at
                the expense of lower image quality.
        """
        # Encode inputs.
        embeddings = self.backbone.encode_step(token_ids, negative_token_ids)

        # Denoise.
        def body_fun(step, latents):
            return self.backbone.denoise_step(
                latents,
                embeddings,
                step,
                num_steps,
                classifier_free_guidance_scale,
            )

        latents = ops.fori_loop(0, num_steps, body_fun, latents)

        # Decode.
        return self.backbone.decode_step(latents)

    def generate(
        self,
        inputs,
        negative_inputs=None,
        num_steps=28,
        classifier_free_guidance_scale=7.0,
        seed=None,
    ):
        return super().generate(
            inputs,
            negative_inputs=negative_inputs,
            num_steps=num_steps,
            classifier_free_guidance_scale=classifier_free_guidance_scale,
            seed=seed,
        )
