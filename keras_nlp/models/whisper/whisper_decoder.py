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
"""Whisper decoder block."""


from keras_nlp.layers.transformer_decoder import TransformerDecoder


class WhisperDecoder(TransformerDecoder):
    """Whisper decoder.

    Inherits from `keras_nlp.layers.TransformerDecoder`, and overrides the
    `_build` method to effectively get rid of the bias term in the key projection
    for attention.
    """

    def _build(self, input_shape, has_cross_attention):
        super()._build(input_shape, has_cross_attention)

        # For simplicity of code, we just set the key layer's bias term to zero
        # and make it untrainable.
        self._self_attention_layer._key_dense.bias.assign(0.0)
        self._self_attention_layer._key_dense.bias.trainable = False
        if has_cross_attention:
            self._cross_attention_layer._key_dense.bias.assign(0.0)
            self._cross_attention_layer._key_dense.bias.trainable = False
