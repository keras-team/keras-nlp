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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.backend import random


@keras_nlp_export("keras_nlp.samplers.Sampler")
class Sampler:
    """Base sampler class.

    This base class can be extended to implement different auto-regressive
    sampling methods. Subclasses can either:

    - Override the `get_next_token()` method, which computes the next token
      based on a probability distribution over all possible vocab entries.
    - Override `start()`, `has_next()`, `next()`, and `finish()` to implement
      more complex sampling routines.

    Args:
        temperature: float. optional. Used to control the
            randomness of the sampling. The higher the temperature, the
            more diverse the samples. Defaults to `1.0`.
    """

    def __init__(
        self,
        temperature=1.0,
    ):
        self.temperature = temperature
        self.generated_padding_id = 2
        self._seed_generators = []

    def __setattr__(self, name, value):
        # We could update to the `Tracker` class from keras-core if our needs
        # become more advanced (e.g. list assignment, nested trackables). For
        # now, we only track `SeedGenerator` instances directly on the sampler.
        if isinstance(value, random.SeedGenerator):
            self._seed_generators.append(value)
        return super().__setattr__(name, value)

    @property
    def variables(self):
        variables = []
        for sg in self._seed_generators:
            variables.append(sg.state)
        return variables

    def start(self, data):
        return data

    def has_next(
        self,
        data,
        index,
        end_token_id=None,
    ):
        # Check if we have reached `max_length`.
        token_ids, padding_mask = data["token_ids"], data["padding_mask"]
        _, max_length = ops.shape(token_ids)
        length_remaining = ops.less(index, max_length - 1)
        if end_token_id is None:
            return length_remaining
        # Check if all sequences have generated a *new* stop token.
        end_tokens = ops.equal(token_ids, end_token_id)
        new_locations = ops.equal(padding_mask, self.generated_padding_id)
        new_end_tokens = ops.logical_and(end_tokens, new_locations)
        sequence_alive = ops.logical_not(ops.any(new_end_tokens, axis=-1))
        any_alive = ops.any(sequence_alive)
        return ops.logical_and(length_remaining, any_alive)

    def next(
        self,
        data,
        index,
        logits,
    ):
        next_index = index + 1
        token_ids, padding_mask = data["token_ids"], data["padding_mask"]
        # Compute the next token.
        probabilities = self.compute_probabilities(logits)
        next_token = self.get_next_token(probabilities)
        # Compute updated padding column.
        padding_column = padding_mask[:, next_index][:, None]
        next_padding = ops.ones_like(padding_column) * self.generated_padding_id
        next_padding = ops.where(padding_column, padding_column, next_padding)
        # Compute updated token id column.
        token_column = token_ids[:, next_index][:, None]
        next_token = ops.cast(next_token, token_ids.dtype)[:, None]
        next_token = ops.where(padding_column, token_column, next_token)
        # Update both in our data dictionary.
        start = [0, next_index]
        return {
            **data,
            "token_ids": ops.slice_update(token_ids, start, next_token),
            "padding_mask": ops.slice_update(padding_mask, start, next_padding),
        }

    def finish(self, data):
        return data

    def compute_probabilities(self, logits):
        """Compute token probabilities from logits.

        This will always be done in full precision, regardless of dtype, and
        scale by `temperature`.
        """
        logits_dtype = logits.dtype
        logits = ops.cast(logits, "float32")
        probs = keras.activations.softmax(logits / self.temperature)
        return ops.cast(probs, logits_dtype)

    def get_next_token(self, probabilities):
        """Get the next token.

        Args:
            probabilities: a Tensor, the probability distribution for next
                token over all vocab tokens.

        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"temperature": self.temperature}
