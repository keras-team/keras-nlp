# Copyright 2022 The KerasNLP Authors
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
"""Greedy Sampler."""

import tensorflow as tf

from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import base_sampler_keyword_args
from keras_nlp.samplers.sampler import call_keyword_docstring
from keras_nlp.samplers.sampler import sample_keyword_docstring


class GreedySampler(Sampler):
    """Greedy Sampler class.

    This sampler is implemented on greedy search, i.e., always picking up the
    token of the largest probability as the next token.

    Args:
        {{base_sampler_keyword_args}}

    Call Args:
        {{call_keyword_args}}
    """

    def __init__(
        self,
        end_token_id=None,
        pad_token_id=0,
        jit_compile=True,
    ):
        super().__init__(end_token_id, pad_token_id, jit_compile)

    def sample(self, token_probability_fn, prompt, mask, num_steps):
        """Sampler's logic implementation.

        Args:
            {{call_keyword_docstring}}
        """
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        max_length = tf.cast(max_length, num_steps.dtype)
        length = max_length - num_steps

        def one_step(length, prompt, mask):

            probs = token_probability_fn(prompt, mask)
            next_token_prob = tf.gather(
                probs, tf.repeat(length - 1, batch_size), axis=1, batch_dims=1
            )
            next_token = tf.cast(
                tf.argmax(next_token_prob, axis=-1), dtype=prompt.dtype
            )
            next_token = tf.where(
                mask[:, length], prompt[:, length], next_token
            )
            mask = tf.tensor_scatter_nd_update(
                tensor=mask,
                indices=tf.stack(
                    (
                        tf.cast(tf.range(batch_size), dtype=length.dtype),
                        tf.repeat(length, batch_size),
                    ),
                    axis=1,
                ),
                updates=tf.repeat(True, batch_size),
            )

            # Append the next token to current sequence.
            prompt = tf.tensor_scatter_nd_update(
                tensor=prompt,
                indices=tf.stack(
                    (
                        tf.cast(tf.range(batch_size), dtype=length.dtype),
                        tf.repeat(length, batch_size),
                    ),
                    axis=1,
                ),
                updates=next_token,
            )

            length = tf.add(length, 1)
            return (length, prompt, mask)

        # Run a while loop till text of length `max_length` has been generated.
        length, prompt, mask = tf.while_loop(
            cond=lambda length, prompt, mask: tf.less(length, max_length),
            body=one_step,
            loop_vars=(length, prompt, mask),
        )
        return prompt


GreedySampler.__doc__ = GreedySampler.__doc__.replace(
    "{{base_sampler_keyword_args}}", base_sampler_keyword_args
)
GreedySampler.__doc__ = GreedySampler.__doc__.replace(
    "{{call_keyword_docstring}}", call_keyword_docstring
)
GreedySampler.sample.__doc__ = GreedySampler.sample.__doc__.replace(
    "{{sample_keyword_docstring}}", sample_keyword_docstring
)
