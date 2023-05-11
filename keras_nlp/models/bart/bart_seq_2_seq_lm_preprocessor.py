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

"""BART Seq2Seq LM preprocessor layer."""

import tensorflow as tf
from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.bart.bart_preprocessor import BartPreprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras_nlp_export("keras_nlp.models.BartSeq2SeqLMPreprocessor")
class BartSeq2SeqLMPreprocessor(BartPreprocessor):
    """BART Seq2Seq LM preprocessor.

    This layer is used as preprocessor for seq2seq tasks using the BART model.
    This class subclasses `keras_nlp.models.BartPreprocessor` and keeps most of
    its functionality. It has two changes from the superclass:

     1. Sets the `y` (label) and `sample_weights` fields by shifting the
        decoder input sequence one step towards the left. Both these fields are
        inferred internally, and any passed values will be ignored.
     2. Drops the last token from the decoder input sequence as it does not have
        a successor.

    Args:
        tokenizer: A `keras_nlp.models.BartTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Call arguments:
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Label data. Should always be `None` as the layer generates labels by
            shifting the decoder input sequence one step to the left.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights by shifting the padding mask one step to the
            left.

    Examples:

    Directly calling the layer on data
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Preprocess unbatched inputs.
    inputs = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)

    # Preprocess batched inputs.
    inputs = {
        "encoder_text": ["The fox was sleeping.", "The lion was quiet."],
        "decoder_text": ["The fox was awake.", "The lion was roaring."]
    }
    preprocessor(inputs)

    # Custom vocabulary.
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.BartPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=20,
        decoder_sequence_length=10,
    )
    inputs = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Map single sentences.
    features = {
        "encoder_text": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_text": tf.constant(
            ["The fox was awake.", "The lion was roaring."]
        )
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def __init__(
        self,
        tokenizer,
        encoder_sequence_length,
        decoder_sequence_length,
        truncate="round_robin",
        **kwargs
    ):
        # Since we truncate the last token from `decoder_token_ids`, we need to
        # forcefully set the `decoder_sequence_length` to one greater than the
        # value passed.
        super().__init__(
            tokenizer=tokenizer,
            encoder_sequence_length=encoder_sequence_length,
            decoder_sequence_length=decoder_sequence_length + 1,
            **kwargs
        )

        # Maintain a private copy of the sequence lengths for config purposes.
        self._encoder_sequence_length = encoder_sequence_length
        self._decoder_sequence_length = decoder_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sequence_length": self._encoder_sequence_length,
                "decoder_sequence_length": self._decoder_sequence_length,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`BartSeq2SeqLMPreprocessor` infers `y` and `sample_weight` "
                "from the provided input data, i.e., `x`. However, non-`None`"
                "values have been passed for `y` or `sample_weight` or both. "
                "These values will be ignored."
            )

        x = super().call(x)
        decoder_token_ids = x.pop("decoder_token_ids")
        decoder_padding_mask = x.pop("decoder_padding_mask")

        # The last token does not have a next token. Hence, we truncate it.
        x = {
            **x,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        # Target `y` will be the decoder input sequence shifted one step to the
        # left (i.e., the next token).
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)

    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        encoder_text = x["encoder_text"]
        decoder_text = x["decoder_text"]

        # Tokenize the encoder inputs. We can use the preprocessor directly
        # here. At the same time, the preprocessor will also do pre-checks on
        # the inputs.
        preprocessed_inputs = self(
            {
                "encoder_text": encoder_text,
                "decoder_text": "dummy text",
            }
        )
        encoder_token_ids = preprocessed_inputs[0]["encoder_token_ids"]
        encoder_padding_mask = preprocessed_inputs[0]["encoder_padding_mask"]

        # Tokenize decoder inputs. We cannot use the preprocessor call directly
        # since the `max_length` might be different. Moreover, the
        # `BartSeq2SeqLMPreprocessor` layer handles the training case.
        decoder_text = convert_inputs_to_list_of_tensor_segments(decoder_text)[
            0
        ]
        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, _ = self.decoder_packer(
            decoder_token_ids,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        end_tokens = tf.fill(
            (tf.shape(decoder_token_ids)[0], 1),
            value=self.preprocessor.tokenizer.end_token_id,
        )
        decoder_token_ids = tf.concat([end_tokens, decoder_token_ids], axis=1)
        decoder_padding_mask = decoder_token_ids != self.tokenizer.pad_token_id

        return {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        # Strip any special tokens during detokenization (e.g. the start and
        # end markers). In the future we could make this configurable.
        padding_mask = padding_mask & (token_ids != self.tokenizer.end_token_id)
        token_ids = tf.ragged.boolean_mask(token_ids, padding_mask)
        return self.tokenizer.detokenize(token_ids)
