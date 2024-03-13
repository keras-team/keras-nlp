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

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.bert.bert_presets import classifier_presets
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer
from keras_nlp.utils.python_utils import classproperty

PRESET_NAMES = ", ".join(list(backbone_presets) + list(classifier_presets))


@keras_nlp_export("keras_nlp.models.BertTokenizer")
class BertTokenizer(WordPieceTokenizer):
    """A BERT tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.WordPieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by BERT
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a BERT preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.BertPreprocessor` layer for input packing.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        lowercase: If `True`, the input text will be first lowered before
            tokenization.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to False.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_nlp.models.BertTokenizer.from_preset(
        "bert_base_en_uncased",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_nlp.models.BertTokenizer(vocabulary=vocab)
    tokenizer("The quick brown fox jumped.")
    ```
    """

    def __init__(
        self,
        vocabulary=None,
        lowercase=False,
        special_tokens_in_strings=False,
        **kwargs,
    ):
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            special_tokens=[
                self.cls_token,
                self.sep_token,
                self.pad_token,
                self.mask_token,
            ],
            special_tokens_in_strings=special_tokens_in_strings,
            **kwargs,
        )

    def set_vocabulary(self, vocabulary):
        super().set_vocabulary(vocabulary)

        if vocabulary is not None:
            self.cls_token_id = self.token_to_id(self.cls_token)
            self.sep_token_id = self.token_to_id(self.sep_token)
            self.pad_token_id = self.token_to_id(self.pad_token)
            self.mask_token_id = self.token_to_id(self.mask_token)
        else:
            self.cls_token_id = None
            self.sep_token_id = None
            self.pad_token_id = None
            self.mask_token_id = None

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets, **classifier_presets})

    def get_config(self):
        config = super().get_config()
        # In the constructor, we pass the list of special tokens to the
        # `special_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["special_tokens"]
        return config
