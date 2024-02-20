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

import os

import pytest

from keras_nlp.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_nlp.tests.test_case import TestCase


class XLMRobertaTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_xlm_roberta_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=XLMRobertaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_tokenizer_special_tokens(self):
        input_data = ["<s> the quick <mask> brown RANDOM_WORD fox </s> <pad>"]
        tokenizer = XLMRobertaTokenizer(**self.init_kwargs)
        start_token_id = tokenizer.start_token_id
        end_token_id = tokenizer.end_token_id
        pad_token_id = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id
        unk_token_id = tokenizer.unk_token_id
        expected_output = [
            [
                start_token_id,
                4,
                9,
                mask_token_id,
                5,
                unk_token_id,
                7,
                end_token_id,
                pad_token_id,
            ]
        ]
        self.assertAllEqual(tokenizer(input_data), expected_output)

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=XLMRobertaTokenizer,
            preset="xlm_roberta_base_multi",
            input_data=["The quick brown fox."],
            expected_output=[[581, 63773, 119455, 6, 147797, 5]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in XLMRobertaTokenizer.presets:
            self.run_preset_test(
                cls=XLMRobertaTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
