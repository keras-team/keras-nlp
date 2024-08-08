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
"""ResNetVd model preset configurations."""

backbone_presets_no_weights = {
    "resnet18_vd": {
        "metadata": {
            "description": (
                "ResNet_vd model with 18 layers."
            ),
            "params": 11186112,
            "official_name": "ResNetVd",
            "path": "resnet_vd",
        },
    #    "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet18/2",
    },
    "resnet34_vd": {
        "metadata": {
            "description": (
                "ResNet_vd model with 34 layers."
            ),
            "params": 21301696,
            "official_name": "ResNetVd",
            "path": "resnet_vd",
        },
    #    "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet34/2",
    },
    "resnet50_vd": {
        "metadata": {
            "description": (
                "ResNet_vd model with 50 layers."
            ),
            "params": 23561152,
            "official_name": "ResNetVd",
            "path": "resnet_vd",
        },
    #    "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet50/2",
    },
    "resnet101_vd": {
        "metadata": {
            "description": (
                "ResNet_vd model with 101 layers."
            ),
            "params": 42605504,
            "official_name": "ResNetVd",
            "path": "resnet_vd",
        },
    #    "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet101/2",
    },
    "resnet152_vd": {
        "metadata": {
            "description": (
                "ResNet_vd model with 152 layers."
            ),
            "params": 58295232,
            "official_name": "ResNetVd",
            "path": "resnet_vd",
        },
    #    "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet152/2",
    },
}

backbone_presets_with_weights = {

}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
