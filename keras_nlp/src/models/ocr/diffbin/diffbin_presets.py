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
"""DiffBin model preset configurations."""

from keras_nlp.src.models.backbones.resnet_vd import resnet_vd_backbone_presets

presets_no_weights = {
    "diffbin_resnet50vd": {
        "metadata": {
            "description": "DiffBin with a ResNet50_vd backbone.",
            "params": 25482722,
            "official_name": "DiffBin",
            "path": "diffbin_resnet50vd",
        },
        "config": {
            "backbone": resnet_vd_backbone_presets.backbone_presets["resnet50_vd"],
            "num_classes": 1,
            "input_shape": (640, 640, 3),
        },
    },
}

presets_with_weights = {
    # TODO: Add DiffBin preset with weights
}

basnet_presets = {**presets_no_weights, **presets_with_weights}
