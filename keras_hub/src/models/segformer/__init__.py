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
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_backbone_presets import (
    presets as backbone_presets,
)
from keras_hub.src.models.segformer.segformer_image_segmenter import (
    SegFormerImageSegmenter,
)
from keras_hub.src.models.segformer.segformer_presets import presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(presets, SegFormerImageSegmenter)
register_presets(backbone_presets, SegFormerBackbone)
