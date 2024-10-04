import pytest
from keras import ops

from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.tests.test_case import TestCase


class RetinaNetBackboneTest(TestCase):
    def setUp(self):
        resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "image_shape": (None, None, 3),
            "block_type": "bottleneck_block",
            "use_pre_activation": False,
        }
        image_encoder = ResNetBackbone(**resnet_kwargs)

        self.init_kwargs = {
            "image_encoder": image_encoder,
            "min_level": 3,
            "max_level": 4,
        }

        self.input_size = 256
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=RetinaNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "P3": (2, 32, 32, 256),
                "P4": (2, 16, 16, 256),
            },
            expected_pyramid_output_keys=False,
            run_mixed_precision_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=RetinaNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
