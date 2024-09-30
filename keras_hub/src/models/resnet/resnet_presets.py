"""ResNet preset configurations."""

backbone_presets = {
    "resnet_18_imagenet": {
        "metadata": {
            "description": (
                "18-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 11186112,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_18_imagenet/3",
    },
    "resnet_50_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 23561152,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_50_imagenet/3",
    },
    "resnet_101_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 42605504,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_101_imagenet/3",
    },
    "resnet_152_imagenet": {
        "metadata": {
            "description": (
                "152-layer ResNet model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 58295232,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv1/keras/resnet_152_imagenet/3",
    },
    "resnet_v2_50_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetV2 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "params": 23561152,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_50_imagenet/3",
    },
    "resnet_v2_101_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNetV2 model pre-trained on the ImageNet 1k "
                "dataset at a 224x224 resolution."
            ),
            "params": 42605504,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/2110.00476",
        },
        "kaggle_handle": "kaggle://kerashub/resnetv2/keras/resnet_v2_101_imagenet/3",
    },
    "resnet_vd_18_imagenet": {
        "metadata": {
            "description": (
                "18-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 11722824,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet18_vd",
    },
    "resnet_vd_34_imagenet": {
        "metadata": {
            "description": (
                "34-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 21838408,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet34_vd",
    },
    "resnet_vd_50_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 25629512,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet50_vd",
    },
    "resnet_vd_50_ssld_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution with knowledge distillation."
            ),
            "params": 25629512,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet50_vd_ssld",
    },
    "resnet_vd_50_ssld_v2_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution with knowledge distillation and AutoAugment."
            ),
            "params": 25629512,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet50_vd_ssld_v2",
    },
    "resnet_vd_50_ssld_v2_fix_imagenet": {
        "metadata": {
            "description": (
                "50-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution with knowledge distillation, AutoAugment and "
                "additional fine-tuning of the classification head."
            ),
            "params": 25629512,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/fix_resnet50_vd_ssld_v2",
    },
    "resnet_vd_101_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 44673864,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet101_vd",
    },
    "resnet_vd_101_ssld_imagenet": {
        "metadata": {
            "description": (
                "101-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution with knowledge distillation."
            ),
            "params": 44673864,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet101_vd_ssld",
    },
    "resnet_vd_152_imagenet": {
        "metadata": {
            "description": (
                "152-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 60363592,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet152_vd",
    },
    "resnet_vd_200_imagenet": {
        "metadata": {
            "description": (
                "200-layer ResNetVD (ResNet with bag of tricks) model "
                "pre-trained on the ImageNet 1k dataset at a 224x224 "
                "resolution."
            ),
            "params": 74933064,
            "official_name": "ResNet",
            "path": "resnet",
            "model_card": "https://arxiv.org/abs/1812.01187",
        },
        "kaggle_handle": "kaggle://kerashub/resnetvd/keras/resnet200_vd",
    },
}
