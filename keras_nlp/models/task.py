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
"""Base class for Task models."""

import os

from tensorflow import keras

from keras_nlp.utils.keras_utils import print_msg
from keras_nlp.utils.keras_utils import print_row
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class Task(PipelineModel):
    """Base class for Task models."""

    def __init__(self, *args, **kwargs):
        self._backbone = None
        self._preprocessor = None
        super().__init__(*args, **kwargs)

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @property
    def backbone(self):
        """A `keras.Model` instance providing the backbone submodel."""
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    @property
    def preprocessor(self):
        """A `keras.layers.Layer` instance used to preprocess inputs."""
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self.include_preprocessing = value is not None
        self._preprocessor = value

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Task constructors.
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "preprocessor": keras.layers.serialize(self.preprocessor),
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.layers.deserialize(
                config["preprocessor"]
            )
        return cls(**config)

    @classproperty
    def backbone_cls(cls):
        return None

    @classproperty
    def preprocessor_cls(cls):
        return None

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate {{model_task_name}} model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = {{model_task_name}}.from_preset("{{example_preset_name}}")

        # Load randomly initialized model from preset architecture
        model = {{model_task_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """
        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = cls.preprocessor_cls.from_preset(preset)

        # Check if preset is backbone-only model
        if preset in cls.backbone_cls.presets:
            backbone = cls.backbone_cls.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        metadata = cls.presets[preset]
        config = metadata["config"]
        if (
            "activation" in config
            and "activation" in kwargs
            and config["activation"] != kwargs["activation"]
        ):
            config_activation = config["activation"]
            kwargs_activation = kwargs["activation"]
            raise ValueError(
                f"`activation` cannot be changed for preset '{preset}' which "
                f"was trained with `activation` of `{config_activation}`. "
                "Please do not specify the argument. Received `activation` of "
                f"`{kwargs_activation}`"
            )
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define `from_preset`, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Task.from_preset.__doc__
            format_docstring(
                model_task_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)

    @property
    def layers(self):
        # Remove preprocessor from layers so it does not show up in the summary.
        layers = super().layers
        if self.preprocessor and self.preprocessor in layers:
            layers.remove(self.preprocessor)
        return layers

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        **kwargs,
    ):
        """Override `model.summary()` to show a preprocessor if set."""
        # Defaults are copied from core Keras; we should try to stay in sync.
        line_length = line_length or 98
        positions = positions or [0.33, 0.55, 0.67, 1.0]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        if print_fn is None:
            print_fn = print_msg

        if self.preprocessor:
            column_names = ["Tokenizer (type)", "Vocab #"]
            tokenizer = self.preprocessor.tokenizer
            column_values = [
                f"{tokenizer.name} ({tokenizer.__class__.__name__})",
                f"{tokenizer.vocabulary_size()}",
            ]

            print_fn(f'Preprocessor: "{self.preprocessor.name}"')
            print_fn("_" * line_length)
            print_row(column_names, positions[1:3], print_fn)
            print_fn("=" * line_length)
            print_row(column_values, positions[1:3], print_fn)
            print_fn("_" * line_length)
            print_fn(" " * line_length)

        super().summary(
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            **kwargs,
        )
