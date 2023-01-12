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

from tensorflow import keras

from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class Preprocessor(keras.layers.Layer):
    """Base class for model preprocessors."""

    def __init_subclass__(cls, **kwargs):
        # We use the __init_subclass__ hook to properly format the from_preset
        # docstring on subclasses.
        if cls.from_preset.__func__.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Preprocessor.from_preset.__doc__
            preset_names = list(cls.presets.keys())
            format_docstring(
                preprocessor_name=cls.__name__,
                example_preset_name=preset_names[0],
                preset_names='", "'.join(preset_names),
            )(cls.from_preset.__func__)
        super().__init_subclass__(**kwargs)

    @property
    def tokenizer(self):
        """The tokenizer used to tokenize strings."""
        return self._tokenizer

    def get_config(self):
        config = super().get_config()
        config["tokenizer"] = keras.layers.serialize(self.tokenizer)
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    @classproperty
    def tokenizer_cls(cls):
        return None

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{preprocessor_name}} from preset architecture.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load preprocessor from preset
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
        )
        preprocessor("The quick brown fox jumped.")

        # Override sequence_length
        preprocessor = keras_nlp.models.{{preprocessor_name}}.from_preset(
            "{{example_preset_name}}",
            sequence_length=64
        )
        preprocessor("The quick brown fox jumped.")
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

        tokenizer = cls.tokenizer_cls.from_preset(preset)

        metadata = cls.presets[preset]
        # For task model presets, the backbone config is nested.
        if "backbone" in metadata["config"]:
            backbone_config = metadata["config"]["backbone"]["config"]
        else:
            backbone_config = metadata["config"]

        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        sequence_length = kwargs.pop("sequence_length", None)
        max_sequence_length = backbone_config["max_sequence_length"]
        if sequence_length is not None:
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {sequence_length}."
                )
        else:
            sequence_length = max_sequence_length

        return cls(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            **kwargs,
        )
