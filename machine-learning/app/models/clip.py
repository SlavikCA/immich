import json
import os.path as path
from abc import abstractmethod, abstractproperty
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer

from app.config import clean_name, log
from app.models.transforms import crop, get_pil_resampling, normalize, resize, to_numpy
from app.schemas import ModelType, ndarray_f32, ndarray_i32, ndarray_i64

from .ann import AnnModel
from .base import InferenceModel
from .onnx import OnnxModel


class BaseCLIPEncoder(InferenceModel):
    _model_type = ModelType.CLIP

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        mode: Literal["text", "vision"] | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.mode = mode
        super().__init__(model_name, cache_dir, **model_kwargs)

    def _load(self) -> None:
        if self.mode == "text" or self.mode is None:
            log.debug(f"Loading clip text model '{self.model_name}'")
            self._load_text()

        if self.mode == "vision" or self.mode is None:
            log.debug(f"Loading clip vision model '{self.model_name}'")
            self._load_vision()

    @abstractmethod
    def _load_text(self) -> None:
        pass

    @abstractmethod
    def _load_vision(self) -> None:
        pass

    def _predict(self, image_or_text: Image.Image | str) -> ndarray_f32:
        if isinstance(image_or_text, bytes):
            image_or_text = Image.open(BytesIO(image_or_text))

        match image_or_text:
            case Image.Image():
                if self.mode == "text":
                    raise TypeError("Cannot encode image as text-only model")
                return self._predict_vision(self.transform(image_or_text))
            case str():
                if self.mode == "vision":
                    raise TypeError("Cannot encode text as vision-only model")
                return self._predict_text(self.tokenize(image_or_text))
            case _:
                raise TypeError(f"Expected Image or str, but got: {type(image_or_text)}")

    @abstractmethod
    def _predict_vision(self, input: dict[str, ndarray_f32]) -> ndarray_f32:
        pass

    @abstractmethod
    def _predict_text(self, input: dict[str, ndarray_i32]) -> ndarray_f32:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> dict[str, ndarray_i32]:
        pass

    @abstractmethod
    def transform(self, image: Image.Image) -> dict[str, ndarray_f32]:
        pass

    @property
    def textual_dir(self) -> Path:
        return self.cache_dir / "textual"

    @property
    def visual_dir(self) -> Path:
        return self.cache_dir / "visual"

    @property
    def model_cfg_path(self) -> Path:
        return self.cache_dir / "config.json"

    @abstractproperty
    def textual_path(self) -> Path:
        pass

    @abstractproperty
    def visual_path(self) -> Path:
        pass

    @property
    def preprocess_cfg_path(self) -> Path:
        return self.visual_dir / "preprocess_cfg.json"

    @property
    def cached(self) -> bool:
        return self.textual_path.is_file() and self.visual_path.is_file()


class BaseOnnxCLIPTextEncoder(BaseCLIPEncoder, OnnxModel):
    @property
    def textual_path(self) -> Path:
        return self.textual_dir / "model.onnx"

    def _load_text(self) -> None:
        super()._load_text()
        self.text_model = ort.InferenceSession(
            self.textual_path.as_posix(),
            sess_options=self.sess_options,
            providers=self.providers,
            provider_options=self.provider_options,
        )

    def _predict_text(self, input: dict[str, ndarray_i32]) -> ndarray_f32:
        return self.text_model.run(None, input)[0][0]


class BaseOnnxCLIPVisionEncoder(BaseCLIPEncoder, OnnxModel):
    @property
    def visual_path(self) -> Path:
        return self.visual_dir / "model.onnx"

    def _load_vision(self) -> None:
        super()._load_vision()
        self.vision_model = ort.InferenceSession(
            self.visual_path.as_posix(),
            sess_options=self.sess_options,
            providers=self.providers,
            provider_options=self.provider_options,
        )

    def _predict_vision(self, input: dict[str, ndarray_f32]) -> ndarray_f32:
        return self.vision_model.run(None, input)[0][0]


class BaseAnnCLIPVisionEncoder(BaseCLIPEncoder, AnnModel):
    @property
    def visual_path(self) -> Path:
        return self.visual_dir / "model.armnn"

    def _load_vision(self) -> None:
        super()._load_vision()
        model_file = self.visual_path.as_posix()
        cache_file = model_file + ".anncache"
        save = False
        if not path.exists(cache_file):
            save = True
            with open(cache_file, mode="a"):
                pass

        self.vision_model = self.ann.load(model_file, save_cached_network=save, cached_network_path=cache_file)

    def _predict_vision(self, input: dict[str, ndarray_f32]) -> ndarray_f32:
        img = next(iter(input.values()))
        img = np.moveaxis(img, 1, 3)  # Ann expects input as NHWC
        return self.ann.embed(self.vision_model, img)[0]


class BaseOpenCLIPEncoder(BaseCLIPEncoder):
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        mode: Literal["text", "vision"] | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(clean_name(model_name), cache_dir, mode, **model_kwargs)

    def _load_text(self) -> None:
        super()._load_text()
        self.tokenizer = AutoTokenizer.from_pretrained(self.textual_dir)
        self.sequence_length = self.model_cfg["text_cfg"]["context_length"]

    def _load_vision(self) -> None:
        super()._load_vision()
        self.size = (
            self.preprocess_cfg["size"][0] if type(self.preprocess_cfg["size"]) == list else self.preprocess_cfg["size"]
        )
        self.resampling = get_pil_resampling(self.preprocess_cfg["interpolation"])
        self.mean = np.array(self.preprocess_cfg["mean"], dtype=np.float32)
        self.std = np.array(self.preprocess_cfg["std"], dtype=np.float32)

    def tokenize(self, text: str) -> dict[str, ndarray_i32]:
        input_ids: ndarray_i64 = self.tokenizer(
            text,
            max_length=self.sequence_length,
            return_tensors="np",
            return_attention_mask=False,
            padding="max_length",
            truncation=True,
        ).input_ids
        return {"text": input_ids.astype(np.int32)}

    def transform(self, image: Image.Image) -> dict[str, ndarray_f32]:
        image = resize(image, self.size)
        image = crop(image, self.size)
        image_np = to_numpy(image)
        image_np = normalize(image_np, self.mean, self.std)
        return {"image": np.expand_dims(image_np.transpose(2, 0, 1), 0)}

    @cached_property
    def model_cfg(self) -> dict[str, Any]:
        model_cfg: dict[str, Any] = json.load(self.model_cfg_path.open())
        return model_cfg

    @cached_property
    def preprocess_cfg(self) -> dict[str, Any]:
        preprocess_cfg: dict[str, Any] = json.load(self.preprocess_cfg_path.open())
        return preprocess_cfg


class OpenCLIPEncoderOnnx(BaseOnnxCLIPTextEncoder, BaseOnnxCLIPVisionEncoder, BaseOpenCLIPEncoder):
    pass


class OpenCLIPEncoderAnn(BaseOnnxCLIPTextEncoder, BaseAnnCLIPVisionEncoder, BaseOpenCLIPEncoder):
    pass


class MCLIPEncoderOnnx(OpenCLIPEncoderOnnx):
    def tokenize(self, text: str) -> dict[str, ndarray_i32]:
        tokens: dict[str, ndarray_i64] = self.tokenizer(text, return_tensors="np")
        return {k: v.astype(np.int32) for k, v in tokens.items()}
