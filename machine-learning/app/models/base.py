from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Any

from huggingface_hub import snapshot_download

from ..config import get_cache_dir, get_hf_model_name, log
from ..schemas import ModelType


class InferenceModel(ABC):
    _model_type: ModelType

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.loaded = False
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def download(self) -> None:
        if not self.cached:
            log.info(
                (f"Downloading {self.model_type.replace('-', ' ')} model '{self.model_name}'." "This may take a while.")
            )
            self._download()

    def load(self) -> None:
        if self.loaded:
            return
        self.download()
        log.info(f"Loading {self.model_type.replace('-', ' ')} model '{self.model_name}'")
        self._load()
        self.loaded = True

    def predict(self, inputs: Any, **model_kwargs: Any) -> Any:
        self.load()
        if model_kwargs:
            self.configure(**model_kwargs)
        return self._predict(inputs)

    @abstractmethod
    def _predict(self, inputs: Any) -> Any:
        ...

    def configure(self, **model_kwargs: Any) -> None:
        pass

    def _download(self) -> None:
        snapshot_download(
            get_hf_model_name(self.model_name),
            cache_dir=self.cache_dir,
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )

    @abstractmethod
    def _load(self) -> None:
        ...

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir if self._cache_dir is not None else get_cache_dir(self.model_name, self.model_type)

    @cache_dir.setter
    def cache_dir(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    @property
    def cached(self) -> bool:
        return self.cache_dir.exists() and any(self.cache_dir.iterdir())

    @classmethod
    def from_model_type(cls, model_type: ModelType, model_name: str, **model_kwargs: Any) -> InferenceModel:
        subclasses = {subclass._model_type: subclass for subclass in cls.__subclasses__()}
        if model_type not in subclasses:
            raise ValueError(f"Unsupported model type: {model_type}")

        return subclasses[model_type](model_name, **model_kwargs)

    def clear_cache(self) -> None:
        if not self.cache_dir.exists():
            log.warn(
                f"Attempted to clear cache for model '{self.model_name}' but cache directory does not exist.",
            )
            return
        if not rmtree.avoids_symlink_attacks:
            raise RuntimeError("Attempted to clear cache, but rmtree is not safe on this platform.")

        if self.cache_dir.is_dir():
            log.info(f"Cleared cache directory for model '{self.model_name}'.")
            rmtree(self.cache_dir)
        else:
            log.warn(
                (
                    f"Encountered file instead of directory at cache path "
                    f"for '{self.model_name}'. Removing file and replacing with a directory."
                ),
            )
            self.cache_dir.unlink()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
