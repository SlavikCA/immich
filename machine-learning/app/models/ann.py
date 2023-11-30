from __future__ import annotations

from pathlib import Path
from typing import Any

from ann.ann import Ann

from ..config import log, settings
from .base import InferenceModel


class AnnModel(InferenceModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(model_name, cache_dir, **model_kwargs)
        tuning_file = Path(settings.cache_folder) / "gpu-tuning.ann"
        with open(tuning_file, 'a'):
            pass
        self.ann = Ann(tuning_level=3, tuning_file=tuning_file.as_posix())
