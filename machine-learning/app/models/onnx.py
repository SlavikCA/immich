from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import onnxruntime as ort
from typing_extensions import Buffer

from ..config import log, settings
from .base import InferenceModel


class OnnxModel(InferenceModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        inter_op_num_threads: int = settings.model_inter_op_threads,
        intra_op_num_threads: int = settings.model_intra_op_threads,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(model_name, cache_dir, **model_kwargs)
        self.providers = model_kwargs.pop("providers", ["CPUExecutionProvider"])
        #  don't pre-allocate more memory than needed
        self.provider_options = model_kwargs.pop(
            "provider_options", [{"arena_extend_strategy": "kSameAsRequested"}] * len(self.providers)
        )
        log.debug(
            (
                f"Setting '{self.model_name}' execution providers to {self.providers}"
                "in descending order of preference"
            ),
        )
        log.debug(f"Setting execution provider options to {self.provider_options}")
        self.sess_options = PicklableSessionOptions()
        # avoid thread contention between models
        if inter_op_num_threads > 1:
            self.sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        log.debug(f"Setting execution_mode to {self.sess_options.execution_mode.name}")
        log.debug(f"Setting inter_op_num_threads to {inter_op_num_threads}")
        log.debug(f"Setting intra_op_num_threads to {intra_op_num_threads}")
        self.sess_options.inter_op_num_threads = inter_op_num_threads
        self.sess_options.intra_op_num_threads = intra_op_num_threads
        self.sess_options.enable_cpu_mem_arena = False


# HF deep copies configs, so we need to make session options picklable
class PicklableSessionOptions(ort.SessionOptions):  # type: ignore[misc]
    def __getstate__(self) -> bytes:
        return pickle.dumps([(attr, getattr(self, attr)) for attr in dir(self) if not callable(getattr(self, attr))])

    def __setstate__(self, state: Buffer) -> None:
        self.__init__()  # type: ignore[misc]
        attrs: list[tuple[str, Any]] = pickle.loads(state)
        for attr, val in attrs:
            setattr(self, attr, val)
