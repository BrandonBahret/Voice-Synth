"""Kokoro local-provider integration.

Kokoro is optional and imported lazily so voice_conductor can still be installed
without the model dependency. The provider keeps a pipeline instance after first
use and configures Hugging Face tokens just before model loading.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from voice_conductor.config import Settings
from voice_conductor.exceptions import ConfigurationError, DependencyError, ProviderError
from voice_conductor.providers.base import TTSProvider
from voice_conductor.types import SynthesizedAudio, VoiceInfo

_DEFAULT_KOKORO_VOICES = [
    "af_heart",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
]


class KokoroProvider(TTSProvider):
    """Local text-to-speech backend powered by the optional Kokoro package."""

    name = "kokoro"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider_settings = settings.providers.kokoro
        self._pipeline: Any | None = None

    def is_available(self) -> bool:
        """Return whether Kokoro dependencies and credentials are configured."""

        return self._has_kokoro_package() and self._has_huggingface_token()

    def _has_kokoro_package(self) -> bool:
        return importlib.util.find_spec("kokoro") is not None

    def _has_huggingface_token(self) -> bool:
        token = self._provider_settings.hf_token
        return bool(token and token.strip())

    def default_voice(self) -> str | None:
        """Return the configured Kokoro voice id."""

        return self._provider_settings.default_voice

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self._has_kokoro_package():
            raise DependencyError(
                "Kokoro backend requires the optional 'kokoro' package. "
                "Install it with 'pip install \"VoiceConductor[kokoro]\"'."
            )
        if not self._has_huggingface_token():
            raise ConfigurationError("Kokoro requires providers.kokoro.hf_token.")

        self._configure_huggingface_token()

        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code=self._provider_settings.language_code)
        return self._pipeline

    def _configure_huggingface_token(self) -> None:
        token = self._provider_settings.hf_token
        if not token:
            return

        try:
            from huggingface_hub import login
        except ImportError:
            return

        try:
            login(token=token, add_to_git_credential=False, skip_if_logged_in=True)
        except Exception as exc:
            raise ProviderError(f"Failed to authenticate with Hugging Face: {exc}") from exc

    def list_voices(self=None, settings: Settings | None = None) -> list[VoiceInfo]:
        """Return the curated built-in Kokoro voices known to this package."""

        return [
            VoiceInfo(id=voice, name=voice, provider=KokoroProvider.name)
            for voice in _DEFAULT_KOKORO_VOICES
        ]

    def cache_settings(self) -> dict[str, float]:
        """Partition phrase cache entries by Kokoro speed setting."""

        return {"speed": self._provider_settings.speed}

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        """Synthesize text with Kokoro and concatenate streamed audio chunks."""

        pipeline = self._ensure_pipeline()
        voice_name = voice or self.default_voice() or "af_heart"
        try:
            generator = pipeline(text, voice=voice_name, speed=self._provider_settings.speed)
            chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in generator]
        except Exception as exc:
            raise ProviderError(f"Kokoro synthesis failed: {exc}") from exc

        if not chunks:
            raise ProviderError("Kokoro did not return any audio data.")

        merged = np.concatenate(chunks).reshape(-1, 1)
        return SynthesizedAudio(
            samples=merged,
            sample_rate=24000,
            channels=1,
            provider=self.name,
            voice=voice_name,
            text=text,
            metadata={
                "language_code": self._provider_settings.language_code,
                "speed": self._provider_settings.speed,
                "normalized_voice_key": self.cache_voice_key(voice_name),
            },
        )
