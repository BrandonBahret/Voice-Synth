"""Abstract provider contract for text-to-speech backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from voice_conductor.config import Settings, load_settings
from voice_conductor.types import SynthesizedAudio, VoiceInfo
from voice_conductor.voice_keys import normalize_voice_key


def settings_from_provider_or_arg(
    provider_or_settings: Any | None = None,
    settings: Settings | None = None,
) -> Settings:
    """Resolve settings for provider methods that also support class calls."""

    if settings is not None:
        return settings
    if isinstance(provider_or_settings, Settings):
        return provider_or_settings
    provider_settings = getattr(provider_or_settings, "settings", None)
    if isinstance(provider_settings, Settings):
        return provider_settings
    return load_settings()


class TTSProvider(ABC):
    """Interface every synthesis backend must implement."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether dependencies and credentials are present enough to use."""

        raise NotImplementedError

    def default_voice(self) -> str | None:
        """Return the provider's configured fallback voice, if any."""

        return None

    def cache_settings(self) -> str | dict[str, Any] | None:
        """Return synthesis option state that should split phrase-cache entries."""

        return ""

    def cache_voice_key(self, voice: str | None) -> str:
        """Return the normalized voice identity used by phrase caching."""

        return normalize_voice_key(self.name, voice)

    @abstractmethod
    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        """Synthesize ``text`` into normalized ``SynthesizedAudio``."""

        raise NotImplementedError

    @abstractmethod
    def list_voices(self, settings: Settings | None = None) -> list[VoiceInfo]:
        """Return the voices currently exposed by the provider."""

        raise NotImplementedError
