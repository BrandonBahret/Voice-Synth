"""ElevenLabs provider implementation.

ElevenLabs returns raw PCM when requested with a ``pcm_*`` output format, so the
provider converts bytes directly into ``SynthesizedAudio``. Voice and model
lists are cached by API key because those responses are account-scoped.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import error, parse, request

from voice_conductor.api_cache import APICache
from voice_conductor.api_cache import build_scoped_cache_key
from voice_conductor.api_cache import ELEVENLABS_MODEL_LIST_TTL_SECONDS
from voice_conductor.api_cache import ELEVENLABS_VOICE_LIST_TTL_SECONDS
from voice_conductor.config import Settings
from voice_conductor.exceptions import ConfigurationError, ProviderError
from voice_conductor.providers.base import TTSProvider, settings_from_provider_or_arg
from voice_conductor.types import SynthesizedAudio, VoiceInfo
from voice_conductor.voice_keys import normalize_voice_key


class ElevenLabsProvider(TTSProvider):
    """Text-to-speech backend backed by the ElevenLabs REST API."""

    name = "elevenlabs"
    _base_url = "https://api.elevenlabs.io/v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider_settings = settings.providers.elevenlabs
        self._cache_settings = settings.voice_conductor.cache
        self._api_cache = APICache(self.name, self._cache_settings.api_dir)

    def is_available(self) -> bool:
        """Return whether an ElevenLabs API key is configured."""

        return bool(self._provider_settings.api_key)

    def _require_config(self) -> str:
        if not self.is_available():
            raise ConfigurationError("ElevenLabs requires providers.elevenlabs.api_key.")
        return self._provider_settings.api_key or ""

    def default_voice(self) -> str | None:
        """Return the configured default ElevenLabs voice name or id."""

        return self._provider_settings.default_voice

    def _output_format(self) -> str:
        return self._provider_settings.output_format

    def _voice_settings_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"speed": self._provider_settings.speed}
        if self._provider_settings.stability is not None:
            payload["stability"] = self._provider_settings.stability
        if self._provider_settings.similarity_boost is not None:
            payload["similarity_boost"] = self._provider_settings.similarity_boost
        if self._provider_settings.style is not None:
            payload["style"] = self._provider_settings.style
        if self._provider_settings.speaker_boost is not None:
            payload["use_speaker_boost"] = self._provider_settings.speaker_boost
        return payload

    def cache_settings(self) -> dict[str, Any]:
        """Encode synthesis options that affect phrase-cache audio output."""

        return {
            "model_id": self._provider_settings.model_id,
            "output_format": self._output_format(),
            "language_code": self._provider_settings.language_code,
            "voice_settings": self._voice_settings_payload(),
        }

    def cache_voice_key(self, voice: str | None) -> str:
        """Use ElevenLabs' stable voice id, never the display name, for caching."""

        voice_id, _ = self._resolve_voice_id(voice)
        return normalize_voice_key(self.name, voice_id)

    def _pcm_sample_rate(self, output_format: str) -> int:
        parts = output_format.split("_")
        if len(parts) < 2 or parts[0] != "pcm":
            raise ConfigurationError(
                "ELEVENLABS_OUTPUT_FORMAT must use a PCM format such as 'pcm_24000'."
            )
        try:
            return int(parts[1])
        except ValueError as exc:
            raise ConfigurationError(
                "ELEVENLABS_OUTPUT_FORMAT must include a numeric sample rate, "
                "for example 'pcm_24000'."
            ) from exc

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self._require_config(),
            "Content-Type": "application/json",
            "User-Agent": "VoiceConductor",
        }

    def _api_cache_ttl(self, default_ttl_seconds: int) -> int:
        return (
            self._cache_settings.ttl_seconds
            if self._cache_settings.ttl_seconds is not None
            else default_ttl_seconds
        )

    def _cache_key(self, base_key: str) -> str:
        return build_scoped_cache_key(base_key, self._provider_settings.api_key)

    def list_voices(self=None, settings: Settings | None = None) -> list[VoiceInfo]:
        """Return account voices, using the provider metadata cache."""

        provider = (
            self
            if isinstance(self, ElevenLabsProvider) and settings is None
            else ElevenLabsProvider(settings_from_provider_or_arg(self, settings))
        )
        payload = provider._api_cache.get_or_fetch(
            provider._cache_key("voices:list"),
            provider._fetch_voices_payload,
            ttl_seconds=provider._api_cache_ttl(ELEVENLABS_VOICE_LIST_TTL_SECONDS),
        )
        return [
            VoiceInfo(
                id=item["voice_id"],
                name=item["name"],
                provider=ElevenLabsProvider.name,
                metadata={"category": item.get("category")},
            )
            for item in payload.get("voices", [])
        ]

    def _fetch_voices_payload(self) -> dict[str, Any]:
        req = request.Request(f"{self._base_url}/voices", headers=self._headers())
        try:
            with request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"ElevenLabs voice list request failed: {exc.code} {body}"
            ) from exc
        except error.URLError as exc:
            raise ProviderError(f"ElevenLabs voice list request failed: {exc.reason}") from exc

        if not isinstance(payload, dict):
            raise ProviderError("ElevenLabs voice list request returned an unexpected payload.")
        return payload

    def list_models(self) -> list[dict[str, Any]]:
        """Return ElevenLabs model metadata, using the provider metadata cache."""

        return self._api_cache.get_or_fetch(
            self._cache_key("models:list"),
            self._fetch_models_payload,
            ttl_seconds=self._api_cache_ttl(ELEVENLABS_MODEL_LIST_TTL_SECONDS),
        )

    def _fetch_models_payload(self) -> list[dict[str, Any]]:
        req = request.Request(f"{self._base_url}/models", headers=self._headers())
        try:
            with request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"ElevenLabs model list request failed: {exc.code} {body}"
            ) from exc
        except error.URLError as exc:
            raise ProviderError(f"ElevenLabs model list request failed: {exc.reason}") from exc

        if not isinstance(payload, list):
            raise ProviderError("ElevenLabs model list request returned an unexpected payload.")
        return payload

    def _resolve_voice_id(self, voice: str | None) -> tuple[str, str]:
        requested = voice or self.default_voice()
        voices = self.list_voices()
        for item in voices:
            if item.id == requested or item.name.lower() == requested.lower():
                return item.id, item.name
        if voices:
            fallback = voices[0]
            return fallback.id, fallback.name
        raise ProviderError("ElevenLabs returned no voices for this account.")

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        """Synthesize text with ElevenLabs and return mono normalized PCM audio."""

        voice_id, voice_name = self._resolve_voice_id(voice)
        output_format = self._output_format()
        query = parse.urlencode({"output_format": output_format})
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self._provider_settings.model_id,
            "voice_settings": self._voice_settings_payload(),
        }
        if self._provider_settings.language_code:
            payload["language_code"] = self._provider_settings.language_code
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self._base_url}/text-to-speech/{voice_id}?{query}",
            headers=self._headers(),
            data=body,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                pcm_bytes = response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"ElevenLabs synthesis request failed: {exc.code} {body}"
            ) from exc
        except error.URLError as exc:
            raise ProviderError(f"ElevenLabs synthesis request failed: {exc.reason}") from exc

        return SynthesizedAudio.from_pcm16_bytes(
            pcm_bytes,
            sample_rate=self._pcm_sample_rate(output_format),
            channels=1,
            provider=self.name,
            voice=voice_name,
            text=text,
            metadata={
                "model_id": self._provider_settings.model_id,
                "voice_id": voice_id,
                "normalized_voice_key": normalize_voice_key(self.name, voice_id),
                "output_format": output_format,
                "language_code": self._provider_settings.language_code,
                "voice_settings": self._voice_settings_payload(),
            },
        )
