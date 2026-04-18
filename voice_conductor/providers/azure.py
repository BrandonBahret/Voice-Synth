"""Azure Speech provider implementation.

Azure synthesis is sent as SSML and requested as WAV/RIFF PCM so the shared
``SynthesizedAudio`` parser can normalize it. Voice list cache entries are
scoped by both region and key because available voices can vary by account and
endpoint.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape

from voice_conductor.api_cache import APICache
from voice_conductor.api_cache import build_scoped_cache_key
from voice_conductor.api_cache import AZURE_VOICE_LIST_TTL_SECONDS
from voice_conductor.config import Settings
from voice_conductor.exceptions import ConfigurationError, ProviderError
from voice_conductor.providers.base import TTSProvider, settings_from_provider_or_arg
from voice_conductor.types import SynthesizedAudio, VoiceInfo


class AzureSpeechProvider(TTSProvider):
    """Text-to-speech backend backed by Azure Cognitive Services Speech."""

    name = "azure"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider_settings = settings.providers.azure
        self._cache_settings = settings.voice_conductor.cache
        self._api_cache = APICache(self.name, self._cache_settings.api_dir)

    def is_available(self) -> bool:
        """Return whether both Azure speech key and region are configured."""

        return bool(self._provider_settings.speech_key and self._provider_settings.region)

    def _require_config(self) -> tuple[str, str]:
        if not self.is_available():
            raise ConfigurationError(
                "Azure Speech requires providers.azure.speech_key and providers.azure.region."
            )
        return self._provider_settings.speech_key or "", self._provider_settings.region or ""

    def default_voice(self) -> str | None:
        """Return the configured Azure neural voice."""

        return self._provider_settings.default_voice

    @property
    def _tts_url(self) -> str:
        _, region = self._require_config()
        return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    @property
    def _voices_url(self) -> str:
        _, region = self._require_config()
        return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"

    def _api_cache_ttl(self, default_ttl_seconds: int) -> int:
        return (
            self._cache_settings.ttl_seconds
            if self._cache_settings.ttl_seconds is not None
            else default_ttl_seconds
        )

    def _cache_key(self, base_key: str) -> str:
        return build_scoped_cache_key(
            base_key,
            self._provider_settings.region,
            self._provider_settings.speech_key,
        )

    def _prosody_rate(self) -> float:
        return min(max(self._provider_settings.speed, 0.5), 2.0)

    def cache_settings(self) -> dict[str, Any]:
        """Encode Azure synthesis options that alter phrase-cache audio."""

        return {
            "speed": self._prosody_rate(),
            "language_code": self._provider_settings.language_code,
        }

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        """Synthesize text with Azure SSML and return normalized WAV audio."""

        key, _ = self._require_config()
        voice_name = voice or self.default_voice()
        language = self._provider_settings.language_code or "en-US"
        escaped_text = escape(text)
        ssml = (
            "<speak version='1.0' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='https://www.w3.org/2001/mstts' "
            f"xml:lang='{escape(language)}'>"
            f"<voice name='{escape(voice_name)}'>"
            f"<prosody rate='{self._prosody_rate():.2f}'>{escaped_text}</prosody>"
            "</voice>"
            "</speak>"
        ).encode("utf-8")

        req = request.Request(
            self._tts_url,
            data=ssml,
            method="POST",
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
                "User-Agent": "VoiceConductor",
            },
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                wav_bytes = response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Azure Speech request failed: {exc.code} {body}") from exc
        except error.URLError as exc:
            raise ProviderError(f"Azure Speech request failed: {exc.reason}") from exc

        return SynthesizedAudio.from_wav_bytes(
            wav_bytes,
            provider=self.name,
            voice=voice_name,
            text=text,
            metadata={
                "format": "riff-16khz-16bit-mono-pcm",
                "language_code": self._provider_settings.language_code,
                "speed": self._prosody_rate(),
            },
        )

    def list_voices(self=None, settings: Settings | None = None) -> list[VoiceInfo]:
        """Return Azure voices for the configured region, using metadata cache."""

        provider = (
            self
            if isinstance(self, AzureSpeechProvider) and settings is None
            else AzureSpeechProvider(settings_from_provider_or_arg(self, settings))
        )
        payload = provider._api_cache.get_or_fetch(
            provider._cache_key("voices:list"),
            provider._fetch_voices_payload,
            ttl_seconds=provider._api_cache_ttl(AZURE_VOICE_LIST_TTL_SECONDS),
        )
        return [
            VoiceInfo(
                id=item["ShortName"],
                name=item["DisplayName"],
                provider=AzureSpeechProvider.name,
                language=item.get("Locale"),
                metadata={"local_name": item.get("LocalName")},
            )
            for item in payload
        ]

    def _fetch_voices_payload(self) -> list[dict[str, Any]]:
        key, _ = self._require_config()
        req = request.Request(
            self._voices_url,
            headers={"Ocp-Apim-Subscription-Key": key, "User-Agent": "VoiceConductor"},
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Azure voice list request failed: {exc.code} {body}") from exc
        except error.URLError as exc:
            raise ProviderError(f"Azure voice list request failed: {exc.reason}") from exc

        if not isinstance(payload, list):
            raise ProviderError("Azure voice list request returned an unexpected payload.")
        return payload
