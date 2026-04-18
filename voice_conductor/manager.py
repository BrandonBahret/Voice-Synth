"""High-level orchestration for synthesis, caching, and playback routing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from voice_conductor.phrase_cache import CacheKey, CacheLookupMode, PhraseCache, canonical_settings_json
from voice_conductor.audio.devices import list_output_devices
from voice_conductor.audio.playback import PlaybackQueue
from voice_conductor.audio.router import RouteConfig, _RoutePlaybackEngine
from voice_conductor.config import Settings, load_settings
from voice_conductor.exceptions import ConfigurationError, ProviderError
from voice_conductor.providers import (
    TTSProvider,
    build_registered_providers,
)
from voice_conductor.types import (
    AudioDevice,
    PlaybackHooks,
    PlaybackResult,
    PlaybackTask,
    SynthesizedAudio,
    VoiceInfo,
)

import logging.config
import logging

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)


class TTSManager:
    """High-level text-to-speech manager for synthesis, caching, and playback.

    ``TTSManager`` is the main public entry point for applications using
    VoiceConductor. It loads configured providers, selects an available provider
    for each request, caches synthesized phrases, and routes generated audio to
    speakers, microphone devices, or both.

    Most callers can construct it with defaults. Tests and host applications
    may inject settings or provider instances to customize behavior.
    """

    def __init__(
        self,
        *,
        settings: Settings | str | Path | None = None,
        providers: dict[str, TTSProvider] | None = None,
    ) -> None:
        """Build a manager from settings and injectable provider adapters.

        The optional constructor arguments are primarily for tests or host apps
        that want to inject providers directly.

        Args:
            config: Optional complete configuration object or config file path.
                This positional argument is a convenience for notebook and script
                usage such as ``TTSManager("config.jsonc")``.
            settings: Optional complete configuration object or config file path.
                When omitted, settings are loaded from the default config path
                and environment.
            providers: Optional provider registry keyed by provider name. When
                omitted, providers are built from the configured provider
                registry.
        """
        settings_source = settings
        if isinstance(settings_source, Settings):
            self.settings = settings_source
        elif isinstance(settings_source, (str, Path)):
            self.settings = Settings.from_file(settings_source)
        elif settings_source is None:
            self.settings = load_settings()
        else:
            raise TypeError(
                "settings must be a Settings instance, config path, or None; "
                f"got {type(settings_source).__name__}."
            )
        
        self._providers = providers or build_registered_providers(self.settings)
        self._resolve_default_provider()
        self._playback_queue = PlaybackQueue()
        resolved_route_config = self.settings.voice_conductor.route_config
        resolved_route_config.resolve_missing_devices()
        self._route_engine = _RoutePlaybackEngine(
            resolved_route_config,
            playback_queue=self._playback_queue,
            route_config_provider=lambda: self.settings.voice_conductor.route_config,
        )
        self._cache = PhraseCache(self.settings.voice_conductor.cache.path)

    def close(self) -> None:
        """Release resources held by the manager."""

        self._cache.close()

    def __enter__(self) -> TTSManager:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def list_providers(self) -> list[str]:
        """Return names of registered providers that are currently available."""

        return [name for name, provider in self._providers.items() if provider.is_available()]

    def list_output_devices(self) -> list[AudioDevice]:
        """Return output-capable host audio devices."""

        return list_output_devices()

    def refresh_audio_devices(
        self,
        *,
        use_system_defaults: bool = True,
        devices: list[AudioDevice] | None = None,
    ) -> RouteConfig:
        """Refresh route device names from the current output device list.

        Set ``use_system_defaults=True`` to discard configured route devices and
        re-resolve the built-in speaker and virtual-mic routes from the current
        OS defaults and available virtual cables.
        """

        if use_system_defaults:
            self.settings.voice_conductor.route_config = RouteConfig.default()
        route_config = self.settings.voice_conductor.route_config
        route_config.resolve_missing_devices(devices=devices)
        self._route_engine.route_config = route_config
        return route_config

    def list_voices(self, provider: str) -> list[VoiceInfo]:
        """Return voices exposed by an available provider."""

        return self._get_provider(provider).list_voices()

    def synthesize(
        self,
        text: str,
        *,
        provider: str | None = None,
        voice: str | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        cache_lookup: CacheLookupMode = "strict",
    ) -> SynthesizedAudio:
        """Compatibility alias for ``synthesize_voice``."""

        return self.synthesize_voice(
            text,
            provider=provider,
            voice=voice,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_lookup=cache_lookup,
        )

    def synthesize_voice(
        self,
        text: str,
        *,
        provider: str | None = None,
        voice: str | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        cache_lookup: CacheLookupMode = "strict",
    ) -> SynthesizedAudio:
        """Synthesize text with cache lookup keyed by provider and resolved voice."""

        selected = self._get_provider(provider)
        resolved_voice = voice or selected.default_voice()

        settings_json = self._cache_settings_json(selected)
        cache_key = CacheKey(
            text=text,
            provider=selected.name,
            voice_key=selected.cache_voice_key(resolved_voice),
            settings_json=settings_json,
        )
        cached = (
            self._cache.get(cache_key, lookup_mode=cache_lookup)
            if use_cache and not refresh_cache
            else None
        )
        if cached is not None:
            return cached

        audio = selected.synthesize(text, voice=resolved_voice)
        audio_voice_key = self._audio_voice_key(selected, audio, resolved_voice)
        if resolved_voice != audio.voice and audio_voice_key != cache_key.voice_key:
            logger.warning(
                "Voice '%s' was not a direct match for any available voice for provider '%s'; using '%s'.",
                resolved_voice,
                selected.name,
                audio.voice,
            )
            resolved_voice = selected.default_voice()

        if use_cache:
            self._cache.set(cache_key, audio)
            if audio_voice_key != cache_key.voice_key:
                self._cache.set(
                    CacheKey(
                        text=text,
                        provider=selected.name,
                        voice_key=audio_voice_key,
                        settings_json=settings_json,
                    ),
                    audio,
                )
        return audio

    def route(
        self,
        audio: SynthesizedAudio,
        routes: str | Iterable[str] = "speakers",
        *,
        background: bool = False,
        hooks: PlaybackHooks | None = None,
    ) -> PlaybackResult | PlaybackTask[PlaybackResult]:
        """Play an already synthesized clip through one or more named routes."""

        return self._route_engine.route(
            audio,
            routes,
            background=background,
            hooks=hooks,
        )

    def speak(
        self,
        text: str,
        routes: str | Iterable[str] = "speakers",
        *,
        provider: str | None = None,
        voice: str | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
        cache_lookup: CacheLookupMode = "strict",
        background: bool = False,
        hooks: PlaybackHooks | None = None,
    ) -> PlaybackResult | PlaybackTask[PlaybackResult]:
        """Synthesize text and route it to speakers, mic cable, or both."""

        if background:
            return self._playback_queue.submit(
                self._speak_sync,
                text,
                routes,
                provider=provider,
                voice=voice,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                cache_lookup=cache_lookup,
                hooks=hooks,
            )
        return self._speak_sync(
            text,
            routes,
            provider=provider,
            voice=voice,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_lookup=cache_lookup,
            hooks=hooks,
        )

    def _speak_sync(
        self,
        text: str,
        routes: str | Iterable[str],
        *,
        provider: str | None,
        voice: str | None,
        use_cache: bool,
        refresh_cache: bool,
        cache_lookup: CacheLookupMode,
        hooks: PlaybackHooks | None,
    ) -> PlaybackResult:
        audio = self.synthesize_voice(
            text,
            provider=provider,
            voice=voice,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_lookup=cache_lookup,
        )
        result = self._route_engine.route(audio, routes, hooks=hooks)
        assert isinstance(result, PlaybackResult)
        return result

    def invalidate_synthesis_cache(
        self,
        *,
        text: str | None = None,
        provider: str | None = None,
        voice: str | None = None,
    ) -> int:
        """Remove phrase cache entries matching the provided filters."""

        voice_key = None
        if provider is not None and voice is not None:
            voice_key = self._require_provider(provider).cache_voice_key(voice)
        elif voice is not None:
            voice_key = voice
        return self._cache.invalidate(text=text, provider=provider, voice_key=voice_key)

    def clear_synthesis_cache(self) -> None:
        """Clear all cached synthesized phrases."""

        self._cache.clear()

    def _get_provider(self, provider: str | None) -> TTSProvider:
        if provider is not None:
            return self._require_provider(provider)

        configured_chain = self._provider_chain()
        for provider_name in configured_chain:
            provider_candidate = self._providers.get(provider_name)
            if provider_candidate is None:
                raise ProviderError(f"Unknown provider: {provider_name}")
            if provider_candidate.is_available():
                return provider_candidate

        if configured_chain:
            raise ConfigurationError(
                "None of the configured providers are available. Update voice_conductor.provider_chain or configure the missing backends."
            )

        raise ConfigurationError(
            "No TTS provider is available. Configure ElevenLabs credentials, install Kokoro, configure Azure Speech, use Windows Speech, or opt into the demo provider."
        )

    def _require_provider(self, provider_name: str) -> TTSProvider:
        provider_name = provider_name.lower()
        if provider_name not in self._providers:
            raise ProviderError(f"Unknown provider: {provider_name}")

        selected = self._providers[provider_name]
        if not selected.is_available():
            raise ConfigurationError(
                f"Provider {provider_name!r} is not available. Configure credentials."
            )
        return selected

    def _resolve_default_provider(self) -> None:
        """Infer an unset default from the first available configured provider."""

        if self.settings.voice_conductor.default_provider:
            return

        for provider_name in self.settings.voice_conductor.provider_chain:
            provider_candidate = self._providers.get(provider_name)
            if provider_candidate is not None and provider_candidate.is_available():
                self.settings.voice_conductor.default_provider = provider_name
                return

    def _provider_chain(self) -> list[str]:
        """Return provider fallback order from explicit settings or built-in defaults."""

        if self.settings.voice_conductor.default_provider:
            return [
                self.settings.voice_conductor.default_provider,
                *(
                    provider
                    for provider in self.settings.voice_conductor.provider_chain
                    if provider != self.settings.voice_conductor.default_provider
                ),
            ]
        return self.settings.voice_conductor.provider_chain

    def _cache_settings_json(self, provider: TTSProvider) -> str:
        """Return deterministic provider option state for phrase-cache settings."""

        return canonical_settings_json(provider.cache_settings())

    def _audio_voice_key(
        self,
        provider: TTSProvider,
        audio: SynthesizedAudio,
        fallback_voice: str | None,
    ) -> str:
        metadata_key = audio.metadata.get("normalized_voice_key")
        if isinstance(metadata_key, str) and metadata_key.strip():
            return metadata_key.strip()
        return provider.cache_voice_key(audio.voice or fallback_voice)
