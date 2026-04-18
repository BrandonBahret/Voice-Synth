from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_conductor.config import CacheSettings, Settings, VoiceConductorSettings
from voice_conductor.config import _default_cache_path
from voice_conductor.config import load_settings
from voice_conductor.config import settings_from_dict
from voice_conductor.config_registry import register_provider_config, unregister_provider_config
from voice_conductor.exceptions import ConfigurationError, DeviceResolutionError, ProviderError
from voice_conductor.manager import TTSManager
from voice_conductor.audio.router import AudioRoute, RouteConfig
from voice_conductor.providers.base import TTSProvider
from voice_conductor.providers.registry import ProviderRegistry, unregister_provider
from voice_conductor.types import AudioDevice, PlaybackHooks, SynthesizedAudio, VoiceInfo


class FakeProvider(TTSProvider):
    def __init__(self, name: str, *, available: bool, default_voice: str | None = None) -> None:
        self.name = name
        self.available = available
        self._default_voice = default_voice
        self.calls: list[tuple[str, str | None]] = []

    def is_available(self) -> bool:
        return self.available

    def default_voice(self) -> str | None:
        return self._default_voice

    def cache_settings(self) -> str:
        return ""

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        self.calls.append((text, voice))
        return SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider=self.name,
            voice=voice,
            text=text,
        )

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="test", name="Test", provider=self.name)]


class FakeCanonicalVoiceProvider(FakeProvider):
    def __init__(
        self,
        name: str,
        *,
        available: bool,
        default_voice: str | None = None,
        returned_voice: str | None = None,
    ) -> None:
        super().__init__(name, available=available, default_voice=default_voice)
        self._returned_voice = returned_voice

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        self.calls.append((text, voice))
        return SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider=self.name,
            voice=self._returned_voice,
            text=text,
        )


class FakeProviderWithSettings(FakeProvider):
    def __init__(
        self,
        name: str,
        *,
        available: bool,
        cache_settings: str,
        default_voice: str | None = None,
    ) -> None:
        super().__init__(name, available=available, default_voice=default_voice)
        self._cache_settings = cache_settings

    def cache_settings(self) -> str:
        return self._cache_settings


@dataclass(slots=True)
class TypedFakeProviderSettings:
    endpoint: str
    speed: float = 1.0


class TypedConfigProvider(TTSProvider):
    name = "typed_fake"

    def __init__(self, settings: Settings) -> None:
        config = settings.provider_settings(self.name)
        assert isinstance(config, TypedFakeProviderSettings)
        self.endpoint = config.endpoint
        self.speed = config.speed

    def is_available(self) -> bool:
        return True

    def default_voice(self) -> str | None:
        return None

    def cache_settings(self) -> dict[str, str | float]:
        return {"endpoint": self.endpoint, "speed": self.speed}

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        return SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider=self.name,
            voice=voice,
            text=text,
            metadata={"endpoint": self.endpoint, "speed": self.speed},
        )

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="typed", name="Typed", provider=self.name)]


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, audio: SynthesizedAudio, device: AudioDevice) -> None:
        self.calls.append((id(audio), device.id))


class BlockingWriter(RecordingWriter):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.second_started = Event()
        self.release = Event()
        self.second_release = Event()

    def __call__(self, audio: SynthesizedAudio, device: AudioDevice) -> None:
        super().__call__(audio, device)
        if len(self.calls) == 1:
            self.started.set()
            self.release.wait(timeout=5)
            return
        self.second_started.set()
        self.second_release.wait(timeout=5)


class FailingWriter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio: SynthesizedAudio, device: AudioDevice) -> None:
        self.calls += 1
        raise RuntimeError("writer boom")


class FakeSoundDevice:
    def __init__(self) -> None:
        self.default = type("Default", (), {"device": [0, 0]})()

    @staticmethod
    def query_devices():
        return [
            {
                "name": "Speakers",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
            {
                "name": "CABLE Input",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]

    @staticmethod
    def query_hostapis():
        return [{"name": "Windows WASAPI"}]


class FakeSoundDeviceNoVirtual:
    def __init__(self) -> None:
        self.default = type("Default", (), {"device": [0, 0]})()

    @staticmethod
    def query_devices():
        return [
            {
                "name": "Speakers",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]

    @staticmethod
    def query_hostapis():
        return [{"name": "Windows WASAPI"}]


class BrokenSoundDevice:
    default = type("Default", (), {"device": [0, 0]})()

    @staticmethod
    def query_devices():
        raise RuntimeError("device probe failed")

    @staticmethod
    def query_hostapis():
        return [{"name": "Windows WASAPI"}]


def _settings(
    *,
    cache_path: str | None = None,
    provider_chain: list[str] | None = None,
    default_provider: str | None = None,
    speaker_device: str | None = None,
    mic_device: str | None = None,
) -> Settings:
    voice_conductor: dict[str, object] = {}
    if cache_path is not None:
        voice_conductor["cache"] = {"path": cache_path}
    if provider_chain is not None:
        voice_conductor["provider_chain"] = provider_chain
    if default_provider is not None:
        voice_conductor["default_provider"] = default_provider
    devices: dict[str, object] = {}
    if speaker_device is not None:
        devices["speaker"] = speaker_device
    if mic_device is not None:
        devices["mic"] = mic_device
    if devices:
        routes: dict[str, object] = {}
        if speaker_device is not None:
            routes["speakers"] = {"device": speaker_device}
        if mic_device is not None:
            routes["mic"] = {"device": mic_device, "prefer_virtual_cable": True}
        voice_conductor["route_config"] = {"routes": routes}
    return settings_from_dict({"voice_conductor": voice_conductor})


class TTSManagerTests(unittest.TestCase):
    def test_load_settings_reads_local_json_config(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "voice_conductor.config.jsonc"
            config_path.write_text(
                (
                    "// JSONC comments are allowed here.\n"
                    "{"
                    '"voice_conductor":{"provider_chain":["azure","windows"]},'
                    '"providers":{"azure":{"speech_key":"file-key","region":"westus"}}'
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.azure.speech_key, "file-key")
        self.assertEqual(settings.providers.azure.region, "westus")
        self.assertEqual(settings.voice_conductor.provider_chain, ["azure", "windows"])

    def test_load_settings_prefers_jsonc_over_json(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                '{"voice_conductor":{"provider_chain":["windows"]}}',
                encoding="utf-8",
            )
            (temp_path / "voice_conductor.config.jsonc").write_text(
                (
                    "{"
                    '"voice_conductor":{'
                    '"provider_chain":["azure"],'
                    "},"
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.voice_conductor.provider_chain, ["azure"])

    def test_load_settings_rejects_flat_json_config(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                '{"azure_speech_key":"file-key","provider_chain":"azure,windows"}',
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                with self.assertRaisesRegex(ValueError, "nested JSON"):
                    load_settings()
            finally:
                os.chdir(original_cwd)

    def test_cache_file_setting_loads_from_json(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                '{"voice_conductor":{"cache":{"path":"phrases.db"}}}',
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.voice_conductor.cache.path, str(temp_path / "phrases.db"))

    def test_hf_token_loads_from_json(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                '{"providers":{"kokoro":{"hf_token":"hf-test-token"}}}',
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.kokoro.hf_token, "hf-test-token")

    def test_elevenlabs_output_format_loads_from_json(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                (
                    "{"
                    '"providers":{"elevenlabs":{'
                    '"output_format":"pcm_24000",'
                    '"stability":0.3,'
                    '"similarity_boost":0.8,'
                    '"style":0.25,'
                    '"speaker_boost":true'
                    "}}"
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.elevenlabs.output_format, "pcm_24000")
        self.assertEqual(settings.providers.elevenlabs.stability, 0.3)
        self.assertEqual(settings.providers.elevenlabs.similarity_boost, 0.8)
        self.assertEqual(settings.providers.elevenlabs.style, 0.25)
        self.assertTrue(settings.providers.elevenlabs.speaker_boost)

    def test_windows_settings_load_from_json(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                (
                    "{"
                    '"providers":{"windows":{'
                    '"default_voice":"Microsoft Zira Desktop",'
                    '"volume":80'
                    "}}"
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.windows.default_voice, "zira")
        self.assertEqual(settings.providers.windows.volume, 80)

    def test_provider_voice_settings_load_from_json_scalars(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.json").write_text(
                (
                    "{"
                    '"providers":{"azure":{'
                    '"speed":1.15,'
                    '"language_code":"en"'
                    "}}"
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.azure.speed, 1.15)
        self.assertEqual(settings.providers.azure.language_code, "en")

    def test_empty_string_config_values_fall_back_cleanly(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            expected_cache_path = str(temp_path / "voice_conductor_cache.db")
            (temp_path / "voice_conductor.config.json").write_text(
                (
                    "{"
                    '"providers":{"azure":{"default_voice":""},"kokoro":{"language_code":""}},'
                    '"voice_conductor":{"cache":{"path":""},"route_config":{"routes":{"speakers":{"device":""}}},"default_provider":""}'
                    "}"
                ),
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(settings.providers.azure.default_voice, "en-US-AvaNeural")
        self.assertEqual(settings.providers.kokoro.language_code, "a")
        self.assertEqual(settings.voice_conductor.cache.path, expected_cache_path)
        self.assertIsNone(settings.voice_conductor.route_config.get("speakers").device)
        self.assertIsNone(settings.voice_conductor.default_provider)

    def test_cache_key_changes_when_provider_settings_change(self) -> None:
        provider_fast = FakeProviderWithSettings("kokoro", available=True, cache_settings="speed=1.0")
        provider_slow = FakeProviderWithSettings("kokoro", available=True, cache_settings="speed=1.5")
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = str(Path(temp_dir) / "voice_conductor_cache.db")
            first = TTSManager(
                settings=_settings(cache_path=cache_path),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider_fast,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            second = TTSManager(
                settings=_settings(cache_path=cache_path),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider_slow,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            first.synthesize("Need backup")
            second.synthesize("Need backup")
            first.close()
            second.close()

        self.assertEqual(provider_fast.calls, [("Need backup", None)])
        self.assertEqual(provider_slow.calls, [("Need backup", None)])

    def test_relaxed_cache_lookup_can_reuse_different_provider_settings(self) -> None:
        provider_fast = FakeProviderWithSettings("kokoro", available=True, cache_settings="speed=1.0")
        provider_slow = FakeProviderWithSettings("kokoro", available=True, cache_settings="speed=1.5")
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = str(Path(temp_dir) / "voice_conductor_cache.db")
            first = TTSManager(
                settings=_settings(cache_path=cache_path),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider_fast,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            first.synthesize("Need backup")

            second = TTSManager(
                settings=_settings(cache_path=cache_path),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider_slow,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            cached = second.synthesize("Need backup", cache_lookup="relaxed")
            first.close()
            second.close()

        self.assertEqual(cached.text, "Need backup")
        self.assertEqual(provider_slow.calls, [])

    def test_example_config_is_not_loaded_as_live_config(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "voice_conductor.config.example.json").write_text(
                '{"providers":{"azure":{"speech_key":"should-not-load"}}}',
                encoding="utf-8",
            )
            os.chdir(temp_path)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertIsNone(settings.providers.azure.speech_key)

    def test_list_providers_only_returns_available_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=True),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                },
            )

        self.assertEqual(manager.list_providers(), ["azure", "kokoro"])

    def test_provider_registry_is_a_singleton(self) -> None:
        self.assertIs(ProviderRegistry(), ProviderRegistry())

    def test_provider_registry_registers_builtins_once(self) -> None:
        first = ProviderRegistry()
        second = ProviderRegistry()

        self.assertEqual(first.names(), second.names())
        self.assertEqual(first.names().count("elevenlabs"), 1)
        self.assertEqual(first.names().count("kokoro"), 1)
        self.assertEqual(first.names().count("azure"), 1)
        self.assertEqual(first.names().count("windows"), 1)

    def test_provider_registry_can_build_provider_by_name(self) -> None:
        settings = Settings()
        provider = ProviderRegistry().build_provider("windows", settings)

        self.assertEqual(provider.name, "windows")

    def test_provider_registry_build_provider_raises_for_unknown_name(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderRegistry().build_provider("nope", Settings())

    def test_provider_registry_builds_manager_providers_by_name(self) -> None:
        provider = FakeProvider("fake", available=True)
        registry = ProviderRegistry()
        registry.register("fake", lambda settings: provider)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = TTSManager(
                    settings=_settings(
                        cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                        provider_chain=["fake"],
                    ),
                )

                audio = manager.synthesize("hello")
                manager.close()
        finally:
            unregister_provider("fake")

        self.assertEqual(audio.provider, "fake")
        self.assertEqual(provider.calls, [("hello", None)])

    def test_registered_provider_can_consume_typed_custom_config(self) -> None:
        registry = ProviderRegistry()
        registry.register("typed_fake", TypedConfigProvider)
        register_provider_config("typed_fake", TypedFakeProviderSettings)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = TTSManager(
                    settings=settings_from_dict(
                        {
                            "voice_conductor": {
                                "cache": {"path": str(Path(temp_dir) / "voice_conductor_cache.db")},
                                "provider_chain": ["typed_fake"],
                            },
                            "providers": {
                                "typed_fake": {
                                    "endpoint": "http://localhost:7000",
                                    "speed": 1.4,
                                }
                            },
                        }
                    ),
                )

                audio = manager.synthesize("hello")
                manager.close()
        finally:
            unregister_provider_config("typed_fake")
            unregister_provider("typed_fake")

        provider = manager._providers["typed_fake"]
        self.assertIsInstance(provider, TypedConfigProvider)
        self.assertEqual(provider.endpoint, "http://localhost:7000")
        self.assertEqual(provider.speed, 1.4)
        self.assertEqual(audio.metadata["endpoint"], "http://localhost:7000")
        self.assertEqual(audio.metadata["speed"], 1.4)

    def test_default_provider_prefers_elevenlabs(self) -> None:
        azure = FakeProvider("azure", available=False)
        kokoro = FakeProvider("kokoro", available=False)
        elevenlabs = FakeProvider("elevenlabs", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": azure,
                    "elevenlabs": elevenlabs,
                    "kokoro": kokoro,
                },
            )

            audio = manager.synthesize("hello")
            manager.close()

        self.assertEqual(audio.provider, "elevenlabs")
        self.assertEqual(elevenlabs.calls, [("hello", None)])
        self.assertEqual(kokoro.calls, [])
        self.assertEqual(azure.calls, [])

    def test_default_provider_falls_back_to_kokoro(self) -> None:
        azure = FakeProvider("azure", available=True)
        kokoro = FakeProvider("kokoro", available=True)
        elevenlabs = FakeProvider("elevenlabs", available=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": azure,
                    "elevenlabs": elevenlabs,
                    "kokoro": kokoro,
                },
            )

            audio = manager.synthesize("hello")
            manager.close()

        self.assertEqual(audio.provider, "kokoro")
        self.assertEqual(manager.settings.voice_conductor.default_provider, "kokoro")
        self.assertEqual(elevenlabs.calls, [])
        self.assertEqual(kokoro.calls, [("hello", None)])
        self.assertEqual(azure.calls, [])

    def test_configured_default_provider_is_not_changed(self) -> None:
        azure = FakeProvider("azure", available=False)
        elevenlabs = FakeProvider("elevenlabs", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    provider_chain=["elevenlabs", "azure"],
                    default_provider="azure",
                ),
                providers={
                    "azure": azure,
                    "elevenlabs": elevenlabs,
                },
            )

            audio = manager.synthesize("hello")
            manager.close()

        self.assertEqual(manager.settings.voice_conductor.default_provider, "azure")
        self.assertEqual(audio.provider, "elevenlabs")
        self.assertEqual(elevenlabs.calls, [("hello", None)])
        self.assertEqual(azure.calls, [])

    def test_configured_default_falls_back_when_unavailable(self) -> None:
        azure = FakeProvider("azure", available=False)
        kokoro = FakeProvider("kokoro", available=False)
        elevenlabs = FakeProvider("elevenlabs", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    provider_chain=["azure", "elevenlabs", "windows"],
                ),
                providers={
                    "azure": azure,
                    "elevenlabs": elevenlabs,
                    "kokoro": kokoro,
                },
            )

            audio = manager.synthesize("hello")
            manager.close()

        self.assertEqual(audio.provider, "elevenlabs")
        self.assertEqual(manager.settings.voice_conductor.default_provider, "elevenlabs")
        self.assertEqual(elevenlabs.calls, [("hello", None)])
        self.assertEqual(kokoro.calls, [])
        self.assertEqual(azure.calls, [])

    def test_provider_chain_uses_windows_last_resort(self) -> None:
        azure = FakeProvider("azure", available=False)
        kokoro = FakeProvider("kokoro", available=False)
        elevenlabs = FakeProvider("elevenlabs", available=False)
        windows = FakeProvider("windows", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    provider_chain=["azure", "elevenlabs", "kokoro", "windows"],
                ),
                providers={
                    "azure": azure,
                    "elevenlabs": elevenlabs,
                    "kokoro": kokoro,
                    "windows": windows,
                },
            )

            audio = manager.synthesize("hello")
            manager.close()

        self.assertEqual(audio.provider, "windows")
        self.assertEqual(windows.calls, [("hello", None)])

    def test_cache_uses_provider_default_voice(self) -> None:
        provider = FakeProvider("kokoro", available=True, default_voice="af_heart")
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            first = manager.synthesize("Need backup")
            second = manager.synthesize("need backup")
            manager.close()

        self.assertEqual(provider.calls, [("Need backup", "af_heart")])
        self.assertEqual(first.voice, "af_heart")
        self.assertEqual(second.text, "Need backup")

    def test_phrase_cache_reuses_generation_for_same_lowercase_text(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            first = manager.synthesize("Need backup")
            second = manager.synthesize("need backup")
            manager.close()

        self.assertEqual(provider.calls, [("Need backup", None)])
        self.assertEqual(first.text, "Need backup")
        self.assertEqual(second.text, "Need backup")

    def test_synthesize_can_bypass_phrase_cache(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            manager.synthesize("Need backup", use_cache=False)
            manager.synthesize("Need backup", use_cache=False)
            manager.close()

        self.assertEqual(provider.calls, [("Need backup", None), ("Need backup", None)])

    def test_synthesize_can_refresh_phrase_cache(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            manager.synthesize("Need backup")
            manager.synthesize("Need backup", refresh_cache=True)
            manager.synthesize("Need backup")
            manager.close()

        self.assertEqual(provider.calls, [("Need backup", None), ("Need backup", None)])

    def test_manager_exposes_phrase_cache_invalidation(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )

            manager.synthesize("Need backup")
            removed = manager.invalidate_synthesis_cache(text="Need backup")
            manager.synthesize("Need backup")
            manager.close()

        self.assertEqual(removed, 1)
        self.assertEqual(provider.calls, [("Need backup", None), ("Need backup", None)])

    def test_phrase_cache_reuses_generation_when_provider_returns_different_voice_label(self) -> None:
        provider = FakeCanonicalVoiceProvider(
            "elevenlabs",
            available=True,
            default_voice="voice-id-123",
            returned_voice="Rachel",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": provider,
                    "kokoro": FakeProvider("kokoro", available=False),
                    "windows": FakeProvider("windows", available=False),
                },
            )

            first = manager.synthesize("Team, rotate to B.")
            second = manager.synthesize("Team, rotate to B.")
            manager.close()

        self.assertEqual(provider.calls, [("Team, rotate to B.", "voice-id-123")])
        self.assertEqual(first.voice, "Rachel")
        self.assertEqual(second.voice, "Rachel")

    def test_route_plays_existing_audio_to_named_routes(self) -> None:
        writer = RecordingWriter()
        audio = SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                    mic_device="CABLE Input",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            result = manager.route(audio, ["speakers", "mic"])

        self.assertEqual(result.routes, ["speakers", "mic"])
        self.assertEqual(set(result.devices), {"speakers", "mic"})
        self.assertEqual({call[1] for call in writer.calls}, {0, 1})
        self.assertEqual({call[0] for call in writer.calls}, {id(audio)})

    def test_manager_discovers_default_route_devices_on_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        routes = manager._route_engine.route_config.routes
        self.assertEqual(routes["speakers"].device, "Speakers")
        self.assertEqual(routes["mic"].device, "CABLE Output")

    def test_manager_uses_route_config_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=Settings(
                    voice_conductor=VoiceConductorSettings(
                        cache=CacheSettings(path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                        route_config=RouteConfig(
                            routes={
                                "stream": AudioRoute(
                                    name="stream",
                                    device="CABLE Input",
                                    prefer_virtual_cable=True,
                                )
                            }
                        ),
                    )
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        routes = manager._route_engine.route_config.routes
        self.assertEqual(set(routes), {"stream"})
        self.assertEqual(routes["stream"].device, "CABLE Input")

    def test_manager_route_engine_reflects_reassigned_settings_route_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                    mic_device="CABLE Input",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        replacement = RouteConfig.default(speaker="Headphones", mic="VB-Cable")
        manager.settings.voice_conductor.route_config = replacement

        self.assertIs(manager._route_engine.route_config, replacement)
        self.assertEqual(manager._route_engine.route_config.get("speakers").device, "Headphones")

    def test_refresh_audio_devices_can_reset_to_system_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Old Speakers",
                    mic_device="Old Cable",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        refreshed = manager.refresh_audio_devices(use_system_defaults=True, devices=[
            AudioDevice(
                id=0,
                name="Headphones",
                hostapi="WASAPI",
                max_output_channels=2,
                default_samplerate=48000,
                is_default=True,
            ),
            AudioDevice(
                id=1,
                name="CABLE Input",
                hostapi="WASAPI",
                max_output_channels=2,
                default_samplerate=48000,
                is_virtual_cable=True,
            ),
        ])

        self.assertIs(manager.settings.voice_conductor.route_config, refreshed)
        self.assertIs(manager._route_engine.route_config, refreshed)
        self.assertEqual(refreshed.get("speakers").device, "Headphones")
        self.assertEqual(refreshed.get("mic").device, "CABLE Output")

    def test_manager_construction_keeps_synthesize_only_workflows_when_discovery_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=BrokenSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        routes = manager._route_engine.route_config.routes
        self.assertIsNone(routes["speakers"].device)
        self.assertIsNone(routes["mic"].device)

    def test_manager_route_fails_loudly_when_discovered_config_is_not_viable(self) -> None:
        writer = RecordingWriter()
        audio = SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDeviceNoVirtual(),
        ):
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            with self.assertRaisesRegex(DeviceResolutionError, "No virtual cable output device"):
                manager.route(audio, "mic")

        self.assertEqual(writer.calls, [])

    def test_speak_synthesizes_once_and_routes_once(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        writer = RecordingWriter()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                    mic_device="CABLE Input",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            result = manager.speak("push now", ["speakers", "mic"])
            manager.close()

        self.assertEqual(provider.calls, [("push now", None)])
        self.assertEqual(result.routes, ["speakers", "mic"])
        self.assertEqual(set(result.devices), {"speakers", "mic"})
        self.assertEqual({call[1] for call in writer.calls}, {0, 1})

    def test_speak_passes_hooks_through_to_route_playback(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        writer = RecordingWriter()
        ready_events = []
        complete_events = []
        hooks = PlaybackHooks(
            on_audio_ready=lambda event: ready_events.append(event),
            on_playback_complete=lambda event: complete_events.append(event),
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            result = manager.speak("push now", hooks=hooks)
            manager.close()

        self.assertEqual(len(ready_events), 1)
        self.assertEqual(ready_events[0].routes, ["speakers"])
        self.assertEqual(len(complete_events), 1)
        self.assertIs(complete_events[0].result, result)
        self.assertEqual(writer.calls[0][1], 0)

    def test_speak_with_background_flag_synthesizes_and_routes_in_background(self) -> None:
        provider = FakeProvider("kokoro", available=True)
        writer = BlockingWriter()
        hooks = PlaybackHooks()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": provider,
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            task = manager.speak("push now", background=True, hooks=hooks)
            self.assertTrue(writer.started.wait(timeout=1))
            self.assertFalse(task.done())
            writer.release.set()
            result = task.result(timeout=2)
            manager.close()

        self.assertEqual(provider.calls, [("push now", None)])
        self.assertEqual(result.routes, ["speakers"])
        self.assertEqual(writer.calls[0][1], 0)

    def test_background_routes_share_fifo_queue(self) -> None:
        writer = BlockingWriter()
        audio = SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                    mic_device="CABLE Input",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            first = manager.route(audio, "speakers", background=True)
            second = manager.route(audio, "mic", background=True)
            self.assertTrue(writer.started.wait(timeout=1))
            self.assertFalse(writer.second_started.wait(timeout=0.1))
            writer.release.set()
            first.result(timeout=2)
            self.assertTrue(writer.second_started.wait(timeout=1))
            self.assertFalse(second.done())
            writer.second_release.set()
            second_result = second.result(timeout=2)

        self.assertEqual(second_result.routes, ["mic"])
        self.assertEqual([call[1] for call in writer.calls], [0, 1])

    def test_route_background_exception_surfaces_on_task(self) -> None:
        writer = FailingWriter()
        audio = SynthesizedAudio(
            samples=np.zeros((8, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider="test",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "voice_conductor.audio.devices._load_sounddevice",
            return_value=FakeSoundDevice(),
        ):
            manager = TTSManager(
                settings=_settings(
                    cache_path=str(Path(temp_dir) / "voice_conductor_cache.db"),
                    speaker_device="Speakers",
                ),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=True),
                    "windows": FakeProvider("windows", available=False),
                },
            )
            manager._route_engine._audio_writer = writer

            task = manager.route(audio, background=True)

        with self.assertRaisesRegex(RuntimeError, "writer boom"):
            task.result(timeout=2)
        self.assertIsInstance(task.exception(timeout=0), RuntimeError)

    def test_unknown_provider_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=True),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=False),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        with self.assertRaises(ProviderError):
            manager.synthesize("hello", provider="nope")

    def test_no_available_providers_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TTSManager(
                settings=_settings(cache_path=str(Path(temp_dir) / "voice_conductor_cache.db")),
                providers={
                    "azure": FakeProvider("azure", available=False),
                    "elevenlabs": FakeProvider("elevenlabs", available=False),
                    "kokoro": FakeProvider("kokoro", available=False),
                    "windows": FakeProvider("windows", available=False),
                },
            )

        with self.assertRaises(ConfigurationError):
            manager.synthesize("hello")
