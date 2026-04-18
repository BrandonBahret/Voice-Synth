from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from voice_conductor.config import (
    AzureSettings,
    CacheSettings,
    DemoProviderSettings,
    ElevenLabsSettings,
    KokoroSettings,
    ProviderSettings,
    Settings,
    VoiceConductorSettings,
    WindowsSettings,
)
from voice_conductor.config import load_settings
from voice_conductor.config import _format_voice_comment_label
from voice_conductor.config import _loads_config_text
from voice_conductor.config import settings_from_dict
from voice_conductor.config import settings_to_dict
from voice_conductor.config_registry import register_provider_config, unregister_provider_config
from voice_conductor.audio.router import AudioRoute, RouteConfig
from voice_conductor.types import VoiceInfo


@dataclass(slots=True)
class CustomProviderSettings:
    endpoint: str
    api_token: str | None = None
    speed: float = 1.0


class ConfigAPITests(unittest.TestCase):
    def test_format_voice_comment_label_compacts_verbose_display_names(self) -> None:
        voices = [
            VoiceInfo(
                id="CwhRBWXzGAHq8TQ4Fs17",
                name="Roger - Laid-Back, Casual, Resonant",
                provider="elevenlabs",
            ),
            VoiceInfo(
                id="EXAVITQu4vr4xnSDxMaL",
                name="Sarah - Mature, Reassuring, Confident",
                provider="elevenlabs",
            ),
        ]

        label = _format_voice_comment_label("elevenlabs", voices)

        self.assertIn("CwhRBWXzGAHq8TQ4Fs17 (Roger)", label)
        self.assertIn("EXAVITQu4vr4xnSDxMaL (Sarah)", label)
        self.assertNotIn("Laid-Back", label)
        self.assertTrue(label.startswith("elevenlabs:<name> ["))

    def test_format_voice_comment_label_omits_redundant_names_for_readable_ids(self) -> None:
        demo_label = _format_voice_comment_label(
            "demo",
            [
                VoiceInfo(id="demo:animalese", name="Animalese-ish", provider="demo"),
                VoiceInfo(id="demo:robot", name="Robot Radio", provider="demo"),
            ],
        )
        windows_label = _format_voice_comment_label(
            "windows",
            [
                VoiceInfo(
                    id="Microsoft David Desktop",
                    name="Microsoft David Desktop",
                    provider="windows",
                ),
                VoiceInfo(
                    id="Microsoft Zira Desktop",
                    name="Microsoft Zira Desktop",
                    provider="windows",
                ),
            ],
        )

        self.assertEqual(demo_label, "demo:<name> [animalese, robot]")
        self.assertEqual(windows_label, "windows:<name> [david, zira]")

    def test_settings_fields_are_documented_for_consumers(self) -> None:
        setting_types = (
            CacheSettings,
            VoiceConductorSettings,
            ElevenLabsSettings,
            AzureSettings,
            KokoroSettings,
            WindowsSettings,
            DemoProviderSettings,
            ProviderSettings,
            Settings,
        )

        for setting_type in setting_types:
            with self.subTest(setting_type=setting_type.__name__):
                for item in fields(setting_type):
                    self.assertIsInstance(item.metadata.get("doc"), str)
                    self.assertTrue(item.metadata["doc"].strip())

    def test_settings_defaults_include_provider_chain(self) -> None:
        self.assertEqual(
            Settings().voice_conductor.provider_chain,
            ["elevenlabs", "kokoro", "azure", "windows"],
        )

    def test_cache_settings_root_derives_default_cache_locations(self) -> None:
        settings = Settings(
            voice_conductor=VoiceConductorSettings(cache=CacheSettings(root="runtime-cache"))
        )

        self.assertEqual(settings.voice_conductor.cache.root, "runtime-cache")
        self.assertEqual(
            Path(settings.voice_conductor.cache.path),
            Path("runtime-cache") / "voice_conductor_cache.db",
        )
        self.assertEqual(
            Path(settings.voice_conductor.cache.api_dir),
            Path("runtime-cache") / "api-caches",
        )

    def test_cache_settings_accepts_path_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            settings = CacheSettings(root=root)

        self.assertEqual(settings.root, str(root))
        self.assertEqual(settings.path, str(root / "voice_conductor_cache.db"))
        self.assertEqual(settings.api_dir, str(root / "api-caches"))

    def test_cache_settings_root_does_not_override_explicit_cache_locations(self) -> None:
        settings = Settings(
            voice_conductor=VoiceConductorSettings(
                cache=CacheSettings(
                    path="phrases.db",
                    api_dir="provider-cache",
                    root="runtime-cache",
                )
            )
        )

        self.assertEqual(settings.voice_conductor.cache.root, "runtime-cache")
        self.assertEqual(settings.voice_conductor.cache.path, "phrases.db")
        self.assertEqual(settings.voice_conductor.cache.api_dir, "provider-cache")

    def test_direct_settings_construction_derives_default_route_config_from_devices(self) -> None:
        settings = Settings(
            voice_conductor=VoiceConductorSettings(
                route_config=RouteConfig(
                    routes={
                        "speakers": AudioRoute("speakers", device="Speakers"),
                        "mic": AudioRoute("mic", device="CABLE Input"),
                    }
                )
            )
        )

        self.assertEqual(settings.voice_conductor.route_config.get("speakers").device, "Speakers")
        self.assertEqual(settings.voice_conductor.route_config.get("mic").device, "CABLE Input")

    def test_load_settings_defaults_include_provider_chain(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                settings = load_settings()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            settings.voice_conductor.provider_chain,
            ["elevenlabs", "kokoro", "azure", "windows"],
        )

    def test_settings_from_file_loads_existing_jsonc_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo-voice_conductor.config.jsonc"
            path.write_text(
                """
                {
                  // JSONC should be accepted here.
                  "voice_conductor": {
                    "provider_chain": ["demo"],
                  },
                  "providers": {
                    "demo": {
                      "default_voice": "robot",
                    },
                  },
                }
                """,
                encoding="utf-8",
            )

            settings = Settings.from_file(path)

        self.assertEqual(settings.voice_conductor.provider_chain, ["demo"])
        self.assertEqual(settings.providers.demo.default_voice, "robot")

    def test_settings_from_file_resolves_relative_cache_paths_from_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "voice_conductor.config.jsonc"
            path.parent.mkdir()
            path.write_text(
                '{"voice_conductor":{"cache":{"root":".runtime"}}}',
                encoding="utf-8",
            )

            settings = Settings.from_file(path)

        self.assertEqual(settings.voice_conductor.cache.root, str(path.parent / ".runtime"))
        self.assertEqual(
            settings.voice_conductor.cache.path,
            str(path.parent / ".runtime" / "voice_conductor_cache.db"),
        )
        self.assertEqual(
            settings.voice_conductor.cache.api_dir,
            str(path.parent / ".runtime" / "api-caches"),
        )

    def test_settings_from_file_writes_defaults_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo" / "demo-voice_conductor.config.jsonc"

            settings = Settings.from_file(path)
            self.assertTrue(path.exists())
            payload = _loads_config_text(path.read_text(encoding="utf-8"), path)

        self.assertEqual(settings.voice_conductor.provider_chain, ["elevenlabs", "kokoro", "azure", "windows"])
        self.assertIn("voice_conductor", payload)
        self.assertIn("providers", payload)

    def test_settings_from_file_keeps_relative_cache_paths_stable_across_save_loops(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            os.chdir(temp_path)
            try:
                path = Path("demo_files") / "demo-voice_conductor.config.jsonc"

                settings = Settings.from_file(path)
                settings.save_settings(path)
                settings = Settings.from_file(path)
                settings.save_settings(path)

                saved_path = temp_path / path
                payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)
                cache = payload["voice_conductor"]["cache"]
                reloaded = Settings.from_file(path)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(cache["root"], ".")
        self.assertEqual(cache["path"], "voice_conductor_cache.db")
        self.assertEqual(cache["api_dir"], "api-caches")
        self.assertEqual(
            Path(reloaded.voice_conductor.cache.path),
            temp_path / "demo_files" / "voice_conductor_cache.db",
        )
        self.assertEqual(
            Path(reloaded.voice_conductor.cache.api_dir),
            temp_path / "demo_files" / "api-caches",
        )

    def test_settings_from_file_collapses_legacy_cwd_relative_cache_paths(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            os.chdir(temp_path)
            try:
                path = Path("demo_files") / "demo-voice_conductor.config.jsonc"
                path.parent.mkdir()
                path.write_text(
                    (
                        '{"voice_conductor":{"cache":{'
                        '"root":"demo_files/demo_files",'
                        '"path":"demo_files/demo_files/voice_conductor_cache.db",'
                        '"api_dir":"demo_files/demo_files/api-caches"'
                        "}}}"
                    ),
                    encoding="utf-8",
                )

                settings = Settings.from_file(path)
                settings.save_settings(path)

                saved_path = temp_path / path
                payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)
                cache = payload["voice_conductor"]["cache"]
            finally:
                os.chdir(original_cwd)

        self.assertEqual(cache["root"], ".")
        self.assertEqual(cache["path"], "voice_conductor_cache.db")
        self.assertEqual(cache["api_dir"], "api-caches")

    def test_settings_from_dict_constructs_nested_config(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "provider_chain": ["kokoro", "windows"],
                    "route_config": {
                        "routes": {
                            "speakers": {"device": "Speakers"},
                            "mic": {"device": "CABLE Input"},
                        }
                    },
                    "cache": {"path": "phrases.db", "api_dir": "api-cache", "ttl_seconds": 30},
                },
                "providers": {
                    "elevenlabs": {
                        "api_key": "secret",
                        "default_voice": "Rachel",
                        "speed": 1.2,
                        "speaker_boost": True,
                    },
                    "windows": {"default_voice": "Zira", "volume": 85},
                },
            }
        )

        self.assertEqual(settings.voice_conductor.provider_chain, ["kokoro", "windows"])
        self.assertEqual(settings.voice_conductor.route_config.get("speakers").device, "Speakers")
        self.assertEqual(settings.voice_conductor.route_config.get("mic").device, "CABLE Input")
        self.assertEqual(settings.voice_conductor.cache.path, "phrases.db")
        self.assertEqual(settings.voice_conductor.cache.api_dir, "api-cache")
        self.assertEqual(settings.voice_conductor.cache.ttl_seconds, 30)
        self.assertEqual(settings.providers.elevenlabs.speed, 1.2)
        self.assertTrue(settings.providers.elevenlabs.speaker_boost)
        self.assertEqual(settings.providers.elevenlabs.api_key, "secret")
        self.assertEqual(settings.providers.windows.volume, 85)
        self.assertEqual(settings.voice_conductor.route_config.get("speakers").device, "Speakers")
        self.assertEqual(settings.voice_conductor.route_config.get("mic").device, "CABLE Input")

    def test_settings_from_dict_derives_cache_defaults_from_configured_root(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "cache": {"root": "runtime-cache", "ttl_seconds": 30},
                }
            }
        )

        self.assertEqual(settings.voice_conductor.cache.root, "runtime-cache")
        self.assertEqual(
            Path(settings.voice_conductor.cache.path),
            Path("runtime-cache") / "voice_conductor_cache.db",
        )
        self.assertEqual(
            Path(settings.voice_conductor.cache.api_dir),
            Path("runtime-cache") / "api-caches",
        )
        self.assertEqual(settings.voice_conductor.cache.ttl_seconds, 30)

    def test_settings_from_dict_normalizes_windows_device_prefixed_cache_paths(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "cache": {
                        "root": r"\\?\m:\cache",
                        "path": r"\\?\m:\cache\voice_conductor_cache.db",
                        "api_dir": r"\\?\UNC\server\share\voice-conductor\api-caches",
                    }
                }
            }
        )

        self.assertEqual(settings.voice_conductor.cache.root, r"M:\cache")
        self.assertEqual(settings.voice_conductor.cache.path, r"M:\cache\voice_conductor_cache.db")
        self.assertEqual(settings.voice_conductor.cache.api_dir, r"\\server\share\voice-conductor\api-caches")

    def test_settings_from_dict_accepts_explicit_route_config(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "route_config": {
                        "routes": {
                            "stream": {
                                "device": "CABLE Input",
                                "prefer_virtual_cable": True,
                            }
                        }
                    }
                }
            }
        )

        route = settings.voice_conductor.route_config.get("stream")

        self.assertEqual(route.device, "CABLE Input")
        self.assertTrue(route.prefer_virtual_cable)

    def test_settings_to_dict_serializes_secrets(self) -> None:
        settings = Settings(
            providers=ProviderSettings(
                azure=AzureSettings(speech_key="azure-secret"),
                kokoro=KokoroSettings(hf_token="hf-secret"),
            )
        )
        settings.providers.elevenlabs.api_key = "eleven-secret"

        payload = settings_to_dict(settings)

        self.assertEqual(payload["providers"]["elevenlabs"]["api_key"], "eleven-secret")
        self.assertEqual(payload["providers"]["azure"]["speech_key"], "azure-secret")
        self.assertEqual(payload["providers"]["kokoro"]["hf_token"], "hf-secret")
        self.assertEqual(settings.provider_config("elevenlabs")["api_key"], "eleven-secret")

    def test_serialized_settings_omit_removed_bloat_keys(self) -> None:
        payload = settings_to_dict(Settings())

        self.assertNotIn("defaults", payload["voice_conductor"])
        self.assertNotIn("voice_gender", payload["providers"]["windows"])
        self.assertNotIn("voice_age", payload["providers"]["windows"])
        self.assertNotIn("voice_culture", payload["providers"]["windows"])
        self.assertNotIn("voice_alternate", payload["providers"]["windows"])

    def test_settings_to_dict_serializes_route_config(self) -> None:
        settings = Settings(
            voice_conductor=VoiceConductorSettings(
                route_config=RouteConfig(
                    routes={
                        "stream": AudioRoute(
                            name="stream",
                            device="CABLE Input",
                            prefer_virtual_cable=True,
                        )
                    }
                )
            )
        )

        payload = settings_to_dict(settings)

        route_payload = payload["voice_conductor"]["route_config"]["routes"]["stream"]
        self.assertEqual(route_payload["device"], "CABLE Input")
        self.assertTrue(route_payload["prefer_virtual_cable"])

    def test_settings_to_dict_fills_default_devices_from_resolved_routes(self) -> None:
        settings = Settings(
            voice_conductor=VoiceConductorSettings(
                route_config=RouteConfig(
                    routes={
                        "speakers": AudioRoute("speakers", device="Speakers (High Definition Audio"),
                        "mic": AudioRoute("mic", device="VoiceMeeter Aux Output (VB-Audio"),
                    }
                )
            )
        )

        payload = settings_to_dict(settings)

        self.assertEqual(
            payload["voice_conductor"]["route_config"]["routes"]["speakers"]["device"],
            "Speakers (High Definition Audio",
        )
        self.assertEqual(
            payload["voice_conductor"]["route_config"]["routes"]["mic"]["device"],
            "VoiceMeeter Aux Output (VB-Audio",
        )

    def test_save_settings_round_trips_secrets(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "route_config": {"routes": {"speakers": {"device": "Speakers"}}},
                    "provider_chain": ["elevenlabs", "windows"],
                },
                "providers": {"elevenlabs": {"api_key": "eleven-secret"}},
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            saved_path = settings.save_settings(path)
            payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)
            reloaded = settings_from_dict(payload)

        self.assertEqual(payload["providers"]["elevenlabs"]["api_key"], "eleven-secret")
        self.assertEqual(reloaded.providers.elevenlabs.api_key, "eleven-secret")
        self.assertEqual(reloaded.voice_conductor.route_config.get("speakers").device, "Speakers")
        self.assertEqual(reloaded.voice_conductor.provider_chain, ["elevenlabs", "windows"])

    def test_save_settings_normalizes_windows_device_prefixed_cache_paths(self) -> None:
        settings = settings_from_dict(
            {
                "voice_conductor": {
                    "cache": {
                        "root": r"\\?\m:\cache",
                        "path": r"\\?\m:\cache\voice_conductor_cache.db",
                        "api_dir": r"\\?\UNC\server\share\voice-conductor\api-caches",
                    }
                }
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            saved_path = settings.save_settings(path)
            payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)

        self.assertEqual(
            payload["voice_conductor"]["cache"]["root"],
            r"M:\cache",
        )
        self.assertEqual(
            payload["voice_conductor"]["cache"]["path"],
            r"M:\cache\voice_conductor_cache.db",
        )
        self.assertEqual(
            payload["voice_conductor"]["cache"]["api_dir"],
            r"\\server\share\voice-conductor\api-caches",
        )

    def test_save_settings_preserves_existing_provider_credentials_when_unset(self) -> None:
        settings = Settings()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            path.write_text(
                (
                    "{"
                    '"providers": {'
                    '"elevenlabs": {"api_key": "existing-eleven-key"},'
                    '"azure": {'
                    '"speech_key": "existing-azure-key",'
                    '"region": "westus",'
                    "},"
                    '"kokoro": {"hf_token": "existing-hf-token"},'
                    "},"
                    "}"
                ),
                encoding="utf-8",
            )

            saved_path = settings.save_settings(path)
            payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)

        self.assertEqual(payload["providers"]["elevenlabs"]["api_key"], "existing-eleven-key")
        self.assertEqual(payload["providers"]["azure"]["speech_key"], "existing-azure-key")
        self.assertEqual(payload["providers"]["azure"]["region"], "westus")
        self.assertEqual(payload["providers"]["kokoro"]["hf_token"], "existing-hf-token")

    def test_save_settings_prefers_explicit_provider_credentials(self) -> None:
        settings = settings_from_dict(
            {
                "providers": {
                    "elevenlabs": {"api_key": "new-eleven-key"},
                    "azure": {"speech_key": "new-azure-key", "region": "eastus"},
                    "kokoro": {"hf_token": "new-hf-token"},
                }
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            path.write_text(
                (
                    "{"
                    '"providers": {'
                    '"elevenlabs": {"api_key": "existing-eleven-key"},'
                    '"azure": {'
                    '"speech_key": "existing-azure-key",'
                    '"region": "westus",'
                    "},"
                    '"kokoro": {"hf_token": "existing-hf-token"},'
                    "},"
                    "}"
                ),
                encoding="utf-8",
            )

            saved_path = settings.save_settings(path)
            payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)

        self.assertEqual(payload["providers"]["elevenlabs"]["api_key"], "new-eleven-key")
        self.assertEqual(payload["providers"]["azure"]["speech_key"], "new-azure-key")
        self.assertEqual(payload["providers"]["azure"]["region"], "eastus")
        self.assertEqual(payload["providers"]["kokoro"]["hf_token"], "new-hf-token")

    def test_save_settings_adds_concise_default_voice_hints_and_help_file(self) -> None:
        settings = Settings()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            saved_path = settings.save_settings(path)
            saved_text = saved_path.read_text(encoding="utf-8")
            help_text = (Path(temp_dir) / "config-help.md").read_text(encoding="utf-8")

        self.assertIn("// available voices: demo:<name> [animalese, pilot, robot]", saved_text)
        self.assertIn(
            "// available voices unavailable: "
            "Provider 'kokoro' is not available. Configure credentials.",
            saved_text,
        )
        self.assertIn("## Field reference", help_text)
        self.assertIn("`voice_conductor.route_config`", help_text)
        self.assertIn("## Available voices", help_text)
        self.assertIn("| Animalese-ish | `animalese` | gibberish |", help_text)
        self.assertIn("phrase caching stores the stable voice id", help_text)

    def test_save_settings_resolves_default_voice_from_provider_availability(self) -> None:
        settings = Settings()

        class _FakeProvider:
            def __init__(self, available: bool, voices: list[VoiceInfo]) -> None:
                self._available = available
                self._voices = voices

            def is_available(self) -> bool:
                return self._available

            def list_voices(self) -> list[VoiceInfo]:
                return list(self._voices)

        def _build_provider(name: str, _settings: Settings) -> _FakeProvider:
            if name == "azure":
                return _FakeProvider(
                    True,
                    [
                        VoiceInfo(id="en-US-JennyNeural", name="Jenny", provider="azure"),
                        VoiceInfo(id="en-US-GuyNeural", name="Guy", provider="azure"),
                    ],
                )
            if name == "kokoro":
                return _FakeProvider(False, [])
            if name == "windows":
                return _FakeProvider(
                    True,
                    [VoiceInfo(id="Microsoft Zira Desktop", name="Zira", provider="windows")],
                )
            if name == "elevenlabs":
                return _FakeProvider(
                    True,
                    [VoiceInfo(id="voice-id-1", name="Rachel", provider="elevenlabs")],
                )
            if name == "demo":
                return _FakeProvider(
                    True,
                    [VoiceInfo(id="demo:robot", name="Robot Radio", provider="demo")],
                )
            raise AssertionError(f"Unexpected provider: {name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice_conductor.config.jsonc"
            with patch(
                "voice_conductor.providers.registry.ProviderRegistry.build_provider",
                side_effect=_build_provider,
            ):
                saved_path = settings.save_settings(path)
            payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)

        self.assertEqual(payload["providers"]["azure"]["default_voice"], "en-US-JennyNeural")
        self.assertEqual(payload["providers"]["windows"]["default_voice"], "zira")
        self.assertEqual(payload["providers"]["elevenlabs"]["default_voice"], "voice-id-1")
        self.assertEqual(payload["providers"]["demo"]["default_voice"], "robot")
        self.assertIsNone(payload["providers"]["kokoro"]["default_voice"])
        self.assertEqual(payload["providers"]["azure"]["speed"], 1.0)

    def test_unregistered_provider_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown provider config: custom"):
            settings_from_dict(
                {
                    "providers": {
                        "custom": {
                            "endpoint": "http://localhost:5000",
                            "api_token": "custom-secret",
                        }
                    }
                }
            )

    def test_registered_provider_config_parses_to_typed_object_and_round_trips(self) -> None:
        register_provider_config("custom_typed", CustomProviderSettings)
        try:
            settings = settings_from_dict(
                {
                    "providers": {
                        "custom_typed": {
                            "endpoint": "http://localhost:5001",
                            "api_token": "typed-secret",
                            "speed": 1.25,
                        }
                    }
                }
            )
            typed = settings.provider_settings("custom_typed")
            payload = settings_to_dict(settings)

            self.assertIsInstance(typed, CustomProviderSettings)
            self.assertEqual(typed.endpoint, "http://localhost:5001")
            self.assertEqual(typed.speed, 1.25)
            self.assertEqual(settings.provider_config("custom_typed")["endpoint"], "http://localhost:5001")
            self.assertEqual(payload["providers"]["custom_typed"]["api_token"], "typed-secret")
        finally:
            unregister_provider_config("custom_typed")

    def test_save_settings_persists_registered_custom_provider_config(self) -> None:
        register_provider_config("custom_typed", CustomProviderSettings)
        try:
            settings = Settings(
                providers=ProviderSettings(
                    extra={
                        "custom_typed": CustomProviderSettings(
                            endpoint="http://localhost:5002",
                            api_token="saved-secret",
                            speed=1.5,
                        )
                    }
                )
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "voice_conductor.config.jsonc"
                saved_path = settings.save_settings(path)
                payload = _loads_config_text(saved_path.read_text(encoding="utf-8"), saved_path)

            self.assertEqual(
                payload["providers"]["custom_typed"]["endpoint"],
                "http://localhost:5002",
            )
            self.assertEqual(payload["providers"]["custom_typed"]["api_token"], "saved-secret")
            self.assertEqual(payload["providers"]["custom_typed"]["speed"], 1.5)
        finally:
            unregister_provider_config("custom_typed")

    def test_provider_settings_returns_built_in_typed_settings(self) -> None:
        settings = Settings()

        self.assertIsInstance(settings.provider_settings("windows"), WindowsSettings)

    def test_missing_registered_custom_provider_settings_return_none(self) -> None:
        register_provider_config("custom_typed", CustomProviderSettings)
        try:
            settings = Settings()
            self.assertIsNone(settings.provider_settings("custom_typed"))
        finally:
            unregister_provider_config("custom_typed")

    def test_unregistered_provider_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown provider config: custom"):
            Settings().provider_settings("custom")

        with self.assertRaisesRegex(ValueError, "Unknown provider config: custom"):
            Settings().provider_config("custom")

    def test_unregistered_extra_provider_config_is_rejected_on_save(self) -> None:
        settings = Settings(providers=ProviderSettings(extra={"custom": {"endpoint": "test"}}))

        with self.assertRaisesRegex(ValueError, "Unknown provider config: custom"):
            settings_to_dict(settings)

    def test_builtin_extra_provider_config_is_rejected_on_save(self) -> None:
        settings = Settings(providers=ProviderSettings(extra={"demo": DemoProviderSettings()}))

        with self.assertRaisesRegex(ValueError, "demo is a built-in provider config"):
            settings_to_dict(settings)

    def test_flat_config_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nested JSON"):
            settings_from_dict({"azure_speech_key": "file-key"})
