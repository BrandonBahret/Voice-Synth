from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_conductor.config import Settings, settings_from_dict
from voice_conductor.providers.windows import WindowsSpeechProvider


class RecordingWindowsSpeechProvider(WindowsSpeechProvider):
    def __init__(self, settings: Settings, *, script_output: str = "") -> None:
        super().__init__(settings)
        self.script_output = script_output
        self.scripts: list[str] = []

    def _run_powershell(self, script: str) -> str:
        self.scripts.append(script)
        return self.script_output


class ScriptedWindowsSpeechProvider(WindowsSpeechProvider):
    def __init__(self, settings: Settings, *, outputs: dict[str, str]) -> None:
        super().__init__(settings)
        self.outputs = outputs
        self.scripts: list[str] = []

    def _run_powershell(self, script: str) -> str:
        self.scripts.append(script)
        if "GetInstalledVoices()" in script:
            return self.outputs.get("voices", "")
        if "$synth.Voice.Name" in script:
            return self.outputs.get("default_voice", "")
        return ""


def _settings(
    *,
    windows: dict[str, object] | None = None,
) -> Settings:
    return settings_from_dict(
        {
            "providers": {"windows": windows or {}},
        }
    )


class WindowsSpeechProviderTests(unittest.TestCase):
    def test_default_voice_resolves_from_settings(self) -> None:
        provider = RecordingWindowsSpeechProvider(
            _settings(windows={"default_voice": "zira"}),
            script_output='[{"name":"Microsoft Zira Desktop","culture":"en-US","description":"English voice","enabled":true}]',
        )

        self.assertEqual(provider.default_voice(), "Microsoft Zira Desktop")

    def test_default_voice_falls_back_to_system_default_when_configured_voice_is_missing(self) -> None:
        provider = ScriptedWindowsSpeechProvider(
            _settings(windows={"default_voice": "nonexistent"}),
            outputs={
                "voices": '[{"name":"Microsoft Zira Desktop","culture":"en-US","description":"English voice","enabled":true}]',
                "default_voice": "Microsoft David Desktop",
            },
        )

        self.assertEqual(provider.default_voice(), "Microsoft David Desktop")

    def test_list_voices_includes_voice_metadata(self) -> None:
        provider = RecordingWindowsSpeechProvider(
            Settings(),
            script_output='[{"name":"Microsoft Zira Desktop","culture":"en-US","description":"English voice","enabled":true}]',
        )

        voices = provider.list_voices()

        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].name, "Microsoft Zira Desktop")
        self.assertEqual(voices[0].language, "en-US")
        self.assertEqual(voices[0].metadata["description"], "English voice")
        self.assertTrue(voices[0].metadata["enabled"])

    def test_list_voices_can_be_called_without_instantiating_provider(self) -> None:
        from unittest.mock import patch

        payload = (
            '[{"name":"Microsoft Zira Desktop","culture":"en-US",'
            '"description":"English voice","enabled":true}]'
        )

        with patch.object(
            WindowsSpeechProvider,
            "_run_powershell_script",
            return_value=payload,
        ) as mock_run:
            WindowsSpeechProvider._cached_voice_list_output.cache_clear()
            try:
                voices = WindowsSpeechProvider.list_voices()
            finally:
                WindowsSpeechProvider._cached_voice_list_output.cache_clear()

        self.assertEqual(voices[0].name, "Microsoft Zira Desktop")
        mock_run.assert_called_once()

    def test_cache_settings_tracks_windows_specific_settings(self) -> None:
        provider = RecordingWindowsSpeechProvider(
            _settings(
                windows={
                    "default_voice": "Zira",
                    "volume": 65,
                    "speed": 1.2,
                },
            )
        )

        cache_settings = provider.cache_settings()

        self.assertEqual(
            cache_settings,
            {"rate": 2, "volume": 65, "default_voice": "zira"},
        )

    def test_synthesize_uses_default_voice_when_configured(self) -> None:
        provider = RecordingWindowsSpeechProvider(
            _settings(
                windows={
                    "default_voice": "Zira",
                    "volume": 80,
                },
            ),
            script_output='[{"name":"Microsoft Zira Desktop","culture":"en-US","description":"English voice","enabled":true}]',
        )
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "tiny.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(np.zeros(160, dtype=np.int16).tobytes())

            class TempWaveFile:
                def __init__(self, path: Path) -> None:
                    self.name = str(path)

                def __enter__(self) -> "TempWaveFile":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

            with patch.object(tempfile, "NamedTemporaryFile", return_value=TempWaveFile(wav_path)):
                audio = provider.synthesize("push now")

        script = provider.scripts[-1]
        self.assertIn("$synth.SelectVoice('Microsoft Zira Desktop')", script)
        self.assertIn("$synth.Volume = 80", script)
        self.assertEqual(audio.metadata["volume"], 80)
