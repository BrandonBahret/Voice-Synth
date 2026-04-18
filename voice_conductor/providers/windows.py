"""Windows System.Speech provider implementation.

The provider shells out to PowerShell because ``System.Speech`` is a .NET API.
It renders to a temporary WAV file and then normalizes that file into the common
``SynthesizedAudio`` shape used by the rest of the package.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from voice_conductor.config import Settings
from voice_conductor.exceptions import ProviderError
from voice_conductor.providers.base import TTSProvider
from voice_conductor.types import SynthesizedAudio, VoiceInfo

_LIST_VOICES_SCRIPT = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() |
    ForEach-Object {
        [PSCustomObject]@{
            name = $_.VoiceInfo.Name
            culture = $_.VoiceInfo.Culture.Name
            description = $_.VoiceInfo.Description
            enabled = $_.Enabled
        }
    } |
    ConvertTo-Json -Compress
"""

_DEFAULT_VOICE_SCRIPT = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Voice.Name
"""


class WindowsSpeechProvider(TTSProvider):
    """Offline TTS backend using Windows ``System.Speech.Synthesis``."""

    name = "windows"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider_settings = settings.providers.windows

    def is_available(self) -> bool:
        """Return whether the current interpreter is running on Windows."""

        return sys.platform.startswith("win")

    def default_voice(self) -> str | None:
        """Resolve the configured default voice to an installed voice name."""

        configured = self._resolve_voice(self._provider_settings.default_voice)
        if configured:
            return configured
        return self._system_default_voice()

    @staticmethod
    def _run_powershell_script(script: str) -> str:
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProviderError("Windows Speech requires PowerShell to be available.") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
            raise ProviderError(f"Windows Speech failed: {message}") from exc
        return completed.stdout

    def _run_powershell(self, script: str) -> str:
        return self._run_powershell_script(script)

    def list_voices(self=None, settings: Settings | None = None) -> list[VoiceInfo]:
        """Return installed Windows speech voices via PowerShell."""

        if isinstance(self, WindowsSpeechProvider):
            output = self._list_voices_with_runner(self._run_powershell)
        else:
            output = WindowsSpeechProvider._cached_voice_list_output()
        return WindowsSpeechProvider._parse_voice_list_output(output)

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_voice_list_output() -> str:
        return WindowsSpeechProvider._run_powershell_script(_LIST_VOICES_SCRIPT)

    @staticmethod
    def _list_voices_with_runner(runner: Callable[[str], str]) -> str:
        return runner(_LIST_VOICES_SCRIPT)

    @staticmethod
    def _parse_voice_list_output(output: str) -> list[VoiceInfo]:
        output = output.strip()
        if not output:
            return []
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            payload = [{"name": output}]
        if isinstance(payload, dict):
            payload = [payload]
        if isinstance(payload, str):
            payload = [{"name": payload}]

        voices: list[VoiceInfo] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            voices.append(
                VoiceInfo(
                    id=name,
                    name=name,
                    provider=WindowsSpeechProvider.name,
                    language=WindowsSpeechProvider._optional_string(item.get("culture")),
                    metadata={
                        "description": WindowsSpeechProvider._optional_string(
                            item.get("description")
                        ),
                        "enabled": bool(item.get("enabled", True)),
                    },
                )
            )
        return voices

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _resolve_voice(self, voice: str | None) -> str | None:
        requested = voice
        if not requested:
            return None
        normalized_requested = requested.strip().lower()
        for item in self.list_voices():
            normalized_id = item.id.lower()
            normalized_name = item.name.lower()
            if (
                normalized_id == normalized_requested
                or normalized_name == normalized_requested
                or normalized_requested in normalized_id
                or normalized_requested in normalized_name
            ):
                return item.name
        return None

    def _system_default_voice(self) -> str | None:
        """Return Windows SpeechSynthesizer's current default voice name."""

        try:
            voice_name = self._run_powershell(_DEFAULT_VOICE_SCRIPT).strip()
        except ProviderError:
            return None
        return voice_name or None

    def _rate(self) -> int:
        speed = min(max(self._provider_settings.speed, 0.5), 2.0)
        scaled = (speed - 1.0) * 10.0
        return max(-10, min(10, math.floor(scaled) if scaled < 0 else math.ceil(scaled)))

    def _volume(self) -> int:
        configured = self._provider_settings.volume
        if configured is None:
            return 100
        return max(0, min(100, configured))

    def _voice_selection_script(self, selected_voice: str | None) -> str:
        if selected_voice:
            escaped_voice = selected_voice.replace("'", "''")
            return f"$synth.SelectVoice('{escaped_voice}')"
        return ""

    def cache_settings(self) -> dict[str, int | str]:
        """Encode Windows speech options that affect rendered audio."""

        settings: dict[str, int | str] = {"rate": self._rate(), "volume": self._volume()}
        if self._provider_settings.default_voice:
            settings["default_voice"] = self._provider_settings.default_voice.strip().lower()
        return settings

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        """Render speech to a temporary WAV file and return normalized audio."""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)

        selected_voice = self._resolve_voice(voice) or self.default_voice()
        escaped_text = text.replace("'", "''")
        voice_line = self._voice_selection_script(selected_voice)
        script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
{voice_line}
$synth.Rate = {self._rate()}
$synth.Volume = {self._volume()}
$synth.SetOutputToWaveFile('{str(wav_path).replace("'", "''")}')
$synth.Speak('{escaped_text}')
$synth.Dispose()
"""
        try:
            self._run_powershell(script)
            wav_bytes = wav_path.read_bytes()
        finally:
            wav_path.unlink(missing_ok=True)

        return SynthesizedAudio.from_wav_bytes(
            wav_bytes,
            provider=self.name,
            voice=selected_voice,
            text=text,
            metadata={
                "engine": "System.Speech.Synthesis",
                "rate": self._rate(),
                "volume": self._volume(),
                "normalized_voice_key": self.cache_voice_key(selected_voice),
            },
        )
