"""Intermediate example: add and use a custom voice_conductor provider.

Run from the repository root after installing the package in editable mode:

    python examples/custom_provider_demo/custom_provider_demo.py

This entry point now runs the richer ASCII song demo in ``main.py``.
That provider exposes a small ensemble of synthesized instrument voices, writes
stem tracks plus a merged WAV, then routes the merged song to the speaker route.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # This keeps the example runnable straight from a source checkout. Once the
    # package is installed, the normal environment import would work too.
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from voice_conductor import (
    CacheSettings,
    ProviderSettings,
    Settings,
    SynthesizedAudio,
    TTSManager,
    VoiceInfo,
    VoiceConductorSettings,
    register_provider,
    register_provider_config,
    unregister_provider,
    unregister_provider_config,
)
from voice_conductor.providers import TTSProvider


PROVIDER_NAME = "tone"


@dataclass(slots=True)
class ToneProviderSettings:
    """Typed settings that will be available through settings.provider_settings()."""

    default_voice: str = "tone:sine"
    base_frequency: float = 220.0
    sample_rate: int = 16_000
    seconds_per_character: float = 0.035


class ToneProvider(TTSProvider):
    """Tiny provider that converts characters into deterministic tones."""

    name = PROVIDER_NAME

    def __init__(self, settings: Settings) -> None:
        # Provider factories receive the complete Settings object. For custom
        # provider config, register_provider_config() teaches Settings how to
        # parse and return a typed object for this provider name.
        config = settings.provider_settings(self.name)
        if config is None:
            config = ToneProviderSettings()
        if not isinstance(config, ToneProviderSettings):
            raise TypeError(f"{self.name!r} config must be ToneProviderSettings.")
        self.config = config

    def is_available(self) -> bool:
        # Real providers usually check imports, credentials, endpoint health, or
        # model files here. This example is pure NumPy, so it is always usable.
        return True

    def default_voice(self) -> str | None:
        return self._normalize_voice(self.config.default_voice)

    def cache_settings(self) -> dict[str, float | int]:
        # Cache keys should include settings that change generated audio. If the
        # base frequency changes, cached clips should not be reused.
        return {
            "base_frequency": self.config.base_frequency,
            "sample_rate": self.config.sample_rate,
            "seconds_per_character": self.config.seconds_per_character,
        }

    def list_voices(self) -> list[VoiceInfo]:
        # VoiceInfo powers list_voices() in host applications and configuration
        # helpers. The id is what users pass as the voice argument.
        return [
            VoiceInfo(
                id="tone:sine",
                name="Sine Tone",
                provider=self.name,
                language="tones",
                metadata={"shape": "smooth"},
            ),
            VoiceInfo(
                id="tone:square",
                name="Square Tone",
                provider=self.name,
                language="tones",
                metadata={"shape": "retro"},
            ),
        ]

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        # Providers return normalized float32 samples. voice_conductor handles
        # phrase caching, routing, and WAV/PCM export from this shared type.
        selected_voice = self._normalize_voice(voice or self.default_voice())
        sample_rate = int(self.config.sample_rate)
        frame_count = self._frame_count(text, sample_rate=sample_rate)
        timeline = np.arange(frame_count, dtype=np.float32) / float(sample_rate)

        # Use the text to make each phrase sound a little different while still
        # remaining deterministic. That keeps the example obvious but testable.
        text_value = sum(ord(character) for character in text)
        frequency = float(self.config.base_frequency) + (text_value % 240)
        waveform = np.sin(2.0 * np.pi * frequency * timeline)
        if selected_voice == "tone:square":
            waveform = np.sign(waveform)

        fade_frames = min(frame_count // 8, int(sample_rate * 0.04))
        if fade_frames > 0:
            fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
            waveform[:fade_frames] *= fade_in
            waveform[-fade_frames:] *= fade_out

        samples = (0.18 * waveform).astype(np.float32).reshape(-1, 1)
        return SynthesizedAudio(
            samples=samples,
            sample_rate=sample_rate,
            channels=1,
            provider=self.name,
            voice=selected_voice,
            text=text,
            metadata={
                "frequency": frequency,
                "normalized_voice_key": self.cache_voice_key(selected_voice),
            },
        )

    def _frame_count(self, text: str, *, sample_rate: int) -> int:
        seconds = max(0.25, len(text) * float(self.config.seconds_per_character))
        return max(1, int(sample_rate * seconds))

    def _normalize_voice(self, voice: str | None) -> str:
        requested = str(voice or "").strip().lower()
        if requested in {"square", "tone:square"}:
            return "tone:square"
        return "tone:sine"


def build_manager() -> TTSManager:
    """Register the custom provider and create a manager that prefers it."""

    register_provider_config(PROVIDER_NAME, ToneProviderSettings)
    register_provider(PROVIDER_NAME, ToneProvider)

    settings = Settings(
        voice_conductor=VoiceConductorSettings(
            provider_chain=[PROVIDER_NAME],
            cache=CacheSettings(root=EXAMPLE_DIR / ".runtime"),
        ),
        providers=ProviderSettings(
            extra={
                PROVIDER_NAME: ToneProviderSettings(
                    default_voice="square",
                    base_frequency=180.0,
                    seconds_per_character=0.028,
                ),
            }
        ),
    )
    return TTSManager(settings=settings)


from main import (  # noqa: E402
    NOTE_CHARS,
    ToneProvider,
    ToneProviderSettings,
    build_manager,
    main,
    merge_tracks,
    notation_help,
    parse_ascii_score,
)
from midi_to_ascii import (  # noqa: E402
    MidiAsciiSong,
    MidiAsciiTrack,
    convert_midi_to_ascii,
    format_midi_ascii_song,
    midi_note_to_ascii,
)


__all__ = [
    "NOTE_CHARS",
    "PROVIDER_NAME",
    "ToneProvider",
    "ToneProviderSettings",
    "build_manager",
    "main",
    "merge_tracks",
    "notation_help",
    "parse_ascii_score",
    "MidiAsciiSong",
    "MidiAsciiTrack",
    "convert_midi_to_ascii",
    "format_midi_ascii_song",
    "midi_note_to_ascii",
]


if __name__ == "__main__":
    main()
