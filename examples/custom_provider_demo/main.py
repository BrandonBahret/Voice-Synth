"""Custom provider demo that turns ASCII notation into a multi-track song."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from song import song


EXAMPLE_DIR = Path(__file__).resolve().parent
PROVIDER_NAME = "tone"
VOICE_PREFIX = f"{PROVIDER_NAME}:"

NOTE_CHARS = {
    "z": ("C", 3),
    "x": ("D", 3),
    "c": ("E", 3),
    "v": ("F", 3),
    "b": ("G", 3),
    "n": ("A", 3),
    "m": ("B", 3),
    "a": ("C", 4),
    "s": ("D", 4),
    "d": ("E", 4),
    "f": ("F", 4),
    "g": ("G", 4),
    "h": ("A", 4),
    "j": ("B", 4),
    "q": ("C", 5),
    "w": ("D", 5),
    "e": ("E", 5),
    "r": ("F", 5),
    "t": ("G", 5),
    "y": ("A", 5),
    "u": ("B", 5),
}
REST_CHARS = {".", "-"}
NOTE_OFFSETS = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


@dataclass(frozen=True, slots=True)
class ScoreEvent:
    """One note, chord, or rest in the ASCII score."""

    notes: tuple[float, ...]
    beats: float
    volume: float


@dataclass(slots=True)
class ToneProviderSettings:
    """Typed settings available through settings.provider_settings()."""

    default_voice: str = "tone:piano"
    sample_rate: int = 44_100
    bpm: float = 132.0
    master_volume: float = 0.65


@dataclass(frozen=True, slots=True)
class WrittenTrack:
    """One synthesized stem and the WAV file written for it."""

    voice_name: str
    audio: SynthesizedAudio
    target: Path


class ToneProvider(TTSProvider):
    """Tiny provider that converts ASCII notation into deterministic music."""

    name = PROVIDER_NAME

    _VOICES = {
        "tone:piano": {"name": "ASCII Piano", "language": "music", "family": "keyed"},
        "tone:electric_piano": {
            "name": "ASCII Electric Piano",
            "language": "music",
            "family": "electric keyed",
        },
        "tone:banjo": {
            "name": "ASCII Banjo",
            "language": "music",
            "family": "plucked string",
        },
        "tone:bandoneon": {
            "name": "ASCII Bandoneon",
            "language": "music",
            "family": "free reed",
        },
        "tone:bass": {
            "name": "ASCII Bass",
            "language": "music",
            "family": "string",
        },
        "tone:electric_bass": {
            "name": "ASCII Electric Bass",
            "language": "music",
            "family": "electric string",
        },
        "tone:synth_bass": {
            "name": "ASCII Synth Bass",
            "language": "music",
            "family": "synth",
        },
        "tone:clarinet": {
            "name": "ASCII Clarinet",
            "language": "music",
            "family": "reed",
        },
        "tone:marimba": {
            "name": "ASCII Marimba",
            "language": "music",
            "family": "percussion",
        },
        "tone:drum_kit": {
            "name": "ASCII Drum Kit",
            "language": "music",
            "family": "percussion",
        },
        "tone:oboe": {
            "name": "ASCII Oboe",
            "language": "music",
            "family": "double reed",
        },
        "tone:recorder": {
            "name": "ASCII Recorder",
            "language": "music",
            "family": "fipple flute",
        },
        "tone:tenor_sax": {
            "name": "ASCII Tenor Sax",
            "language": "music",
            "family": "single reed",
        },
        "tone:square_lead": {
            "name": "ASCII Square Lead",
            "language": "music",
            "family": "synth",
        },
        "tone:synth_strings": {
            "name": "ASCII Synth Strings",
            "language": "music",
            "family": "synth",
        },
        "tone:trumpet": {
            "name": "ASCII Cornet-Trumpet",
            "language": "music",
            "family": "brass",
        },
        "tone:tuba": {
            "name": "ASCII Tuba",
            "language": "music",
            "family": "brass",
        },
    }

    def __init__(self, settings: Settings) -> None:
        config = settings.provider_settings(self.name)
        if config is None:
            config = ToneProviderSettings()
        if not isinstance(config, ToneProviderSettings):
            raise TypeError(f"{self.name!r} config must be ToneProviderSettings.")
        self.config = config

    def is_available(self) -> bool:
        return True

    def default_voice(self) -> str | None:
        return self._normalize_voice(self.config.default_voice)

    def cache_settings(self) -> dict[str, float | int]:
        return {
            "bpm": self.config.bpm,
            "master_volume": self.config.master_volume,
            "sample_rate": self.config.sample_rate,
        }

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=voice_id,
                name=meta["name"],
                provider=self.name,
                language=meta["language"],
                metadata=meta,
            )
            for voice_id, meta in self._VOICES.items()
        ]

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        selected_voice = self._normalize_voice(voice or self.default_voice())
        base_voice, part_number = _split_voice_part(selected_voice)
        sample_rate = int(self.config.sample_rate)
        events, bpm = parse_ascii_score(text, default_bpm=float(self.config.bpm))
        beat_seconds = 60.0 / bpm
        total_frames = sum(
            max(1, int(event.beats * beat_seconds * sample_rate)) for event in events
        )
        samples = np.zeros(max(1, total_frames), dtype=np.float32)

        cursor = 0
        for event in events:
            frame_count = max(1, int(event.beats * beat_seconds * sample_rate))
            if event.notes:
                rendered = [
                    self._render_note(
                        frequency,
                        frame_count=frame_count,
                        sample_rate=sample_rate,
                        voice=base_voice,
                        part_number=part_number,
                    )
                    for frequency in event.notes
                ]
                chord = np.sum(rendered, axis=0) / max(1, len(rendered))
                samples[cursor : cursor + frame_count] += chord * event.volume
            cursor += frame_count

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.0:
            samples = samples / peak
        samples = (
            (samples * float(self.config.master_volume))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        return SynthesizedAudio(
            samples=samples,
            sample_rate=sample_rate,
            channels=1,
            provider=self.name,
            voice=selected_voice,
            text=text,
            metadata={
                "bpm": bpm,
                "events": len(events),
                "notation": notation_help(),
                "base_voice": base_voice,
                "part_number": part_number,
                "normalized_voice_key": self.cache_voice_key(selected_voice),
            },
        )

    def _normalize_voice(self, voice: str | None) -> str:
        requested = str(voice or "").strip().lower()
        voice_id = (
            requested
            if requested.startswith(VOICE_PREFIX)
            else f"{VOICE_PREFIX}{requested}"
        )
        base_voice, _part_number = _split_voice_part(voice_id)
        if base_voice in self._VOICES:
            return voice_id
        return "tone:piano"

    def _render_note(
        self,
        frequency: float,
        *,
        frame_count: int,
        sample_rate: int,
        voice: str,
        part_number: int,
    ) -> np.ndarray:
        timeline = np.arange(frame_count, dtype=np.float32) / float(sample_rate)
        duration = max(frame_count / float(sample_rate), 0.001)
        phase = 2.0 * np.pi * frequency * timeline
        part_pan_gain = 1.0 if part_number == 1 else max(0.72, 1.0 - part_number * 0.08)

        if voice == "tone:banjo":
            detune = 1.0 + (part_number - 1) * 0.0025
            bright_phase = phase * detune
            waveform = _harmonic_wave(
                bright_phase,
                [1.0, 0.82, 0.54, 0.34, 0.24, 0.16, 0.10, 0.07],
            )
            waveform += 0.18 * _harmonic_wave(
                bright_phase * 2.015,
                [1.0, 0.42, 0.21],
            )
            pick = np.exp(-135.0 * timeline)
            waveform += pick * (
                0.26 * np.sin(18.0 * phase) + 0.12 * np.sin(31.0 * phase)
            )
            envelope = np.exp(-5.7 * timeline / duration)
            envelope *= _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.002,
                decay=0.045,
                sustain=0.22,
                release=0.045,
            )
        elif voice == "tone:bandoneon":
            tremolo = 1.0 + 0.07 * np.sin(2.0 * np.pi * 5.4 * timeline)
            reed_detune = 1.006 + (part_number - 1) * 0.0015
            waveform = (
                _harmonic_wave(phase, [1.0, 0.58, 0.38, 0.25, 0.17, 0.10])
                + 0.42
                * _harmonic_wave(
                    phase * reed_detune, [1.0, 0.45, 0.28, 0.18, 0.12]
                )
            ) * tremolo
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.055,
                decay=0.05,
                sustain=0.84,
                release=0.11,
            )
        elif voice == "tone:electric_piano":
            tremolo = 1.0 + 0.035 * np.sin(2.0 * np.pi * 6.2 * timeline)
            bell = np.exp(-7.0 * timeline / duration)
            waveform = _harmonic_wave(phase, [1.0, 0.34, 0.16, 0.08])
            waveform += bell * (
                0.42 * np.sin(2.01 * phase)
                + 0.18 * np.sin(3.98 * phase)
                + 0.09 * np.sin(7.02 * phase)
            )
            waveform *= tremolo
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.004,
                decay=0.18,
                sustain=0.42,
                release=0.08,
            )
        elif voice == "tone:bass":
            waveform = _harmonic_wave(phase, [1.0, 0.72, 0.36, 0.18, 0.10])
            waveform += 0.12 * np.sin(0.5 * phase)
            envelope = np.exp(-2.4 * timeline / duration)
            envelope *= _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.009,
                decay=0.09,
                sustain=0.55,
                release=0.08,
            )
        elif voice == "tone:electric_bass":
            pluck = np.exp(-4.0 * timeline / duration)
            waveform = _harmonic_wave(phase, [1.0, 0.62, 0.30, 0.16])
            waveform += 0.22 * np.sin(0.5 * phase)
            waveform = np.tanh(1.7 * waveform) * pluck
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.005,
                decay=0.07,
                sustain=0.58,
                release=0.07,
            )
        elif voice == "tone:synth_bass":
            sub = np.sin(0.5 * phase)
            square = np.sign(np.sin(phase))
            pulse = np.sign(np.sin(phase + 0.42 * np.sin(2.0 * np.pi * 2.1 * timeline)))
            waveform = 0.45 * sub + 0.38 * square + 0.17 * pulse
            waveform = np.tanh(1.9 * waveform)
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.003,
                decay=0.045,
                sustain=0.70,
                release=0.055,
            )
        elif voice == "tone:clarinet":
            waveform = (
                np.sin(phase) + 0.35 * np.sin(3.0 * phase) + 0.18 * np.sin(5.0 * phase)
            )
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.045,
                decay=0.08,
                sustain=0.75,
                release=0.08,
            )
        elif voice == "tone:marimba":
            waveform = (
                np.sin(phase)
                + 0.36 * np.sin(4.0 * phase)
                + 0.19 * np.sin(9.8 * phase)
                + 0.07 * np.sin(14.3 * phase)
            )
            strike = np.exp(-95.0 * timeline)
            waveform += strike * (0.22 * np.sin(17.0 * phase) + 0.12 * np.sin(23.0 * phase))
            envelope = np.exp(-3.1 * timeline / duration)
            envelope *= _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.003,
                decay=0.045,
                sustain=0.18,
                release=0.07,
            )
        elif voice == "tone:drum_kit":
            noise = _deterministic_noise(timeline, frequency)
            if frequency < 180.0:
                drop = np.exp(-34.0 * timeline)
                drum_phase = 2.0 * np.pi * (55.0 + 58.0 * drop) * timeline
                waveform = np.sin(drum_phase) * np.exp(-11.0 * timeline)
                waveform += 0.08 * noise * np.exp(-55.0 * timeline)
                envelope = np.ones(frame_count, dtype=np.float32)
            elif frequency < 420.0:
                body = np.sin(2.0 * np.pi * 190.0 * timeline)
                snap = noise * np.exp(-24.0 * timeline)
                waveform = 0.38 * body * np.exp(-12.0 * timeline) + 0.72 * snap
                envelope = np.ones(frame_count, dtype=np.float32)
            elif frequency < 620.0:
                drum_phase = 2.0 * np.pi * (95.0 + frequency * 0.22) * timeline
                waveform = np.sin(drum_phase) * np.exp(-9.0 * timeline)
                waveform += 0.20 * noise * np.exp(-20.0 * timeline)
                envelope = np.ones(frame_count, dtype=np.float32)
            else:
                metallic = (
                    np.sin(2.0 * np.pi * 5211.0 * timeline)
                    + np.sin(2.0 * np.pi * 7417.0 * timeline)
                    + np.sin(2.0 * np.pi * 9127.0 * timeline)
                )
                waveform = (0.72 * noise + 0.28 * metallic) * np.exp(-36.0 * timeline)
                envelope = np.ones(frame_count, dtype=np.float32)
        elif voice == "tone:oboe":
            reed_buzz = _harmonic_wave(
                phase, [0.95, 0.28, 0.88, 0.24, 0.52, 0.16, 0.28, 0.10]
            )
            nasal_formant = 0.18 * np.sin(2.0 * np.pi * 1150.0 * timeline)
            waveform = reed_buzz + nasal_formant * np.sin(phase)
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.035,
                decay=0.07,
                sustain=0.82,
                release=0.075,
            )
        elif voice == "tone:recorder":
            breath = _breath_noise(timeline, frequency) * 0.045
            waveform = _harmonic_wave(phase, [1.0, 0.08, 0.38, 0.04, 0.16, 0.03])
            waveform = waveform + breath
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.018,
                decay=0.05,
                sustain=0.80,
                release=0.065,
            )
        elif voice == "tone:tenor_sax":
            growl = 1.0 + 0.012 * np.sin(2.0 * np.pi * 28.0 * timeline)
            sax_phase = phase * growl
            waveform = _harmonic_wave(
                sax_phase, [1.0, 0.72, 0.55, 0.35, 0.24, 0.18, 0.11]
            )
            waveform += 0.055 * _breath_noise(timeline, frequency * 0.5)
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.038,
                decay=0.09,
                sustain=0.78,
                release=0.09,
            )
        elif voice == "tone:square_lead":
            vibrato = 1.0 + 0.004 * np.sin(2.0 * np.pi * 5.8 * timeline)
            lead_phase = 2.0 * np.pi * frequency * vibrato * timeline
            waveform = (
                np.sign(np.sin(lead_phase))
                + 0.38 * np.sign(np.sin(2.0 * lead_phase))
                + 0.16 * np.sin(3.0 * lead_phase)
            )
            waveform = np.tanh(1.25 * waveform)
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.002,
                decay=0.035,
                sustain=0.74,
                release=0.045,
            )
        elif voice == "tone:synth_strings":
            slow = 1.0 - np.exp(-12.0 * timeline)
            chorus = (
                _harmonic_wave(phase, [1.0, 0.42, 0.24, 0.15])
                + 0.52 * _harmonic_wave(phase * 1.004, [1.0, 0.36, 0.20])
                + 0.36 * _harmonic_wave(phase * 0.997, [1.0, 0.28, 0.14])
            )
            shimmer = 1.0 + 0.028 * np.sin(2.0 * np.pi * 0.8 * timeline)
            waveform = chorus * slow * shimmer
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.080,
                decay=0.18,
                sustain=0.82,
                release=0.16,
            )
        elif voice == "tone:trumpet":
            wobble = 1.0 + 0.003 * np.sin(2.0 * np.pi * 4.6 * timeline)
            bright_phase = 2.0 * np.pi * frequency * wobble * timeline
            cornet_bloom = 1.0 - np.exp(-28.0 * timeline)
            waveform = (
                1.0 * np.sin(bright_phase)
                + 0.46 * np.sin(2.0 * bright_phase)
                + 0.22 * np.sin(3.0 * bright_phase)
                + 0.09 * np.sin(4.0 * bright_phase)
                + 0.04 * np.sin(5.0 * bright_phase)
            ) * cornet_bloom
            waveform += 0.10 * np.sin(0.5 * bright_phase)
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.042,
                decay=0.15,
                sustain=0.82,
                release=0.095,
            )
        elif voice == "tone:tuba":
            slow_bloom = 1.0 - np.exp(-18.0 * timeline)
            waveform = _harmonic_wave(phase, [1.0, 0.64, 0.42, 0.30, 0.20, 0.12])
            waveform += 0.18 * np.sin(0.5 * phase)
            waveform *= slow_bloom
            envelope = _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.060,
                decay=0.10,
                sustain=0.88,
                release=0.12,
            )
        else:
            waveform = (
                np.sin(phase) + 0.42 * np.sin(2.0 * phase) + 0.18 * np.sin(3.0 * phase)
            )
            envelope = np.exp(-4.2 * timeline / duration)
            envelope *= _adsr_envelope(
                frame_count,
                sample_rate,
                attack=0.006,
                decay=0.04,
                sustain=0.38,
                release=0.05,
            )
        return (waveform * envelope * 0.35 * part_pan_gain).astype(np.float32)


def _split_voice_part(voice: str) -> tuple[str, int]:
    """Return the playable voice plus an optional numbered part suffix."""

    base_voice, separator, suffix = voice.rpartition("_")
    if separator and suffix.isdigit():
        return base_voice, max(1, int(suffix))
    return voice, 1


def _harmonic_wave(phase: np.ndarray, amplitudes: list[float]) -> np.ndarray:
    waveform = np.zeros_like(phase)
    for index, amplitude in enumerate(amplitudes, start=1):
        waveform += float(amplitude) * np.sin(float(index) * phase)
    return waveform


def _breath_noise(timeline: np.ndarray, seed_frequency: float) -> np.ndarray:
    seed = max(1.0, seed_frequency)
    return (
        np.sin(2.0 * np.pi * (1703.0 + seed * 0.07) * timeline)
        + 0.55 * np.sin(2.0 * np.pi * (2411.0 + seed * 0.11) * timeline)
        + 0.35 * np.sin(2.0 * np.pi * (3187.0 + seed * 0.05) * timeline)
    )


def _deterministic_noise(timeline: np.ndarray, seed_frequency: float) -> np.ndarray:
    seed = max(1.0, seed_frequency)
    return np.tanh(
        1.4
        * (
            np.sin(2.0 * np.pi * (1217.0 + seed * 0.31) * timeline)
            + 0.72 * np.sin(2.0 * np.pi * (2659.0 + seed * 0.17) * timeline)
            + 0.48 * np.sin(2.0 * np.pi * (4801.0 + seed * 0.09) * timeline)
            + 0.31 * np.sin(2.0 * np.pi * (7919.0 + seed * 0.04) * timeline)
        )
    )


def parse_ascii_score(
    score: str, *, default_bpm: float
) -> tuple[list[ScoreEvent], float]:
    """Parse compact ASCII notation into timed score events.

    Syntax: `zxcvbnm` is C3..B3, `asdfghj` is C4..B4, `qwertyu` is C5..B5,
    `.` or `-` is a rest, `#`/`b` accidentals follow notes, numbers and `/N`
    set duration, `[adg]` is a chord, `^0.55` sets track gain, `!0.5`
    changes event volume, `@144` changes tempo, and `(phrase)*2` repeats a
    phrase.
    """

    expanded = _expand_repeats(score)
    events: list[ScoreEvent] = []
    bpm = default_bpm
    track_gain = 1.0
    volume = 1.0
    index = 0

    while index < len(expanded):
        character = expanded[index]
        if character.isspace() or character == "|":
            index += 1
            continue
        if character == "@":
            value, index = _read_number(expanded, index + 1, default=bpm)
            bpm = max(40.0, min(240.0, value))
            continue
        if character == "^":
            value, index = _read_number(expanded, index + 1, default=track_gain)
            track_gain = max(0.0, min(1.5, value))
            continue
        if character == "!":
            value, index = _read_number(expanded, index + 1, default=volume)
            volume = max(0.0, min(1.5, value))
            continue
        if character == "[":
            close_index = expanded.find("]", index + 1)
            if close_index == -1:
                close_index = len(expanded)
            notes = _parse_chord_notes(expanded[index + 1 : close_index])
            index = close_index + 1
            beats, index = _read_duration(expanded, index)
            events.append(
                ScoreEvent(notes=notes, beats=beats, volume=volume * track_gain)
            )
            continue
        if character in NOTE_CHARS:
            accidental = ""
            index += 1
            if index < len(expanded) and expanded[index] in {"#", "b"}:
                accidental = expanded[index]
                index += 1
            beats, index = _read_duration(expanded, index)
            events.append(
                ScoreEvent(
                    notes=(_note_frequency(character, accidental=accidental),),
                    beats=beats,
                    volume=volume * track_gain,
                )
            )
            continue
        if character in REST_CHARS:
            index += 1
            beats, index = _read_duration(expanded, index)
            events.append(ScoreEvent(notes=(), beats=beats, volume=volume * track_gain))
            continue
        index += 1

    return events or [ScoreEvent(notes=(), beats=1.0, volume=0.0)], bpm


def notation_help() -> str:
    return "zxcvbnm=C3..B3, asdfghj=C4..B4, qwertyu=C5..B5, .=rest, []=chord, ^=gain, @=bpm, !=volume, (phrase)*N=repeat"


def merge_tracks(
    tracks: list[SynthesizedAudio], *, text: str = "merged song"
) -> SynthesizedAudio:
    """Mix several mono tracks into one normalized clip."""

    if not tracks:
        raise ValueError("At least one track is required.")
    sample_rate = tracks[0].sample_rate
    if any(track.sample_rate != sample_rate for track in tracks):
        raise ValueError("All tracks must use the same sample rate.")

    frame_count = max(track.frame_count for track in tracks)
    mix = np.zeros((frame_count, 1), dtype=np.float32)
    for track in tracks:
        mix[: track.frame_count, :1] += track.samples[:, :1]
    mix /= max(1.0, np.sqrt(len(tracks)))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.95:
        mix *= 0.95 / peak
    return SynthesizedAudio(
        samples=mix,
        sample_rate=sample_rate,
        channels=1,
        provider=PROVIDER_NAME,
        voice="tone:ensemble",
        text=text,
        metadata={
            "tracks": [track.voice for track in tracks],
            "normalized_voice_key": "tone:ensemble",
        },
    )


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
                    default_voice="piano",
                    bpm=132.0,
                    master_volume=0.62,
                ),
            }
        ),
    )
    return TTSManager(settings=settings)


def write_track(
    manager: TTSManager, runtime_dir: Path, voice_name: str, score: str
) -> WrittenTrack:
    """Synthesize one score and write its stem WAV."""

    track = manager.synthesize_voice(score, provider=PROVIDER_NAME, voice=voice_name)
    target = runtime_dir / f"{voice_name}-track.wav"
    track.copy_to(target)
    return WrittenTrack(voice_name=voice_name, audio=track, target=target)


def main() -> None:
    manager = build_manager()
    try:
        print("Available providers:", manager.list_providers())
        print("Tone voices:")
        for voice in manager.list_voices(PROVIDER_NAME):
            print(f"  - {voice.id}: {voice.name}")
        print(f"Notation: {notation_help()}")

        # song = {
        #     "electric_piano": "@132 !0.72 ([adg] [sfh] [dgj] [fha])*2",
        #     "synth_bass": "@132 !0.66 z/2 -/2 z/2 b/2 | z/2 -/2 b/2 z/2",
        #     "square_lead": "@132 !0.58 -- q e t | y2 t e q",
        #     "synth_strings": "@132 !0.42 [adg]4 | [sfh]4",
        #     "drum_kit": "@132 !0.70 (z/2 r/4 s/4 r/4 s/4 y/2)*2",
        # }

        runtime_dir = EXAMPLE_DIR / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        
        
        
        written_tracks = list(
                map(
                    write_track,
                    repeat(manager),
                    repeat(runtime_dir),
                    song.keys(),
                    song.values(),
                )
            )
        
            

        tracks = [written.audio for written in written_tracks]
        for written in written_tracks:
            print(
                f"Wrote {written.voice_name:8} track: {written.target} "
                f"({written.audio.duration_seconds:.2f}s)"
            )

        merged = merge_tracks(tracks, text="\n".join(song.values()))
        merged_target = runtime_dir / "ascii-ensemble-song.wav"
        merged.copy_to(merged_target)
        print(f"Wrote merged song: {merged_target} ({merged.duration_seconds:.2f}s)")

        try:
            routes = ["speakers", "mic"]
            # routes = ["mic"]
            result = manager.route(merged, routes=routes)
            for route in routes:
                print("Routed merged song to:", result.devices[route].name)
        except Exception as exc:
            print(f"Could not route to speakers in this environment: {exc}")
    finally:
        unregister_provider(PROVIDER_NAME)
        unregister_provider_config(PROVIDER_NAME)


def _adsr_envelope(
    frame_count: int,
    sample_rate: int,
    *,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    envelope = np.full(frame_count, sustain, dtype=np.float32)
    attack_frames = min(frame_count, max(1, int(attack * sample_rate)))
    decay_frames = min(max(0, frame_count - attack_frames), int(decay * sample_rate))
    release_frames = min(
        max(0, frame_count - attack_frames - decay_frames), int(release * sample_rate)
    )

    envelope[:attack_frames] = np.linspace(0.0, 1.0, attack_frames, dtype=np.float32)
    decay_end = attack_frames + decay_frames
    if decay_frames > 0:
        envelope[attack_frames:decay_end] = np.linspace(
            1.0, sustain, decay_frames, dtype=np.float32
        )
    if release_frames > 0:
        envelope[-release_frames:] *= np.linspace(
            1.0, 0.0, release_frames, dtype=np.float32
        )
    return envelope


def _expand_repeats(score: str) -> str:
    pattern = re.compile(r"\(([^()]*)\)\*(\d+)")
    expanded = score
    while True:
        next_expanded = pattern.sub(
            lambda match: match.group(1) * int(match.group(2)), expanded
        )
        if next_expanded == expanded:
            return expanded
        expanded = next_expanded


def _note_frequency(note_char: str, *, accidental: str) -> float:
    note_name, octave = NOTE_CHARS[note_char]
    semitone = NOTE_OFFSETS[note_name] + (octave - 4) * 12
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    return 440.0 * (2.0 ** (semitone / 12.0))


def _parse_chord_notes(chord: str) -> tuple[float, ...]:
    notes: list[float] = []
    index = 0
    while index < len(chord):
        note_char = chord[index]
        if note_char not in NOTE_CHARS:
            index += 1
            continue
        index += 1
        accidental = ""
        if index < len(chord) and chord[index] in {"#", "b"}:
            accidental = chord[index]
            index += 1
        notes.append(_note_frequency(note_char, accidental=accidental))
    return tuple(notes)


def _read_duration(score: str, index: int) -> tuple[float, int]:
    beats = 1.0
    if index < len(score) and score[index].isdigit():
        beats, index = _read_number(score, index, default=beats)
    if index < len(score) and score[index] == "/":
        divisor, index = _read_number(score, index + 1, default=1.0)
        if divisor > 0:
            beats /= divisor
    return max(0.0625, beats), index


def _read_number(score: str, index: int, *, default: float) -> tuple[float, int]:
    start = index
    while index < len(score) and (score[index].isdigit() or score[index] == "."):
        index += 1
    if start == index:
        return default, index
    return float(score[start:index]), index


if __name__ == "__main__":
    main()
