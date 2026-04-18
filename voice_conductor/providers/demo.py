"""Dependency-free demo provider with deterministic retro speech output."""

from __future__ import annotations

import re

import numpy as np

from voice_conductor.config import Settings
from voice_conductor.providers.base import TTSProvider
from voice_conductor.types import SynthesizedAudio, VoiceInfo

_WORD_TOKEN_PATTERN = re.compile(r"\S+")
_VOWEL_PATTERN = re.compile(r"[aeiouy]+")
_SENTENCE_PUNCTUATION = ".!?"
_MID_PUNCTUATION = ",;:"
_DEFAULT_VOICE = "demo:animalese"


class DemoProvider(TTSProvider):
    """Tiny retro-game voice provider for documentation, tests, and demos."""

    _voice_profiles = {
        "demo:animalese": {"name": "Animalese-ish", "base": 520.0, "scale": 1.00, "noise": 0.018, "speed": 0.70},
        "demo:pilot": {"name": "SNES Pilot Comms", "base": 310.0, "scale": 0.82, "noise": 0.035, "speed": 0.6},
        "demo:robot": {"name": "Robot Radio", "base": 180.0, "scale": 0.58, "noise": 0.010, "speed": 0.7},
    }

    def __init__(
        self,
        settings: Settings,
        *,
        name: str = "demo",
        available: bool = True,
        default_voice: str | None = None,
    ) -> None:
        self.settings = settings
        self.name = name
        self.available = available
        self._default_voice = default_voice
        self.calls: list[tuple[str, str | None]] = []

    def is_available(self) -> bool:
        return self.available

    def default_voice(self) -> str | None:
        return self._resolve_voice(self._default_voice or self._configured_default_voice())

    def _configured_default_voice(self) -> str:
        config = self.settings.provider_settings(self.name)
        if isinstance(config, dict):
            voice = config.get("default_voice", _DEFAULT_VOICE)
        else:
            voice = getattr(config, "default_voice", _DEFAULT_VOICE)
        return str(voice or _DEFAULT_VOICE)

    def _speed(self) -> float:
        config = self.settings.provider_settings(self.name)
        if isinstance(config, dict):
            speed = config.get("speed", 1.0)
        else:
            speed = getattr(config, "speed", 1.0)
        try:
            return float(speed)
        except (TypeError, ValueError):
            return 1.0

    def cache_settings(self) -> dict[str, str | float]:
        speed = self._speed()
        return {
            "engine": "retro-blip-v2",
            "speed": round(speed, 3),
            "voice": self.default_voice() or _DEFAULT_VOICE,
        }

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=voice_id, name=profile["name"], provider=self.name, language="gibberish")
            for voice_id, profile in self._voice_profiles.items()
        ]

    def _resolve_voice(self, voice: str | None) -> str:
        requested = str(voice or "").strip()
        if not requested:
            return _DEFAULT_VOICE
        if requested in self._voice_profiles:
            return requested

        normalized_requested = requested.lower()
        provider_prefixed = f"{self.name}:{normalized_requested}"
        if provider_prefixed in self._voice_profiles:
            return provider_prefixed

        for voice_id, profile in self._voice_profiles.items():
            short_id = voice_id.split(":", 1)[-1].lower()
            display_name = str(profile["name"]).lower()
            if normalized_requested in {short_id, display_name}:
                return voice_id
        return _DEFAULT_VOICE

    def _word_timing(
        self,
        text: str,
        *,
        char_starts: list[float],
        total_frames: int,
        sample_rate: int,
    ) -> list[dict[str, float | str]]:
        tokens: list[tuple[str, int]] = []
        for match in _WORD_TOKEN_PATTERN.finditer(text):
            token_text = match.group(0)
            for relative_index, character in enumerate(token_text):
                if character.isalnum():
                    tokens.append((token_text, match.start() + relative_index))
                    break

        if not tokens:
            return []

        total_seconds = total_frames / float(sample_rate) if sample_rate > 0 else 0.0
        timing: list[dict[str, float | str]] = []
        for index, (token_text, start_char_index) in enumerate(tokens):
            start_seconds = char_starts[start_char_index]
            if index + 1 < len(tokens):
                end_seconds = char_starts[tokens[index + 1][1]]
            else:
                end_seconds = total_seconds
            timing.append(
                {
                    "text": token_text,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                }
            )
        return timing

    def _split_syllable_chunks(self, word: str) -> list[str]:
        """Split a word into simple syllable-like chunks."""
        if len(word) <= 3:
            return [word]

        vowel_groups = list(_VOWEL_PATTERN.finditer(word.lower()))
        if len(vowel_groups) <= 1:
            return [word]

        chunks: list[str] = []
        start = 0
        for group_index, vowel_group in enumerate(vowel_groups):
            boundary = vowel_group.end() if group_index + 1 < len(vowel_groups) else len(word)
            if boundary <= start:
                continue
            chunk = word[start:boundary]
            if chunk:
                chunks.append(chunk)
            start = boundary

        if start < len(word):
            chunks.append(word[start:])

        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2] += chunks.pop()

        return chunks or [word]

    def _render_syllable(
        self,
        syllable: str,
        *,
        profile: dict[str, float | str],
        sample_rate: int,
        rng: np.random.Generator,
        voice_value: str,
        word_index: int,
        syllable_index: int,
        syllable_total: int,
        contour_pitch: float,
        contour_duration: float,
    ) -> np.ndarray:
        syllable_lower = syllable.lower()
        syllable_value = sum(ord(ch) for ch in syllable_lower if ch.isalnum())
        vowel_count = sum(1 for ch in syllable_lower if ch in "aeiouy")
        consonant_count = sum(1 for ch in syllable_lower if ch.isalnum()) - vowel_count
        syllable_span = max(1, syllable_total)
        local_contour = syllable_index - ((syllable_span - 1) / 2.0)

        duration_seconds = 0.038 + 0.0075 * len(syllable) + 0.006 * max(1, vowel_count)
        duration_seconds /= max(self._speed(), 0.25)
        duration_seconds /= float(profile["speed"])
        duration_seconds *= 1.0 + 0.03 * min(4, syllable_total - 1)
        duration_seconds *= 1.0 + min(contour_duration, 0.2)
        duration_seconds *= 1.0 + 0.012 * ((syllable_value + word_index + len(voice_value)) % 5)
        frame_count = max(80, int(sample_rate * duration_seconds))

        t = np.arange(frame_count, dtype=np.float32) / sample_rate
        seed_offset = syllable_value + len(syllable) * 31 + word_index * 97 + syllable_index * 53
        pitch_step = ((seed_offset * 7) % 19) - 9
        pitch_step += local_contour * 1.2
        pitch_step += contour_pitch
        frequency = float(profile["base"]) * float(profile["scale"])
        frequency *= 2 ** (pitch_step / 24.0)

        carrier = np.sin(2 * np.pi * frequency * t)
        harmonic = 0.38 * np.sin(2 * np.pi * frequency * 2.01 * t)
        buzz = 0.18 * np.sign(np.sin(2 * np.pi * frequency * 0.5 * t))
        noise_scale = float(profile["noise"])
        noise_scale *= 0.82 + 0.06 * ((seed_offset // 5) % 5)
        noise_scale *= 1.0 + 0.03 * max(0, consonant_count - vowel_count)
        noise = noise_scale * rng.normal(0.0, 1.0, frame_count)

        envelope_source = np.sin(np.linspace(0.0, np.pi, frame_count, dtype=np.float32))
        envelope_power = 0.72 + 0.05 * ((seed_offset + syllable_index) % 5)
        if syllable_index == 0:
            envelope_power -= 0.04
        if syllable_index + 1 == syllable_total:
            envelope_power += 0.03
        envelope = np.maximum(envelope_source, 0.0) ** envelope_power

        amplitude = 0.20 + 0.015 * ((seed_offset // 3) % 7)
        amplitude *= 1.0 + 0.02 * local_contour
        amplitude *= 1.0 + min(contour_duration, 0.14)
        waveform = amplitude * envelope * (carrier + harmonic + buzz + noise)
        return waveform.astype(np.float32)

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        self.calls.append((text, voice))
        selected_voice = self._resolve_voice(voice) if voice else self.default_voice() or _DEFAULT_VOICE
        profile = self._voice_profiles.get(selected_voice, self._voice_profiles[_DEFAULT_VOICE])
        sample_rate = 16_000
        chunks: list[np.ndarray] = []
        char_starts: list[float] = []
        frame_cursor = 0
        rng = np.random.default_rng(sum(ord(ch) for ch in f"{selected_voice}|{text}"))
        contour_pitch = 0.0
        contour_duration = 0.0
        word_index = 0

        def append_silence(seconds: float) -> None:
            nonlocal frame_cursor
            pause_frames = int(sample_rate * seconds)
            if pause_frames <= 0:
                return
            chunks.append(np.zeros(pause_frames, dtype=np.float32))
            frame_cursor += pause_frames

        index = 0
        while index < len(text):
            char = text[index]
            if char.isalnum():
                run_end = index + 1
                while run_end < len(text) and text[run_end].isalnum():
                    run_end += 1
                for _ in range(index, run_end):
                    char_starts.append(frame_cursor / float(sample_rate))

                word_text = text[index:run_end]
                syllables = self._split_syllable_chunks(word_text)
                syllable_total = len(syllables)
                for syllable_index, syllable in enumerate(syllables):
                    syllable_samples = self._render_syllable(
                        syllable,
                        profile=profile,
                        sample_rate=sample_rate,
                        rng=rng,
                        voice_value=selected_voice,
                        word_index=word_index,
                        syllable_index=syllable_index,
                        syllable_total=syllable_total,
                        contour_pitch=contour_pitch,
                        contour_duration=contour_duration,
                    )
                    chunks.append(syllable_samples)
                    frame_cursor += syllable_samples.shape[0]

                    if syllable_index + 1 < syllable_total:
                        gap_seconds = 0.009 + 0.0015 * min(len(syllable), 5)
                        append_silence(gap_seconds)

                    contour_pitch *= 0.82
                    contour_duration *= 0.86

                word_index += 1
                index = run_end
                continue

            char_starts.append(frame_cursor / float(sample_rate))
            lower_character = char.lower()
            if char.isspace():
                append_silence(0.075)
                index += 1
                continue
            if lower_character in _MID_PUNCTUATION:
                append_silence(0.14)
                contour_pitch = max(contour_pitch - 0.75, -4.0)
                contour_duration = min(contour_duration + 0.07, 0.22)
                index += 1
                continue
            if lower_character in _SENTENCE_PUNCTUATION:
                append_silence(0.24)
                contour_pitch = max(contour_pitch - 1.45, -5.0)
                contour_duration = min(contour_duration + 0.12, 0.30)
                index += 1
                continue
            if not char.isalnum():
                index += 1
                continue

            index += 1

        if not chunks:
            silence_frames = int(sample_rate * 0.2)
            chunks.append(np.zeros(silence_frames, dtype=np.float32))
            frame_cursor += silence_frames

        samples = np.concatenate(chunks)
        word_timing = self._word_timing(
            text,
            char_starts=char_starts,
            total_frames=frame_cursor,
            sample_rate=sample_rate,
        )
        return SynthesizedAudio(
            samples=np.clip(samples, -0.9, 0.9).reshape(-1, 1),
            sample_rate=sample_rate,
            channels=1,
            provider=self.name,
            voice=selected_voice,
            text=text,
            metadata={
                "demo": True,
                "style": "retro talking blips",
                "voice_profile": profile["name"],
                "normalized_voice_key": self.cache_voice_key(selected_voice),
                "cache_settings": self.cache_settings(),
                "word_timing": word_timing,
            },
        )
