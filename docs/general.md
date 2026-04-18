# General architecture

VoiceConductor is organized around one high-level object, a small set of settings dataclasses, pluggable provider adapters, and shared audio/playback types. Most applications only need `TTSManager`, but the lower-level pieces are exported so host apps can customize provider selection, audio routing, and provider registration.

## TTSManager

`TTSManager` is the main public entry point. It coordinates four jobs:

- Loading `Settings` when no settings object is provided.
- Building registered TTS providers from the provider registry.
- Synthesizing text into normalized `SynthesizedAudio`.
- Routing synthesized audio to named playback routes such as `speakers` and `mic`.

The common path is:

```python
from voice_conductor import TTSManager

tts = TTSManager()
tts.speak("Hello from VoiceConductor.", routes=["speakers", "mic"])
```

`speak()` synthesizes and plays in one call. `synthesize_voice()` produces a `SynthesizedAudio` object without playing it, and `route()` plays an existing audio object:

```python
audio = tts.synthesize_voice("This clip can be reused.", provider="windows")
result = tts.route(audio, routes="speakers")
```

Provider selection follows the configured fallback chain. If a call does not pass `provider=...`, the manager tries the configured providers in order and uses the first available one. Cache lookups are keyed by the text, provider, normalized voice key, and provider settings that affect the generated audio.

Useful manager methods:

- `list_providers()` returns registered providers that are currently available.
- `list_voices(provider)` returns `VoiceInfo` records from a provider.
- `list_output_devices()` returns output-capable `AudioDevice` records.
- `refresh_audio_devices()` re-resolves configured routes against current devices.
- `invalidate_synthesis_cache(...)` removes matching phrase-cache entries.
- `clear_synthesis_cache()` clears every cached phrase.

## Settings

`Settings` is the complete configuration object consumed by `TTSManager`. It has two main sections:

- `settings.voice_conductor`: package-level behavior such as provider order, route config, and cache paths.
- `settings.providers`: provider-specific credentials, voices, model choices, speed settings, and custom provider config.

By default, `load_settings()` looks for `voice_conductor.config.jsonc` or `voice_conductor.config.json` in the current working directory. The file uses the same nested shape as the dataclasses:

```jsonc
{
  "voice_conductor": {
    "default_provider": null,
    "provider_chain": ["elevenlabs", "kokoro", "azure", "windows"],
    "route_config": {
      "routes": {
        "speakers": {
          "device": null,
          "prefer_virtual_cable": false
        },
        "mic": {
          "device": null,
          "prefer_virtual_cable": true
        }
      }
    },
    "cache": {
      "path": null,
      "api_dir": null,
      "ttl_seconds": null
    }
  },
  "providers": {
    "windows": {
      "default_voice": null,
      "volume": null,
      "speed": 1.0
    }
  }
}
```

`voice_conductor.provider_chain` is the normal way to choose fallback order. `voice_conductor.default_provider` can force one preferred provider to the front of the chain.

`voice_conductor.route_config.routes` maps route names to audio device preferences. The built-in route names are:

- `speakers`: defaults to the host default output device.
- `mic`: prefers a virtual cable output, because chat apps receive that cable as a microphone input.

`voice_conductor.cache` controls two cache locations: the SQLite phrase cache used for generated audio and the provider metadata cache used for things like voice lists. `ttl_seconds` applies to provider metadata cache entries.

Settings can also be constructed directly:

```python
from voice_conductor import Settings, TTSManager, VoiceConductorSettings

settings = Settings(
    voice_conductor=VoiceConductorSettings(provider_chain=["demo", "windows"]),
)

tts = TTSManager(settings=settings)
```

## Providers

Providers implement the `TTSProvider` interface. A provider is responsible for checking whether it can run, reporting available voices, and turning text into `SynthesizedAudio`.

Built-in providers:

| Provider | Purpose | Typical requirement |
| --- | --- | --- |
| `elevenlabs` | Hosted ElevenLabs voices. | API key. |
| `kokoro` | Local Kokoro synthesis. | Kokoro extra/model access. |
| `azure` | Azure Speech neural voices. | Speech key and region. |
| `windows` | Installed Windows System.Speech voices. | Windows speech support. |
| `demo` | Dependency-free test provider. | None. |

The provider contract is intentionally small:

```python
class TTSProvider:
    name: str

    def is_available(self) -> bool: ...
    def default_voice(self) -> str | None: ...
    def cache_settings(self) -> str | dict[str, Any] | None: ...
    def cache_voice_key(self, voice: str | None) -> str: ...
    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio: ...
    def list_voices(self, settings: Settings | None = None) -> list[VoiceInfo]: ...
```

`cache_settings()` should include provider options that change the generated audio, such as model, speed, language, endpoint, or output format. Returning a dict is recommended because Voice Conductor serializes it deterministically before hashing. `cache_voice_key()` normalizes voice identifiers so the phrase cache can reuse entries even when users pass equivalent voice names.

## Custom Providers

Custom providers are registered in the process-wide `ProviderRegistry`. The factory receives the parsed `Settings` object and returns a `TTSProvider`.

```python
from dataclasses import dataclass

import numpy as np

from voice_conductor import (
    Settings,
    SynthesizedAudio,
    TTSManager,
    VoiceInfo,
    register_provider,
    register_provider_config,
    settings_from_dict,
)
from voice_conductor.providers import TTSProvider


@dataclass(slots=True)
class LocalProviderSettings:
    endpoint: str
    speed: float = 1.0


class LocalProvider(TTSProvider):
    name = "local"

    def __init__(self, settings: Settings) -> None:
        config = settings.provider_settings(self.name)
        assert isinstance(config, LocalProviderSettings)
        self.endpoint = config.endpoint
        self.speed = config.speed

    def is_available(self) -> bool:
        return True

    def default_voice(self) -> str | None:
        return "default"

    def cache_settings(self) -> dict[str, str | float]:
        return {"endpoint": self.endpoint, "speed": self.speed}

    def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        return SynthesizedAudio(
            samples=np.zeros((16000, 1), dtype=np.float32),
            sample_rate=16000,
            channels=1,
            provider=self.name,
            voice=voice or self.default_voice(),
            text=text,
        )

    def list_voices(self, settings: Settings | None = None) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="Default", provider=self.name)]


register_provider_config("local", LocalProviderSettings)
register_provider("local", LocalProvider)

settings = settings_from_dict(
    {
        "voice_conductor": {"provider_chain": ["local"]},
        "providers": {
            "local": {
                "endpoint": "http://localhost:7000",
                "speed": 1.25,
            }
        },
    }
)

tts = TTSManager(settings=settings)
tts.speak("Local provider is ready.", routes="speakers")
```

Use `register_provider_config()` when a custom provider has typed settings that should be parsed from the `providers.<name>` config block and round-tripped by `settings_to_dict()` or `Settings.save_settings()`.

## Shared Types

### SynthesizedAudio

`SynthesizedAudio` is the normalized audio object used by providers, caches, and playback routing. Providers should return this type rather than provider-native audio bytes.

Important fields:

- `samples`: a NumPy float32 array clipped to `[-1.0, 1.0]`.
- `sample_rate`: sample rate in Hz.
- `channels`: channel count; 1D samples are reshaped into mono.
- `provider`: provider name that generated the clip.
- `voice`: resolved voice id or name.
- `text`: original text, when available.
- `metadata`: provider-specific details such as normalized voice keys.

Useful helpers:

- `duration_seconds` and `frame_count` describe clip length.
- `to_pcm16_bytes()` encodes raw signed 16-bit PCM.
- `to_wav_bytes()` encodes a WAV payload.
- `copy_to(path, format="wav")` writes WAV or PCM16 to disk.
- `from_wav_bytes(...)` and `from_pcm16_bytes(...)` normalize provider-native bytes.

### AudioDevice

`AudioDevice` describes an output-capable host audio device discovered through the audio layer. Even the `mic` route resolves to an output device, because virtual cables expose a playback endpoint that other applications can use as microphone input.

Important fields:

- `id`: host audio device index.
- `name`: device display name.
- `hostapi` / `hostapi_name`: host API name.
- `max_output_channels`: number of output channels.
- `default_samplerate`: default sample rate reported by the host.
- `is_default`: whether this is the system default output.
- `is_virtual_cable`: whether the name looks like a virtual cable or VoiceMeeter endpoint.
- `raw`: original device metadata.

### Hook Events

`PlaybackHooks` lets callers observe playback lifecycle events:

```python
from voice_conductor import PlaybackHooks

hooks = PlaybackHooks(
    on_audio_ready=lambda event: press_push_to_talk(),
    on_playback_complete=lambda event: release_push_to_talk(),
)

tts.speak("Playback is starting.", routes="mic", hooks=hooks)
```

`on_audio_ready` receives a `PlaybackReadyEvent` after route names have been resolved to devices and before playback starts.

`on_playback_complete` receives a `PlaybackCompleteEvent` after playback succeeds or fails. On success, `event.result` contains a `PlaybackResult`. On failure, `event.error` contains the exception.

Both events include:

- `routes`: normalized route names.
- `audio`: the `SynthesizedAudio` being played.
- `devices`: mapping of route name to resolved `AudioDevice`.

### PlaybackTask

When `background=True`, `speak()` and `route()` return a `PlaybackTask[PlaybackResult]` instead of a direct `PlaybackResult`.

```python
task = tts.speak("Non-blocking call.", routes="speakers", background=True)

if not task.done():
    result = task.result(timeout=10)
```

`PlaybackTask` is a small wrapper around a future. It exposes `done()`, `result(timeout=None)`, `exception(timeout=None)`, and `cancel()`.
