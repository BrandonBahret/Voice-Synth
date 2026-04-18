"""Configuration loading and coercion for voice_conductor.

Settings come from defaults plus an optional ``voice_conductor.config.jsonc`` file
in the current working directory. The public dataclasses use the nested config
shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
import os
from pathlib import Path
from typing import Any

from pypercache.query import JsonInjester

from .audio.router import AudioRoute, RouteConfig
from .config_registry import ProviderConfigRegistry, register_provider_config
from .voice_keys import normalize_voice_config_value, normalize_voice_key

_CONFIG_FILENAMES = ("voice_conductor.config.jsonc", "voice_conductor.config.json")
_TOP_LEVEL_KEYS = {"voice_conductor", "providers"}
_BUILT_IN_PROVIDERS = {"elevenlabs", "azure", "kokoro", "windows", "demo"}
_DEFAULT_PROVIDER_CHAIN = ["elevenlabs", "kokoro", "azure", "windows"]
_PRESERVED_PROVIDER_FIELDS = {
    "elevenlabs": {"api_key"},
    "azure": {"speech_key", "region"},
    "kokoro": {"hf_token"},
}
_IMPLICIT_DEFAULT_VOICES = {
    "elevenlabs": "JBFqnCBsd6RMkjVDRZzb",
    "azure": "en-US-AvaNeural",
    "kokoro": "af_heart",
    "demo": "animalese",
}


def _cache_root(root: str | Path | None = None) -> Path:
    if root is None:
        return Path.cwd()
    return Path(root)


def _default_cache_root() -> str:
    return str(_cache_root())


def _default_cache_path(root: str | Path | None = None) -> str:
    return str(_cache_root(root) / "voice_conductor_cache.db")


def _default_api_cache_dir(root: str | Path | None = None) -> str:
    return str(_cache_root(root) / "api-caches")


def _normalize_provider_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("provider name must not be empty")
    return normalized


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _coerce_string(value: Any, field_name: str) -> str | None:
    value = _normalize_config_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_windows_config_path(value: str | os.PathLike[str]) -> str:
    """Return a user-friendly Windows path by dropping long-path device prefixes."""

    value = os.fspath(value)
    while value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[len("\\\\?\\UNC\\") :]
    while value.startswith("\\\\?\\"):
        value = value[len("\\\\?\\") :]
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        value = value[0].upper() + value[1:]
    return value


def _absolute_config_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return (Path.cwd() / target).resolve(strict=False)


def _path_is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
    except ValueError:
        return False
    return True


def _config_relative_path(
    value: str | os.PathLike[str] | None,
    base_dir: Path,
) -> str | None:
    if value is None:
        return None

    raw_path = Path(value).expanduser()
    should_relativize = not raw_path.is_absolute()
    if raw_path.is_absolute():
        absolute_path = raw_path.resolve(strict=False)
        should_relativize = _path_is_relative_to(absolute_path, base_dir)
    else:
        absolute_path = (Path.cwd() / raw_path).resolve(strict=False)

    if not should_relativize:
        return _normalize_windows_config_path(absolute_path)

    try:
        relative_path = os.path.relpath(absolute_path, base_dir)
    except ValueError:
        return _normalize_windows_config_path(absolute_path)
    return _normalize_windows_config_path(relative_path)


def _resolve_config_cache_path(value: str | os.PathLike[str], base_dir: Path) -> str:
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return _normalize_windows_config_path(raw_path.resolve(strict=False))

    legacy_path = _legacy_cwd_relative_cache_path(raw_path, base_dir)
    if legacy_path is not None:
        return _normalize_windows_config_path(legacy_path)

    return _normalize_windows_config_path(base_dir / raw_path)


def _legacy_cwd_relative_cache_path(raw_path: Path, base_dir: Path) -> Path | None:
    """Collapse cache paths written relative to cwd by older nested config saves."""

    cwd = Path.cwd().resolve(strict=False)
    base_dir = base_dir.resolve(strict=False)
    try:
        base_parts = base_dir.relative_to(cwd).parts
    except ValueError:
        return None
    if not base_parts:
        return None

    raw_parts = raw_path.parts
    if raw_parts[: len(base_parts)] != base_parts:
        return None

    while raw_parts[: len(base_parts)] == base_parts:
        raw_parts = raw_parts[len(base_parts) :]

    if not raw_parts:
        return base_dir
    return base_dir.joinpath(*raw_parts)


def _coerce_int(value: Any, field_name: str) -> int | None:
    value = _normalize_config_value(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc


def _coerce_float(value: Any, field_name: str) -> float | None:
    value = _normalize_config_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}.") from exc


def _coerce_bool(value: Any, field_name: str) -> bool | None:
    value = _normalize_config_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}.")


def _parse_provider_chain(value: Any) -> list[str]:
    value = _normalize_config_value(value)
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, list):
        candidates = value
    else:
        raise ValueError(
            "voice_conductor.provider_chain must be a list or comma-separated string, "
            f"got {value!r}."
        )
    return [str(item).strip().lower() for item in candidates if str(item).strip()]


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def _validate_top_level_keys(payload: dict[str, Any]) -> None:
    unsupported = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if not unsupported:
        return
    keys = ", ".join(unsupported)
    raise ValueError(
        "Config must use nested JSON with only 'voice_conductor' and 'providers' "
        f"top-level keys. Unsupported top-level keys: {keys}."
    )


def _strip_jsonc_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                result.append("\n" if text[index] in "\r\n" else " ")
                index += 1
            index += 2 if index + 1 < len(text) else 0
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _strip_jsonc_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        result.append(char)
        index += 1

    return "".join(result)


def _loads_config_text(text: str, path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonc":
        text = _strip_jsonc_trailing_commas(_strip_jsonc_comments(text))
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a JSON object.")
    return payload


def _load_file_payload() -> dict[str, Any]:
    path = _find_config_file()
    if path is None:
        return {}
    payload = _loads_config_text(path.read_text(encoding="utf-8"), path)
    _validate_top_level_keys(payload)
    return payload


def _find_config_file() -> Path | None:
    for filename in _CONFIG_FILENAMES:
        path = Path.cwd() / filename
        if path.exists():
            return path
    return None


def _load_settings_file_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = _loads_config_text(target.read_text(encoding="utf-8"), target)
    _validate_top_level_keys(payload)
    return payload


@dataclass(slots=True)
class CacheSettings:
    """Filesystem locations and optional TTLs for phrase and provider caches."""

    path: str | None = field(
        default=None,
        metadata={
            "doc": "SQLite phrase-cache file used to reuse synthesized audio; defaults under root."
        },
    )
    api_dir: str | None = field(
        default=None,
        metadata={
            "doc": "Directory for provider metadata caches such as voice and model lists; defaults under root."
        },
    )
    ttl_seconds: int | None = field(
        default=None,
        metadata={
            "doc": "Optional provider metadata cache lifetime in seconds; None keeps entries until manually invalidated."
        },
    )
    root: str | None = field(
        default=None,
        metadata={
            "doc": "Base directory used to derive default phrase and provider cache locations."
        },
    )

    def __post_init__(self) -> None:
        root = _normalize_windows_config_path(self.root or _default_cache_root())
        self.root = root
        self.path = _normalize_windows_config_path(self.path or _default_cache_path(root))
        self.api_dir = _normalize_windows_config_path(self.api_dir or _default_api_cache_dir(root))


@dataclass(slots=True)
class VoiceConductorSettings:
    """Top-level VoiceConductor settings for provider selection, routing, and caching."""

    default_provider: str | None = field(
        default=None,
        metadata={
            "doc": "Single preferred provider name used when provider_chain is not configured."
        },
    )
    provider_chain: list[str] = field(
        default_factory=lambda: list(_DEFAULT_PROVIDER_CHAIN),
        metadata={
            "doc": "Ordered provider names to try for synthesis before falling back to availability-based selection."
        },
    )
    route_config: RouteConfig = field(
        default_factory=RouteConfig,
        metadata={"doc": "Default audio device targets used for speaker and virtual-mic routing."},
    )
    cache: CacheSettings = field(
        default_factory=CacheSettings,
        metadata={"doc": "Phrase-cache and provider-metadata cache settings."},
    )

@dataclass(slots=True)
class ElevenLabsSettings:
    """Credentials and request defaults for the ElevenLabs provider."""

    api_key: str | None = field(
        default=None,
        metadata={"doc": "ElevenLabs API key sent as the xi-api-key request header."},
    )
    default_voice: str = field(
        default="JBFqnCBsd6RMkjVDRZzb",
        metadata={"doc": "ElevenLabs voice name or voice id used when synthesis does not specify one."},
    )
    model_id: str = field(
        default="eleven_multilingual_v2",
        metadata={"doc": "ElevenLabs text-to-speech model id used in synthesis requests."},
    )
    output_format: str = field(
        default="pcm_24000",
        metadata={
            "doc": "ElevenLabs generated-audio format string, such as codec_sample-rate_bitrate or a pcm_* format."
        },
    )
    speed: float = field(
        default=1.0,
        metadata={"doc": "ElevenLabs voice-settings speed multiplier; 1.0 means normal speed."},
    )
    language_code: str | None = field(
        default=None,
        metadata={"doc": "Optional ElevenLabs language code included in synthesis requests."},
    )
    stability: float | None = field(
        default=None,
        metadata={
            "doc": "Optional ElevenLabs voice-setting override controlling stability and generation randomness."
        },
    )
    similarity_boost: float | None = field(
        default=None,
        metadata={
            "doc": "Optional ElevenLabs voice-setting override controlling how closely output adheres to the original voice."
        },
    )
    style: float | None = field(
        default=None,
        metadata={
            "doc": "Optional ElevenLabs voice-setting override that exaggerates the source voice style."
        },
    )
    speaker_boost: bool | None = field(
        default=None,
        metadata={
            "doc": "Optional ElevenLabs voice-setting override for use_speaker_boost, trading latency for speaker similarity."
        },
    )


@dataclass(slots=True)
class AzureSettings:
    """Credentials and default voice for Azure Speech."""

    speech_key: str | None = field(
        default=None,
        metadata={"doc": "Azure Speech resource key sent as the Ocp-Apim-Subscription-Key header."},
    )
    region: str | None = field(
        default=None,
        metadata={"doc": "Azure Speech resource region used to build text-to-speech REST endpoints."},
    )
    default_voice: str = field(
        default="en-US-AvaNeural",
        metadata={"doc": "Azure neural voice name used in the SSML voice element by default."},
    )
    speed: float = field(
        default=1.0,
        metadata={"doc": "Azure SSML prosody rate multiplier clamped to the supported speech range."},
    )
    language_code: str | None = field(
        default=None,
        metadata={"doc": "Optional Azure SSML xml:lang code; None uses en-US."},
    )


@dataclass(slots=True)
class KokoroSettings:
    """Local Kokoro provider settings, including Hugging Face auth."""

    hf_token: str | None = field(
        default=None,
        metadata={"doc": "Hugging Face token required before loading Kokoro model assets."},
    )
    default_voice: str = field(
        default="af_heart",
        metadata={"doc": "Kokoro voice preset or voice tensor name used when synthesis does not specify one."},
    )
    language_code: str = field(
        default="a",
        metadata={"doc": "Kokoro KPipeline language code; it should match the configured Kokoro voice."},
    )
    speed: float = field(
        default=1.0,
        metadata={"doc": "Kokoro synthesis speed multiplier passed to the local pipeline."},
    )


@dataclass(slots=True)
class WindowsSettings:
    """Settings for the Windows System.Speech provider."""

    default_voice: str | None = field(
        default=None,
        metadata={
            "doc": "Provider-local Windows System.Speech voice id selected before speaking."
        },
    )
    volume: int | None = field(
        default=100,
        metadata={
            "doc": "Windows SpeechSynthesizer output volume from 0 through 100; None uses 100."
        },
    )
    speed: float = field(
        default=1.0,
        metadata={"doc": "Windows speech rate multiplier mapped onto the System.Speech -10 through 10 rate."},
    )


@dataclass(slots=True)
class DemoProviderSettings:
    """Settings for the dependency-free demo provider."""

    default_voice: str = field(
        default="animalese",
        metadata={
            "doc": "Demo provider-local voice id used when synthesis does not specify one."
        },
    )
    speed: float = field(
        default=1.0,
        metadata={"doc": "Demo synthesis speed multiplier; 1.0 means normal speed."},
    )


@dataclass(slots=True)
class ProviderSettings:
    """Provider-specific settings plus registered custom provider payloads."""

    elevenlabs: ElevenLabsSettings = field(
        default_factory=ElevenLabsSettings,
        metadata={"doc": "ElevenLabs provider credentials, voice, model, format, and voice-setting overrides."},
    )
    azure: AzureSettings = field(
        default_factory=AzureSettings,
        metadata={"doc": "Azure Speech provider credentials and default neural voice."},
    )
    kokoro: KokoroSettings = field(
        default_factory=KokoroSettings,
        metadata={"doc": "Kokoro local-provider auth, voice, and language settings."},
    )
    windows: WindowsSettings = field(
        default_factory=WindowsSettings,
        metadata={"doc": "Windows System.Speech provider voice and volume settings."},
    )
    demo: DemoProviderSettings = field(
        default_factory=DemoProviderSettings,
        metadata={"doc": "Dependency-free demo provider voice and speed settings."},
    )
    extra: dict[str, Any] = field(
        default_factory=dict,
        metadata={"doc": "Typed config objects for explicitly registered custom provider names."},
    )


register_provider_config("elevenlabs", ElevenLabsSettings)
register_provider_config("azure", AzureSettings)
register_provider_config("kokoro", KokoroSettings)
register_provider_config("windows", WindowsSettings)
register_provider_config("demo", DemoProviderSettings)


@dataclass(slots=True)
class Settings:
    """Complete VoiceConductor configuration object consumed by ``TTSManager``."""

    voice_conductor: VoiceConductorSettings = field(
        default_factory=VoiceConductorSettings,
        metadata={"doc": "Package-level provider selection, routing, and cache settings."},
    )
    providers: ProviderSettings = field(
        default_factory=ProviderSettings,
        metadata={"doc": "Built-in and custom provider-specific configuration."},
    )

    @staticmethod
    def from_file(path: str | Path) -> "Settings":
        """Load settings from a file, writing defaults first when it is missing."""

        target = _absolute_config_path(path)
        if target.exists():
            return _settings_from_config_file(target)

        root_dir = target.parent
        settings = Settings(
            voice_conductor=VoiceConductorSettings(
                cache=CacheSettings(root=root_dir)
            )
        )        
        settings.save_settings(target)
        return settings

    def __post_init__(self) -> None:
        default_route_names = {"speakers", "mic"}
        has_default_routes = set(self.voice_conductor.route_config.routes) == default_route_names
        routes_are_unset = all(
            route.device is None for route in self.voice_conductor.route_config.routes.values()
        )
        if has_default_routes and routes_are_unset:
            self.voice_conductor.route_config = RouteConfig().resolve_missing_devices()
            

    def provider_settings(self, name: str) -> Any:
        """Return typed provider settings for built-ins or registered custom config."""

        normalized = _normalize_provider_name(name)
        if normalized in _BUILT_IN_PROVIDERS:
            return getattr(self.providers, normalized)
        if normalized in self.providers.extra:
            return self.providers.extra[normalized]
        if ProviderConfigRegistry().is_registered(normalized):
            return None
        raise ValueError(f"Unknown provider config: {normalized}. Register it before use.")

    def provider_config(self, name: str) -> dict[str, Any]:
        """Return a provider config dict for built-in or registered custom names."""

        normalized = _normalize_provider_name(name)
        if normalized not in _BUILT_IN_PROVIDERS and not ProviderConfigRegistry().is_registered(
            normalized
        ):
            raise ValueError(f"Unknown provider config: {normalized}. Register it before use.")
        payload = settings_to_dict(self)["providers"]
        return dict(payload.get(normalized, {}))

    def save_settings(self, path: str | Path = "voice_conductor.config.jsonc") -> Path:
        """Write settings to JSON and return the destination path."""

        target = _absolute_config_path(path)
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        payload = settings_to_dict(self)
        _make_cache_paths_config_relative(payload, target.parent)
        _preserve_existing_provider_fields(payload, target)
        _resolve_provider_default_voices(payload, self, target)
        target.write_text(_dumps_settings_config(payload, self, target), encoding="utf-8")
        if target.suffix.lower() == ".jsonc":
            _write_config_help(self, target.with_name("config-help.md"))
        return target


def _read_string(
    payload: dict[str, Any],
    key: str,
    field_name: str,
    default: str | None = None,
) -> str | None:
    if key not in payload:
        return default
    return _coerce_string(payload[key], field_name) or default


def _read_default_voice(
    provider: str,
    payload: dict[str, Any],
    key: str,
    field_name: str,
    default: str | None = None,
) -> str | None:
    return normalize_voice_config_value(
        provider,
        _read_string(payload, key, field_name, default),
    )


def _read_float(payload: dict[str, Any], key: str, field_name: str) -> float | None:
    if key not in payload:
        return None
    return _coerce_float(payload[key], field_name)


def _read_float_default(
    payload: dict[str, Any],
    key: str,
    field_name: str,
    default: float,
) -> float:
    value = _read_float(payload, key, field_name)
    if value is None:
        return default
    return value


def _parse_voice_conductor(payload: dict[str, Any]) -> VoiceConductorSettings:
    voice_conductor = _require_mapping(payload.get("voice_conductor"), "voice_conductor")
    cache = _require_mapping(voice_conductor.get("cache"), "voice_conductor.cache")
    cache_root = _coerce_string(cache.get("root"), "voice_conductor.cache.root")

    return VoiceConductorSettings(
        default_provider=_coerce_string(
            voice_conductor.get("default_provider"),
            "voice_conductor.default_provider",
        ),
        provider_chain=_parse_provider_chain(
            voice_conductor.get("provider_chain", _DEFAULT_PROVIDER_CHAIN)
        ),
        route_config=_parse_route_config(voice_conductor),
        cache=CacheSettings(
            path=_coerce_string(cache.get("path"), "voice_conductor.cache.path"),
            api_dir=_coerce_string(cache.get("api_dir"), "voice_conductor.cache.api_dir"),
            ttl_seconds=_coerce_int(cache.get("ttl_seconds"), "voice_conductor.cache.ttl_seconds"),
            root=_normalize_windows_config_path(cache_root) if cache_root is not None else None,
        ),
    )


# def _parse_route_config(payload: dict[str, Any], voice_conductor: VoiceConductorSettings) -> RouteConfig:
def _parse_route_config(payload: dict[str, Any]) -> RouteConfig:
    if "route_config" not in payload:
        return RouteConfig()

    route_config = _require_mapping(payload.get("route_config"), "route_config")
    if "routes" not in route_config:
        return RouteConfig()

    routes_payload = _require_mapping(route_config.get("routes"), "route_config.routes")
    routes: dict[str, AudioRoute] = {}
    for name, route_payload in routes_payload.items():
        field_name = f"route_config.routes.{name}"
        route = _require_mapping(route_payload, field_name)
        normalized = str(name).strip().lower()
        routes[normalized] = AudioRoute(
            name=normalized,
            device=_normalize_config_value(route.get("device")),
            prefer_virtual_cable=_coerce_bool(
                route.get("prefer_virtual_cable"),
                f"{field_name}.prefer_virtual_cable",
            )
            or False,
        )
    return RouteConfig(routes=routes)


def _parse_registered_provider_config(name: str, payload: dict[str, Any]) -> Any:
    registry = ProviderConfigRegistry()
    normalized = _normalize_provider_name(name)
    if normalized in _BUILT_IN_PROVIDERS:
        raise ValueError(f"{normalized} is a built-in provider config.")
    if not registry.is_registered(normalized):
        raise ValueError(f"Unknown provider config: {normalized}. Register it before loading.")
    return registry.parse(normalized, payload)


def _parse_providers(payload: dict[str, Any]) -> ProviderSettings:
    providers = _require_mapping(payload.get("providers"), "providers")
    elevenlabs = _require_mapping(providers.get("elevenlabs"), "providers.elevenlabs")
    azure = _require_mapping(providers.get("azure"), "providers.azure")
    kokoro = _require_mapping(providers.get("kokoro"), "providers.kokoro")
    windows = _require_mapping(providers.get("windows"), "providers.windows")
    demo = _require_mapping(providers.get("demo"), "providers.demo")
    extra = {
        _normalize_provider_name(name): _parse_registered_provider_config(
            name,
            _require_mapping(config, f"providers.{_normalize_provider_name(name)}"),
        )
        for name, config in providers.items()
        if _normalize_provider_name(name) not in _BUILT_IN_PROVIDERS
    }


    return ProviderSettings(
        elevenlabs=ElevenLabsSettings(
            api_key=_coerce_string(elevenlabs.get("api_key"), "providers.elevenlabs.api_key"),
            default_voice=_read_default_voice(
                "elevenlabs",
                elevenlabs,
                "default_voice",
                "providers.elevenlabs.default_voice",
                "JBFqnCBsd6RMkjVDRZzb",
            )
            or "JBFqnCBsd6RMkjVDRZzb",
            model_id=_read_string(
                elevenlabs,
                "model_id",
                "providers.elevenlabs.model_id",
                "eleven_multilingual_v2",
            )
            or "eleven_multilingual_v2",
            output_format=_read_string(
                elevenlabs,
                "output_format",
                "providers.elevenlabs.output_format",
                "pcm_24000",
            )
            or "pcm_24000",
            speed=_read_float_default(
                elevenlabs,
                "speed",
                "providers.elevenlabs.speed",
                1.0,
            ),
            language_code=_coerce_string(
                elevenlabs.get("language_code"),
                "providers.elevenlabs.language_code",
            ),
            stability=_coerce_float(
                elevenlabs.get("stability"),
                "providers.elevenlabs.stability",
            ),
            similarity_boost=_coerce_float(
                elevenlabs.get("similarity_boost"),
                "providers.elevenlabs.similarity_boost",
            ),
            style=_coerce_float(elevenlabs.get("style"), "providers.elevenlabs.style"),
            speaker_boost=_coerce_bool(
                elevenlabs.get("speaker_boost"),
                "providers.elevenlabs.speaker_boost",
            ),
        ),
        azure=AzureSettings(
            speech_key=_coerce_string(azure.get("speech_key"), "providers.azure.speech_key"),
            region=_coerce_string(azure.get("region"), "providers.azure.region"),
            default_voice=_read_default_voice(
                "azure",
                azure,
                "default_voice",
                "providers.azure.default_voice",
                "en-US-AvaNeural",
            )
            or "en-US-AvaNeural",
            speed=_read_float_default(
                azure,
                "speed",
                "providers.azure.speed",
                1.0,
            ),
            language_code=_coerce_string(azure.get("language_code"), "providers.azure.language_code"),
        ),
        kokoro=KokoroSettings(
            hf_token=_coerce_string(kokoro.get("hf_token"), "providers.kokoro.hf_token"),
            default_voice=_read_default_voice(
                "kokoro",
                kokoro,
                "default_voice",
                "providers.kokoro.default_voice",
                "af_heart",
            )
            or "af_heart",
            language_code=_read_string(
                kokoro,
                "language_code",
                "providers.kokoro.language_code",
                "a",
            )
            or "a",
            speed=_read_float_default(
                kokoro,
                "speed",
                "providers.kokoro.speed",
                1.0,
            ),
        ),
        windows=WindowsSettings(
            default_voice=_read_default_voice(
                "windows",
                windows,
                "default_voice",
                "providers.windows.default_voice",
            ),
            volume=_coerce_int(windows.get("volume"), "providers.windows.volume"),
            speed=_read_float_default(
                windows,
                "speed",
                "providers.windows.speed",
                1.0,
            ),
        ),
        demo=DemoProviderSettings(
            default_voice=_read_default_voice(
                "demo",
                demo,
                "default_voice",
                "providers.demo.default_voice",
                "animalese",
            )
            or "animalese",
            speed=_read_float_default(
                demo,
                "speed",
                "providers.demo.speed",
                1.0,
            ),
        ),
        extra=extra,
    )


def load_settings() -> Settings:
    """Load settings from defaults and the local config file."""

    path = _find_config_file()
    if path is None:
        return settings_from_dict({})
    return _settings_from_config_file(path)


def _settings_from_config_file(path: str | Path) -> Settings:
    target = _absolute_config_path(path)
    return _resolve_relative_cache_paths(
        settings_from_dict(_load_settings_file_payload(target)),
        target.parent,
    )


def _resolve_relative_cache_paths(settings: Settings, base_dir: Path) -> Settings:
    base_dir = _absolute_config_path(base_dir)
    cache = settings.voice_conductor.cache
    for field_name in ("root", "path", "api_dir"):
        value = getattr(cache, field_name)
        if value is not None and not Path(value).is_absolute():
            setattr(cache, field_name, _resolve_config_cache_path(value, base_dir))
    return settings


def _make_cache_paths_config_relative(payload: dict[str, Any], base_dir: Path) -> None:
    voice_conductor = payload.get("voice_conductor")
    if not isinstance(voice_conductor, dict):
        return
    cache = voice_conductor.get("cache")
    if not isinstance(cache, dict):
        return
    for field_name in ("root", "path", "api_dir"):
        cache[field_name] = _config_relative_path(cache.get(field_name), base_dir)


def settings_from_dict(payload: dict[str, Any]) -> Settings:
    """Parse nested settings from a plain mapping, validating supported keys."""

    if not isinstance(payload, dict):
        raise ValueError("Settings payload must be a JSON object.")
    _validate_top_level_keys(payload)
    voice_conductor = _parse_voice_conductor(payload)
    return Settings(
        voice_conductor=voice_conductor,
        providers=_parse_providers(payload)
    )


def _preserve_existing_provider_fields(payload: dict[str, Any], target: Path) -> None:
    if not target.exists():
        return

    try:
        existing = _loads_config_text(target.read_text(encoding="utf-8"), target)
    except (OSError, ValueError, json.JSONDecodeError):
        return

    existing_providers = existing.get("providers")
    payload_providers = payload.get("providers")
    if not isinstance(existing_providers, dict) or not isinstance(payload_providers, dict):
        return

    for provider_name, field_names in _PRESERVED_PROVIDER_FIELDS.items():
        existing_provider = existing_providers.get(provider_name)
        payload_provider = payload_providers.get(provider_name)
        if not isinstance(existing_provider, dict) or not isinstance(payload_provider, dict):
            continue
        for field_name in field_names:
            existing_value = _normalize_config_value(existing_provider.get(field_name))
            if existing_value is None:
                continue
            if _normalize_config_value(payload_provider.get(field_name)) is None:
                payload_provider[field_name] = existing_value


def _dumps_settings_config(payload: dict[str, Any], settings: Settings, target: Path) -> str:
    text = json.dumps(payload, indent=2) + "\n"
    if target.suffix.lower() != ".jsonc":
        return text
    comments = _provider_voice_comments(payload, settings)
    if not comments:
        return text
    return _append_default_voice_comments(text, comments)


def _load_existing_payload_if_present(target: Path) -> dict[str, Any]:
    if not target.exists():
        return {}
    try:
        payload = _loads_config_text(target.read_text(encoding="utf-8"), target)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_available_voice_key(voices: list[Any]) -> str | None:
    if not voices:
        return None
    first = voices[0]
    provider_name = _normalize_provider_name(str(getattr(first, "provider", "") or ""))
    voice_id = _normalize_config_value(getattr(first, "id", None))
    if voice_id is not None:
        if provider_name in {"demo", "kokoro", "windows"}:
            normalized = normalize_voice_key(provider_name, str(voice_id))
            prefix = f"{provider_name}:"
            if normalized.startswith(prefix):
                return normalized[len(prefix) :]
        return str(voice_id)
    voice_name = _normalize_config_value(getattr(first, "name", None))
    if voice_name is not None:
        return str(voice_name)
    return None


def _resolve_provider_default_voices(payload: dict[str, Any], settings: Settings, target: Path) -> None:
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return

    existing = _load_existing_payload_if_present(target)
    existing_providers = existing.get("providers") if isinstance(existing.get("providers"), dict) else {}

    try:
        from voice_conductor.providers.registry import ProviderRegistry
    except ImportError:
        return

    registry = ProviderRegistry()
    for provider_name, provider_payload in providers.items():
        if not isinstance(provider_payload, dict) or "default_voice" not in provider_payload:
            continue

        normalized = _normalize_provider_name(provider_name)
        existing_provider = existing_providers.get(normalized)
        existing_default = None
        if isinstance(existing_provider, dict):
            existing_default = _normalize_config_value(existing_provider.get("default_voice"))
        has_explicit_existing_default = existing_default is not None

        current_default = _normalize_config_value(provider_payload.get("default_voice"))
        is_unset = current_default is None
        if (
            not is_unset
            and not has_explicit_existing_default
            and current_default == _IMPLICIT_DEFAULT_VOICES.get(normalized)
        ):
            is_unset = True

        try:
            provider = registry.build_provider(normalized, settings)
            if not provider.is_available():
                provider_payload["default_voice"] = None
                continue
            voices = provider.list_voices()
        except Exception:
            provider_payload["default_voice"] = None
            continue

        if not voices:
            provider_payload["default_voice"] = None
            continue

        if is_unset:
            provider_payload["default_voice"] = _first_available_voice_key(voices)


def _provider_voice_comments(payload: dict[str, Any], settings: Settings) -> dict[str, str]:
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return {}

    try:
        from voice_conductor.providers.registry import ProviderRegistry
    except ImportError:
        return {}

    registry = ProviderRegistry()
    comments: dict[str, str] = {}
    for provider_name, provider_payload in providers.items():
        if not isinstance(provider_payload, dict) or "default_voice" not in provider_payload:
            continue
        normalized = _normalize_provider_name(provider_name)
        try:
            provider = registry.build_provider(normalized, settings)
            if not provider.is_available():
                comments[normalized] = _sanitize_jsonc_line_comment(
                    f"available voices unavailable: Provider {normalized!r} is not available. Configure credentials."
                )
                continue
            voices = provider.list_voices()
        except Exception as exc:
            comments[normalized] = _sanitize_jsonc_line_comment(
                f"available voices unavailable: {exc}"
            )
            continue

        voice_label = _format_voice_comment_label(normalized, voices)
        if voice_label:
            comments[normalized] = _sanitize_jsonc_line_comment(
                "available voices: " + voice_label
            )
        else:
            comments[normalized] = "available voices: none found"
    return comments


def _format_voice_comment_label(provider_name: str, voices: list[Any]) -> str:
    voice_pairs: list[tuple[str, str]] = []
    for voice in voices:
        raw_voice_key = str(getattr(voice, "id", "") or getattr(voice, "name", "") or "")
        key = normalize_voice_key(provider_name, raw_voice_key)
        if key == f"{provider_name}:":
            continue
        display_name = _compact_voice_display_name(
            provider_name,
            str(getattr(voice, "name", "") or "").strip(),
        )
        voice_pairs.append((key, display_name))

    if not voice_pairs:
        return ""

    prefix = f"{provider_name}:"
    if all(key.startswith(prefix) for key, _ in voice_pairs):
        names: list[str] = []
        for key, display_name in voice_pairs:
            provider_key = key[len(prefix) :]
            if (
                display_name
                and display_name.lower() != provider_key.lower()
                and _voice_key_needs_display_name(provider_key)
            ):
                names.append(f"{provider_key} ({display_name})")
            else:
                names.append(provider_key)
        return f"{provider_name}:<name> [{', '.join(names)}]"
    return ", ".join(key for key, _ in voice_pairs)


def _voice_key_needs_display_name(provider_key: str) -> bool:
    compact_key = provider_key.strip()
    if not compact_key:
        return False

    hexish = compact_key.replace("-", "").replace("_", "")
    if len(hexish) >= 16 and all(ch in "0123456789abcdefABCDEF" for ch in hexish):
        return True

    has_letter = any(ch.isalpha() for ch in compact_key)
    has_digit = any(ch.isdigit() for ch in compact_key)
    has_separator = any(ch in "-_ ." for ch in compact_key)
    return len(compact_key) >= 16 and has_letter and has_digit and not has_separator


def _compact_voice_display_name(provider_name: str, display_name: str) -> str:
    compact = " ".join(display_name.split())
    if not compact:
        return ""

    normalized_provider = _normalize_provider_name(provider_name)
    if normalized_provider == "elevenlabs":
        # ElevenLabs names often include long marketing descriptors after " - ".
        head, separator, _tail = compact.partition(" - ")
        if separator and head.strip():
            compact = head.strip()
    elif normalized_provider == "windows":
        if compact.startswith("Microsoft "):
            compact = compact[len("Microsoft ") :]
        if compact.endswith(" Desktop"):
            compact = compact[: -len(" Desktop")]
        compact = compact.strip()

    max_length = 32
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip(" ,") + "..."


def _sanitize_jsonc_line_comment(comment: str) -> str:
    normalized = " ".join(str(comment).replace("\r", " ").replace("\n", " ").split())
    max_length = 120
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip(" ,") + "..."


def _append_default_voice_comments(text: str, comments: dict[str, str]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_providers = False
    current_provider: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == '"providers": {':
            in_providers = True
            current_provider = None
        elif in_providers and line.startswith("  }"):
            in_providers = False
            current_provider = None
        elif in_providers and line.startswith("    ") and line.endswith("{"):
            current_provider = stripped.split(":", 1)[0].strip('"')

        if (
            in_providers
            and current_provider in comments
            and line.startswith("      ")
            and stripped.startswith('"default_voice":')
        ):
            output.append(f"{line[: len(line) - len(line.lstrip())]}// {comments[current_provider]}")
        output.append(line)

    return "\n".join(output) + "\n"


def _write_config_help(settings: Settings, path: Path) -> None:
    """Write companion Markdown explaining generated config fields."""

    lines = [
        "# VoiceConductor config help",
        "",
        "Generated next to `voice_conductor.config.jsonc` so the config can stay compact and this guide can stay easy to scan.",
        "",
        "## Edit map",
        "",
        "| Area | Use it for | Start with |",
        "| --- | --- | --- |",
        "| `voice_conductor` | Provider order, routing, and cache behavior. | `provider_chain` then `route_config` |",
        "| `voice_conductor.route_config` | Named speaker and virtual-mic playback targets. | `speakers` and `mic` routes |",
        "| `providers` | Credentials, voices, models, and provider-specific tuning. | The provider named first in `provider_chain` |",
        "",
        "## Quick changes",
        "",
        "| When you want to... | Edit... |",
        "| --- | --- |",
        "| Try providers in a different order | `voice_conductor.provider_chain` |",
        "| Force one provider when no chain is set | `voice_conductor.default_provider` |",
        "| Send playback to speakers or a virtual mic | `voice_conductor.route_config` |",
        "| Pick a new default voice | `providers.<name>.default_voice` |",
        "| Move or expire caches | `voice_conductor.cache` |",
        "",
        "## Available voices",
        "",
        "Use the value in the `Voice id` column for `providers.<name>.default_voice`. Live provider lists come from the provider metadata cache when possible.",
        "",
    ]
    lines.extend(_provider_voice_sections(settings))
    lines.extend(
        [
            "",
            "## Field reference",
            "",
            "Field names match the nested JSON path in `voice_conductor.config.jsonc`.",
            "",
        ]
    )
    lines.extend(_settings_help_sections("voice_conductor", settings.voice_conductor))
    lines.extend(_settings_help_sections("providers", settings.providers))
    lines.extend(
        [
            "",
            "## Provider notes",
            "",
            "- `providers.elevenlabs.default_voice` accepts an ElevenLabs voice name or id; phrase caching stores the stable voice id.",
            "- `providers.kokoro.default_voice` should be a Kokoro voice id such as `af_heart`.",
            "- `providers.windows.default_voice` should be a short provider-local voice id such as `david` or `zira`; installed Windows voice names still work.",
            "- `providers.demo.default_voice` should be a short provider-local voice id such as `animalese`.",
            "- Inline JSONC voice hints are intentionally truncated; call `list_voices(provider)` for the complete live list.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _provider_voice_sections(settings: Settings) -> list[str]:
    try:
        from voice_conductor.providers.registry import ProviderRegistry
    except ImportError:
        return [
            "Provider voice lookup is unavailable because the provider registry could not be imported.",
            "",
        ]

    registry = ProviderRegistry()
    output: list[str] = []
    for provider_name in _configured_provider_names(settings):
        output.extend([f"### {_provider_help_title(provider_name)}", ""])
        try:
            provider = registry.build_provider(provider_name, settings)
            voices = provider.list_voices()
        except Exception as exc:
            output.extend([f"Voice list unavailable: {_markdown_table_cell(str(exc))}", ""])
            continue

        if not voices:
            output.extend(["No voices were reported by this provider.", ""])
            continue

        plural = "voice" if len(voices) == 1 else "voices"
        output.extend(
            [
                f"{len(voices)} {plural} found.",
                "",
                "| Voice | Voice id | Details |",
                "| --- | --- | --- |",
            ]
        )
        output.extend(_voice_table_row(voice) for voice in voices)
        output.append("")
    return output


def _configured_provider_names(settings: Settings) -> list[str]:
    names = [item.name for item in fields(settings.providers) if item.name != "extra"]
    names.extend(sorted(settings.providers.extra))
    return names


def _voice_table_row(voice: Any) -> str:
    voice_id = str(getattr(voice, "id", "") or "").strip()
    voice_name = str(getattr(voice, "name", "") or "").strip()
    language = str(getattr(voice, "language", "") or "").strip()
    metadata = getattr(voice, "metadata", {}) or {}
    details = []
    if language:
        details.append(language)
    if isinstance(metadata, dict):
        details.extend(
            str(value).strip()
            for key, value in sorted(metadata.items())
            if key in {"category", "description", "local_name"}
            and str(value or "").strip()
            and str(value).strip() != voice_name
            and not str(value).strip().startswith(f"{voice_name} - ")
        )
    detail_text = ", ".join(details) if details else "-"
    display_name = voice_name or voice_id or "-"
    provider_name = str(getattr(voice, "provider", "") or "").strip()
    config_voice = (
        normalize_voice_config_value(provider_name, voice_id or voice_name)
        if provider_name
        else None
    )
    display_id = config_voice or voice_id or voice_name or "-"
    return (
        f"| {_markdown_table_cell(display_name)} "
        f"| `{_markdown_table_cell(display_id)}` "
        f"| {_markdown_table_cell(detail_text)} |"
    )


def _settings_help_sections(prefix: str, value: Any) -> list[str]:
    if not is_dataclass(value):
        return []

    output: list[str] = []
    sections = _collect_settings_help_sections(prefix, value)
    for section, rows in sections:
        if not rows:
            continue
        output.extend(
            [
                f"### {_settings_help_title(section)}",
                "",
                f"`{section}`",
                "",
                "| Field | Notes |",
                "| --- | --- |",
            ]
        )
        output.extend(f"| `{name}` | {doc} |" for name, doc in rows)
        output.append("")
    return output


def _collect_settings_help_sections(
    prefix: str,
    value: Any,
) -> list[tuple[str, list[tuple[str, str]]]]:
    rows: list[tuple[str, str]] = []
    nested_sections: list[tuple[str, list[tuple[str, str]]]] = []

    if not is_dataclass(value):
        return []

    for item in fields(value):
        if item.name == "extra":
            continue
        doc = str(item.metadata.get("doc", "")).strip()
        field_value = getattr(value, item.name)
        name = f"{prefix}.{item.name}"
        if doc:
            rows.append((name, doc))
        if is_dataclass(field_value):
            nested_sections.extend(_collect_settings_help_sections(name, field_value))

    return [(prefix, rows), *nested_sections]


def _settings_help_title(section: str) -> str:
    label = section.removeprefix("providers.")
    label = label.removeprefix("voice_conductor.")
    label = label.replace("_", " ").replace(".", " / ")
    if section.startswith("providers.") and section != "providers":
        return f"{_provider_help_title(label)} settings"
    if section == "voice_conductor":
        return "Voice conductor settings"
    if section == "providers":
        return "Provider settings"
    return label.title()


def _provider_help_title(provider_name: str) -> str:
    known = {
        "elevenlabs": "ElevenLabs",
        "azure": "Azure Speech",
        "kokoro": "Kokoro",
        "windows": "Windows Speech",
        "demo": "Demo",
    }
    normalized = str(provider_name).strip().lower()
    return known.get(normalized, normalized.replace("_", " ").title())


def _markdown_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """Serialize settings back to the nested JSON shape."""

    registry = ProviderConfigRegistry()

    providers = {}
    for name, config in settings.providers.extra.items():
        normalized = _normalize_provider_name(name)
        if normalized in _BUILT_IN_PROVIDERS:
            raise ValueError(f"{normalized} is a built-in provider config.")
        if not registry.is_registered(normalized):
            raise ValueError(f"Unknown provider config: {normalized}. Register it before saving.")
        providers[normalized] = registry.serialize(normalized, config)
    providers["elevenlabs"] = {
        "api_key": settings.providers.elevenlabs.api_key,
        "default_voice": normalize_voice_config_value(
            "elevenlabs",
            settings.providers.elevenlabs.default_voice,
        ),
        "model_id": settings.providers.elevenlabs.model_id,
        "output_format": settings.providers.elevenlabs.output_format,
        "speed": settings.providers.elevenlabs.speed,
        "language_code": settings.providers.elevenlabs.language_code,
        "stability": settings.providers.elevenlabs.stability,
        "similarity_boost": settings.providers.elevenlabs.similarity_boost,
        "style": settings.providers.elevenlabs.style,
        "speaker_boost": settings.providers.elevenlabs.speaker_boost,
    }
    providers["azure"] = {
        "speech_key": settings.providers.azure.speech_key,
        "region": settings.providers.azure.region,
        "default_voice": normalize_voice_config_value(
            "azure",
            settings.providers.azure.default_voice,
        ),
        "speed": settings.providers.azure.speed,
        "language_code": settings.providers.azure.language_code,
    }
    providers["kokoro"] = {
        "hf_token": settings.providers.kokoro.hf_token,
        "default_voice": normalize_voice_config_value(
            "kokoro",
            settings.providers.kokoro.default_voice,
        ),
        "language_code": settings.providers.kokoro.language_code,
        "speed": settings.providers.kokoro.speed,
    }
    providers["windows"] = {
        "default_voice": normalize_voice_config_value(
            "windows",
            settings.providers.windows.default_voice,
        ),
        "volume": settings.providers.windows.volume,
        "speed": settings.providers.windows.speed,
    }
    providers["demo"] = {
        "default_voice": normalize_voice_config_value(
            "demo",
            settings.providers.demo.default_voice,
        ),
        "speed": settings.providers.demo.speed,
    }

    return {
        "voice_conductor": {
            "default_provider": settings.voice_conductor.default_provider,
            "provider_chain": list(settings.voice_conductor.provider_chain),
            "route_config": {
                "routes": {
                    name: {
                        "device": route.device,
                        "prefer_virtual_cable": route.prefer_virtual_cable,
                    }
                    for name, route in settings.voice_conductor.route_config.routes.items()
                },
            },
            "cache": {
                "root": _normalize_windows_config_path(settings.voice_conductor.cache.root or _default_cache_root()),
                "path": _normalize_windows_config_path(settings.voice_conductor.cache.path),
                "api_dir": _normalize_windows_config_path(settings.voice_conductor.cache.api_dir),
                "ttl_seconds": settings.voice_conductor.cache.ttl_seconds,
            },
        },
        "providers": providers,
    }
