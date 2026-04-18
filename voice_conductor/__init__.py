"""Public package API for voice_conductor.

Most callers should start with ``TTSManager``. The re-exported settings,
provider registry, routing, and shared data types are exposed for applications
that need to customize provider registration, fallback, or audio output
behavior.
"""

from .audio.router import AudioRoute, RouteConfig
from .config import (
    AzureSettings,
    CacheSettings,
    DemoProviderSettings,
    ElevenLabsSettings,
    KokoroSettings,
    ProviderSettings,
    Settings,
    VoiceConductorSettings,
    WindowsSettings,
    load_settings,
    settings_from_dict,
    settings_to_dict,
)
from .config_registry import (
    ProviderConfigRegistry,
    register_provider_config,
    unregister_provider_config,
)
from .manager import TTSManager
from .providers.demo import DemoProvider
from .providers.registry import ProviderRegistry, register_provider, unregister_provider
from .types import (
    AudioDevice,
    PlaybackCompleteEvent,
    PlaybackHooks,
    PlaybackReadyEvent,
    PlaybackResult,
    PlaybackTask,
    SynthesizedAudio,
    VoiceInfo,
)
from .voice_keys import normalize_voice_key

__version__ = "0.1.2"

__all__ = [
    "AudioDevice",
    "AudioRoute",
    "AzureSettings",
    "CacheSettings",
    "DemoProvider",
    "DemoProviderSettings",
    "ElevenLabsSettings",
    "KokoroSettings",
    "PlaybackCompleteEvent",
    "PlaybackHooks",
    "PlaybackReadyEvent",
    "PlaybackResult",
    "PlaybackTask",
    "ProviderSettings",
    "ProviderConfigRegistry",
    "ProviderRegistry",
    "RouteConfig",
    "Settings",
    "SynthesizedAudio",
    "TTSManager",
    "VoiceInfo",
    "VoiceConductorSettings",
    "WindowsSettings",
    "load_settings",
    "normalize_voice_key",
    "register_provider",
    "register_provider_config",
    "settings_from_dict",
    "settings_to_dict",
    "unregister_provider",
    "unregister_provider_config",
]
