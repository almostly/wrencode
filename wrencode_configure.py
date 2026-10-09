"""Backend and model selection for wrencode: the first-run chooser, /configure, /model,
API-key prompts and verification, model lists, and the saved config in ~/.wrencode.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import pathlib
import platform
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import wrencode_backends as backends
import wrencode_ui as ui
from wrencode_ui import BLUE, DIM, RED, RESET, YELLOW

CUSTOM_MODEL_OPTION = "— type a custom model id —"
_MLX_UNCHANGED = object()


# -----------------------------------------------------------------------------------------------
# Backend selection & config persistence
# -----------------------------------------------------------------------------------------------
def is_frozen() -> bool:
    """Set true when running as a PyInstaller standalone binary."""
    return bool(getattr(sys, "frozen", False))


def available_backends() -> list[str]:
    """Backends offerable in the current runtime.

    The standalone binary can't bundle the ML stack, so local-ml backends
    (mlx/transformers) are only offered from a source install. MLX is further
    limited to Apple Silicon.
    """
    return [
        name
        for name, spec in backends.BACKEND_SPECS.items()
        if not (spec["kind"] == "agent-sdk" and is_frozen())
        and not (
            spec["kind"] == "local-ml"
            and (
                is_frozen()
                or (
                    name == "mlx"
                    and not (
                        platform.system() == "Darwin" and platform.machine() == "arm64"
                    )
                )
            )
        )
    ]


def _read_models_cache(cache: pathlib.Path) -> list[str] | None:
    """Return cached model ids if the cache file is fresh (<24h), else None."""
    if not cache.exists():
        return None
    with contextlib.suppress(Exception):
        age = time.time() - cache.stat().st_mtime
        if age < 86400:
            cached = json.loads(cache.read_text())
            if isinstance(cached, list) and cached:
                return [str(m) for m in cached]
    return None


def _write_models_cache(cache: pathlib.Path, ids: list[str]) -> None:
    """Persist model ids for the next configure session."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids, indent=2))
    os.chmod(cache, 0o600)


def _catalog_price(entry: dict[str, Any]) -> list[float] | None:
    """[input, output, cache_read, cache_write] per million tokens from a catalog entry.

    OpenRouter (and catalogs in its format) list `pricing` in USD per token as
    strings: prompt, completion, and optionally input_cache_read and
    input_cache_write. Missing cache rates count as the input rate.
    """
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    try:
        inp = float(pricing.get("prompt")) * 1e6
        out = float(pricing.get("completion")) * 1e6
        read = pricing.get("input_cache_read")
        write = pricing.get("input_cache_write")
        rates = [
            inp,
            out,
            float(read) * 1e6 if read is not None else inp,
            float(write) * 1e6 if write is not None else inp,
        ]
    except (TypeError, ValueError):
        return None
    if any(r < 0 for r in rates):
        return None
    return [round(r, 6) for r in rates]


def _save_catalog_prices(backend: str, entries: list[Any]) -> int:
    """Merge the prices a /models catalog lists into prices.json; returns how many."""
    found = {
        f"{backend}/{m['id']}": rates
        for m in entries
        if isinstance(m, dict) and m.get("id") and (rates := _catalog_price(m))
    }
    if not found:
        return 0
    path = backends.PRICES_FILE
    saved: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            saved = loaded
    saved.update(found)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2, sort_keys=True))
    os.chmod(path, 0o600)
    return len(found)


def _api_key_for_backend(backend: str) -> str:
    """Resolve an API key for model fetches without requiring BACKEND == backend."""
    if env_key := backends._env_api_key(backend):
        return env_key
    if backend == backends.BACKEND and backends.API_KEY:
        return backends.API_KEY
    cfg = backends.load_config()
    if cfg.get("backend") == backend:
        return cfg.get("api_key", "")
    return ""


def _workspace_id_for_anthropic() -> str:
    """Resolve Anthropic workspace id: env > active session > saved config."""
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        return os.environ["ANTHROPIC_WORKSPACE_ID"]
    if backends.ANTHROPIC_WORKSPACE_ID:
        return backends.ANTHROPIC_WORKSPACE_ID
    return backends.load_config().get("anthropic_workspace_id", "")


def _fetch_hosted_models(
    backend: str, url: str, cache: pathlib.Path, label: str
) -> list[str]:
    """Fetch an OpenAI-style /models list for a hosted backend, with a 24h local cache."""
    fallback = list(backends.BACKEND_MODELS.get(backend, []))
    cached = _read_models_cache(cache)
    if cached is not None:
        return cached

    key = _api_key_for_backend(backend)
    if not key:
        return fallback

    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        entries = data.get("data", [])
        ids = sorted(m.get("id", "") for m in entries if m.get("id"))
        if ids:
            _write_models_cache(cache, ids)
            _save_catalog_prices(backend, entries)
        return ids or fallback
    except Exception as err:  # any failure falls back to the built-in list
        print(f"{YELLOW}Could not fetch {label} models: {err}{RESET}")
        return fallback


def fetch_anthropic_models() -> list[str]:
    """Fetch Anthropic model ids from GET /v1/models (newest first), with a 24h cache."""
    fallback = list(backends.BACKEND_MODELS.get("anthropic", []))
    cached = _read_models_cache(backends.ANTHROPIC_MODELS_CACHE)
    if cached is not None:
        return cached

    key = _api_key_for_backend(
        backends.BACKEND
        if backends.BACKEND in backends.ANTHROPIC_KEY_BACKENDS
        else "anthropic"
    )
    if not key:
        return fallback

    headers = backends._anthropic_headers(
        api_key=key, workspace_id=_workspace_id_for_anthropic()
    )
    try:
        ids: list[str] = []
        after: str | None = None
        while True:
            query = urllib.parse.urlencode(
                {"limit": "1000", **({"after_id": after} if after else {})}
            )
            req = urllib.request.Request(
                f"https://api.anthropic.com/v1/models?{query}", headers=headers
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            page = [m["id"] for m in data.get("data", []) if m.get("id")]
            ids.extend(page)
            if not data.get("has_more") or not data.get("last_id"):
                break
            after = data["last_id"]
        if ids:
            _write_models_cache(backends.ANTHROPIC_MODELS_CACHE, ids)
        return ids or fallback
    except Exception as err:  # any failure falls back to the built-in list
        print(f"{YELLOW}Could not fetch Anthropic models: {err}{RESET}")
        return fallback


def fetch_openai_models() -> list[str]:
    """Fetch OpenAI chat-oriented model ids from GET /v1/models, with a 24h cache."""
    fallback = list(backends.BACKEND_MODELS.get("openai", []))
    cached = _read_models_cache(backends.OPENAI_MODELS_CACHE)
    if cached is not None:
        return cached

    key = _api_key_for_backend("openai")
    if not key:
        return fallback

    # /v1/models also lists embeddings, audio, images, etc. — keep chat-ish ids.
    skip_substrings = (
        "embedding",
        "whisper",
        "tts",
        "dall-e",
        "moderation",
        "transcribe",
        "realtime",
        "audio",
        "image",
        "search",
        "babbage",
        "davinci",
        "curie",
        "ada",
    )
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        ids = sorted(
            m["id"]
            for m in data.get("data", [])
            if m.get("id") and not any(s in m["id"].lower() for s in skip_substrings)
        )
        if ids:
            _write_models_cache(backends.OPENAI_MODELS_CACHE, ids)
        return ids or fallback
    except Exception as err:  # any failure falls back to the built-in list
        print(f"{YELLOW}Could not fetch OpenAI models: {err}{RESET}")
        return fallback


def fetch_openrouter_models() -> list[str]:
    """Fetch OpenRouter model ids, with a 24h local cache."""
    return _fetch_hosted_models(
        "openrouter",
        "https://openrouter.ai/api/v1/models",
        backends.OPENROUTER_MODELS_CACHE,
        "OpenRouter",
    )


def fetch_nanogpt_models() -> list[str]:
    """Fetch NanoGPT model ids, with a 24h local cache."""
    return _fetch_hosted_models(
        "nanogpt",
        "https://nano-gpt.com/api/v1/models",
        backends.NANOGPT_MODELS_CACHE,
        "NanoGPT",
    )


def fetch_ollama_models() -> list[str]:
    """List models reported by a local Ollama server."""
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            data = json.load(resp)
        names = sorted(
            m.get("name", "") for m in data.get("models", []) if m.get("name")
        )
        return names or [backends.BACKEND_SPECS["ollama"]["model"]]
    except Exception as err:  # any failure falls back to the default model
        print(f"{YELLOW}Could not reach Ollama at {base}: {err}{RESET}")
        return [backends.BACKEND_SPECS["ollama"]["model"]]


def fetch_openai_compatible_models() -> list[str]:
    """Return the model ids an openai-compatible server serves, or [] if unavailable."""
    return backends._list_openai_compatible_models()[0]


def list_models_for_backend(backend: str) -> list[str]:
    """Return selectable models for a backend (includes a custom-id option)."""
    spec = backends.BACKEND_SPECS[backend]
    if backend in backends.ANTHROPIC_KEY_BACKENDS:
        models = fetch_anthropic_models()
    elif backend == "openai":
        models = fetch_openai_models()
    elif backend == "openrouter":
        models = fetch_openrouter_models()
    elif backend == "nanogpt":
        models = fetch_nanogpt_models()
    elif backend == "ollama":
        models = fetch_ollama_models()
    elif backend == "openai-compatible":
        models = fetch_openai_compatible_models()
    else:
        models = list(backends.BACKEND_MODELS.get(backend, [spec["model"]]))

    if not models:
        models = [spec["model"]] if spec["model"] else []
    models = list(dict.fromkeys(models))
    current = backends.MODEL if backend == backends.BACKEND else spec["model"]
    if current and current not in models:
        models.insert(0, current)
    if CUSTOM_MODEL_OPTION not in models:
        models.append(CUSTOM_MODEL_OPTION)
    return models


def _prompt_api_key_if_needed(
    backend: str, existing: dict[str, str], cfg: dict[str, str]
) -> None:
    """Prompt for an API key when switching to a hosted backend."""
    spec = backends.BACKEND_SPECS[backend]
    if spec["kind"] == "aws":
        # Bedrock uses AWS environment credentials, not a saved key.
        ak, sk, _ = backends._aws_credentials()
        if ak and sk:
            ui.print_system(
                f"✓ Using AWS credentials from environment @ {backends._aws_region()}"
            )
        else:
            print(
                f"{YELLOW}Bedrock uses your AWS credentials — set AWS_ACCESS_KEY_ID, "
                f"AWS_SECRET_ACCESS_KEY, and AWS_REGION.{RESET}"
            )
        return
    if spec["kind"] not in backends.KEYED_KINDS:
        return
    key_env = spec["key_env"]
    if env_key := os.environ.get(key_env):
        key = getpass.getpass(
            f"{BLUE}❯{RESET} {key_env} found in environment (…{env_key[-4:]}). "
            "Enter to use it, or paste a new key (input hidden): "
        ).strip()
        if key:
            _use_entered_key(key_env, key, cfg)
        else:
            cfg["api_key_overrides_env"] = ""
            ui.print_system(f"✓ Using {key_env} from environment")
        return
    saved_key = (
        existing.get("api_key", "") if existing.get("backend") == backend else ""
    )
    keep_hint = " (leave blank to keep saved key)" if saved_key else ""
    key = getpass.getpass(
        f"{BLUE}❯{RESET} {key_env}{keep_hint} (input hidden): "
    ).strip()
    if key:
        _use_entered_key(key_env, key, cfg)
    elif saved_key:
        cfg["api_key"] = saved_key
        ui.print_system(f"✓ Keeping saved {key_env}")
    else:
        print(f"{YELLOW}No key entered — set {key_env} or re-run with /backend.{RESET}")


def _use_entered_key(key_env: str, key: str, cfg: dict[str, str]) -> None:
    """Save a key typed in /configure so it beats a stale one in the env or .env."""
    cfg["api_key"] = key
    if os.environ.get(key_env) and os.environ[key_env] != key:
        cfg["api_key_overrides_env"] = "1"
        os.environ[key_env] = key  # this process, before the config is saved
        ui.print_system(
            f"✓ Saved key will be used instead of {key_env} from the environment"
        )


def _prompt_anthropic_workspace_if_needed(
    existing: dict[str, str], cfg: dict[str, str]
) -> None:
    """Prompt for anthropic-workspace-id when using a multi-workspace API key."""
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        ui.print_system("✓ Using ANTHROPIC_WORKSPACE_ID from environment")
        return
    saved = (
        existing.get("anthropic_workspace_id", "")
        if existing.get("backend") in backends.ANTHROPIC_KEY_BACKENDS
        else ""
    )
    keep_hint = (
        " (leave blank to keep saved)"
        if saved
        else " (optional; required for multi-workspace keys)"
    )
    raw = input(f"{BLUE}❯{RESET} Anthropic workspace id{keep_hint}: ").strip()
    if raw:
        cfg["anthropic_workspace_id"] = raw
    elif saved:
        cfg["anthropic_workspace_id"] = saved
        ui.print_system("✓ Keeping saved Anthropic workspace id")
    else:
        # Clear a stale value if the user left it blank on a fresh anthropic setup.
        cfg.pop("anthropic_workspace_id", None)


def persist_backend_choice(cfg: dict[str, str]) -> None:
    """Merge cfg into saved config and apply module-level backend globals."""
    merged = {**backends.load_config(), **cfg}
    # Drop empty workspace id so a blank configure answer clears a prior value.
    if "anthropic_workspace_id" in cfg and not cfg["anthropic_workspace_id"]:
        merged.pop("anthropic_workspace_id", None)
    backends.save_config(merged)
    backends.apply_backend(
        merged["backend"],
        merged.get("model", ""),
        merged.get("api_key", ""),
        anthropic_workspace_id=merged.get("anthropic_workspace_id", ""),
    )


def verify_api_key() -> tuple[str, str]:
    """Probe the current backend with API_KEY to see if the key actually works.

    Returns (state, detail): "ok" (accepted), "invalid" (rejected by the
    service), or "unknown" (couldn't reach it — network/timeout). Local
    backends and missing keys short-circuit to "ok"/"invalid" without a call.
    """
    spec = backends.BACKEND_SPECS[backends.BACKEND]
    if spec["kind"] == "aws":
        # Probe Bedrock's control plane (ListFoundationModels) with a signed
        # GET: confirms the AWS creds + region work without invoking a model.
        ak, sk, _ = backends._aws_credentials()
        if not (ak and sk):
            return ("invalid", "no AWS credentials")
        region = backends._aws_region()
        url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
        try:
            headers = backends._sigv4_signed_headers("GET", url, b"", "bedrock", region)
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read(1)
            return ("ok", "")
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                return ("invalid", f"HTTP {err.code}")
            return ("unknown", f"HTTP {err.code}")
        except Exception as err:  # verification is advisory
            return ("unknown", str(err))
    if spec["kind"] not in backends.KEYED_KINDS:
        return ("ok", "")
    if not backends.API_KEY:
        return ("invalid", "no key")
    anthropic_probe = (
        "https://api.anthropic.com/v1/models",
        backends._anthropic_headers(),
        "GET",
    )
    probes = {
        "anthropic": anthropic_probe,
        backends.AGENT_SDK_BACKEND: anthropic_probe,
        "openai": (
            "https://api.openai.com/v1/models",
            backends._openai_headers(),
            "GET",
        ),
        "openrouter": (
            "https://openrouter.ai/api/v1/key",
            backends._openai_headers(),
            "GET",
        ),
        # NanoGPT's /models is public, so check the key against the balance endpoint.
        "nanogpt": (
            "https://nano-gpt.com/api/check-balance",
            backends._openai_headers(),
            "POST",
        ),
    }
    if backends.BACKEND not in probes:
        return ("unknown", "")
    url, headers, method = probes[backends.BACKEND]
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read(1)
        return ("ok", "")
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace")
        if err.code in (401, 403):
            return ("invalid", f"HTTP {err.code}")
        if err.code == 400 and "workspace" in body.lower():
            return (
                "invalid",
                (
                    "API key needs anthropic-workspace-id "
                    "(set ANTHROPIC_WORKSPACE_ID or enter it in /configure)"
                ),
            )
        return ("unknown", f"HTTP {err.code}")
    except Exception as err:  # verification is advisory
        return ("unknown", str(err))


def pick_model_interactive(backend: str) -> str | None:
    """Arrow-key model picker for a backend; returns model id or None."""
    models = list_models_for_backend(backend)
    labels = _model_labels(backend, models)
    initial = models.index(backends.MODEL) if backends.MODEL in models else 0
    idx = ui.pick_from_list(
        f"Choose model ({backend})",
        models,
        labels=labels,
        initial_index=initial,
    )
    if idx is None:
        return None
    choice = models[idx]
    if choice == CUSTOM_MODEL_OPTION:
        default = backends.BACKEND_SPECS[backend]["model"]
        custom = input(f"{BLUE}❯{RESET} model id [{default}]: ").strip()
        return custom or default
    return choice


def _model_labels(backend: str, models: list[str]) -> list[str]:
    """Model ids with their price alongside, when one is known."""
    width = max((len(m) for m in models), default=0)
    labels = []
    for model in models:
        price = backends.price_for(model, backend)
        if price is None or model == CUSTOM_MODEL_OPTION:
            labels.append(model)
        else:
            labels.append(f"{model:<{width}}  {DIM}{price.label()}{RESET}")
    return labels


def try_reload_model() -> Any:
    """Load weights for local-ml backends; return None for API/proxy backends."""
    if backends.BACKEND in backends.LOCAL_ML_BACKENDS:
        try:
            return backends.load_model()
        except SystemExit:
            return _MLX_UNCHANGED
    if (
        backends.BACKEND in backends.API_BACKENDS | {backends.AGENT_SDK_BACKEND}
        and not backends.API_KEY
    ):
        key_env = backends.BACKEND_SPECS[backends.BACKEND]["key_env"]
        print(f"{RED}{key_env} not set — cannot use {backends.BACKEND}.{RESET}")
        return _MLX_UNCHANGED
    return None


def switch_model_runtime(model_id: str = "") -> Any:
    """Switch model on the current backend; reload local weights if needed."""
    if model_id:
        model = model_id
    else:
        if not sys.stdin.isatty():
            print(f"{RED}/model needs an interactive terminal.{RESET}")
            return _MLX_UNCHANGED
        picked = pick_model_interactive(backends.BACKEND)
        if picked is None:
            ui.print_system("Cancelled.")
            return _MLX_UNCHANGED
        model = picked

    if os.environ.get("MODEL"):
        print(f"{YELLOW}MODEL env var overrides saved choice.{RESET}")

    persist_backend_choice({"backend": backends.BACKEND, "model": model})
    ui.print_system(f"✓ Model → {backends.MODEL}")
    return try_reload_model()


def _apply_credentials_for_model_fetch(
    backend: str, cfg: dict[str, str], existing: dict[str, str]
) -> None:
    """Apply key/workspace so live /models fetches can authenticate."""
    backends.apply_backend(
        backend,
        existing.get("model", "") if existing.get("backend") == backend else "",
        cfg.get("api_key")
        or (existing.get("api_key", "") if existing.get("backend") == backend else ""),
        anthropic_workspace_id=cfg.get(
            "anthropic_workspace_id",
            existing.get("anthropic_workspace_id", "")
            if existing.get("backend") in backends.ANTHROPIC_KEY_BACKENDS
            else "",
        ),
    )


def switch_backend_runtime() -> Any:
    """Switch backend (and model) mid-session; reload local weights if needed."""
    if not sys.stdin.isatty():
        print(f"{RED}/backend needs an interactive terminal.{RESET}")
        return _MLX_UNCHANGED

    existing = backends.load_config()
    names = available_backends()
    labels = [
        f"{backends.BACKEND_SPECS[n]['label']}  [{backends.BACKEND_SPECS[n]['model'] or 'server default'}]"
        for n in names
    ]
    initial = names.index(backends.BACKEND) if backends.BACKEND in names else 0
    idx = ui.pick_from_list(
        "Choose backend", names, labels=labels, initial_index=initial
    )
    if idx is None:
        ui.print_system("Cancelled.")
        return _MLX_UNCHANGED

    backend = names[idx]
    cfg: dict[str, str] = {"backend": backend}
    _prompt_api_key_if_needed(backend, existing, cfg)
    if backend in backends.ANTHROPIC_KEY_BACKENDS:
        _prompt_anthropic_workspace_if_needed(existing, cfg)
    # Credentials first so the model picker can fetch live /models lists.
    _apply_credentials_for_model_fetch(backend, cfg, existing)
    model = pick_model_interactive(backend)
    if model is None:
        ui.print_system("Cancelled.")
        return _MLX_UNCHANGED

    cfg["model"] = model
    persist_backend_choice(cfg)
    ui.print_system(f"✓ Backend → {backends.BACKEND}:{backends.MODEL}")
    return try_reload_model()


def choose_backend_interactive() -> None:
    """Prompt the user to pick a backend, persist the choice, and apply it."""
    existing = backends.load_config()
    names = available_backends()
    labels = [
        f"{backends.BACKEND_SPECS[n]['label']}  [{backends.BACKEND_SPECS[n]['model'] or 'server default'}]"
        for n in names
    ]
    if not is_frozen():
        ui.print_system("Local model backends need mlx-lm or transformers installed.")
        print()

    idx = ui.pick_from_list("Choose backend", names, labels=labels, initial_index=0)
    if idx is None:
        print(f"{RED}Backend selection required.{RESET}")
        raise SystemExit(1)

    choice = names[idx]
    cfg: dict[str, str] = {"backend": choice}
    spec = backends.BACKEND_SPECS[choice]
    _prompt_api_key_if_needed(choice, existing, cfg)
    if choice in backends.ANTHROPIC_KEY_BACKENDS:
        _prompt_anthropic_workspace_if_needed(existing, cfg)
    # Credentials first so the model picker can fetch live /models lists.
    _apply_credentials_for_model_fetch(choice, cfg, existing)
    model = pick_model_interactive(choice)
    if model is None:
        print(f"{RED}Model selection required.{RESET}")
        raise SystemExit(1)

    cfg["model"] = model
    persist_backend_choice(cfg)
    ui.print_system(f"✓ Saved backend choice to {backends.CONFIG_FILE}")

    # Verify the credential actually works before declaring the backend ready,
    # so a typo/revoked key (or missing AWS creds) surfaces here instead of
    # mid-chat. Re-prompt API keys on a hard rejection; Bedrock creds come from
    # the environment, so there's nothing to re-prompt. Don't block on a
    # transient network failure.
    cred_name = spec.get("key_env", "AWS credentials")
    for attempt in range(3):
        state, detail = verify_api_key()
        if state == "ok":
            ui.print_system(f"✓ {choice}:{backends.MODEL} ready")
            return
        if state == "unknown":
            print(f"{YELLOW}⚠ Couldn't verify {cred_name} ({detail}).{RESET}")
            return
        print(f"{RED}✗ {cred_name} was rejected ({detail}).{RESET}")
        if (
            spec["kind"] not in backends.KEYED_KINDS
            or not sys.stdin.isatty()
            or attempt == 2
        ):
            if spec["kind"] == "aws":
                print(
                    f"{DIM}Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (and AWS_REGION) and re-run.{RESET}"
                )
            return
        if choice in backends.ANTHROPIC_KEY_BACKENDS and "workspace" in detail.lower():
            ws = input(
                f"{BLUE}❯{RESET} Anthropic workspace id (from Settings → Workspaces): "
            ).strip()
            if not ws:
                return
            cfg["anthropic_workspace_id"] = ws
            persist_backend_choice(cfg)
            continue
        newkey = getpass.getpass(
            f"{BLUE}❯{RESET} re-enter {cred_name} (input hidden): "
        ).strip()
        if not newkey:
            return
        _use_entered_key(spec["key_env"], newkey, cfg)
        persist_backend_choice(cfg)


def resolve_configuration() -> None:
    """Decide which backend to use: env override > saved config > interactive > error."""
    # 1. Explicit BACKEND env var — power users / CI. Unchanged from prior behaviour.
    env_backend = os.environ.get("BACKEND")
    if env_backend:
        if env_backend not in backends.BACKEND_SPECS:
            valid = ", ".join(backends.BACKEND_SPECS)
            print(f"{RED}Unknown BACKEND '{env_backend}'.{RESET} Valid: {valid}")
            raise SystemExit(1)
        backends.apply_backend(env_backend)
        return

    # 2. A choice saved from a previous run.
    cfg = backends.load_config()
    if cfg.get("backend") in backends.BACKEND_SPECS:
        backend = cfg["backend"]
        spec = backends.BACKEND_SPECS[backend]
        # A hosted backend with no usable key — env var unset and nothing
        # saved (e.g. configured in a dir whose .env supplied the key, so it
        # was never persisted) — would dead-end in load_model() with a
        # SystemExit. In a terminal, re-run the chooser so the user can pick a
        # backend and enter a key instead of the tool exiting immediately.
        key_missing = (
            spec["kind"] in backends.KEYED_KINDS
            and not backends._env_api_key(backend)
            and not cfg.get("api_key")
        )
        if key_missing and sys.stdin.isatty():
            print(
                f"{YELLOW}Saved backend '{backend}' has no API key "
                f"({spec['key_env']} is unset and none was saved).{RESET}"
            )
            choose_backend_interactive()
            return
        backends.apply_backend(
            backend,
            cfg.get("model", ""),
            cfg.get("api_key", ""),
            anthropic_workspace_id=cfg.get("anthropic_workspace_id", ""),
        )
        return

    # 3. First run with a real terminal — ask the user.
    if sys.stdin.isatty():
        choose_backend_interactive()
        return

    # 4. Non-interactive with nothing configured — fail with guidance.
    print(f"{RED}No backend configured.{RESET}")
    print(
        "Set BACKEND=<name> (plus the matching API key), "
        "or run `wrencode` in a terminal to choose one."
    )
    raise SystemExit(1)


# -----------------------------------------------------------------------------------------------
# Uninstall
# -----------------------------------------------------------------------------------------------
def uninstall() -> None:
    """Delete saved state and print how to remove the program for this install."""
    if backends.CONFIG_DIR.exists():
        try:
            shutil.rmtree(backends.CONFIG_DIR)
            ui.print_system(f"✓ Removed saved config {backends.CONFIG_DIR}")
        except OSError as err:
            print(f"{YELLOW}Could not remove {backends.CONFIG_DIR}: {err}{RESET}")
    else:
        ui.print_system(f"No saved config at {backends.CONFIG_DIR}")

    ui.print_system("To remove the program itself:")
    if is_frozen():
        # install.sh drops a standalone binary; sys.executable is that file.
        print(f"rm {sys.executable}")
    else:
        print(f"{DIM}uv tool install:{RESET} uv tool uninstall wrencode")
        print(f"{DIM}pip install:{RESET} pip uninstall wrencode")
        print(f"{DIM}uvx cache:{RESET} uv cache clean wrencode")
