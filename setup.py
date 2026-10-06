#!/usr/bin/env python3
"""First-run environment installer for Aemon-AI.

Sets up llama.cpp's ``llama-server`` and a local GGUF model on a brand new
device, then writes a complete ``.env`` so ``python agent.py`` works.

Works on macOS, Windows and Linux using only the Python standard library --
nothing needs to be installed before running this script.

Usage
    python setup.py                  # guided, interactive wizard
    python setup.py --check          # audit an existing install, change nothing
    python setup.py --yes            # accept defaults, no prompts
    python setup.py --skip-deps      # do not create a venv / install packages
    python setup.py --models-dir DIR --model DIR/model.gguf   # fully specified

Note: this is an environment setup script, NOT a setuptools packaging file.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

RELEASE_REPO = "ggml-org/llama.cpp"
API_URL = f"https://api.github.com/repos/{RELEASE_REPO}/releases"
DOWNLOAD_HOST = "https://github.com"
USER_AGENT = "aemon-ai-setup/1.0"

DEFAULT_PORT = 8080
DEFAULT_CTX = 8192
DEFAULT_EXTRA_ARGS = "--flash-attn on --jinja"
HEALTH_TIMEOUT = 180  # seconds to wait for a large model to load

DEP_PACKAGES = [
    "langgraph>=1.0",
    "langgraph-checkpoint-postgres>=3.1",
    "langchain>=1.0",
    "langchain-core>=1.6",
    "langchain-openai>=1.1",
    "langchain-anthropic>=1.4",
    "python-dotenv>=1.0",
    "psycopg[binary,pool]>=3.3",
    "sentence-transformers>=3.0",
    "questionary>=2.0",
    "prompt-toolkit>=3.0",
    "rich>=13.0",
]

# `general.file_type` has been written with three incompatible enums over the
# history of the format (current ggml.h collapses K-quants into families, 2024-era
# llama.cpp used Q4_K_M=15, the GGUF spec doc used Q4_K_M=16). Only these values
# agree across all of them, so the filename tag stays the primary source -- a
# mismatched label here would be worse than no label at all.
STABLE_FILE_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1",
}

# Keys that sit at the very start of the KV section, cheap to reach.
EARLY_KEYS = {"general.name", "general.architecture"}
FAST_READ = 8 << 20
DEEP_READ = 64 << 20

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
BIN_SUFFIX = ".exe" if IS_WINDOWS else ""
EXE_GLOB = "*.exe" if IS_WINDOWS else "*"


# --------------------------------------------------------------------------- #
# Terminal helpers
# --------------------------------------------------------------------------- #

class Style:
    """ANSI styling, disabled when stdout is not a TTY or on legacy consoles."""

    _USE = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    def __init__(self) -> None:
        if IS_WINDOWS:
            os.system("")  # enables VT processing on Windows 10+ consoles
            if not os.environ.get("WT_SESSION") and "ANSCON" not in os.environ:
                # Best effort: legacy cmd.exe ignores ANSI escapes.
                try:
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.GetStdHandle(-11)
                    mode = ctypes.c_uint32()
                    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                except Exception:
                    pass

    def c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._USE else text

    def bold(self, t): return self.c(t, "1")
    def dim(self, t): return self.c(t, "2")
    def red(self, t): return self.c(t, "31")
    def green(self, t): return self.c(t, "32")
    def yellow(self, t): return self.c(t, "33")
    def cyan(self, t): return self.c(t, "36")


S = Style()


def banner(title: str) -> None:
    print(f"\n{S.bold(S.cyan('── ' + title + ' ' + '─' * max(0, 58 - len(title))))}")


def step(msg: str) -> None:
    print(f"{S.cyan('  >>')} {msg}")


def ok(msg: str) -> None:
    print(f"{S.green('  [ok]')} {msg}")


def warn(msg: str) -> None:
    print(f"{S.yellow('  [!!]')} {msg}")


def fail(msg: str) -> None:
    print(f"{S.red('  [xx]')} {msg}")


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def ask(text: str, default: str | None = None) -> str:
    """Prompt for free text, Enter accepts the default."""
    suffix = f" {S.dim(f'[{default}]')}" if default else ""
    while True:
        try:
            raw = input(f"  {S.bold('?')}{text}{suffix}: ").strip()
        except EOFError:
            return default or ""
        if raw:
            return raw
        if default is not None:
            return default
        print(S.yellow("    Please enter a value."))


def confirm(text: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {S.bold('?')} {text} {S.dim(f'[{hint}]')}: ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def select(text: str, options: list[str], default: int = 0) -> int:
    """Numbered picker; returns the chosen index."""
    print(f"  {S.bold(text)}")
    for i, opt in enumerate(options):
        mark = S.cyan("->") if i == default else "  "
        print(f"  {mark} {i + 1}. {opt}")
    while True:
        raw = ask(f"Choose 1-{len(options)}", str(default + 1))
        try:
            idx = int(raw) - 1
        except ValueError:
            idx = default
        if 0 <= idx < len(options):
            return idx
        print(S.yellow(f"    Pick a number between 1 and {len(options)}."))


# --------------------------------------------------------------------------- #
# .env handling -- preserve unknown keys, back up before writing
# --------------------------------------------------------------------------- #

def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)", line)
        if m:
            env[m.group(1)] = m.group(2).strip().strip("'\"")
    return env


def _quote(value: str) -> str:
    if value == "":
        return "''"
    if re.search(r"""[\s#'"]""", value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


def write_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Upsert keys into .env, leaving comments and unrelated keys untouched.

    A timestamped copy of the previous file is kept next to it -- the file is
    never replaced without a recoverable backup.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []

    if path.exists():
        backup = path.with_name(f"{path.name}.bak.{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        ok(f"Backed up existing config to {backup.name}")

    remaining = dict(updates)
    out = []
    for line in lines:
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = m.group(1) if m else None
        if key in remaining:
            out.append(f"{key}={_quote(remaining.pop(key))}")
        else:
            out.append(line)

    if remaining:
        out.append("")
        out.extend(f"{k}={_quote(v)}" for k, v in remaining.items())

    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    ok(f"Wrote {len(updates)} key(s) to {path.name}")


# --------------------------------------------------------------------------- #
# Platform / hardware detection
# --------------------------------------------------------------------------- #

def detect_platform() -> tuple[str, str]:
    """Return (os_key, arch) where os_key is 'macos' | 'linux' | 'windows'."""
    system = platform.system().lower()
    os_key = {"darwin": "macos", "linux": "linux"}.get(system, "windows")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64", "armv8") else "x64"
    if machine in ("s390x",):
        arch = "s390x"
    if os_key == "macos" and machine == "x86_64":
        arch = "x64"
    return os_key, arch


def total_ram_bytes() -> int | None:
    try:
        if IS_WINDOWS:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
            return None
        if IS_MACOS:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            return int(out) if out.isdigit() else None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _runQuiet(cmd: list[str], timeout: int = 15) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (res.stdout or "") + (res.stderr or "")
    except Exception:
        return ""


def detect_cuda_version() -> float | None:
    """Highest CUDA runtime the driver supports, or None when no NVIDIA GPU."""
    out = _runQuiet(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
    if m:
        return float(m.group(1))
    return None


def detect_vulkan() -> bool:
    if shutil.which("vulkaninfo"):
        return True
    if IS_WINDOWS:
        return Path(r"C:\Windows\System32\vulkan-1.dll").exists()
    if IS_MACOS:
        return any(p.startswith("MoltenVK") for p in _moltenvk_paths())
    return bool(glob.glob("/usr/lib/x86_64-linux-gnu/libvulkan.so*") +
                glob.glob("/usr/lib/aarch64-linux-gnu/libvulkan.so*") +
                glob.glob("/usr/local/lib/libvulkan.so*"))


def _moltenvk_paths() -> list[str]:
    return glob.glob("/usr/local/lib/*MoltenVK*") + glob.glob("/opt/homebrew/lib/*MoltenVK*") + \
        glob.glob(os.path.expanduser("~/Library/Frameworks/MoltenVK*"))


def detect_gpu_summary() -> str:
    bits = []
    cuda = detect_cuda_version()
    if cuda:
        bits.append(f"NVIDIA CUDA {cuda}")
    elif IS_MACOS:
        bits.append("Apple Silicon (Metal)")
    if detect_vulkan():
        bits.append("Vulkan")
    if not bits:
        bits.append("CPU only")
    return ", ".join(bits)


# --------------------------------------------------------------------------- #
# llama.cpp release resolution
# --------------------------------------------------------------------------- #

def _gh_get(url: str) -> object:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.load(resp)


def _is_build_tag(tag: str) -> bool:
    return bool(re.fullmatch(r"b\d+", tag))


def latest_binary_release(preferred_tag: str | None = None) -> dict:
    """Find the newest llama.cpp release that actually ships prebuilt binaries.

    The repo's ``latest`` release is a version tag carrying only a pointer file,
    so scan the release list (newest first) for the first ``b<build>`` entry
    that has ``llama-*`` assets attached.
    """
    step("Querying GitHub for the newest llama.cpp binary release")
    url = f"{API_URL}/tags/{preferred_tag}" if preferred_tag else f"{API_URL}?per_page=20"
    try:
        payload = _gh_get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "GitHub rate limit hit (403). Set GITHUB_TOKEN and rerun, or pass "
                "--llama-dir to point at an existing llama-server."
            ) from exc
        if exc.code == 404:
            raise RuntimeError(f"Release tag '{preferred_tag}' not found on GitHub.") from exc
        raise RuntimeError(f"GitHub API error: HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"No network access to GitHub: {exc}") from exc

    candidates = [payload] if isinstance(payload, dict) else list(payload)
    for rel in candidates:
        tag = rel.get("tag_name", "")
        if not (_is_build_tag(tag) or tag.startswith("v")):
            continue
        if _assets_for(rel):
            ok(f"Using llama.cpp {tag} ({len(_assets_for(rel))} packages)")
            return rel
    raise RuntimeError("No llama.cpp release with prebuilt binaries was found.")


def _variant_kind(name: str) -> str:
    name = name.lower()
    for kind in ("cuda", "rocm", "sycl", "openvino", "vulkan"):
        if kind in name:
            return kind
    return "cpu"


def _matches_os_arch(name: str, os_key: str, arch: str) -> bool:
    """True when an asset name targets this OS and CPU architecture.

    Asset names look like ``llama-b11433-bin-ubuntu-cuda-12.8-x64.tar.gz``,
    ``llama-b11433-bin-macos-arm64.tar.gz`` or ``llama-b11433-bin-win-cpu-x64.zip``
    -- every package ends in ``-<arch>.<ext>`` for the platform it targets.
    """
    n = name.lower()
    if not n.startswith("llama-"):
        return False
    if any(junk in n for junk in ("android", "snapdragon", "s390x", "xcframework", "adreno")):
        return False
    ext = ".zip" if os_key == "windows" else ".tar.gz"
    if not n.endswith(f"-{arch}{ext}"):
        return False
    tokens = {"windows": ("-win-",), "linux": ("-ubuntu-", "-linux-"), "macos": ("-macos-",)}
    return any(tok in n for tok in tokens[os_key])


def _assets_for(release: dict) -> list[dict]:
    return [a for a in release.get("assets", []) if a.get("name", "").startswith("llama-")]


def rank_build_candidates(release: dict, os_key: str, arch: str) -> list[dict]:
    """Return build packages usable on this machine, best GPU match first.

    Scores encode preference: a CUDA build the installed driver can actually run
    beats Vulkan, which beats CPU-only. Builds whose CUDA version exceeds the
    driver's are demoted below CPU rather than offered first.
    """
    cuda = detect_cuda_version()
    cuda_build = (int(cuda), int(round((cuda - int(cuda)) * 10))) if cuda else None
    vulkan = detect_vulkan()
    scored: list[tuple[int, dict]] = []

    for asset in _assets_for(release):
        name = asset["name"]
        if not _matches_os_arch(name, os_key, arch):
            continue
        kind = _variant_kind(name)

        if os_key == "macos":
            score = 100  # Metal is already compiled into the macOS builds
        elif kind == "cuda":
            m = re.search(r"cuda-(\d+)\.(\d+)", name)
            if not cuda or not m:
                score = 15  # no NVIDIA GPU (or unknown) -- keep it as a last resort
            else:
                score = 120 if (int(m.group(1)), int(m.group(2))) <= cuda_build else 15
        elif kind == "vulkan":
            score = 90 if vulkan else 30
        elif kind == "rocm":
            score = 60 if _has_amd_gpu() else 10
        elif kind in ("sycl", "openvino"):
            score = 25
        else:
            score = 70

        scored.append((score, asset))

    # Prefer the highest fitting CUDA line when several are compatible.
    scored.sort(key=lambda p: (-p[0], -_cuda_version_key(p[1]["name"]), p[1]["name"]))
    return [asset for _, asset in scored]


def _cuda_version_key(name: str) -> int:
    m = re.search(r"cuda-(\d+)\.(\d+)", name)
    return int(m.group(1)) * 100 + int(m.group(2)) if m else 0


def _has_amd_gpu() -> bool:
    out = _runQuiet(["rocm-smi"]) + _runQuiet(["rocminfo"])
    if re.search(r"gfx[0-9a-f]{4}", out, re.I):
        return True
    if not IS_WINDOWS and Path("/sys/class/drm").exists():
        for card in glob.glob("/sys/class/drm/card*/device/vendor"):
            try:
                if Path(card).read_text().strip() == "0x1002":
                    return True
            except Exception:
                pass
    return False


def describe_asset(asset: dict) -> str:
    kind = _variant_kind(asset["name"])
    if "-macos-" in asset["name"].lower():
        label = "Metal (Apple GPU)"
    else:
        label = {"cuda": "CUDA (NVIDIA GPU)", "rocm": "ROCm (AMD GPU)",
                 "vulkan": "Vulkan (any GPU)", "sycl": "SYCL (Intel GPU)",
                 "openvino": "OpenVINO (Intel)", "cpu": "CPU only"}[kind]
    return f"{label} · {human_size(asset['size'])} · {asset['name']}"


# --------------------------------------------------------------------------- #
# Download + extract
# --------------------------------------------------------------------------- #

def download(url: str, dest: Path, label: str) -> Path:
    """Stream a URL to disk with a stdlib progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        started = time.time()
        last = 0.0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= 0.2 or got == total:
                elapsed = max(now - started, 1e-6)
                speed = got / elapsed
                if total:
                    pct = got / total
                    bar = "#" * int(pct * 30)
                    print(f"\r      {label[:28]:<28} [{bar:<30}] {pct * 100:3.0f}% "
                          f"{speed / 1e6:5.1f} MB/s", end="", flush=True)
                else:
                    print(f"\r      {label[:28]:<28} {human_size(got)} "
                          f"{speed / 1e6:5.1f} MB/s", end="", flush=True)
                last = now
    print()
    tmp.replace(dest)
    return dest


def _safe_members(archive_names: list[str]) -> list[str]:
    bad = [n for n in archive_names if n.startswith(("/", "\\")) or ".." in Path(n).parts]
    if bad:
        raise RuntimeError(f"Refusing to extract unsafe archive entries: {bad[:3]}")
    return archive_names


def extract(archive: Path, dest: Path) -> Path:
    """Extract tar.gz/zip into dest and return the single top-level dir if any."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_members(zf.namelist())
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            _safe_members(tf.getnames())
            try:
                tf.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12 has no filter kwarg
                tf.extractall(dest)

    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        target = inner.name
        for item in inner.iterdir():
            shutil.move(str(item), dest / item.name)
        inner.rmdir()
        # keep a stable, readable install root
        return dest
    return dest


def find_server_binary(root: Path) -> Path | None:
    candidates = [root / f"llama-server{BIN_SUFFIX}"]
    candidates += [Path(p) for p in glob.glob(str(root / "*" / f"llama-server{BIN_SUFFIX}"))]
    candidates += [Path(p) for p in glob.glob(str(root / f"*{BIN_SUFFIX}"))]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def make_executable(root: Path) -> None:
    if IS_WINDOWS:
        return
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in (".so", ".dylib") or path.stat().st_mode & 0o111
                               or path.name.startswith(("llama-", "gg"))):
            try:
                mode = path.stat().st_mode
                path.chmod(mode | 0o111)
            except OSError:
                pass


def install_llama_server(release: dict, asset: dict, install_root: Path, force: bool) -> Path:
    """Download the chosen build (plus the CUDA runtime when paired) and unpack it.

    Everything lands under ``~/.aemon-ai/llama/<tag>/<package>/`` so a re-run is
    idempotent: an already-extracted build is reused, already-downloaded archives
    are not fetched again.
    """
    stem = re.sub(r"\.(tar\.gz|zip)$", "", asset["name"])
    dest = install_root / release["tag_name"] / stem
    cache = install_root / "_downloads"

    if not force:
        cached = find_server_binary(dest)
        if cached:
            ok(f"Already installed in {dest}")
            return cached.resolve()

    dest.mkdir(parents=True, exist_ok=True)
    needed = [asset] + cudart_for(release, asset)

    for item in needed:
        archive = cache / item["name"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            step(f"Downloading {item['name']} ({human_size(item['size'])})")
            download(item["browser_download_url"], archive, item["name"])
        else:
            ok(f"Reusing cached {item['name']}")
        step(f"Extracting {item['name']}")
        extract(archive, dest)

    make_executable(dest)
    server = find_server_binary(dest)
    if not server:
        raise RuntimeError(f"llama-server was not found inside {asset['name']}")
    return server.resolve()


def cudart_for(release: dict, asset: dict) -> list[dict]:
    """Return the cudart package that must sit beside a CUDA build.

    Linux runtime bundles keep the build number in their name while the Windows
    ones do not, so the build tag is stripped before comparing suffixes.
    """
    if _variant_kind(asset["name"]) != "cuda":
        return []
    suffix = re.sub(r"^llama-(b\d+-)?", "", asset["name"])
    matches = [a for a in release["assets"]
               if a["name"].startswith("cudart-") and a["name"].endswith(suffix)]
    if not matches:
        warn(f"No cudart package ending in '{suffix}'; the CUDA build may not start.")
    return matches


def verify_binary(binary: Path) -> str:
    """Run `llama-server --version` as a sanity check."""
    try:
        res = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                             timeout=60, cwd=str(binary.parent))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot execute {binary.name}: {exc}") from exc
    text = (res.stdout + res.stderr).strip()
    if res.returncode != 0:
        raise RuntimeError(f"{binary.name} --version failed:\n{text[-600:]}")
    version = re.search(r"version:\s*(\S+)", text, re.I)
    return version.group(1) if version else text.splitlines()[-1]


def find_existing_install() -> Path | None:
    """Locate a usable llama-server already on this machine."""
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    home = Path.home()
    patterns = [
        home / "llama*" / "build*" / "bin" / f"llama-server{BIN_SUFFIX}",
        home / "llama*" / f"llama-server{BIN_SUFFIX}",
        home / ".aemon-ai" / "llama" / "*" / "*" / f"llama-server{BIN_SUFFIX}",
        home / "Downloads" / "llama-*" / f"llama-server{BIN_SUFFIX}",
        PROJECT_ROOT / "llama*" / "build*" / "bin" / f"llama-server{BIN_SUFFIX}",
    ]
    found: list[Path] = []
    for pat in patterns:
        found += [Path(p) for p in glob.glob(str(pat))]
    return sorted(found, key=lambda p: str(p))[-1] if found else None


# --------------------------------------------------------------------------- #
# GGUF model discovery
# --------------------------------------------------------------------------- #

class _Short(Exception):
    """Raised when the GGUF header extends past what has been read.

    Carries the byte offset to rewind to, so parsing resumes at the start of the
    record that failed instead of restarting from byte zero.
    """

    def __init__(self, rewind: int) -> None:
        super().__init__("incomplete GGUF header")
        self.rewind = rewind


class _Reader:
    __slots__ = ("buf", "i", "mark")

    def __init__(self, buf: bytearray) -> None:
        self.buf = buf
        self.i = 0
        self.mark = 0

    def begin(self) -> None:
        """Remember where the current atomic record started."""
        self.mark = self.i

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.buf):
            raise _Short(self.mark)
        out = self.buf[self.i:self.i + n]
        self.i += n
        return bytes(out)

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))


_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                 10: "Q", 11: "q", 12: "d"}


def _gguf_string(rd: _Reader) -> str:
    (length,) = rd.unpack("Q")
    return rd.take(length).decode("utf-8", errors="replace")


def _gguf_skip(rd: _Reader, vtype: int) -> None:
    """Advance past a value without decoding it (arrays stay cheap this way)."""
    if vtype == 7:  # bool, one byte
        rd.take(1)
    elif vtype == 8:  # string
        (length,) = rd.unpack("Q")
        rd.take(length)
    elif vtype == 9:  # array
        (etype,) = rd.unpack("I")
        (count,) = rd.unpack("Q")
        if etype == 8:
            for _ in range(count):
                (length,) = rd.unpack("Q")
                rd.take(length)
        elif etype == 7:
            rd.take(count)
        elif etype in _GGUF_SCALARS:
            rd.take(struct.calcsize("<" + _GGUF_SCALARS[etype]) * count)
        elif etype == 9:
            for _ in range(count):
                _gguf_skip(rd, 9)
        else:
            raise _Short(rd.mark)
    elif vtype in _GGUF_SCALARS:
        rd.unpack(_GGUF_SCALARS[vtype])
    else:
        raise _Short(rd.mark)


def _gguf_value(rd: _Reader, vtype: int):
    if vtype == 7:
        return rd.unpack("B")[0] != 0
    if vtype == 8:
        return _gguf_string(rd)
    if vtype == 9:
        (etype,) = rd.unpack("I")
        (count,) = rd.unpack("Q")
        items = [_gguf_value(rd, etype) for _ in range(min(count, 4))]
        if count > 4:
            raise _Short(rd.mark)  # caller only wants small arrays
        return items
    return rd.unpack(_GGUF_SCALARS[vtype])[0]


def read_gguf_metadata(path: Path, wanted: set[str], max_read: int = FAST_READ) -> dict:
    """Parse the GGUF key/value header without loading the whole model.

    Metadata sits at the front of the file but its size varies with vocabulary
    length, so the buffer grows in 4 MB steps and parsing resumes exactly where
    it ran out of bytes. Reading stops as soon as every wanted key is found.
    """
    out: dict = {}
    buf = bytearray()
    rd = _Reader(buf)
    header_done = False
    kv_count = 0
    parsed = 0

    with open(path, "rb") as fh:
        while True:
            more = fh.read(4 << 20)
            if more:
                buf.extend(more)

            try:
                if not header_done:
                    rd.begin()
                    if rd.take(4) != b"GGUF":
                        return out
                    (version,) = rd.unpack("I")
                    if version >= 2:
                        rd.unpack("Q")  # n_tensors
                        (kv_count,) = rd.unpack("Q")
                    else:
                        rd.unpack("I")
                        (kv_count,) = rd.unpack("I")
                    header_done = True

                while parsed < kv_count:
                    rd.begin()
                    key = _gguf_string(rd)
                    (vtype,) = rd.unpack("I")
                    if key in wanted and key not in out:
                        out[key] = _gguf_value(rd, vtype)
                        if wanted <= out.keys():
                            return out
                    else:
                        _gguf_skip(rd, vtype)
                    parsed += 1
                return out
            except _Short as exc:
                rd.i = exc.rewind
            except (struct.error, KeyError, UnicodeDecodeError, ValueError):
                return out

            if not more or len(buf) >= max_read:
                return out


def quant_from_name(name: str) -> str:
    """Pull the quant tag out of a llama.cpp-style filename (``...-Q4_K_M.gguf``)."""
    matches = re.findall(r"(?i)\b(iq[0-9]+(?:_[a-z0-9]+)*|q[0-9]+(?:_[a-z0-9]+)*|"
                         r"bf16|fp16|f16|f32)\b", name)
    return matches[-1].upper() if matches else ""


def model_quant_tag(path: Path, meta: dict) -> str:
    """Quant label for a file: filename first, header value only as a fallback."""
    quant = quant_from_name(path.name)
    if quant:
        return quant
    # general.file_type lives at the end of the KV block, so pay for the deep
    # read only when the filename carries no tag.
    deep = read_gguf_metadata(path, {"general.file_type"}, max_read=DEEP_READ)
    return STABLE_FILE_TYPE.get(deep.get("general.file_type"), "")


def scan_gguf_models(root: Path, progress: bool = True) -> list[dict]:
    """Collect loadable GGUF models under ``root``.

    Vocabulary dumps (``ggml-vocab-*.gguf``) and vision adapters (``clip``
    architecture, used as ``--mmproj`` rather than ``-m``) are excluded because
    llama-server cannot serve them as the main model.
    """
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in
                       ("node_modules", "__pycache__", ".git")]
        for fn in filenames:
            if fn.lower().endswith(".gguf"):
                paths.append(Path(dirpath, fn))
    paths.sort()

    models = []
    for idx, path in enumerate(paths, 1):
        if progress:
            print(f"\r      scanning {idx}/{len(paths)} files", end="", flush=True)
        if path.name.lower().startswith("ggml-vocab"):
            continue
        meta = read_gguf_metadata(path, EARLY_KEYS)
        arch = meta.get("general.architecture") or ""
        if not arch:
            # unusual key order -- retry with the whole metadata block before
            # deciding this is not a servable model
            meta = read_gguf_metadata(path, EARLY_KEYS, max_read=DEEP_READ)
            arch = meta.get("general.architecture") or ""
        if not arch or arch == "clip":
            continue  # not a standalone model
        models.append({
            "path": path,
            "name": meta.get("general.name") or path.stem,
            "arch": arch,
            "quant": model_quant_tag(path, meta) or "unknown",
            "size": path.stat().st_size,
        })
    if progress and paths:
        print()
    return models


def default_model_dirs() -> list[Path]:
    home = Path.home()
    guesses = [PROJECT_ROOT / "models", home / "Models", home / "llama.cpp" / "models",
               home / ".cache" / "llama.cpp", home / "Downloads"]
    return [g for g in guesses if g.is_dir()]


def suggest_model_dir() -> Path:
    """Most likely models directory on this machine, or a sane place to create."""
    guesses = default_model_dirs()
    return guesses[0] if guesses else Path.home() / "Models"


def format_model(m: dict) -> str:
    return (f"{m['name']} · {m['arch']} · {m['quant']} · "
            f"{human_size(m['size'])} · {m['path'].parent}")


def pick_model(models: list[dict], ram: int | None) -> tuple[str, dict | None]:
    """Paginated model picker with a RAM-fit hint.

    Returns ``('pick', model)``, or ``('rescan', None)`` / ``('newdir', None)``
    so the caller knows whether to re-scan the same tree or ask for a new one.
    """
    models = sorted(models, key=lambda m: m["size"])
    page, per = 0, 15
    while True:
        start = page * per
        chunk = models[start:start + per]
        total_ram = human_size(ram) if ram else "unknown"
        print(S.dim(f"  {len(models)} model(s), smallest first · system RAM {total_ram} "
                    f"· page {page + 1}"))
        for i, m in enumerate(chunk, start + 1):
            tight = S.yellow("   (tight for RAM)") if ram and m["size"] > ram * 0.75 else ""
            print(f"    {S.cyan(f'{i:>3}.')} {format_model(m)}{tight}")

        nav = ask(S.dim("model number · n next · p prev · r rescan · d new dir"),
                  str(start + 1)).strip().lower()
        if nav in ("n", "next"):
            if start + per < len(models):
                page += 1
            else:
                warn("Already on the last page.")
            continue
        if nav in ("p", "prev"):
            page = max(0, page - 1)
            continue
        if nav in ("r", "rescan"):
            return "rescan", None
        if nav in ("d", "newdir", "dir"):
            return "newdir", None
        try:
            idx = int(nav) - 1
        except ValueError:
            print(S.yellow("    Enter a listed number, or n / p / r / d."))
            continue
        if not 0 <= idx < len(models):
            print(S.yellow(f"    Pick a number between 1 and {len(models)}."))
            continue
        chosen = models[idx]
        if ram and chosen["size"] > ram:
            warn(f"{chosen['name']} ({human_size(chosen['size'])}) is bigger than your "
                 f"{human_size(ram)} of RAM -- expect slow loads or a failure to start.")
        return "pick", chosen


# --------------------------------------------------------------------------- #
# Server smoke test
# --------------------------------------------------------------------------- #

def health_check(binary: Path, model: Path, host: str, port: int, extra: dict) -> bool:
    """Boot llama-server with the chosen model and poll /health."""
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    cmd = [str(binary), "-m", str(model), "--host", host, "--port", str(port),
           "-ngl", str(extra.get("ngl", 999)), "-c", str(extra.get("ctx", DEFAULT_CTX))]
    cmd += extra.get("args", DEFAULT_EXTRA_ARGS).split()

    log = PROJECT_ROOT / ".setup-server.log"
    step(f"Starting test server: {binary.name} with {model.name}")
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(binary.parent))
        url = f"http://{connect_host}:{port}/health"
        deadline = time.time() + HEALTH_TIMEOUT
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    fail(f"Server exited early. Last log lines:\n{log.read_text(errors='replace')[-800:]}")
                    return False
                try:
                    with urllib.request.urlopen(url, timeout=3) as resp:
                        if resp.status == 200:
                            ok("Server responded healthy on " + url)
                            return True
                except Exception:
                    time.sleep(1.5)
                    print(f"\r      waiting for model to load ({int(deadline - time.time())}s left)",
                          end="", flush=True)
        finally:
            print()
            _terminate(proc)
    fail("Timed out waiting for the server. Check .setup-server.log")
    return False


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            _runQuiet(["taskkill", "/F", "/PID", str(proc.pid), "/T"], timeout=20)
        else:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Python dependencies
# --------------------------------------------------------------------------- #

def ensure_requirements() -> None:
    if REQUIREMENTS.exists():
        ok(f"{REQUIREMENTS.name} already exists ({len(REQUIREMENTS.read_text().splitlines())} entries)")
        return
    REQUIREMENTS.write_text(
        "# Generated by setup.py -- direct runtime dependencies for Aemon-AI\n"
        + "\n".join(DEP_PACKAGES) + "\n", encoding="utf-8")
    ok(f"Created {REQUIREMENTS.name}")


def venv_path(root: Path = PROJECT_ROOT) -> Path:
    return root / (".venv/Scripts/python.exe" if IS_WINDOWS else ".venv/bin/python")


def install_dependencies(use_venv: bool) -> None:
    ensure_requirements()
    py = venv_path()
    if use_venv:
        if not py.exists():
            step("Creating virtual environment (.venv)")
            subprocess.run([sys.executable, "-m", "venv", str(PROJECT_ROOT / ".venv")], check=True)
            ok(".venv created")
        target = str(py)
    else:
        target = sys.executable

    step("Installing Python packages (this can take a few minutes)")
    subprocess.run([target, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    res = subprocess.run([target, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if res.returncode:
        warn("pip failed -- install manually: "
             f"{target} -m pip install -r {REQUIREMENTS.name}")
    else:
        ok("Dependencies installed")


def missing_imports() -> list[str]:
    probe = {"langgraph": "langgraph", "langchain": "langchain", "dotenv": "python-dotenv",
             "psycopg": "psycopg[binary,pool]", "sentence_transformers": "sentence-transformers",
             "questionary": "questionary", "rich": "rich",
             "langgraph.checkpoint.postgres.aio": "langgraph-checkpoint-postgres"}
    missing = []
    for module, pkg in probe.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    return missing


# --------------------------------------------------------------------------- #
# Database pre-flight
# --------------------------------------------------------------------------- #

def check_database(url: str) -> bool:
    m = re.match(r"postgresql(?:ql)?://(?:[^@/]*@)?([^:/?]+):?(\d+)?/", url or "")
    if not m:
        warn("DATABASE_URL is missing or not a postgresql:// URL.")
        return False
    host, port = m.group(1), int(m.group(2) or 5432)
    try:
        with socket.create_connection((host, port), timeout=4):
            ok(f"Postgres reachable at {host}:{port}")
            return True
    except OSError as exc:
        fail(f"Cannot reach Postgres at {host}:{port} ({exc.strerror or exc})")
        print(S.dim("""
      Postgres is required for the agent's memory. Hints:
        macOS   brew services start postgresql@17 && createdb aemon-ai
        Linux   sudo systemctl start postgresql && createdb aemon-ai
        Windows install from postgresql.org/download/windows, then:
                "C:\\Program Files\\PostgreSQL\\17\\bin\\createdb.exe" aemon-ai
        Then re-run: python setup.py --check"""))
        return False


# --------------------------------------------------------------------------- #
# Wizard steps
# --------------------------------------------------------------------------- #

def setup_llama_server(non_interactive: bool, force_download: bool,
                       release_tag: str | None) -> dict[str, str]:
    banner("1/4  llama-server")
    os_key, arch = detect_platform()
    print(f"  Platform {S.bold(f'{os_key} {arch}')} · GPU {S.bold(detect_gpu_summary())}")

    install_root = Path.home() / ".aemon-ai" / "llama"

    if not force_download:
        existing = find_existing_install()
        if existing:
            ok(f"Found {existing}")
            try:
                version = verify_binary(existing)
                if non_interactive or confirm(f"Reuse this llama-server ({version})?", True):
                    return {"LLAMA_SERVER_BIN": str(existing), "LLAMA_RELEASE": "pre-existing"}
                step("Will install a fresh build instead")
            except RuntimeError as exc:
                warn(f"{existing.name} is not runnable ({exc}); installing a fresh build")

    release = latest_binary_release(release_tag)
    candidates = rank_build_candidates(release, os_key, arch)
    if not candidates:
        raise RuntimeError(f"No llama.cpp prebuilt package for {os_key}/{arch}. Pass "
                           "--llama-dir with your own llama-server, or build llama.cpp "
                           "from source and point the setup at it.")

    if non_interactive:
        chosen = candidates[0]
        step(f"Auto-selected {chosen['name']}")
    else:
        print()
        chosen = candidates[select("Which llama-server build?",
                                  [describe_asset(c) for c in candidates[:6]], 0)]

    asset = dict(chosen)
    asset.setdefault("browser_download_url",
                     f"{DOWNLOAD_HOST}/{RELEASE_REPO}/releases/download/"
                     f"{release['tag_name']}/{chosen['name']}")
    binary = install_llama_server(release, asset, install_root, force_download)

    version = verify_binary(binary)
    ok(f"llama-server ready: {version}")
    print(S.dim(f"     {binary}"))
    return {"LLAMA_SERVER_BIN": str(binary), "LLAMA_RELEASE": release["tag_name"],
            "LLAMA_BUILD": chosen["name"]}


def _ask_models_dir(current: Path) -> Path:
    raw = ask("Directory that holds your .gguf files", str(current))
    return Path(raw).expanduser()


def setup_model(non_interactive: bool, models_dir: str | None,
                model_path: str | None) -> dict[str, str]:
    banner("2/4  Model directory and GGUF selection")

    if model_path:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"--model not found: {path}")
        ok(f"Using {path.name}")
        return {"LLAMA_MODELS_DIR": str(path.parent), "LOCAL_MODEL_PATH": str(path),
                "LOCAL_MODEL_NAME": path.stem}

    root = Path(models_dir).expanduser() if models_dir else suggest_model_dir()
    ram = total_ram_bytes()
    models: list[dict] = []

    while True:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot use {root}: {exc}") from exc

        step(f"Scanning {root} for .gguf files")
        models = scan_gguf_models(root)
        if models:
            ok(f"Found {len(models)} loadable model(s)")
            break

        warn("No loadable GGUF models here -- vocabulary dumps and mmproj vision "
             "adapters are skipped on purpose.")
        if non_interactive or models_dir:
            raise RuntimeError(f"No GGUF model under {root}. Copy a .gguf file in, "
                               "then rerun: python setup.py")
        root = _ask_models_dir(root)

    if non_interactive:
        fits = [m for m in models if not ram or m["size"] <= ram * 0.75]
        chosen = max(fits or models, key=lambda m: m["size"])
        step(f"Auto-selected {chosen['name']} ({human_size(chosen['size'])})")
    else:
        while True:
            action, picked = pick_model(models, ram)
            if action == "pick":
                chosen = picked
                break
            if action == "newdir":
                root = _ask_models_dir(root)
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                warn(f"Cannot use {root}: {exc}")
                continue
            step(f"Scanning {root} for .gguf files")
            fresh = scan_gguf_models(root)
            if fresh:
                models = fresh
                ok(f"Found {len(fresh)} loadable model(s)")
            else:
                warn("No loadable GGUF models found; keeping the previous list.")

    ok(f"Selected {chosen['name']} ({human_size(chosen['size'])})")
    return {"LLAMA_MODELS_DIR": str(chosen["path"].parent),
            "LOCAL_MODEL_PATH": str(chosen["path"]),
            "LOCAL_MODEL_NAME": chosen["path"].stem}


def setup_runtime(non_interactive: bool) -> dict[str, str]:
    banner("3/4  Runtime defaults")
    host, port, ctx, ngl, extra = "0.0.0.0", DEFAULT_PORT, DEFAULT_CTX, 999, DEFAULT_EXTRA_ARGS
    if not non_interactive:
        host = ask("Bind address (0.0.0.0 exposes the port to your network)", host)
        port = _ask_int("Port", port)
        ctx = _ask_int("Context length in tokens", ctx)
        ngl = _ask_int("GPU layers to offload (-ngl, 999 = all)", ngl)
        extra = ask("Extra llama-server flags", extra)
    return {"LLAMA_HOST": host, "LLAMA_PORT": str(port), "LLAMA_CTX": str(ctx),
            "LLAMA_NGL": str(ngl), "LLAMA_EXTRA_ARGS": extra}


def _ask_int(text: str, default: int) -> int:
    while True:
        raw = ask(text, str(default))
        try:
            return int(raw)
        except ValueError:
            print(S.yellow(f"    Enter a whole number (default {default})."))


def smoke_test(llama: dict[str, str], models: dict[str, str],
               runtime: dict[str, str]) -> bool:
    """Boot the configured server once to prove the setup actually works."""
    banner("4/4  Boot test")
    server = Path(llama["LLAMA_SERVER_BIN"])
    if not server.is_file():
        warn(f"Skipping boot test, no binary at {server}")
        return False
    return health_check(server, Path(models["LOCAL_MODEL_PATH"]), runtime["LLAMA_HOST"],
                        int(runtime["LLAMA_PORT"]),
                        {"ngl": int(runtime["LLAMA_NGL"]), "ctx": int(runtime["LLAMA_CTX"]),
                         "args": runtime["LLAMA_EXTRA_ARGS"]})


# --------------------------------------------------------------------------- #
# --check
# --------------------------------------------------------------------------- #

def run_check() -> int:
    banner("Environment check")
    env = read_env()
    rows: list[tuple[str, bool, str]] = []

    server = env.get("LLAMA_SERVER_BIN", "")
    rows.append(("LLAMA_SERVER_BIN", bool(server) and Path(server).is_file(), server or "not set"))

    model = env.get("LOCAL_MODEL_PATH", "")
    rows.append(("LOCAL_MODEL_PATH", bool(model) and Path(model).is_file(), model or "not set"))

    rows.append(("DATABASE_URL", check_database(env.get("DATABASE_URL", "")),
                 re.sub(r"://[^@]*@", "://***@", env.get("DATABASE_URL", "")) or "not set"))

    missing = missing_imports()
    rows.append(("python packages", not missing, "missing: " + ", ".join(missing) if missing
                 else f"{sys.executable}"))

    for label, good, detail in rows:
        print(f"  {S.green('[ok]') if good else S.red('[xx]')} {label:<18} {S.dim(detail[:78])}")

    if server and Path(server).is_file():
        try:
            ok(f"llama-server runs: {verify_binary(Path(server))}")
        except RuntimeError as exc:
            fail(str(exc))
            return 1
    if model and Path(model).is_file() and server and Path(server).is_file():
        if confirm("Boot the server now for a live test?", False):
            if not health_check(Path(server), Path(model), env.get("LLAMA_HOST", "0.0.0.0"),
                                int(env.get("LLAMA_PORT", DEFAULT_PORT)),
                                {"ngl": int(env.get("LLAMA_NGL", "999")),
                                 "ctx": int(env.get("LLAMA_CTX", str(DEFAULT_CTX))),
                                 "args": env.get("LLAMA_EXTRA_ARGS", DEFAULT_EXTRA_ARGS)}):
                return 1
    print(f"\n{S.bold('Next:')} python agent.py\n")
    return 0 if all(g for _, g, _ in rows) else 1


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="setup.py",
        description="Install llama-server, choose a local GGUF model and write .env "
                    "(macOS / Windows / Linux, stdlib only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[-1] if __doc__ else "")
    p.add_argument("--check", action="store_true", help="verify an existing setup, change nothing")
    p.add_argument("--yes", action="store_true", help="non-interactive, accept defaults")
    p.add_argument("--models-dir", help="directory to scan for .gguf files")
    p.add_argument("--model", help="exact path to a .gguf model")
    p.add_argument("--llama-dir", help="path to an existing llama-server binary or llama.cpp build")
    p.add_argument("--release-tag", help="pin a llama.cpp release, e.g. b11433")
    p.add_argument("--force-download", action="store_true", help="ignore existing llama-server")
    p.add_argument("--skip-deps", action="store_true", help="do not install Python packages")
    return p.parse_args(argv)


def setup_database(existing: str) -> dict[str, str]:
    """Make sure a reachable Postgres URL is configured (memory + checkpoints)."""
    banner("Postgres (checkpoints and long-term memory)")
    if existing and check_database(existing):
        return {}
    if existing:
        warn(f"Current DATABASE_URL does not connect: {existing}")
    default = existing or "postgresql://postgres@localhost:5432/aemon-ai"
    url = ask("DATABASE_URL", default)
    if not check_database(url):
        warn("Keeping the value anyway -- fix Postgres before running the agent.")
    return {"DATABASE_URL": url}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.check:
        print(S.bold("Aemon-AI setup"))
        print(S.dim(f"Project: {PROJECT_ROOT}"))
        print(S.dim(f"Python {platform.python_version()} · {sys.executable}"))
        return run_check()

    print(S.bold("Aemon-AI setup"))
    print(S.dim(f"Project: {PROJECT_ROOT}"))
    print(S.dim(f"Python {platform.python_version()} · {sys.executable}"))

    if not ENV_PATH.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            shutil.copy2(example, ENV_PATH)
            ok("Created .env from .env.example")
        else:
            ENV_PATH.touch()
            warn("Created an empty .env -- fill in any secrets it still needs")

    if args.llama_dir:
        candidate = Path(args.llama_dir).expanduser()
        binary = candidate if candidate.is_file() else find_server_binary(candidate)
        if not binary:
            raise RuntimeError(f"No llama-server found at {candidate}")
        banner("1/4  llama-server")
        llama = {"LLAMA_SERVER_BIN": str(binary.resolve()), "LLAMA_RELEASE": "user-provided"}
        ok(f"Using {binary}")
        verify_binary(binary)
    else:
        llama = setup_llama_server(args.yes, args.force_download, args.release_tag)

    models = setup_model(args.yes, args.models_dir, args.model)
    runtime = setup_runtime(args.yes)

    if not args.skip_deps:
        banner("Python dependencies")
        missing = missing_imports()
        print(f"  Missing packages: {S.dim(', '.join(missing) or 'none, all present')}")
        if args.yes:
            install_dependencies(use_venv=False)
        elif missing or not REQUIREMENTS.exists():
            if confirm("Install dependencies now?", True):
                install_dependencies(
                    confirm("Create an isolated .venv instead of the active environment?", True))
            else:
                ensure_requirements()
        else:
            ensure_requirements()

    updates = {**llama, **models, **runtime}
    write_env(updates)

    if args.yes or confirm("Boot the server once to verify it works?", True):
        if not smoke_test(llama, models, runtime):
            warn("The agent may not start -- rerun the test with: python setup.py --check")

    updates.update(setup_database(read_env().get("DATABASE_URL", "")))
    write_env(updates)

    banner("Done")
    print(f"  Model    {updates.get('LOCAL_MODEL_NAME')}")
    print(f"  Path     {updates.get('LOCAL_MODEL_PATH')}")
    print(f"  Server   {updates.get('LLAMA_SERVER_BIN')}")
    print(f"  Endpoint http://{updates.get('LLAMA_HOST')}:{updates.get('LLAMA_PORT')}")
    print(f"\n{S.bold('Start the agent:')}  python agent.py")
    if venv_path().exists():
        print(S.dim(f"  Or with the venv this script created: "
                    f"{venv_path().relative_to(PROJECT_ROOT)} agent.py"))
    print(S.dim("Re-check anytime with:  python setup.py --check\n"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(S.yellow("\nCancelled -- nothing partially applied was left without a backup."))
        sys.exit(130)
    except RuntimeError as exc:
        fail(str(exc))
        sys.exit(1)
