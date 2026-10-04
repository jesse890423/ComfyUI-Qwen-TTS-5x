# ComfyUI-Qwen-TTS Node Implementation
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import os
import sys
import torch
import numpy as np
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import folder_paths
import types

from comfy import model_management
from comfy.utils import ProgressBar

try:
    from qwen_tts.core.models.modeling_qwen3_tts import set_progress_hook
except Exception:  # model layer unavailable; progress degrades to coarse phases
    set_progress_hook = None

# Register "qwen-tts" model folder for extra_model_paths.yaml support
try:
    folder_paths.add_model_folder_path("qwen-tts", os.path.join(folder_paths.models_dir, "qwen-tts"))
except Exception:
    pass



# Common languages list for UI
DEMO_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]

# Language mapping dictionary to engine codes
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

# Model family options for UI (0.6B / 1.7B)
MODEL_FAMILIES = ["0.6B", "1.7B"]
# Mapping of family to default HuggingFace repo ID
MODEL_FAMILY_TO_HF = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# All required models for batch download
ALL_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
]

_MODELS_CHECKED = False
_MODEL_CACHE = {}

# Handle qwen_tts package import
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_dir = os.path.join(current_dir, "qwen_tts")

# CRITICAL: Add current_dir to sys.path FIRST so Python can resolve 'qwen_tts' as a package
# This allows qwen_tts internal files to use relative imports like 'from ..core.models import ...'
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also ensure the qwen_tts folder itself is importable
if qwen_tts_dir not in sys.path:
    sys.path.insert(0, qwen_tts_dir)

try:
    # 1. Try absolute import first (if user installed via pip)
    import qwen_tts
    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
    VoiceClonePromptItem = qwen_tts.VoiceClonePromptItem
except ImportError:
    try:
        # 2. Fallback to local package import (relative or absolute via sys.path)
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except ImportError as e:
        import traceback
        print(f"\n❌ [Qwen3-TTS] Critical Import Error: {e}")
        if not os.path.exists(qwen_tts_dir):
            print(f"   Missing directory: {qwen_tts_dir}")
            print("   Please clone the repository with submodules or ensure 'qwen_tts' folder exists.")
        else:
            print("   Traceback for debugging:")
            traceback.print_exc()
            print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        
        Qwen3TTSModel = None
        VoiceClonePromptItem = None


# ---------------------------------------------------------------------------
# Path guards
#
# Widget values are stored in the workflow, so whoever shares a workflow also
# decides these strings. Every filesystem path built from a widget is resolved
# against a fixed base directory here, so a shared workflow cannot make this
# plugin read or write files elsewhere on the host.
# ---------------------------------------------------------------------------

def _resolve_within(base_dir, raw, label="path", must_exist=False):
    """Resolve `raw` against `base_dir` and confirm the result stays inside it.

    realpath() collapses symlinks and "..", then commonpath() confirms the
    resolved path really is under the base directory. Both steps are needed:
    realpath alone does not stop a symlink that points outside the base, and
    commonpath alone does not stop an unresolved "..".

    Returns the resolved absolute path, or raises ValueError when the value
    escapes the base directory (or, with must_exist, does not exist).
    """
    base_real = os.path.realpath(base_dir)
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Empty {label}.")

    # Reject absolute paths and Windows drive/UNC prefixes up front so the
    # error message names the actual problem instead of a generic escape.
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        raise ValueError(
            f"{label} must be a relative path inside {base_real}; "
            f"absolute paths are rejected. Got: {text}"
        )

    candidate = os.path.realpath(os.path.join(base_real, text))

    try:
        inside = os.path.commonpath([candidate, base_real]) == base_real
    except ValueError:
        # Raised on Windows when the two paths are on different drives.
        inside = False

    if not inside:
        raise ValueError(
            f"{label} must stay inside {base_real}. Got: {text}"
        )

    if must_exist and not os.path.exists(candidate):
        raise ValueError(f"{label} not found: {candidate}")

    return candidate


def _voices_dir():
    """The single base directory this plugin writes voices into."""
    return os.path.realpath(os.path.join(folder_paths.models_dir, "qwen-tts", "voices"))


def _safe_filename(name, label="filename"):
    """Reduce a user-supplied file name to a bare name inside the base directory.

    Directory separators, drive letters and ".." are rejected outright rather
    than sanitised, so the result is always a plain file name directly inside
    the base directory.
    """
    text = str(name or "").strip()
    if not text:
        raise ValueError(f"Empty {label}.")
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        raise ValueError(f"{label} must be a bare file name, not an absolute path. Got: {text}")
    if "/" in text or "\\" in text:
        raise ValueError(f"{label} must not contain directory separators. Got: {text}")
    # Checked as a path segment, so that "..", "../x" and "a/../b" are all caught.
    if os.pardir in text.replace("\\", "/").split("/"):
        raise ValueError(f"{label} must not contain '..'. Got: {text}")
    if text in (os.curdir, ""):
        raise ValueError(f"{label} must name a file. Got: {text}")
    return text


# The .qvp files hold VoiceClonePromptItem objects, which weights_only=True
# refuses to unpickle unless the class is registered as a safe global.
# Registering it keeps torch.load in safe mode while still letting a user's own
# saved voices load; it does not allow arbitrary code from the file to run.
_SAFE_GLOBALS_REGISTERED = False


def _register_safe_globals():
    """Allow VoiceClonePromptItem to be unpickled by torch.load(weights_only=True)."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    if VoiceClonePromptItem is None:
        return
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:
        return
    try:
        add([VoiceClonePromptItem])
    except Exception as exc:  # pragma: no cover - depends on the torch build
        print(f"[Qwen3-TTS] Could not register VoiceClonePromptItem as a safe global: {exc}")
        return
    _SAFE_GLOBALS_REGISTERED = True


ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]

def check_attention_implementation():
    """Check available attention implementations and return in priority order."""
    available = []

    try:
        from sageattention import sageattn
        available.append("sage_attn")
    except ImportError:
        pass

    try:
        import flash_attn
        available.append("flash_attn")
    except ImportError:
        pass

    available.append("sdpa")
    available.append("eager")

    return available

def get_attention_implementation(selection: str) -> str:
    """Get the actual attention implementation based on selection and availability."""
    available = check_attention_implementation()

    if selection == "auto":
        priority = ["sage_attn", "flash_attn", "sdpa", "eager"]
        for attn in priority:
            if attn in available:
                print(f"[Qwen3-TTS] Auto-selected attention: {attn}")
                return attn
        return "eager"
    else:
        if selection in available:
            print(f"[Qwen3-TTS] Using attention: {selection}")
            return selection
        else:
            print(f"[Qwen3-TTS] Requested attention '{selection}' not available, falling back to sdpa")
            if "sdpa" in available:
                return "sdpa"
            return "eager"


def split_text_by_pauses(text: str, config: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Split text into segments based on punctuation and defined pause durations.
    Returns: List of (segment_text, pause_duration_after)
    """
    if not config:
        return [(text, 0.0)]

    pause_linebreak = config.get("pause_linebreak", 0.5)
    period_pause = config.get("period_pause", 0.4)
    comma_pause = config.get("comma_pause", 0.2)
    question_pause = config.get("question_pause", 0.6)
    hyphen_pause = config.get("hyphen_pause", 0.3)

    # Handle linebreaks: replace \n with break tags before punctuation processing
    if pause_linebreak > 0:
        text = re.sub(r'\n+', f' [break={pause_linebreak}] ', text)

    # Inject break tags for both English and Chinese punctuation
    if period_pause > 0:
        text = re.sub(r'[\.。](?!\d)', rf'\g<0> [break={period_pause}]', text)
    if comma_pause > 0:
        text = re.sub(r'[,，](?!\d)', rf'\g<0> [break={comma_pause}]', text)
    if question_pause > 0:
        text = re.sub(r'[\?？](?!\d)', rf'\g<0> [break={question_pause}]', text)
    if hyphen_pause > 0:
        text = re.sub(r'[-—](?!\d)', rf'\g<0> [break={hyphen_pause}]', text)

    # Process explicit break tags
    pause_pattern = r'\[break=([\d\.]+)\]'
    parts = re.split(pause_pattern, text)
    
    segments = []
    current_segment_text = ""
    
    for i in range(len(parts)):
        chunk = parts[i]
        if i % 2 == 0:
            # This is text
            if chunk.strip():
                if current_segment_text:
                    # We had previous text waiting for a pause, but found text instead.
                    # This shouldn't happen with our regex unless there's no punctuation between them.
                    segments.append((current_segment_text.strip(), 0.0))
                current_segment_text = chunk
        else:
            # This is a pause duration
            try:
                pause_val = float(chunk)
            except ValueError:
                pause_val = 0.0
                
            if current_segment_text:
                segments.append((current_segment_text.strip(), pause_val))
                current_segment_text = ""
            elif segments and pause_val > 0:
                # Add this pause to the previous segment (consecutive pauses)
                prev_txt, prev_pause = segments[-1]
                segments[-1] = (prev_txt, prev_pause + pause_val)

    if current_segment_text.strip():
        segments.append((current_segment_text.strip(), 0.0))
        
    return segments


def unload_cached_model(cache_key=None):
    """Unload cached model(s) and clear GPU memory.

    Args:
        cache_key: If provided, only unload that specific model.
                  If None, unload all cached models.
    """
    global _MODEL_CACHE

    if cache_key and cache_key in _MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading model: {cache_key}...")
        del _MODEL_CACHE[cache_key]
    elif _MODEL_CACHE:
        print(f"[Qwen3-TTS] Unloading {len(_MODEL_CACHE)} cached model(s)...")
        _MODEL_CACHE.clear()

    model_management.soft_empty_cache()

    import gc
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[Qwen3-TTS] Model cache and GPU memory cleared")




def apply_qwen3_patches(model):
    """Apply stability and compatibility patches to the model instance"""
    if model is None:
        return
    
    # 1. Monkeypatch model._normalize_audio_inputs to return mutable lists to avoid upstream tuple assignment bug
    # and to support (waveform, sr) tuple format from ComfyUI
    orig_normalize = getattr(model, "_normalize_audio_inputs", None)
    
    def _safe_normalize(self, audios):
        # Adapted from upstream but returns list entries as lists [waveform, sr]
        # and correctly handles the (waveform, sr) tuple format
        if isinstance(audios, list):
            items = audios
        elif isinstance(audios, tuple) and len(audios) == 2 and isinstance(audios[0], np.ndarray):
            # This is a single audio in tuple format (waveform, sr)
            items = [audios]
        else:
            items = [audios]

        out = []
        for a in items:
            if a is None:
                continue
            if isinstance(a, str):
                wav, sr = self._load_audio_to_np(a)
                out.append([wav.astype(np.float32), int(sr)])
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, list) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                # If we still can't identify it, it might be the cause of our NoneType or other issues
                print(f"⚠️ [Qwen3-TTS] Unknown audio input type: {type(a)}")
                continue

        # ensure mono
        for i in range(len(out)):
            wav, sr = out[i][0], out[i][1]
            if wav.ndim > 1:
                out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
        return out

    try:
        model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS] Failed to apply audio normalization patch: {e}")


def download_model_if_needed(model_id: str, qwen_root: str) -> str:
    """Download a specific model if not found locally"""
    folder_name = model_id.split("/")[-1]
    target_dir = os.path.join(qwen_root, folder_name)
    
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Model already exists
        return target_dir
    
    print(f"\n📥 [Qwen3-TTS] Downloading {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=target_dir)
        print(f"✅ [Qwen3-TTS] {folder_name} downloaded successfully.\n")
        return target_dir
    except ImportError:
        print("⚠️ [Qwen3-TTS] 'huggingface_hub' not found. Please install it to use auto-download.")
        return None
    except Exception as e:
        print(f"❌ [Qwen3-TTS] Failed to download {model_id}: {e}")
        return None


def check_and_download_tokenizer():
    """Check and download tokenizer (shared by all models)"""
    global _MODELS_CHECKED
    if _MODELS_CHECKED:
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ComfyUI root search
    comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
    qwen_root = os.path.join(comfy_models_path, "qwen-tts")
    os.makedirs(qwen_root, exist_ok=True)

    # Download tokenizer (required by all models)
    tokenizer_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    download_model_if_needed(tokenizer_id, qwen_root)
    
    _MODELS_CHECKED = True


def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str, attention: str = "auto", unload_after: bool = False, previous_attention: str = None, custom_model_path: Optional[str] = None):
    """Shared model loading logic with caching and local path priority"""
    global _MODEL_CACHE
    
    if previous_attention is not None and previous_attention != attention:
        print(f"🔄 [Qwen3-TTS] Attention changed from '{previous_attention}' to '{attention}', clearing cache...")
        unload_cached_model()
    
    attn_impl = get_attention_implementation(attention)
    
    # Check and download tokenizer (shared by all models)
    check_and_download_tokenizer()
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # 针对 Mac 的关键修复
        else:
            device = "cpu"
    
    if device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")
        
    # Cache key includes attention implementation and custom model path
    cache_key = (model_type, model_choice, device, precision, attn_impl, custom_model_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Clear old cache only when adding a new model with different config
    if _MODEL_CACHE:
        _MODEL_CACHE.clear()
    
    # --- 1. Determine search directories ---
    base_paths = []
    try:
        # Resolve ComfyUI root
        import folder_paths
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        qwen_tts_dir = os.path.join(comfy_root, "models", "qwen-tts")
        if os.path.exists(qwen_tts_dir):
            base_paths.append(qwen_tts_dir)
        else:
            # Compatibility check: models/qwen-tts in parent dir
            alt_root = os.path.dirname(comfy_root)
            alt_qwen_tts_dir = os.path.join(alt_root, "models", "qwen-tts")
            if os.path.exists(alt_qwen_tts_dir):
                base_paths.append(alt_qwen_tts_dir)
    except Exception:
        pass

    # Check registered "qwen-tts" paths (includes extra_model_paths.yaml)
    try:
        qwen_paths = folder_paths.get_folder_paths("qwen-tts") or []
        for p in qwen_paths:
            if p not in base_paths: base_paths.append(p)
    except Exception: pass

    # Check registered TTS paths in folder_paths (Legacy)
    try:
        registered_tts = folder_paths.get_folder_paths("TTS") or []
        for p in registered_tts:
            if p not in base_paths: base_paths.append(p)
    except Exception: pass

    # --- 2. Search for matching models ---
    HF_MODEL_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    
    final_source = HF_MODEL_MAP.get((model_type, model_choice)) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    found_local = None

    if custom_model_path and isinstance(custom_model_path, str) and custom_model_path.strip():
        if os.path.exists(custom_model_path) and os.path.isdir(custom_model_path):
            print(f"🔧 [Qwen3-TTS] Using custom model path: {custom_model_path}")
            found_local = custom_model_path
        else:
            print(f"⚠️ [Qwen3-TTS] Custom model path not found or invalid: {custom_model_path}")
    
    if not found_local:
        for base in base_paths:
            try:
                if not os.path.isdir(base): continue
                subdirs = os.listdir(base)
                for d in subdirs:
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand):
                        # Match logic: contains model size and type keyword
                        if model_choice in d and model_type.lower() in d.lower():
                            found_local = cand
                            break
                if found_local: break
            except Exception: pass
    
    if found_local:
        final_source = found_local
        print(f"✅ [Qwen3-TTS] Loading local model: {os.path.basename(final_source)}")
    else:
        # Try to download the specific model if not found locally
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        qwen_root = os.path.join(comfy_models_path, "qwen-tts")
        
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-TTS] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            # Fall back to remote loading if download failed
            print(f"🌐 [Qwen3-TTS] Loading remote model: {final_source}")

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "❌ [Qwen3-TTS] Model class is not loaded because the 'qwen_tts' package failed to import. "
            "Please check the ComfyUI console for the detailed 'Critical Import Error' above."
        )

    # Map attention implementation to model loading parameter
    attn_param = None
    use_sage_attn = False
    
    if attn_impl == "flash_attn":
        attn_param = "flash_attention_2"
    elif attn_impl == "sage_attn":
        use_sage_attn = True
    elif attn_impl == "sdpa":
        attn_param = "sdpa"
    elif attn_impl == "eager":
        attn_param = "eager"
    
    # Handle sage_attn (sageattention package)
    if use_sage_attn:
        try:
            from sageattention import sageattn
            print(f"🔧 [Qwen3-TTS] Loading model with sage_attn (sageattention)")
            
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
            
            # Patch attention modules to use sageattention
            patched_count = 0
            for name, module in model.model.named_modules():
                if hasattr(module, 'forward') and 'Attention' in type(module).__name__ or 'attn' in name.lower():
                    try:
                        original_forward = module.forward
                        def make_sage_forward(orig_forward, mod):
                            def sage_forward(*args, **kwargs):
                                # Extract q, k, v from attention call
                                if len(args) >= 3:
                                    q, k, v = args[0], args[1], args[2]
                                else:
                                    return orig_forward(*args, **kwargs)
                                
                                # Handle attention_mask
                                attn_mask = kwargs.get('attention_mask', None)
                                
                                # Call sageattention
                                out = sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                                return out
                            return sage_forward
                        
                        module.forward = make_sage_forward(original_forward, module)
                        patched_count += 1
                    except Exception:
                        pass
            
            print(f"🔧 [Qwen3-TTS] Patched {patched_count} attention modules with sage_attn")
            
        except (ImportError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS] Failed with sage_attn, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
    else:
        try:
            if attn_param:
                print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, attn_implementation=attn_param)
            else:
                print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
        except (ImportError, ValueError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS] Failed with {attn_impl}, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
    
    # Apply patches
    apply_qwen3_patches(model)
    
    _MODEL_CACHE[cache_key] = model
    
    if unload_after:
        def unload_callback():
            unload_cached_model()
        model._unload_callback = unload_callback
    else:
        model._unload_callback = None
    
    return model

def _current_node_id():
    """Return the executing ComfyUI node id, or None outside execution."""
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        if ctx is not None:
            node_id = getattr(ctx, "node_id", None)
            if node_id:
                return node_id
    except Exception:
        pass
    return None


# The active TTSProgress console bar. A module-level one-slot holder because
# _emit_progress is a plain function; set/cleared by register()/unregister()
# so sequential nodes never inherit each other's bar.
_CONSOLE_BAR = [None]


def _emit_progress(value, total):
    """Report progress for the executing node, preferring the official API.

    comfy_api.latest.Execution.set_progress() resolves the node id from the
    execution context and calls ProgressRegistry.update_progress() directly.
    That is the documented path and the one ComfyUI's own code uses for the
    node panel bar, so prefer it.

    comfy.utils.ProgressBar is not used as the transport on purpose: its global
    hook forwards node_id straight through, and ProgressBar's default node_id is
    None (ComfyUI's own code passes none either, e.g.
    comfy_api/latest/_input_impl/video_types.py:772). That writes the update to
    a None-keyed registry entry, which the frontend -- matching progress_state
    entries by node id -- never displays. Measured: the executing node's entry
    stayed {value: 0.0, max: 1.0} for an entire run that way.

    ProgressBar is still worth using for its throttling, so the caller keeps it
    and we only decide where the value goes.
    """
    _CONSOLE_BAR[0].update(value, total) if _CONSOLE_BAR[0] else None
    node_id = _current_node_id()
    if node_id is None:
        return False
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().update_progress(
            node_id=node_id, value=value, max_value=total
        )
        return True
    except Exception:
        return False

class _ConsoleBar:
    """A self-contained stderr progress bar.

    ComfyUI 0.37 no longer registers CLIProgressHandler, so nothing draws a
    console bar for custom nodes any more. This restores one without touching
    global state.
    """

    MIN_INTERVAL = 0.1          # seconds between redraws
    WIDTH = 28                  # bar width in characters

    def __init__(self, label="Qwen3-TTS"):
        self.label = label
        self.total = 0
        self.value = 0
        self._last = 0.0
        self._closed = False
        # Decide once, and re-validate in _draw: a console-less launcher (or a
        # redirected stream) must never receive a byte, no matter how the object
        # is poked from the outside.
        self.tty = self._detect_tty()

    @staticmethod
    def _detect_tty():
        try:
            import sys

            return bool(getattr(sys.stderr, "isatty", lambda: False)())
        except Exception:
            return False

    def start(self, total):
        self.total = max(1, int(total or 1))
        self.value = 0
        self._last = 0.0
        self._closed = False
        self._draw(force=True)

    def update(self, value, total=None):
        if self._closed or not self.tty:
            return
        if total is not None:
            self.total = max(1, int(total))
        self.value = value
        self._draw()

    def close(self, message="done"):
        if self._closed or not self.tty:
            self._closed = True
            return
        self._closed = True
        self._draw(force=True, message=message)

    def _draw(self, force=False, message=None):
        if not self.tty or not self._detect_tty():
            self.tty = False
            return
        try:
            now = time.perf_counter()
            if not force and now - self._last < self.MIN_INTERVAL:
                return
            self._last = now
            frac = 0.0 if not self.total else max(0.0, min(1.0, self.value / self.total))
            filled = int(round(frac * self.WIDTH))
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            # `{}` not `{bar}`: str.format() treats a bare name as a *named*
            # placeholder and looks it up in kwargs, so the local variable `bar`
            # was invisible and every call raised KeyError: 'bar'. The except
            # branch then disabled the bar, so nothing was ever written.
            text = "\r{} [{}] {}/{} ({:.0f}%) {}".format(
                self.label, bar, self.value, self.total, frac * 100.0, message or ""
            )
            # Use the module-level `sys`. A function-local `import sys` rebinds
            # the name to the object ComfyUI installed in this module's globals
            # at import time, and that object does not see a redirected
            # sys.stderr: measured 0 bytes written with the local import vs 144
            # bytes with the module-level one, on the same object.
            sys.stderr.write(text + ("\n" if message else ""))
            sys.stderr.flush()
        except Exception:
            # never let a progress bar break generation
            self.tty = False

class TTSProgress:
    """Real per-token progress for one generation call.

    The bar is 100 slots wide on purpose. ComfyUI renders ProgressBar as a
    percentage, and a coarse bar (the original 3 slots) cannot express
    per-token progress at all: a 30-step generation spread over 3 slots yields
    1 -> 2 -> 3, which looks exactly like the frozen bar this replaces. At 100
    slots the bar tracks the sample count, and ProgressBar's own throttling
    (min interval + min percent change) keeps websocket traffic low.

    The hook is scoped to a single generate() call via register()/unregister(),
    so sequential or concurrent nodes never cross-talk. Everything is guarded:
    progress reporting must never be able to break generation.
    """

    # Measured on Qwen3-TTS-12Hz-1.7B: 12 chars -> 21 steps, 19 -> 52, 40 -> 84.
    STEPS_PER_CHAR = 2.1
    MIN_BUDGET = 24

    @classmethod
    def estimate_budget(cls, text, max_new_tokens):
        """Realistic step budget for `text`, clamped by the configured ceiling.

        max_new_tokens is a hard cap (2048 by default) that generation never
        approaches, because the talker stops on codec_eos_token_id. Using it as
        the progress denominator keeps the ratio near zero for the whole run.
        """
        try:
            chars = len(text or "")
        except Exception:
            chars = 0
        ceiling = max(1, int(max_new_tokens or 1))
        return max(1, min(int(chars * cls.STEPS_PER_CHAR) + cls.MIN_BUDGET, ceiling))

    def __init__(self, max_new_tokens, phase_total=100, chunk_total=None,
                 text=None):
        if text is None:
            self.max_new_tokens = max(1, int(max_new_tokens or 1))
        else:
            self.max_new_tokens = self.estimate_budget(text, max_new_tokens)
        # batched generation: the bar is scaled so every chunk owns a slice
        self.chunk_total = int(chunk_total) if chunk_total else None
        self.phase_total = int(phase_total)
        self.pbar = ProgressBar(phase_total, node_id=_current_node_id())
        self.cbar = _ConsoleBar()
        self._active = False
        self._base = 0
        self._span = self.phase_total

    def register(self, span=None):
        """Start observing a generate() call.

        `span` is how many bar slots this call may fill. Single-shot nodes leave
        it None and hand the entire bar to the upcoming call; batched nodes pass
        the per-chunk slice width.
        """
        if span is not None:
            self._span = max(1, int(span))
            self._base = int(getattr(self.pbar, "current", 0) or 0)
        else:
            self._span = self.phase_total
            self._base = 0
        _CONSOLE_BAR[0] = self.cbar
        self.cbar.start(self.phase_total)
        if set_progress_hook is not None and not self._active:
            try:
                set_progress_hook(self._on_step)
                self._active = True
            except Exception:
                self._active = False
        return self

    def unregister(self):
        if self._active and set_progress_hook is not None:
            try:
                set_progress_hook(None)
            except Exception:
                pass
        self._active = False
        if _CONSOLE_BAR[0] is self.cbar:
            _CONSOLE_BAR[0] = None

    def finish_generation(self):
        """The current generate() call ended; release its slice of the bar."""
        self.cbar.start(self.phase_total)
        self._active = False
        if set_progress_hook is not None:
            try:
                set_progress_hook(None)
            except Exception:
                pass
        try:
            self._base = int(getattr(self.pbar, "current", 0) or 0)
        except Exception:
            self._base = 0

    def _on_step(self, step, total):
        try:
            # The node's max_new_tokens is the ceiling the user configured and
            # therefore the honest denominator. The hook's `total` is the model's
            # own default (8192), which is unrelated to the node setting and
            # would keep the ratio pinned near zero for the whole run.
            budget = self.max_new_tokens
            if budget <= 1 and total:
                budget = max(1, int(total))
            budget = max(1, int(budget))
            value = self._base + int(round(self._span * min(int(step), budget) / budget))
            value = max(1, min(value, self.phase_total))
            if not _emit_progress(value, self.phase_total):
                self.pbar.update_absolute(value, self.phase_total, None)
        except Exception:
            pass

    def phase_loaded(self):
        """Model ready; enter the generation phase (never rewinds the bar)."""
        try:
            current = int(getattr(self.pbar, "current", 0) or 0)
        except Exception:
            current = 0
        if not _emit_progress(min(1, current), self.phase_total):
            self.pbar.update_absolute(min(1, current), self.phase_total, None)

    def phase_done(self):
        """Generation finished; complete the bar."""
        self.cbar.close("audio ready")
        if not _emit_progress(self.phase_total, self.phase_total):
            self.pbar.update_absolute(self.phase_total, self.phase_total, None)

class VoiceDesignNode:
    """
    VoiceDesign Node: Generate custom voices based on text descriptions.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize"}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (required for VoiceDesign)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceDesign: Generate custom voices from descriptions."

    def generate(self, text: str, instruct: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        pbar = TTSProgress(max_new_tokens, text=text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "VoiceDesign":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("VoiceDesign", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.phase_loaded()

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        pbar.register()
        wavs, sr = model.generate_voice_design(
            text=text,
            language=mapped_lang,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        pbar.phase_done()
        pbar.unregister()

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)

            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class VoiceCloneNode:
    """
    VoiceClone (Base) Node: Create clones from reference audio and synthesize target text.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "target_text": ("STRING", {"multiline": True, "default": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT", {"tooltip": "Reusable voice clone prompt from VoiceClonePromptNode"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
                "custom_model_path": ("STRING", {"default": "", "placeholder": "Absolute path to local fine-tuned model"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClone: Clone voice from reference audio."

    def _audio_tensor_to_tuple(self, audio_tensor: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        # Accept multiple possible ComfyUI audio formats
        waveform = None
        sr = None
        try:
            if isinstance(audio_tensor, dict):
                # Common keys: 'waveform'/'sample_rate' or 'data'/'sampling_rate'
                if "waveform" in audio_tensor:
                    waveform = audio_tensor.get("waveform")
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate")
                elif "data" in audio_tensor and "sampling_rate" in audio_tensor:
                    waveform = np.asarray(audio_tensor.get("data"))
                    sr = audio_tensor.get("sampling_rate")
                elif "audio" in audio_tensor and isinstance(audio_tensor["audio"], (tuple, list)):
                    # maybe {'audio': (sr, data)}
                    a = audio_tensor["audio"]
                    if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], (int, float)):
                        sr, waveform = int(a[0]), np.asarray(a[1])
                else:
                    # fallback: try common numeric fields
                    for k in ("samples", "y", "wave"):
                        if k in audio_tensor:
                            waveform = np.asarray(audio_tensor.get(k))
                            break
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate") or audio_tensor.get("rate")
            elif isinstance(audio_tensor, tuple) and len(audio_tensor) == 2:
                # (waveform, sr) or (sr, waveform)
                a0, a1 = audio_tensor
                if isinstance(a0, (int, float)) and isinstance(a1, (list, np.ndarray, torch.Tensor)):
                    sr = int(a0)
                    waveform = np.asarray(a1)
                elif isinstance(a1, (int, float)) and isinstance(a0, (list, np.ndarray, torch.Tensor)):
                    sr = int(a1)
                    waveform = np.asarray(a0)
            elif isinstance(audio_tensor, list):
                # maybe [waveform, sr]
                if len(audio_tensor) == 2 and isinstance(audio_tensor[0], (list, np.ndarray)) and isinstance(audio_tensor[1], (int, float)):
                    waveform = np.asarray(audio_tensor[0])
                    sr = int(audio_tensor[1])
        except Exception:
            pass
        # Normalize to 1-D numpy float32 array (model expects 1-D waveforms)
        if isinstance(waveform, torch.Tensor):
            # ComfyUI audio is often [batch, channels, samples] or [channels, samples]
            if waveform.dim() > 1:
                # Squeeze out any unit dimensions (like batch=1, channel=1)
                waveform = waveform.squeeze()
                # If still multi-dimensional (e.g. stereo), average to mono
                if waveform.dim() > 1:
                    waveform = torch.mean(waveform, dim=0)
            waveform = waveform.cpu().numpy()

        if isinstance(waveform, np.ndarray):
            # Double check for numpy version of same logic
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform)
                if waveform.ndim > 1:
                    # Heuristic: the smaller dimension is likely channels
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = np.mean(waveform, axis=0)
                    else:
                        waveform = np.mean(waveform, axis=1)
            waveform = waveform.astype(np.float32)

        # Final safety flatten to ensure it's 1-D
        if waveform is not None and waveform.ndim > 1:
            waveform = waveform.flatten()

        # Basic validation
        if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0:
            raise RuntimeError("Failed to parse reference audio waveform")
        
        min_samples = 1024
        if waveform.size < min_samples:
            # Pad with zeros to avoid upstream padding errors
            pad_to = min_samples
            pad_amount = pad_to - waveform.size
            waveform = np.concatenate([waveform, np.zeros(pad_amount, dtype=np.float32)])

        # Return as tuple (waveform, sr) with 1-D numpy waveform as expected by the tokenizer
        return (waveform, int(sr))

    def generate(self, target_text: str, model_choice: str, device: str, precision: str, language: str, 
                 ref_audio: Optional[Dict[str, Any]] = None, ref_text: str = "", 
                 voice_clone_prompt: Optional[Any] = None, seed: int = 0, 
                 max_new_tokens: int = 2048,
                 top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05,
                 x_vector_only: bool = False, attention: str = "auto",
                 unload_model_after_generate: bool = False, custom_model_path: str = "") -> Tuple[Dict[str, Any]]:
        if ref_audio is None and voice_clone_prompt is None:
            raise RuntimeError("Either reference audio or voice clone prompt is required")

        pbar = TTSProgress(max_new_tokens, text=target_text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention, custom_model_path)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        pbar.phase_loaded()

        audio_tuple = None
        if ref_audio is not None:
            if isinstance(ref_audio, dict):
                audio_tuple = self._audio_tensor_to_tuple(ref_audio)
            else:
                raise RuntimeError("Unsupported reference audio format")

        try:
            mapped_lang = LANGUAGE_MAP.get(language, "auto")

            voice_clone_prompt_param = None
            ref_audio_param = None
            if voice_clone_prompt is not None:
                voice_clone_prompt_param = voice_clone_prompt
            elif ref_audio is not None:
                ref_audio_param = audio_tuple
            else:
                raise RuntimeError("Either 'ref_audio' or 'voice_clone_prompt' must be provided")

            pbar.register()
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=ref_audio_param,
                ref_text=ref_text if ref_text and ref_text.strip() else None,
                voice_clone_prompt=voice_clone_prompt_param,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        pbar.phase_done()
        pbar.unregister()

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)

            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class CustomVoiceNode:
    """
    CustomVoice (TTS) Node: Generate text-to-speech using preset speakers.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize"}),
                "speaker": (["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"], {"default": "Ryan"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (optional)"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
                "custom_model_path": ("STRING", {"default": "", "placeholder": "Absolute path to local fine-tuned model"}),
                "custom_speaker_name": ("STRING", {"default": "", "placeholder": "Custom speaker name (for fine-tuned models)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "CustomVoice: Generate speech using preset speakers."

    def generate(self, text: str, speaker: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, instruct: str = "", max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False, custom_model_path: str = "", custom_speaker_name: str = "") -> Tuple[Dict[str, Any]]:
        # Prefer custom_speaker_name if provided
        target_speaker = speaker
        if custom_speaker_name and custom_speaker_name.strip():
            target_speaker = custom_speaker_name.strip()
        else:
            target_speaker = speaker.lower().replace(" ", "_")

        if not text:
            raise RuntimeError("Text is required")

        pbar = TTSProgress(max_new_tokens, text=text)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "CustomVoice":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("CustomVoice", model_choice, device, precision, attention, unload_model_after_generate, previous_attention, custom_model_path)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.phase_loaded()

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        pbar.register()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=mapped_lang,
            speaker=target_speaker,
            instruct=instruct if instruct and instruct.strip() else None,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        pbar.phase_done()
        pbar.unregister()

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)

            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class VoiceClonePromptNode:
    """
    VoiceClonePrompt Node: Extract voice features from reference audio for reuse.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "If True, only speaker embedding is extracted (ref_text not needed)"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClonePrompt: Extract and cache voice features for reuse in VoiceClone node."

    def create_prompt(self, ref_audio: Dict[str, Any], ref_text: str, model_choice: str, device: str, precision: str, attention: str, x_vector_only: bool = False, unload_model_after_generate: bool = False) -> Tuple[Any]:
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")

        # prompt extraction is a single encoder pass, not token generation:
        # there is no max_new_tokens here and no per-token progress to report
        pbar = TTSProgress(1, text="")

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.phase_loaded()

        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        pbar.phase_loaded()

        vcn = VoiceCloneNode()
        pbar.register()
        audio_tuple = vcn._audio_tensor_to_tuple(ref_audio)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text if ref_text and ref_text.strip() else None,
            x_vector_only_mode=x_vector_only,
        )

        pbar.phase_done()
        pbar.unregister()

        if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
            model._unload_callback()

        return (prompt_items,)


class RoleBankNode:
    """
    RoleBank Node: Manage a collection of voice prompts mapped to names.
    Supports up to 8 roles per node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {},
            "optional": {}
        }
        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            inputs["optional"][f"prompt_{i}"] = ("VOICE_CLONE_PROMPT",)
        return inputs

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create_bank"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "RoleBank: Collect multiple voice prompts into a named registry for dialogue generation."

    def create_bank(self, **kwargs) -> Tuple[Dict[str, Any]]:
        bank = {}
        for i in range(1, 9):
            name = kwargs.get(f"role_name_{i}", "").strip()
            prompt = kwargs.get(f"prompt_{i}")
            if name and prompt is not None:
                bank[name] = prompt
        return (bank,)


class DialogueInferenceNode:
    """
    DialogueInference Node: Generate multi-role continuous dialogue from a script.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.", "placeholder": "Format: RoleName: Text"}),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                # RENAMED: pause_seconds -> pause_linebreak
                # Linebreak Pause
                "pause_linebreak": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration between lines"}),
                # NEW: Period and Comma pause settings
                "period_pause": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after periods (.)"}),
                "comma_pause": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after commas (,)"}),
                # ### NEW INPUTS ###
                "question_pause": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after question marks (?)"}),
                "hyphen_pause": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after hyphens (-)"}),
                
                "merge_outputs": ("BOOLEAN", {"default": True, "tooltip": "Merge all dialogue segments into a single long audio"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Number of lines to process in parallel. Larger = faster but more VRAM."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens_per_line": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "DialogueInference: Execute a script with multiple roles and generate continuous speech."
    # ### UPDATED ARGUMENTS: Added question_pause, hyphen_pause
    def generate_dialogue(self, script: str, role_bank: Dict[str, Any], model_choice: str, device: str, precision: str, language: str, pause_linebreak: float, period_pause: float, comma_pause: float, question_pause: float, hyphen_pause: float, merge_outputs: bool, batch_size: int, seed: int = 0, max_new_tokens_per_line: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not script or not role_bank:
            raise RuntimeError("Script and Role Bank are required")

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        lines = script.strip().split("\n")

        texts_to_gen = []
        prompts_to_gen = []
        langs_to_gen = []
        pauses_to_gen = []

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        pause_pattern = r'\[break=([\d\.]+)\]'

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or ":" not in line and "：" not in line:
                continue

            pos_en = line.find(":")
            pos_cn = line.find("：")

            if pos_en == -1 and pos_cn == -1:
                continue

            if pos_en != -1 and (pos_cn == -1 or pos_en < pos_cn):
                role_name, text = line.split(":", 1)
            else:
                role_name, text = line.split("：", 1)

            role_name = role_name.strip()
            text = text.strip()

            if role_name not in role_bank:
                continue
            # role_bank[role_name] can be a prompt list OR the new packaged dict (if loaded via RoleBank from LoadSpeaker)
            role_data = role_bank[role_name]
            
            # Extract ref_text if it was passed through RoleBank
            current_prompt = role_data[0] if isinstance(role_data, list) else role_data
            current_ref_text = ""

            config = {
                "pause_linebreak": pause_linebreak,
                "period_pause": period_pause,
                "comma_pause": comma_pause,
                "question_pause": question_pause,
                "hyphen_pause": hyphen_pause
            }
            
            segments = split_text_by_pauses(text, config)

            for segment_text, current_segment_pause in segments:
                pronounceable = re.sub(r'[^\w\u4e00-\u9fa5]', '', segment_text)
                if not pronounceable.strip():
                    continue

                # Strip trailing flat punctuation to prevent robotic intonation dropping on short segments
                # We keep ? and ! because they convey necessary emotive intonation.
                clean_seg_text = re.sub(r'[\.,，。…\-—]+$', '', segment_text).strip()
                if not clean_seg_text:
                    clean_seg_text = segment_text

                texts_to_gen.append(clean_seg_text)
                prompts_to_gen.append(current_prompt)
                langs_to_gen.append(mapped_lang)
                pauses_to_gen.append(current_segment_pause)

            if pauses_to_gen:
                pauses_to_gen[-1] += pause_linebreak

        if not texts_to_gen:
            raise RuntimeError("No valid dialogue lines found matching Role Bank.")

        num_lines = len(texts_to_gen)
        num_chunks = (num_lines + batch_size - 1) // batch_size
        total_stages = num_chunks + 1
        # One bar for the whole node, scaled so every chunk owns a slice:
        # chunk boundaries are coarse jumps, tokens fill in between.
        slots_per_chunk = 100
        bar_total = max(1, num_chunks * slots_per_chunk)
        # Budget from the text that will actually be spoken, not from the
        # unreachable max_new_tokens ceiling (same reason as the single-shot
        # nodes). Re-estimated per chunk inside the loop.
        _spoken_chars = sum(len(t) for t in texts_to_gen)
        pbar = TTSProgress(
            TTSProgress.estimate_budget("x" * _spoken_chars, max_new_tokens_per_line * batch_size),
            phase_total=bar_total,
            chunk_total=num_chunks,
        )
        pbar.phase_loaded()
        _emit_progress(slots_per_chunk, bar_total)
        pbar.register(slots_per_chunk)

        try:
            results = []
            sr = 24000

            for i in range(0, num_lines, batch_size):
                chunk_texts = texts_to_gen[i:i + batch_size]
                chunk_prompts = prompts_to_gen[i:i + batch_size]
                chunk_langs = langs_to_gen[i:i + batch_size]
                chunk_pauses = pauses_to_gen[i:i + batch_size]

                current_chunk = i // batch_size + 1
                # re-estimate for this chunk's own text so the slice fills evenly
                pbar.max_new_tokens = TTSProgress.estimate_budget(
                    "".join(chunk_texts), max_new_tokens_per_line * batch_size
                )
                print(f"[Qwen3-TTS] Running batched inference for chunk {current_chunk} of {num_chunks}...")

                wavs_list, sr = model.generate_voice_clone(
                    text=chunk_texts,
                    language=chunk_langs,
                    voice_clone_prompt=chunk_prompts,
                    max_new_tokens=max_new_tokens_per_line,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

                for j, wav in enumerate(wavs_list):
                    waveform = torch.from_numpy(wav).float()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                        if waveform.shape[1] > 1:
                            waveform = torch.mean(waveform, dim=1, keepdim=True)

                    results.append(waveform)

                    this_pause = chunk_pauses[j]
                    if this_pause > 0:
                        silence_len = int(this_pause * sr)
                        silence = torch.zeros((1, 1, silence_len))
                        results.append(silence)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.finish_generation()
                _emit_progress(current_chunk * slots_per_chunk, bar_total)

        except Exception as e:
            raise RuntimeError(f"Dialogue generation failed during chunked inference: {e}")

        if not results:
            raise RuntimeError("No dialogue lines were successfully generated.")

        _emit_progress(bar_total, bar_total)

        if merge_outputs:
            # Concatenate along the sample dimension (last one)
            merged_waveform = torch.cat(results, dim=-1)
            audio_data = {"waveform": merged_waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
        else:
            # Pad to longest for batch format
            max_len = max(w.shape[-1] for w in results)
            padded_results = []
            for w in results:
                curr_len = w.shape[-1]
                if curr_len < max_len:
                    padding = torch.zeros((w.shape[0], w.shape[1], max_len - curr_len))
                    w = torch.cat([w, padding], dim=-1)
                padded_results.append(w)
            
            batched_waveform = torch.cat(padded_results, dim=0)
            audio_data = {"waveform": batched_waveform, "sample_rate": sr}
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)


class SaveVoiceNode:
    """
    SaveVoice Node: Persist extracted voice features to a file.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
                "filename": ("STRING", {"default": "my_custom_voice"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional, but recommended for better quality)"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "SaveVoice: Save voice clone prompt features to disk for later use."

    def save(self, voice_clone_prompt, filename, audio=None, ref_text=""):
        import soundfile as sf
        import json

        # filename comes from a widget, so it is attacker-controlled when a
        # workflow is shared. Reduce it to a bare name and keep every write
        # inside the voices directory.
        filename = _safe_filename(filename, "filename")
        if filename.endswith(".qvp"):
            filename_qvp = filename
            stem = filename[: -len(".qvp")]
        else:
            filename_qvp = filename + ".qvp"
            stem = filename
        filename_wav = stem + ".wav"
        filename_json = stem + ".json"

        # Use ComfyUI models/qwen-tts/voices directory
        output_dir = _voices_dir()
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save features (Raw prompt)
        path_qvp = _resolve_within(output_dir, filename_qvp, "voice features file")
        torch.save(voice_clone_prompt, path_qvp)
        print(f"✅ [Qwen3-TTS] Voice features saved to: {path_qvp}")

        # 2. Save metadata (JSON)
        metadata = {
            "ref_text": ref_text.strip() if ref_text else "",
            "source": "SaveVoiceNode",
            "version": "1.0"
        }
        path_json = _resolve_within(output_dir, filename_json, "voice metadata file")
        try:
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            print(f"✅ [Qwen3-TTS] Voice metadata saved to: {path_json}")
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS] Failed to save metadata JSON: {e}")

        # Optional: Save audio preview as reference WAV
        if audio is not None:
            try:
                # audio is usually {"waveform": tensor, "sample_rate": sr}
                waveform = audio["waveform"]
                sr = audio["sample_rate"]
                
                # Convert from [batch, channels, samples] to [samples, channels]
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform)
                
                if waveform_np.ndim == 3: # [B, C, S] -> [S, C] (assume batch size 1)
                    waveform_np = waveform_np[0].T
                
                wav_path = _resolve_within(output_dir, filename_wav, "reference audio file")
                sf.write(wav_path, waveform_np, sr)
                print(f"✅ [Qwen3-TTS] Reference audio (Speaker) saved to: {wav_path}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to save reference audio: {e}")

        return {}


class LoadSpeakerNode:
    """
    LoadSpeaker Node: Directly load a WAV/Speaker file and its associated features.
    Part of the "Voice Design then Clone" best practice workflow.
    """
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = _voices_dir()
        os.makedirs(output_dir, exist_ok=True)
        # List WAV files in the voices directory
        files = [f for f in os.listdir(output_dir) if f.endswith((".wav", ".mp3", ".flac"))]
        
        return {
            "required": {
                "filename": (files if files else ["None"],),
            },
        }
    
    # Updated: Added ref_text (STRING) output
    RETURN_TYPES = ("VOICE_CLONE_PROMPT", "AUDIO", "STRING")
    RETURN_NAMES = ("voice_clone_prompt", "audio", "ref_text")
    FUNCTION = "load_speaker"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "LoadSpeaker: Load saved WAV audio and its metadata. Fast-loads .qvp features if available."

    def load_speaker(self, filename):
        if filename == "None":
            raise RuntimeError("No speaker files found to load.")

        # filename comes from a widget; keep the read inside the voices directory.
        filename = _safe_filename(filename, "filename")
        voices_dir = _voices_dir()
        wav_path = _resolve_within(voices_dir, filename, "speaker audio file", must_exist=True)

        # 1. Load the AUDIO for output/preview
        import librosa
        wav, sr = librosa.load(wav_path, sr=None)
        waveform = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        audio_preview = {"waveform": waveform, "sample_rate": sr}

        # 2. Automatically load metadata and pre-computed features
        qvp_file = os.path.splitext(filename)[0] + ".qvp"
        qvp_path = _resolve_within(voices_dir, qvp_file, "voice features file")
        json_file = os.path.splitext(filename)[0] + ".json"
        json_path = _resolve_within(voices_dir, json_file, "voice metadata file")
        
        prompt_items = None
        ref_text = ""
        
        # 2.1 Load metadata from JSON
        import json
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    if not ref_text or not ref_text.strip():
                        ref_text = metadata.get("ref_text", "")
                        if ref_text:
                            print(f"📖 [Qwen3-TTS] Loaded metadata from JSON: {json_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to load metadata JSON: {e}")

        # 2.2 Check for pre-computed features (.qvp)
        if os.path.exists(qvp_path):
            try:
                # A .qvp holds VoiceClonePromptItem objects. Load it with
                # weights_only=True so a shared voice file cannot execute
                # embedded code, and register VoiceClonePromptItem as a safe
                # global so the user's own saved voices still load.
                _register_safe_globals()
                try:
                    data = torch.load(qvp_path, map_location="cpu", weights_only=True)
                except TypeError as exc:
                    # torch without the weights_only argument (pre-1.13).
                    # Do NOT retry without it: an unrestricted load would let a
                    # shared voice file execute code. Skip the fast path and
                    # recompute the features from the audio instead.
                    print(
                        f"[Qwen3-TTS] This torch build cannot load .qvp safely "
                        f"({exc}); recomputing features from the audio instead."
                    )
                    raise
                
                # Support legacy packaged format and raw format
                if isinstance(data, dict) and "prompt" in data:
                    prompt_items = data["prompt"]
                    if not ref_text or not ref_text.strip():
                        ref_text = data.get("ref_text", "")
                else:
                    # Raw prompt data
                    prompt_items = data
                
                print(f"🚀 [Qwen3-TTS] Fast-loaded pre-computed features from: {qvp_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS] Failed to fast-load .qvp: {e}")

        # Final Return (including the text)
        # Note: If prompt_items is None, the downstream VoiceCloneNode will extract it using its own settings.
        return (prompt_items, audio_preview, ref_text)
