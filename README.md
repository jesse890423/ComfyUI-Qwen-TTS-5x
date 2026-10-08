# ComfyUI-Qwen-TTS-5x

English | [中文版](README_CN.md)

> **✅ transformers 5.x support (this fork)**
>
> This fork adapts the plugin to **transformers 5.x**, verified on
> transformers 5.14.1 / ComfyUI 0.37.0 / torch 2.13.0.
> The upstream v1.0.7 declares `transformers>=4.57.0,<5.0.0`; if you need the
> original 4.x behaviour, use the upstream release pinned to 4.57.3.

---

## About this fork

This repository is a fork of
[flybirdxx/ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)
(Apache-2.0), adapted for the transformers 5.x environment.

### Why

Upstream v1.0.7 declares `transformers>=4.57.0,<5.0.0`. Downgrading the shared
ComfyUI environment was not an option, so the four affected files were adapted
in place. Every fix was verified by measurement, not by inspection.

### Fixes

| # | Issue | Impact |
|---|---|---|
| 1 | `pad_token_id` no longer copied onto `PretrainedConfig` in 5.x | `AttributeError` on load |
| 2 | `ROPE_INIT_FUNCTIONS["default"]` removed in 5.14 | replaced with a local unscaled implementation (mapping it to `"linear"` would be wrong) |
| 3 | `create_causal_mask` signature change (`input_embeds` -> `inputs_embeds`, `cache_position` dropped) | call-time signature adaptation |
| 4 | `cache_position` arrives as `None` on decode | **crash**: `mat1 and mat2 shapes cannot be multiplied (1x55296 and 2048x2048)` |
| 5 | `inv_freq` left as uninitialised meta memory | **silent failure**: audio is produced but the voice is completely wrong, no exception |
| 6 | `layer_type_validation` deprecation shim logs on every call | 59 identical log lines per model load |
| 7 | `ProgressBar(total)` defaults to `node_id=None` | progress never displayed in the panel |
| 8 | `generate()` is one blocking call | progress bar frozen for the whole generation |
| 9 | step counter went negative (`input_ids` excludes prefill) | bar jumped straight to 100% |
| 10 | ComfyUI 0.37 registers no `CLIProgressHandler` | added a self-contained stderr console bar |

Fix 5 is the one worth highlighting: it raises nothing and logs nothing. The
model just emits meaningless acoustic frames, which is easy to mistake for a
bad reference audio.

### Installing

Pick whichever suits you. All four end up with the same thing: a folder called
`ComfyUI-Qwen-TTS-5x` inside `ComfyUI/custom_nodes/`.

#### 1. ComfyUI Manager (easiest)

1. Open ComfyUI, click **Manager**
2. **Custom Nodes Manager** -> **Install Custom Nodes**
3. Paste this into the search / URL box:
   ```
   https://github.com/jesse890423/ComfyUI-Qwen-TTS-5x
   ```
4. Click **Install**, then restart ComfyUI when prompted

Manager handles the clone and the dependencies for you.

#### 2. git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jesse890423/ComfyUI-Qwen-TTS-5x.git
cd ComfyUI-Qwen-TTS-5x
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Adjust the `python_embeded` path if your ComfyUI is not the portable build.

#### 3. Download the zip

1. On the repository page: **Code** -> **Download ZIP**
2. Unzip it. The zip contains a folder called
   `ComfyUI-Qwen-TTS-5x-main` -- **rename it to `ComfyUI-Qwen-TTS-5x`**
   (drop the `-main` suffix). The name matters: ComfyUI derives the node
   identity from the folder name, and this name is also what lets Manager
   update the plugin later.
3. Move the renamed folder into `ComfyUI/custom_nodes/`, so you end up with
   ```
   ComfyUI/custom_nodes/ComfyUI-Qwen-TTS-5x/
   ```
   Do **not** end up with `ComfyUI/custom_nodes/ComfyUI-Qwen-TTS-5x-main/`,
   and do not put the contents loose directly into `custom_nodes/`.
4. Install the dependencies:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-Qwen-TTS-5x
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```
5. Restart ComfyUI

#### 4. Manual copy

Copy the contents of this repository into a new folder
`ComfyUI/custom_nodes/ComfyUI-Qwen-TTS-5x`, install
`requirements.txt`, and restart ComfyUI.

#### Then get the models

This node pack downloads nothing on its own. Model weights go where the
[Model Directory Structure](#model-directory-structure) section below
describes.

### If the nodes do not appear

1. Check the ComfyUI console for an error mentioning this plugin
2. Confirm the folder path is exactly
   `ComfyUI/custom_nodes/ComfyUI-Qwen-TTS-5x/__init__.py`
3. Confirm `transformers` is installed: `pip show transformers`
4. Restart ComfyUI fully (not just the browser tab)

### Do not install both packages

This fork keeps the upstream node class names (`FB_Qwen3TTS*`) so existing
workflows keep working. Upstream and this fork therefore register **the same
class names**, and enabling both at once makes the nodes clash.

Pick one:

| | transformers | Folder |
|---|---|---|
| This fork | 4.57+ **and 5.x** | `ComfyUI-Qwen-TTS-5x` |
| [Upstream](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) | 4.57.x only | `ComfyUI-Qwen-TTS` |

To switch, disable or delete the other one (rename its folder to `<name>.disabled`,
or use Manager's Disable button).

### Verified environment

```
ComfyUI        0.37.0
transformers   5.14.1
torch          2.13.0+cu130
plugin         ComfyUI-Qwen-TTS v1.0.7
```

### License

Apache License 2.0 -- see [LICENSE](LICENSE) and [NOTICE](NOTICE).

This fork only changes the plugin's own code. **Model weights are not included**
and remain governed by the Qwen3-TTS License Agreement from Alibaba Cloud.

![Nodes Screenshot](example/example.png)

ComfyUI custom nodes for speech synthesis, voice cloning, and voice design, based on the open-source **Qwen3-TTS** project by the Alibaba Qwen team.

## 📋 Changelog

- **2026-04-12 (v1.0.7)**: Removed `QwenTTSConfigNode` due to voice inconsistency; fixed MPS precision bug & CustomVoice channel mismatch; code cleanup ([update.md](doc/update.md))
- **2026-02-04**: Added `extra_model_paths.yaml` support ([update.md](doc/update.md))
- **2026-01-29**: Feature Update: Support for loading custom fine-tuned models & speakers ([update.md](doc/update.md))
  - *Note: Fine-tuning is currently experimental; zero-shot cloning is recommended for best results.*
- **2026-01-27**: UI Optimization: Sleek LoadSpeaker UI; fixed PyTorch 2.6+ compatibility ([update.md](doc/update.md))
- **2026-01-26**: Functional Update: New voice persistence system (SaveVoice / LoadSpeaker) ([update.md](doc/update.md))
- **2026-01-24**: Added attention mechanism selection & model memory management features ([update.md](doc/update.md))
- **2026-01-24**: Added generation parameters (top_p, top_k, temperature, repetition_penalty) to all TTS nodes ([update.md](doc/update.md))
- **2026-01-23**: Dependency compatibility & Mac (MPS) support, New nodes: VoiceClonePromptNode, DialogueInferenceNode ([update.md](doc/update.md))

## Example workflows

Ready-to-use workflows are in [`example/`](example/):

| File | What it shows |
|---|---|
| `example.json` | Basic synthesis |
| `Custom Save Voice.json` | Saving a voice for reuse |
| `Multi-character dialogue.json` | Multi-role dialogue |
| `Model Fine-Tuning and Workflow Usage.json` | Fine-tuning setup |

## Key Features

- 🎵 **Speech Synthesis**: High-quality text-to-speech conversion.
- 🎭 **Voice Cloning**: Zero-shot voice cloning from short reference audio.
- 🎨 **Voice Design**: Create custom voice characteristics based on natural language descriptions.
- 🚀 **Efficient Inference**: Supports both 12Hz and 25Hz speech tokenizer architectures.
- 🎯 **Multilingual**: Native support for 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian).
- ⚡ **Integrated Loading**: No separate loader nodes required; model loading is managed on-demand with global caching.
- ⏱️ **Ultra-Low Latency**: Supports high-fidelity speech reconstruction with low-latency streaming.
- 🧠 **Attention Mechanism Selection**: Choose from multiple attention implementations (sage_attn, flash_attn, sdpa, eager) with auto-detection and graceful fallback.
- 💾 **Memory Management**: Optional model unloading after generation to free GPU memory for users with limited VRAM.

## Nodes List

### 1. Qwen3-TTS Voice Design (`VoiceDesignNode`)
Generate unique voices based on text descriptions.
- **Inputs**:
  - `text`: Target text to synthesize.
  - `instruct`: Description of the voice (e.g., "A gentle female voice with a high pitch").
  - `model_choice`: Currently locked to **1.7B** for VoiceDesign features.
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
- **Capabilities**: Best for creating "imaginary" voices or specific character archetypes.

### 2. Qwen3-TTS Voice Clone (`VoiceCloneNode`)
Clone a voice from a reference audio clip.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to clone.
  - `ref_text`: Text spoken in the `ref_audio` (helps improve quality).
  - `target_text`: The new text you want the cloned voice to say.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.

### 3. Qwen3-TTS Custom Voice (`CustomVoiceNode`)
Standard TTS using preset speakers.
- **Inputs**:
  - `text`: Target text.
  - `speaker`: Selection from preset voices (Aiden, Eric, Serena, etc.).
  - `instruct`: Optional style instructions.
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.

### 4. Qwen3-TTS Role Bank (`RoleBankNode`) [New]
Collect and manage multiple voice prompts for dialogue generation.
- **Inputs**:
  - Up to 8 roles, each with:
    - `role_name_N`: Name of the role (e.g., "Alice", "Bob", "Narrator")
    - `prompt_N`: Voice clone prompt from `VoiceClonePromptNode`
- **Capabilities**: Create named voice registry for use in `DialogueInferenceNode`. Supports up to 8 different voices per bank.

### 5. Qwen3-TTS Voice Clone Prompt (`VoiceClonePromptNode`) [New]
Extract and reuse voice features from reference audio.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to extract features from.
  - `ref_text`: Text spoken in the `ref_audio` (highly recommended for better quality).
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
- **Capabilities**: Extract a "prompt item" once and use it multiple times across different `VoiceCloneNode` instances for faster and more consistent generation.

### 6. Qwen3-TTS Multi-role Dialogue (`DialogueInferenceNode`) [New]
Synthesize complex dialogues with multiple speakers.
- **Inputs**:
  - `script`: Dialogue script in format "RoleName: Text".
  - `role_bank`: Role bank from `RoleBankNode` containing voice prompts.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).
  - `attention`: Attention mechanism (auto, sage_attn, flash_attn, sdpa, eager).
  - `unload_model_after_generate`: Unload model from memory after generation to free GPU memory.
  - `pause_seconds`: Silence duration between sentences.
  - `merge_outputs`: Merge all dialogue segments into a single long audio.
  - `batch_size`: Number of lines to process in parallel (larger = faster but more VRAM).
- **Capabilities**: Handles multi-role speech synthesis in a single node, ideal for audiobook narration or roleplay scenarios.

### 7. Qwen3-TTS Load Speaker (`LoadSpeakerNode`) [New]
Load saved voice features and metadata with zero configuration.
- **Capabilities**: Enables a "Select & Play" experience by auto-loading pre-computed features and metadata.

### 8. Qwen3-TTS Save Voice (`SaveVoiceNode`) [New]
Persist extracted voice features and metadata to disk for future use.
- **Capabilities**: Build a permanent voice library for reuse via `LoadSpeakerNode`.

## Attention Mechanisms

All nodes support multiple attention implementations with automatic detection and graceful fallback:

| Mechanism | Description | Speed | Installation |
|-----------|-------------|-------|--------------|
| **sage_attn** | SAGE attention implementation | ⚡⚡⚡ Fastest | `pip install sage_attn` |
| **flash_attn** | Flash Attention 2 | ⚡⚡ Fast | `pip install flash_attn` |
| **sdpa** | Scaled Dot Product Attention (PyTorch built-in) | ⚡ Medium | Built-in (no installation) |
| **eager** | Standard attention (fallback) | 🐢 Slowest | Built-in (no installation) |
| **auto** | Automatically selects best available option | Varies | N/A |

### Auto-Detection Priority

When `attention: "auto"` is selected, the system checks in this order:
1. **sage_attn** → If installed, use SAGE attention (fastest)
2. **flash_attn** → If installed, use Flash Attention 2
3. **sdpa** → Always available (PyTorch built-in)
4. **eager** → Always available (fallback, slowest)

The selected mechanism is logged to the console for transparency.

### Graceful Fallback

If you select an attention mechanism that's not available:
- Falls back to `sdpa` (if available)
- Falls back to `eager` (as last resort)
- Logs the fallback decision with a warning message

### Model Caching

- Models are cached with attention-specific keys
- Changing attention mechanism automatically clears cache and reloads model
- Same model with different attention mechanisms coexists in cache

## Memory Management

### Model Unloading After Generation

The `unload_model_after_generate` toggle is available on all nodes:
- **Enabled**: Clears model cache, GPU memory, and runs garbage collection after generation
- **Disabled**: Model remains in cache for faster subsequent generations (default)

**When to use:**
- ✅ Enable if you have limited VRAM (< 8GB)
- ✅ Enable if you need to run multiple different models sequentially
- ✅ Enable if you're done with generation and want to free memory
- ❌ Disable if you're generating multiple clips with the same model (faster)

**Console Output:**
```
🗑️ [Qwen3-TTS] Unloading 1 cached model(s)...
✅ [Qwen3-TTS] Model cache and GPU memory cleared
```



## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchaudio transformers librosa accelerate
```

### Model Directory Structure

ComfyUI-Qwen-TTS-5x automatically searches for models in the following priority:

```text
ComfyUI/
├── models/
│   └── qwen-tts/
│       ├── Qwen/Qwen3-TTS-12Hz-1.7B-Base/
│       ├── Qwen/Qwen3-TTS-12Hz-0.6B-Base/
│       ├── Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/
│       ├── Qwen/Qwen3-TTS-Tokenizer-12Hz/
│       └── voices/ (Saved presets .wav/.qvp)
```

**Note**: You can also use `extra_model_paths.yaml` to define a custom model path:
```yaml
qwen-tts: D:\MyModels\Qwen
```

### Loading a Fine-Tuned Model (`custom_model_path`)

`VoiceCloneNode` and `CustomVoiceNode` accept a `custom_model_path` value, which is
stored inside the workflow file. Because workflows are shared, this value is only
ever interpreted as a **folder name relative to one of these fixed bases**:

| Base | Typical content |
|---|---|
| `models/qwen-tts/` (and any `extra_model_paths.yaml` entry for `qwen-tts` / `TTS`) | models you placed there yourself |
| `output/qwen3tts_finetune/` | checkpoints produced by the Train node |

Examples: `my_lora`, `nested/my_lora`, `checkpoint-epoch-9`.

Absolute paths (`D:\models\my_lora`), network shares (`\\server\share\my_lora`) and
anything that escapes one of those bases are refused **before the path is touched** —
on Windows even a plain existence check on a share path opens an SMB session and
leaks the current user's credentials to that machine.

The Train node therefore reports its checkpoint as a relative folder name (the
absolute location is printed in the console), so it can be pasted straight into
`custom_model_path`.

## Tips for Best Results

### Audio Quality
- **Cloning**: Use clean, noise-free reference audio (5-15 seconds).
- **Reference Text**: Providing text spoken in reference audio significantly improves quality.
- **Language**: Select the correct language for best pronunciation and prosody.

### Performance & Memory
- **VRAM**: Use `bf16` precision to save significant memory with minimal quality loss.
- **Attention**: Use `attention: "auto"` for automatic selection of fastest available mechanism.
- **Model Unloading**: Enable `unload_model_after_generate` if you have limited VRAM (< 8GB) or need to run multiple different models.
- **Local Models**: Pre-download weights to `models/qwen-tts/` to prioritize local loading and avoid HuggingFace timeouts.

### Attention Mechanisms
- **Best Performance**: Install `sage_attn` or `flash_attn` for 2-3x speedup over sdpa.
- **Compatibility**: Use `sdpa` (default) for maximum compatibility - no installation required.
- **Low VRAM**: Use `eager` with smaller models (0.6B) if other mechanisms cause OOM errors.

### Dialogue Generation
- **Batch Size**: Increase `batch_size` for faster generation (more VRAM usage).
- **Pauses**: Adjust `pause_seconds` to control timing between dialogue segments.
- **Merge**: Enable `merge_outputs` for continuous dialogue; disable for separate clips.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.

## License

- This project is licensed under the **Apache License 2.0**.
- Model weights are subject to the [Qwen3-TTS License Agreement](https://github.com/QwenLM/Qwen3-TTS#License).

## Credits

This project exists thanks to the original author:

- **[flybirdxx/ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)** -- original
  implementation, Apache-2.0. This repository is a fork of it.
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** -- the model itself, by the
  Alibaba Qwen team. Model weights come from there and are not included here.

Follow the upstream projects for updates to the original node pack and to the model.
