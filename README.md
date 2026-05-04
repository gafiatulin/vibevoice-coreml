# VibeVoice CoreML

On-device text-to-speech and speech-to-text using [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) models converted to CoreML.

Inspired by [FluidAudio](https://github.com/FluidInference/FluidAudio) with ideas from [Mobius](https://github.com/FluidInference/mobius).

## Models

| Model | Params | Type | HuggingFace |
|-------|--------|------|-------------|
| Realtime 0.5B | Qwen2.5-0.5B | Streaming TTS, 25 voices | [vibevoice-tts-0.5b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-0.5b-coreml) |
| Multispeaker 1.5B | Qwen2.5-1.5B | TTS + voice cloning | [vibevoice-tts-1.5b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-1.5b-coreml) |
| Multispeaker 7B | Qwen2.5-7B | TTS + voice cloning | [vibevoice-tts-7b-coreml](https://huggingface.co/gafiatulin/vibevoice-tts-7b-coreml) |
| ASR | Qwen2-7B (8.3B) | Speech-to-text, 50+ langs | [vibevoice-asr-coreml](https://huggingface.co/gafiatulin/vibevoice-asr-coreml) |

1.5B, 7B, and ASR models are INT8 quantized (W8A16) with fused LM+head. 0.5B runs at FP16. Models auto-download from HuggingFace on first use. All inference runs on CPU + GPU. Unfortunately I don't expect any step of any pipeline (even 0.5B) to run on ANE (better/faster than `CPU_GPU`).

## Performance

Benchmarked on M4 Max (64GB). RTF = real-time factor (higher is faster; >1x means faster than real-time). RAM measured per-process via `footprint`.

| Model | Backend | RTF | Peak RAM | Avg RAM | GPU |
|-------|---------|-----|----------|---------|-----|
| 0.5B | CoreML (Swift) | 5.6x | **3.2 GB** | **2.6 GB** | 11% |
| 0.5B | MLX ([mlx-audio](https://github.com/ml-explore/mlx-audio)) | **5.9x** | 5.4 GB | 3.2 GB | 25% |
| 1.5B | CoreML (Swift, int8) | 2.4x | **8.3 GB** | **6.8 GB** | 16% |
| 1.5B | MLX ([vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx), int8) | **2.8x** | 11 GB | 7.1 GB | 32% |
| 7B | CoreML (Swift, int8) | **1.2x** | **16 GB** | **13 GB** | 18% |
| 7B | MLX ([vibevoice-mlx](https://github.com/gafiatulin/vibevoice-mlx), int8) | 1.1x | 26 GB | 22 GB | 60% |
| ASR | CoreML (Swift, int8) | 13.4 tok/s | 22 GB | 13 GB | 10% |
| ASR | MLX ([mlx-audio](https://github.com/ml-explore/mlx-audio)) | **36.3 tok/s** | **10 GB** | **10 GB** | 13% |

1.5B/7B TTS use DPM-Solver++ 2M (10 steps) with CoreML semantic encoder feedback.

### When to use what

- **Deploying in Swift apps (iOS/macOS)**: Use CoreML. It's the only option that integrates natively and runs without Python.
- **TTS 7B**: CoreML is faster (1.2x vs 1.1x) and uses less RAM (16 vs 26 GB peak).
- **TTS 1.5B**: MLX is faster (2.8x vs 2.4x RTF). CoreML uses less RAM (8 vs 11 GB peak). MLX doesn't require CoreML conversion.
- **TTS 0.5B**: MLX is slightly faster (5.9x vs 5.6x RTF). CoreML uses much less RAM (3 vs 12 GB peak).
- **ASR**: MLX is ~2.7x faster (36 vs 13 tok/s) and uses less RAM (10 vs 22 GB). Use MLX for batch transcription, CoreML for on-device apps.
- **Memory-constrained (16 GB devices)**: 0.5B CoreML (3 GB) and 1.5B CoreML (8 GB) fit comfortably. 7B CoreML (16 GB) is tight. MLX uses more RAM due to memory pool overhead.

## Swift Library

Requires macOS 15+. Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/gafiatulin/vibevoice-coreml", from: "x.x.x"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "VibeVoiceCoreML", package: "vibevoice-coreml"),
        ]
    ),
]
```

### Text-to-Speech (0.5B Streaming)

```swift
import VibeVoiceCoreML

let tts = try await RealtimeTTS(modelsDir: modelsURL)
let config = TTSConfig(voice: "Emma")

for try await frame in tts.speak("Hello world", config: config) {
    // frame.samples: [Float] at 24kHz
}
```

### Text-to-Speech (1.5B / 7B with Voice Cloning)

```swift
import VibeVoiceCoreML

let tts = try await MultispeakerTTS(modelsDir: modelsURL, architecture: .model7B)
let voices = try await tts.encodeVoices(from: [referenceAudioURL])
let config = MultispeakerConfig(seed: 42)

for try await frame in tts.speak("Hello world", config: config, voices: voices) {
    // frame.samples: [Float] at 24kHz
}
```

### Speech-to-Text (ASR)

```swift
import VibeVoiceCoreML

let stt = try await SpeechToText(modelsDir: modelsURL)
let result = try await stt.transcribe(audioURL)
print(result.text)
```

### Auto-downloading Models

```swift
import VibeVoiceCoreML

let modelsURL = try await HubDownloader.download(repo: "gafiatulin/vibevoice-tts-0.5b-coreml")
```

## CLI

### Swift

```bash
swift build -c release
```

```bash
Usage: vibevoice-cli --model <0.5b|1.5b|7b|asr> [options]

Required:
  --model        Model variant: 0.5b, 1.5b, 7b, or asr

Optional:
  --models-dir   Directory with CoreML models (default: auto-download from HuggingFace)
  --cache-dir    HuggingFace hub cache directory (default: ~/.cache/huggingface/hub/)
  --verbose      Show detailed download and verification logs

TTS options:
  --tokenizer    Directory with tokenizer.json (default: models dir)
  --text         Text to synthesize (use \n for newlines)
  --output       Output WAV path (default: output.wav)
  --seed         Random seed (default: 42)

0.5B only:
  --voice        Voice name (default: Emma)

1.5B/7B only:
  --ref-audio    Reference audio file(s) for voice cloning (one per speaker)
  --max-tokens   Maximum speech tokens (default: auto from text length)

ASR options:
  --audio        Input audio file to transcribe
  --prompt       Optional prompt/context to guide transcription
  --max-tokens   Maximum tokens to generate (default: auto from audio duration)

Examples:
  vibevoice-cli --model 0.5b --text "Hello world"
  vibevoice-cli --model 7b --ref-audio spk1.wav --text "Hello from a cloned voice"
  vibevoice-cli --model asr --audio recording.wav
  vibevoice-cli --model 0.5b --models-dir /path/to/models --text "Use local models"
```

### Python

```bash
cd python/python-cli
uv sync
```

```bash
usage: vibevoice_cli.py [-h] [--model {0.5b,1.5b,7b,asr}] [--models-dir MODELS_DIR] [--cache-dir CACHE_DIR] [--verbose] [--text TEXT] [--output OUTPUT] [--seed SEED] [--voice VOICE]
                        [--ref-audio REF_AUDIO [REF_AUDIO ...]] [--max-tokens MAX_TOKENS] [--audio AUDIO] [--prompt PROMPT]

VibeVoice CoreML CLI — TTS and ASR inference

options:
  -h, --help            show this help message and exit
  --model {0.5b,1.5b,7b,asr}
                        Model variant: 0.5b, 1.5b, 7b, or asr
  --models-dir MODELS_DIR
                        Local models directory (default: auto-download from HuggingFace)
  --cache-dir CACHE_DIR
                        Cache directory for downloaded models
  --verbose, -v         Show detailed logs
  --text TEXT           Text to synthesize
  --output, -o OUTPUT   Output file path
  --seed SEED           Random seed (default: 42)
  --voice VOICE         Voice name (0.5B only, default: Emma)
  --ref-audio REF_AUDIO [REF_AUDIO ...]
                        Reference audio file(s) for voice cloning (1.5B/7B only)
  --max-tokens MAX_TOKENS
                        Maximum speech/generation tokens (default: auto)
  --audio AUDIO         Input audio file for ASR
  --prompt PROMPT       Optional ASR context prompt

Examples:
  vibevoice-cli --model 0.5b --text "Hello world"
  vibevoice-cli --model 7b --ref-audio spk1.wav --text "Hello from a cloned voice"
  vibevoice-cli --model asr --audio recording.wav
  vibevoice-cli --model 0.5b --models-dir /path/to/models --text "Use local models"
```

Both CLIs auto-download models from HuggingFace and share the same cache (`~/.cache/huggingface/hub/`).

## Project Structure

```
Package.swift                     # Swift package
Sources/
  VibeVoiceCoreML/                # Swift library (RealtimeTTS, MultispeakerTTS, SpeechToText)
  vibevoice-cli/                  # Swift CLI
python/
  common/                         # Shared conversion utilities
  tts/                            # TTS model conversion & pipelines
    common/                       # TTS-specific utilities (RoPE, diffusion, etc.)
    vibevoice-realtime-0.5b/      # 0.5B streaming model
    vibevoice-multispeaker/       # 1.5B + 7B multispeaker models
  stt/                            # ASR model conversion & pipelines
  python-cli/                     # Python CLI
  build_hf_repos.py               # HuggingFace repo builder
```

## Converting Models

To convert models from PyTorch to CoreML yourself:

```bash
cd python/tts/vibevoice-multispeaker
uv sync
uv run python convert/convert_all.py --model-id 7b --int8 --fuse-lm-head
```

## License

MIT
