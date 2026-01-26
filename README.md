# PyVoice

A Python-based speech processing tool with ASR (Speech-to-Text) and TTS (Text-to-Speech) capabilities, powered by ONNX Runtime.

![Demo](assets/demo.jpg)

---

## Features

- 🎙 **Speech-to-Text (ASR)** - Paraformer model for Chinese/English recognition
- 🔊 **Text-to-Speech (TTS)** - MeloTTS for natural Chinese/English synthesis
- 🚀 **FastAPI Server** - RESTful API with Swagger documentation
- 📦 **Python SDK** - Easy integration into other projects
- 🎵 **Multi-format Audio** - Supports WAV, MP3, FLAC, OGG (via FFmpeg)
- ⚡ **Optimized Performance** - NumPy vectorization for 10-100x speedup
- 🖥 **CLI Tool** - Simple command-line interface

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd PyVoice

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (for MP3/FLAC support)
brew install ffmpeg  # macOS
# or: sudo apt install ffmpeg  # Linux
```

### Download Models

```bash
git lfs install
git clone https://huggingface.co/getcharzp/go-speech ./temp_models
mv ./temp_models/melo_weights ./melo_weights
mv ./temp_models/paraformer_weights ./paraformer_weights
rm -rf ./temp_models
```

---

## Usage

### 1. CLI Tool

```bash
# Text-to-Speech
python3 main.py tts "Hello, world!" --out output.wav

# Speech-to-Text
python3 main.py asr audio.wav
```

### 2. FastAPI Server

```bash
python api.py
```

Then visit:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/asr` | Upload audio → Get text |
| POST | `/tts` | Send text → Get audio (WAV/MP3) |
| GET | `/records` | Get history records |
| GET | `/audio/{id}` | Download audio file |

### 3. Python SDK

```python
from pyvoice import ASR, TTS

# Speech Recognition
asr = ASR()
text = asr.recognize("audio.mp3")  # Supports WAV, MP3, FLAC, etc.

# Text-to-Speech
tts = TTS()
tts.synthesize_to_file("Hello world", "output.wav")
```

---

## Project Structure

```
PyVoice/
├── api.py              # FastAPI server
├── pyvoice.py          # SDK interface
├── main.py             # CLI tool
├── asr/                # ASR engine (Paraformer)
├── tts/                # TTS engine (MeloTTS)
├── internal/           # Audio processing utilities
├── storage/            # API uploads/outputs (auto-created)
├── paraformer_weights/ # ASR model files
└── melo_weights/       # TTS model files
```

---

## Tech Stack

- **Runtime**: Python 3.8+, ONNX Runtime
- **API**: FastAPI, Uvicorn
- **Audio**: NumPy, SciPy, FFmpeg
- **NLP**: jieba (Chinese segmentation)
- **Storage**: SQLite

---

## Performance

Audio processing optimized with NumPy vectorization:

| Operation | Optimization | Speedup |
|-----------|--------------|---------|
| Pre-emphasis | Vectorized array ops | ~50x |
| FFT | Batch `np.fft.fft` | ~20x |
| Mel filterbank | Matrix multiplication | ~100x |

---

## License

MIT License

---

## Acknowledgments

Based on [getcharzp/go-speech](https://huggingface.co/getcharzp/go-speech), refactored into a Python SDK with FastAPI integration.
