# PyVoice


PyVoice is a Python-based speech processing tool that supports both speech-to-text (ASR) and text-to-speech (TTS) using ONNX Runtime.
This project refactors and extends an open-source speech project into a production-style CLI application, with cross-platform ONNX Runtime integration.

---

## Features

- 🎙 Speech-to-Text (ASR) using Paraformer  
  - Supports Chinese and English speech recognition
- 🔊 Text-to-Speech (TTS)
  - Mandarin Chinese and English TTS using MeloTTS
- 🖥 CLI-first design with simple subcommands
- ⚙️ Cross-platform ONNX Runtime integration
- 🧩 Clean and extensible project structure

---

## Prerequisites

- Python 3.8+
- pip

## Installation

### 1. Ensure Python 3.8+ is installed

```bash
python3 --version  # Should show Python 3.8 or higher
```

### 2. Install dependencies

```bash
# Install using pip
pip install -r requirements.txt

# Or use pip3
pip3 install -r requirements.txt

# If using virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Verify model files

Ensure the following directories and files exist:

```
paraformer_weights/
  ├── model.int8.onnx
  ├── tokens.txt
  └── am.mvn

melo_weights/
  ├── model.onnx
  ├── tokens.txt
  └── lexicon.txt
```

If model files do not exist, download them from Hugging Face (requires git-lfs):

```bash
# Install git-lfs if not already installed
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Linux

git lfs install

# Clone model repository
git clone https://huggingface.co/getcharzp/go-speech ./temp_models

# Move files to correct locations
mv ./temp_models/melo_weights ./melo_weights
mv ./temp_models/paraformer_weights ./paraformer_weights

# Clean up temporary files
rm -rf ./temp_models
```

---

## How to Run

### Basic Usage

Run from the project root directory:

```bash
# Text-to-Speech (TTS)
python3 main.py tts "<text>" [--out output.wav]

# Speech-to-Text (ASR)
python3 main.py asr <wav-file>
```

### Usage Examples

#### 1. Text-to-Speech

```bash
# Chinese TTS (default output to assets/output.wav)
python3 main.py tts "Hello, world!"

# English TTS with specified output file
python3 main.py tts "Hello, world!" --out hello.wav

# Mixed text example
python3 main.py tts "Hello, this is a test!" --out mixed.wav
```

#### 2. Speech Recognition

```bash
# Recognize WAV file
python3 main.py asr assets/output.wav

# Recognize audio file from other location
python3 main.py asr /path/to/your/audio.wav
```

### Help Information

```bash
# View help
python3 main.py --help

# View subcommand help
python3 main.py tts --help
python3 main.py asr --help
```

### Notes

1. **Audio Format Requirements** (ASR):
   - Supports WAV format
   - Automatically converts to 16kHz mono 16-bit
   - If audio format doesn't match, automatic conversion will be performed

2. **Output Files** (TTS):
   - Default output to `assets/output.wav`
   - Output directory will be created automatically if it doesn't exist
   - Output format: 44.1kHz mono 16-bit WAV

3. **Model Paths**:
   - Make sure to run commands from the project root directory
   - Model file paths are relative paths, relative to the project root directory




## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project is based on the open-source project
[getcharzp/go-speech](https://huggingface.co/getcharzp/go-speech),
with significant refactoring and CLI restructuring.

---


