#!/usr/bin/env python3
"""
PyVoice - Python-based speech processing tool
Supports speech-to-text (ASR) and text-to-speech (TTS)
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from asr.paraformer.engine import Engine as ASREngine
from asr.paraformer.config import Config as ASRConfig
from tts.melotts.engine import Engine as TTSEngine
from tts.melotts.config import Config as TTSConfig


def run_asr(wav_path: str) -> None:
    """
    Run speech recognition
    
    Args:
        wav_path: Path to WAV file
    """
    config = ASRConfig(
        model_path="./paraformer_weights/model.int8.onnx",
        tokens_path="./paraformer_weights/tokens.txt",
        cmvn_path="./paraformer_weights/am.mvn",
    )
    
    engine = ASREngine(config)
    
    try:
        text = engine.recognize_file(wav_path)
        print("Recognized text:")
        print(text)
    except Exception as e:
        print(f"ASR error: {e}")
        sys.exit(1)


def run_tts(text: str, out_path: str = None) -> None:
    """
    Run text-to-speech
    
    Args:
        text: Text to convert
        out_path: Output file path (optional)
    """
    if out_path is None:
        out_path = "assets/output.wav"
    
    # Ensure output directory exists (if path contains directory)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    config = TTSConfig(
        model_path="./melo_weights/model.onnx",
        token_path="./melo_weights/tokens.txt",
        lexicon_path="./melo_weights/lexicon.txt",
    )
    
    engine = TTSEngine(config)
    
    try:
        wav_data = engine.synthesize_to_wav(text, speed=1.0)
        
        with open(out_path, 'wb') as f:
            f.write(wav_data)
        
        print(f"TTS output saved to: {out_path}")
    except Exception as e:
        print(f"TTS error: {e}")
        sys.exit(1)


def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python main.py asr <wav-file>")
    print('  python main.py tts "<text>" [--out output.wav]')


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PyVoice - Speech processing tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ASR subcommand
    asr_parser = subparsers.add_parser('asr', help='Run speech recognition')
    asr_parser.add_argument('wav_file', help='Path to WAV file')
    
    # TTS subcommand
    tts_parser = subparsers.add_parser('tts', help='Run text-to-speech')
    tts_parser.add_argument('text', help='Text to synthesize')
    tts_parser.add_argument('--out', default='assets/output.wav', 
                           help='Output WAV file path (default: assets/output.wav)')
    
    args = parser.parse_args()
    
    if not args.command:
        print_usage()
        sys.exit(1)
    
    if args.command == 'asr':
        run_asr(args.wav_file)
    elif args.command == 'tts':
        run_tts(args.text, args.out)
    else:
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()

