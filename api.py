"""
PyVoice FastAPI Server

Endpoints:
    POST /asr          - Speech to Text (file upload)
    WS   /ws/asr       - Streaming ASR (real-time WebSocket)
    POST /tts          - Text to Speech
    GET  /records      - Get all records
    GET  /audio/{id}   - Download audio file

Usage:
    python api.py
    
    # Then visit: http://localhost:8000
    # API docs: http://localhost:8000/docs
"""
import os
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import struct

from pyvoice import ASR, TTS
import subprocess


def convert_to_mp3(wav_path: str, mp3_path: str) -> None:
    """Convert WAV to MP3 using ffmpeg"""
    cmd = [
        'ffmpeg', '-y', '-i', wav_path,
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        mp3_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

# ============ Configuration ============

# Storage directories
UPLOAD_DIR = Path("storage/uploads")    # User uploaded audio
OUTPUT_DIR = Path("storage/outputs")    # Generated audio
DB_PATH = Path("storage/records.db")    # SQLite database

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============ Pydantic Models ============

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    format: str = "wav"  # "wav" or "mp3"

class ASRResponse(BaseModel):
    id: str
    text: str
    audio_path: str

class TTSResponse(BaseModel):
    id: str
    text: str
    audio_path: str

class RecordsResponse(BaseModel):
    records: list

# ============ Database Setup ============

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            input_text TEXT,
            output_text TEXT,
            audio_path TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_record(record_id: str, record_type: str, input_text: str = None, 
                output_text: str = None, audio_path: str = None):
    """Save a record to database"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT INTO records (id, type, input_text, output_text, audio_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (record_id, record_type, input_text, output_text, audio_path, 
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_records():
    """Get all records from database"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT * FROM records ORDER BY created_at DESC LIMIT 100
    """)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records

# ============ FastAPI App ============

app = FastAPI(
    title="PyVoice API",
    description="Speech Recognition (ASR) and Text-to-Speech (TTS) API",
    version="1.0.0"
)

# Initialize database
init_db()

# Initialize engines (lazy loading)
_asr = None
_tts = None

def get_asr():
    global _asr
    if _asr is None:
        print("Loading ASR model...")
        _asr = ASR()
        print("ASR model loaded!")
    return _asr

def get_tts():
    global _tts
    if _tts is None:
        print("Loading TTS model...")
        _tts = TTS()
        print("TTS model loaded!")
    return _tts

# ============ API Endpoints ============

@app.get("/", response_class=HTMLResponse)
async def index():
    """Home page with simple UI"""
    return HOME_PAGE_HTML


@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(audio: UploadFile = File(...)):
    """
    Speech to Text
    
    Upload an audio file and get transcribed text.
    
    - **audio**: Audio file (supports WAV, MP3, FLAC, etc.)
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Generate unique ID
    record_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    ext = Path(audio.filename).suffix or '.wav'
    audio_path = UPLOAD_DIR / f"{record_id}{ext}"
    
    content = await audio.read()
    with open(audio_path, 'wb') as f:
        f.write(content)
    
    try:
        # Recognize speech
        text = get_asr().recognize(str(audio_path))
        
        # Save record
        save_record(record_id, "asr", output_text=text, audio_path=str(audio_path))
        
        return ASRResponse(
            id=record_id,
            text=text,
            audio_path=str(audio_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts", response_model=TTSResponse)
async def tts_endpoint(request: TTSRequest):
    """
    Text to Speech
    
    Input text and generate audio file.
    
    - **text**: Text to synthesize
    - **speed**: Speed (1.0 = normal, >1 = faster, <1 = slower)
    - **format**: Output format ("wav" or "mp3")
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    
    # Validate format
    output_format = request.format.lower()
    if output_format not in ["wav", "mp3"]:
        raise HTTPException(status_code=400, detail="Format must be 'wav' or 'mp3'")
    
    # Generate unique ID
    record_id = str(uuid.uuid4())[:8]
    
    try:
        # Synthesize speech (always generate WAV first)
        wav_path = OUTPUT_DIR / f"{record_id}.wav"
        get_tts().synthesize_to_file(request.text, str(wav_path), request.speed)
        
        # Convert to MP3 if requested
        if output_format == "mp3":
            audio_path = OUTPUT_DIR / f"{record_id}.mp3"
            convert_to_mp3(str(wav_path), str(audio_path))
            # Remove WAV file
            wav_path.unlink(missing_ok=True)
        else:
            audio_path = wav_path
        
        # Save record
        save_record(record_id, "tts", input_text=request.text, audio_path=str(audio_path))
        
        return TTSResponse(
            id=record_id,
            text=request.text,
            audio_path=str(audio_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records", response_model=RecordsResponse)
async def records_endpoint():
    """
    Get History
    
    Returns the latest 100 records.
    """
    records = get_all_records()
    return RecordsResponse(records=records)


@app.get("/audio/{record_id}")
async def audio_endpoint(record_id: str):
    """
    Download Audio
    
    - **record_id**: Record ID
    """
    # Media type mapping
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
    }
    
    # Search in both directories
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for ext in media_types.keys():
            path = directory / f"{record_id}{ext}"
            if path.exists():
                return FileResponse(
                    path=str(path),
                    media_type=media_types[ext],
                    filename=f"{record_id}{ext}"
                )
    
    raise HTTPException(status_code=404, detail="Audio not found")


# ============ WebSocket Streaming ASR ============

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech recognition.
    
    Protocol:
        1. Client connects
        2. Client sends audio data (binary, 16kHz 16-bit PCM)
        3. Server sends back transcription (JSON: {"text": "...", "is_final": bool})
        4. Client sends "END" text message to finish
        5. Server sends final result and closes
    """
    await websocket.accept()
    
    audio_buffer = bytearray()
    is_connected = True
    
    try:
        while is_connected:
            try:
                data = await websocket.receive()
            except RuntimeError:
                is_connected = False
                break
            
            if data.get("type") == "websocket.disconnect":
                is_connected = False
                break
            
            if "text" in data:
                if data["text"] == "END":
                    break
                continue
            
            if "bytes" in data:
                audio_buffer.extend(data["bytes"])
                
                # Process every ~3 seconds of audio (16kHz * 2 bytes * 3 sec = 96000 bytes)
                if len(audio_buffer) >= 96000:
                    text = process_audio_chunk(audio_buffer)
                    if text and is_connected:
                        try:
                            await websocket.send_json({
                                "text": text,
                                "is_final": False
                            })
                        except RuntimeError:
                            is_connected = False
                            break
                    audio_buffer.clear()
        
        # Final transcription
        if audio_buffer and is_connected:
            text = process_audio_chunk(audio_buffer)
            if text:
                try:
                    await websocket.send_json({
                        "text": text,
                        "is_final": True
                    })
                    await websocket.send_json({"status": "done"})
                except RuntimeError:
                    pass
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


def process_audio_chunk(audio_data: bytes) -> str:
    """Process audio chunk and return transcription"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write WAV header for raw PCM data (16kHz, 16-bit, mono)
        sample_rate = 16000
        num_channels = 1
        bits_per_sample = 16
        data_size = len(audio_data)
        
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, num_channels, sample_rate,
                           sample_rate * num_channels * bits_per_sample // 8,
                           num_channels * bits_per_sample // 8, bits_per_sample))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(audio_data)
        temp_path = f.name
    
    try:
        text = get_asr().recognize(temp_path)
        return text
    except Exception as e:
        print(f"ASR error: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ============ Simple HTML UI ============

HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyVoice - Speech Recognition & Synthesis</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            margin-bottom: 16px;
            font-size: 1.2em;
            color: #00d9ff;
        }
        textarea, input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 16px;
            margin-bottom: 12px;
        }
        textarea { min-height: 100px; resize: vertical; }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        button:hover { transform: translateY(-2px); opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-asr { background: linear-gradient(90deg, #00d9ff, #0099cc); color: #fff; }
        .btn-tts { background: linear-gradient(90deg, #00ff88, #00cc6a); color: #000; }
        .result {
            margin-top: 16px;
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            display: none;
        }
        .result.show { display: block; }
        .result-text { word-break: break-all; }
        audio { width: 100%; margin-top: 12px; }
        .records { margin-top: 20px; }
        .record-item {
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .record-type {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 8px;
        }
        .record-type.asr { background: #00d9ff; color: #000; }
        .record-type.tts { background: #00ff88; color: #000; }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        .api-link {
            text-align: center;
            margin-bottom: 20px;
        }
        .api-link a {
            color: #00d9ff;
            text-decoration: none;
        }
        .api-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ PyVoice</h1>
        
        <div class="api-link">
            <a href="/docs" target="_blank">📖 View API Documentation (Swagger UI)</a>
        </div>
        
        <!-- ASR Section -->
        <div class="card">
            <h2>🎤 Speech Recognition (ASR)</h2>
            
            <!-- Live Recording -->
            <div style="margin-bottom: 16px;">
                <button class="btn-asr" id="recordBtn" onclick="toggleRecording()" style="background: linear-gradient(90deg, #ff6b6b, #ee5a5a);">
                    🎙️ Start Recording
                </button>
                <span id="recordStatus" style="margin-left: 12px; color: #888;"></span>
            </div>
            
            <!-- Or upload file -->
            <div style="margin-bottom: 12px; color: #888; font-size: 14px;">— or upload a file —</div>
            <input type="file" id="audioFile" accept="audio/*">
            <button class="btn-asr" onclick="doASR()">Recognize File</button>
            
            <div class="loading" id="asrLoading">Recognizing...</div>
            <div class="result" id="asrResult">
                <strong>Result:</strong>
                <p class="result-text" id="asrText"></p>
            </div>
        </div>
        
        <!-- TTS Section -->
        <div class="card">
            <h2>🔊 Text to Speech (TTS)</h2>
            <textarea id="ttsInput" placeholder="Enter text to synthesize..."></textarea>
            <div style="margin-bottom: 12px;">
                <label style="margin-right: 16px;">
                    <input type="radio" name="format" value="wav" checked> WAV
                </label>
                <label>
                    <input type="radio" name="format" value="mp3"> MP3
                </label>
            </div>
            <button class="btn-tts" onclick="doTTS()">Generate Speech</button>
            <div class="loading" id="ttsLoading">Generating...</div>
            <div class="result" id="ttsResult">
                <strong>Generated!</strong>
                <audio id="ttsAudio" controls></audio>
            </div>
        </div>
        
        <!-- Records Section -->
        <div class="card">
            <h2>📋 History</h2>
            <button class="btn-asr" onclick="loadRecords()" style="background: #666;">Refresh</button>
            <div class="records" id="recordsList"></div>
        </div>
    </div>
    
    <script>
        // ========== Live Recording with WebSocket ==========
        let isRecording = false;
        let audioContext = null;
        let ws = null;
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { 
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                // Connect WebSocket
                ws = new WebSocket(`ws://${window.location.host}/ws/asr`);
                
                ws.onopen = () => {
                    document.getElementById('recordStatus').textContent = 'Connected, speak now...';
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.text) {
                        const asrText = document.getElementById('asrText');
                        asrText.textContent += data.text + ' ';
                        document.getElementById('asrResult').classList.add('show');
                    }
                    if (data.status === 'done') {
                        document.getElementById('recordStatus').textContent = 'Done!';
                    }
                };
                
                ws.onerror = (e) => {
                    console.error('WebSocket error:', e);
                    document.getElementById('recordStatus').textContent = 'Connection error';
                };
                
                // Process audio data
                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const int16Data = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                        }
                        ws.send(int16Data.buffer);
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                // Update UI
                isRecording = true;
                document.getElementById('recordBtn').textContent = '⏹️ Stop Recording';
                document.getElementById('recordBtn').style.background = 'linear-gradient(90deg, #ff4444, #cc3333)';
                document.getElementById('recordStatus').textContent = 'Connecting...';
                document.getElementById('asrText').textContent = '';
                document.getElementById('asrResult').classList.add('show');
                
                // Store for cleanup
                window.currentStream = stream;
                window.currentProcessor = processor;
                window.currentSource = source;
                
            } catch (e) {
                alert('Microphone access denied: ' + e.message);
            }
        }
        
        function stopRecording() {
            if (window.currentStream) {
                window.currentStream.getTracks().forEach(track => track.stop());
            }
            if (window.currentProcessor) {
                window.currentProcessor.disconnect();
            }
            if (window.currentSource) {
                window.currentSource.disconnect();
            }
            if (audioContext) {
                audioContext.close();
            }
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('END');
            }
            
            isRecording = false;
            document.getElementById('recordBtn').textContent = '🎙️ Start Recording';
            document.getElementById('recordBtn').style.background = 'linear-gradient(90deg, #ff6b6b, #ee5a5a)';
            document.getElementById('recordStatus').textContent = 'Processing...';
        }
        
        // ========== File Upload ASR ==========
        async function doASR() {
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files.length) {
                alert('Please select an audio file');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', fileInput.files[0]);
            
            document.getElementById('asrLoading').classList.add('show');
            document.getElementById('asrResult').classList.remove('show');
            
            try {
                const res = await fetch('/asr', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (data.detail) {
                    alert('Error: ' + data.detail);
                } else {
                    document.getElementById('asrText').textContent = data.text;
                    document.getElementById('asrResult').classList.add('show');
                }
            } catch (e) {
                alert('Request failed: ' + e);
            }
            
            document.getElementById('asrLoading').classList.remove('show');
        }
        
        async function doTTS() {
            const text = document.getElementById('ttsInput').value.trim();
            if (!text) {
                alert('Please enter text');
                return;
            }
            
            const format = document.querySelector('input[name="format"]:checked').value;
            
            document.getElementById('ttsLoading').classList.add('show');
            document.getElementById('ttsResult').classList.remove('show');
            
            try {
                const res = await fetch('/tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, format: format })
                });
                const data = await res.json();
                
                if (data.detail) {
                    alert('Error: ' + data.detail);
                } else {
                    document.getElementById('ttsAudio').src = '/audio/' + data.id;
                    document.getElementById('ttsResult').classList.add('show');
                }
            } catch (e) {
                alert('Request failed: ' + e);
            }
            
            document.getElementById('ttsLoading').classList.remove('show');
        }
        
        async function loadRecords() {
            try {
                const res = await fetch('/records');
                const data = await res.json();
                
                const list = document.getElementById('recordsList');
                list.innerHTML = data.records.map(r => `
                    <div class="record-item">
                        <span class="record-type ${r.type}">${r.type.toUpperCase()}</span>
                        <span>${r.input_text || r.output_text || ''}</span>
                        <br><small style="color:#888">${r.created_at}</small>
                    </div>
                `).join('');
            } catch (e) {
                console.error(e);
            }
        }
        
        // Load records on page load
        loadRecords();
    </script>
</body>
</html>
"""

# ============ Run Server ============

if __name__ == '__main__':
    import uvicorn
    
    print("=" * 50)
    print("PyVoice API Server (FastAPI)")
    print("=" * 50)
    print(f"Storage: {Path('storage').absolute()}")
    print(f"  - Uploads: {UPLOAD_DIR.absolute()}")
    print(f"  - Outputs: {OUTPUT_DIR.absolute()}")
    print(f"  - Database: {DB_PATH.absolute()}")
    print("=" * 50)
    print("Server: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host='0.0.0.0', port=8000)
