"""
WhisperX transcriber for Module B1.

Runs full-video ASR → word-level forced alignment → speaker diarization
and saves a structured transcript.json.

Environment variables:
    HF_TOKEN       HuggingFace token for pyannote speaker diarization.
                    Accept model agreement at: hf.co/pyannote/speaker-diarization-3.1
    WHISPER_MODEL  Model size: tiny / base / small / medium (default).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_B_INTERIM = PROJECT_ROOT / "data" / "interim" / "track_b"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# float16 on GPU; float32 on CPU (int8 segfaults on macOS Intel via CTranslate2)
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract mono 16kHz WAV — WhisperX requirement."""
    wav_path = out_dir / "audio.wav"
    if wav_path.exists():
        return wav_path
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


def run_whisperx(audio_path: Path, out_dir: Path, min_speakers: int, max_speakers: int) -> dict:
    """
    ASR → forced alignment → (optional) speaker diarization.

    Saves transcript.json and returns the dict.
    Idempotent: if transcript.json already exists, loads and returns it.
    """
    out_path = out_dir / "transcript.json"
    if out_path.exists():
        print(f"  [B1] transcript.json already exists — loading from cache.")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    import whisperx  # import here so the module is importable without whisperx installed

    # ── Step 1: ASR ──────────────────────────────────────────────
    print(f"  [B1] Loading WhisperX ({WHISPER_MODEL}) on {DEVICE} ({COMPUTE_TYPE})…")
    model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)
    language = result.get("language", "en")
    print(f"  [B1] Detected language: {language} | Segments: {len(result['segments'])}")

    # Free GPU memory before alignment
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Step 2: Forced word-level alignment ──────────────────────
    print("  [B1] Running forced alignment (wav2vec2)…")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, DEVICE,
        return_char_alignments=False,
    )
    del align_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Step 3: Speaker diarization ──────────────────────────────
    if HF_TOKEN:
        print("  [B1] Running speaker diarization (pyannote)…")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("  [B1] Warning: HF_TOKEN not set → diarization skipped. Set HF_TOKEN env var.")
        print("       All segments will be labeled SPEAKER_00.")

    # ── Step 4: Build output structure ───────────────────────────
    segments = []
    for i, seg in enumerate(result["segments"]):
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": round(float(w.get("start") or 0.0), 3),
                "end": round(float(w.get("end") or 0.0), 3),
                "score": round(float(w.get("score") or 0.0), 3),
            })
        segments.append({
            "segment_id": i,
            "text": seg["text"].strip(),
            "start": round(float(seg.get("start") or 0.0), 3),
            "end": round(float(seg.get("end") or 0.0), 3),
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "words": words,
        })

    speakers = sorted({s["speaker"] for s in segments})
    transcript = {
        "video_id": out_dir.name,
        "language": language,
        "whisper_model": WHISPER_MODEL,
        "diarization": bool(HF_TOKEN),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": segments,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"  [B1] Saved → {out_path}")
    return transcript


def process_video(
    video_path: Path,
    out_root: Path = TRACK_B_INTERIM,
    min_speakers: int = 1,
    max_speakers: int = 4,
) -> dict:
    video_id = video_path.stem
    out_dir = out_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[B1] {video_id}")
    print(f"{'='*60}")

    audio_path = extract_audio(video_path, out_dir)
    transcript = run_whisperx(audio_path, out_dir, min_speakers, max_speakers)
    transcript["video_id"] = video_id  # correct even when loaded from cache

    _print_summary(transcript)
    return transcript


def _print_summary(transcript: dict):
    segs = transcript["segments"]
    duration = segs[-1]["end"] if segs else 0.0
    print(f"\n  Duration : ~{duration:.1f}s")
    print(f"  Segments : {len(segs)}")
    print(f"  Speakers : {transcript['speakers']}")
    if segs:
        print("  Preview (first 5 segments):")
        for s in segs[:5]:
            text_preview = s["text"][:80] + ("…" if len(s["text"]) > 80 else "")
            print(f"    [{s['start']:.1f}s–{s['end']:.1f}s] {s['speaker']}: {text_preview}")


