import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from src.config import CHOCODATA_API_KEY

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTimestamp]

def fetch_transcript_chocodata(video_id: str, api_key: Optional[str] = None) -> Optional[List[TranscriptSegment]]:
    """
    Fetches official timestamped YouTube transcript via Chocodata REST API in 0.4s.
    Bypasses all bot-detection, datacenter blocks, and local Whisper CPU compute.
    Cost: $0.0009 / call.
    """
    key = api_key or CHOCODATA_API_KEY or os.getenv("CHOCODATA_API_KEY", "")
    if not key:
        return None

    url = f"https://api.chocodata.com/api/v1/youtube/transcript?api_key={key}&video_id={video_id}"
    try:
        print(f"[*] Fetching transcript via Chocodata API for video {video_id}...")
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            raw_segments = data.get("segments", [])
            if raw_segments:
                print(f"[+] Chocodata returned {len(raw_segments)} timed segments in <1s!")
                return parse_native_transcript(raw_segments)
            else:
                print("[-] Chocodata: No transcript segments found for this video.")
        else:
            print(f"[-] Chocodata transcript error: {res.status_code} - {res.text[:100]}")
    except Exception as e:
        print(f"[-] Chocodata transcript fetch warning: {e}")
    return None

def parse_native_transcript(raw_transcript: List[Dict[str, Any]]) -> List[TranscriptSegment]:
    """Converts YouTubeTranscriptApi format into TranscriptSegment structures."""
    segments = []
    for item in raw_transcript:
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        end = start + duration
        text = str(item.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        
        # Approximate word timestamps from segment duration
        words_list = text.split()
        word_objs = []
        if words_list:
            word_duration = duration / len(words_list)
            for idx, w in enumerate(words_list):
                w_start = start + idx * word_duration
                w_end = w_start + word_duration
                word_objs.append(WordTimestamp(word=w, start=w_start, end=w_end))
        
        segments.append(TranscriptSegment(
            start=start,
            end=end,
            text=text,
            words=word_objs
        ))
    return segments

def transcribe_audio_whisper(
    video_or_audio_path: Path,
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8"
) -> List[TranscriptSegment]:
    """
    Fallback transcription using faster-whisper on CPU.
    Runs free on GitHub Actions using int8 quantization.
    Provides word-level timestamps.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper is required for audio transcription when native captions are missing.")

    print(f"[*] Initializing Whisper ({model_size}, {device}, {compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[*] Transcribing audio from {video_or_audio_path.name}...")
    whisper_segments, _ = model.transcribe(
        str(video_or_audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    results: List[TranscriptSegment] = []
    for seg in whisper_segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end
                ))
        else:
            # Fallback if words not present
            words_split = seg.text.strip().split()
            if words_split:
                dur = (seg.end - seg.start) / len(words_split)
                for i, w in enumerate(words_split):
                    words.append(WordTimestamp(
                        word=w,
                        start=seg.start + i * dur,
                        end=seg.start + (i + 1) * dur
                    ))

        results.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=words
        ))

    print(f"[+] Transcription complete: {len(results)} segments generated.")
    return results

def get_transcript(
    video_path: Path,
    native_transcript: Optional[List[Dict[str, Any]]] = None,
    video_id: Optional[str] = None
) -> List[TranscriptSegment]:
    """Fetches transcript with multi-tier fallback: Native YouTube -> Chocodata API -> Whisper CPU."""
    if native_transcript:
        print("[+] Using native YouTube transcript (0 compute cost).")
        return parse_native_transcript(native_transcript)

    # Fast Tier 1.5: Chocodata API
    target_id = video_id
    if not target_id:
        match = re.search(r'([0-9A-Za-z_-]{11})', video_path.stem)
        if match:
            target_id = match.group(1)

    if target_id and CHOCODATA_API_KEY:
        choco_segments = fetch_transcript_chocodata(target_id)
        if choco_segments:
            return choco_segments

    print("[!] Native/Chocodata transcript not found. Running faster-whisper speech-to-text...")
    return transcribe_audio_whisper(video_path)
