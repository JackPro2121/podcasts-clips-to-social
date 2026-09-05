import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
    native_transcript: Optional[List[Dict[str, Any]]] = None
) -> List[TranscriptSegment]:
    """Fetches transcript either from native YouTube data or falls back to Whisper."""
    if native_transcript:
        print("[+] Using native YouTube transcript (0 compute cost).")
        return parse_native_transcript(native_transcript)
    
    print("[!] Native transcript not found. Running faster-whisper speech-to-text...")
    return transcribe_audio_whisper(video_path)
