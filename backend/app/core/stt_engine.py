"""
stt_engine.py
-------------
Tek sorumluluk: Ses dosyasını yazıya çevirme (Speech-to-Text, STT)

Desteklenen backend'ler:
 - whisper        : OpenAI Whisper (local, "openai-whisper" kütüphanesi)
 - faster_whisper : CTranslate2 tabanlı hızlı Whisper
 - vosk           : Offline STT (Vosk)
 - google         : Google Cloud Speech-to-Text
 - azure          : Azure Cognitive Services Speech-to-Text
 - openai_api     : OpenAI'nin Whisper API'si (Chat Completions değil! Audio Transcriptions)

Ortam değişkenleri:
  BACKEND_STT        : varsayılan backend (whisper|faster_whisper|vosk|google|azure|openai_api) [default: whisper]
  STT_OUTPUT_DIR     : çıktı/çalışma klasörü [default: ./data/stt]

  # OpenAI API
  OPENAI_API_KEY     : Gerekli (openai_api için)

  # Google
  GOOGLE_APPLICATION_CREDENTIALS : Service account JSON dosyası (google için)

  # Azure
  AZURE_SPEECH_KEY    : Azure anahtarı
  AZURE_SPEECH_REGION : Bölge (örn: westeurope)

Notlar:
 - Giriş formatı: wav/mp3/m4a/ogg; bazı backend'ler için wav önerilir.
 - Dönüş: dict -> {"text": str, "language": str, "backend": str, "segments": Optional[List]}
 - Gelişmiş zaman damgalı segmentler: faster-whisper ve vosk'ta mümkün; minimal seviyede expose edilir.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import time
import uuid

# Opsiyonel bağımlılıkları nazikçe yükle
try:
    import whisper as whisper_local
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

try:
    from faster_whisper import WhisperModel as FasterWhisper
    HAS_FASTER = True
except Exception:
    HAS_FASTER = False

try:
    import vosk
    HAS_VOSK = True
except Exception:
    HAS_VOSK = False

try:
    from google.cloud import speech as google_speech
    HAS_GOOGLE = True
except Exception:
    HAS_GOOGLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE = True
except Exception:
    HAS_AZURE = False

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

DEFAULT_BACKEND = os.getenv("BACKEND_STT", "whisper").lower()
OUTPUT_DIR = Path(os.getenv("STT_OUTPUT_DIR", "data/stt"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Public API
# -----------------------------

def transcribe_audio(file_path: str | Path,
                     backend: Optional[str] = None,
                     language: str = "tr",
                     model_size: str = "base",
                     device: Optional[str] = None,
                     fp16: Optional[bool] = None,
                     **kwargs) -> Dict[str, Any]:
    """Ses dosyasını yazıya çevirir.

    Args:
      file_path: Girdi ses dosyası yolu (wav/mp3/m4a/ogg)
      backend  : whisper | faster_whisper | vosk | google | azure | openai_api
      language : Dil kodu ("tr", "tr-TR" vs.)
      model_size: Whisper model boyutu (tiny/base/small/medium/large-v2)
      device   : cuda|cpu (whisper/faster_whisper için)
      fp16     : True/False (whisper local)

    Returns:
      {
        "text": str,
        "language": str,
        "backend": str,
        "segments": Optional[List[dict]]
      }
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {p}")

    if backend == "whisper":
        return _stt_whisper(p, language=language, model_size=model_size, device=device, fp16=fp16)
    elif backend == "faster_whisper":
        return _stt_faster_whisper(p, language=language, model_size=model_size, device=device, **kwargs)
    elif backend == "vosk":
        return _stt_vosk(p, language=language, **kwargs)
    elif backend == "google":
        return _stt_google(p, language=language, **kwargs)
    elif backend == "azure":
        return _stt_azure(p, language=language, **kwargs)
    elif backend == "openai_api":
        return _stt_openai_api(p, language=language, **kwargs)
    else:
        raise ValueError(f"Bilinmeyen backend: {backend}")

# -----------------------------
# Whisper (local)
# -----------------------------

def _stt_whisper(path: Path, language: str, model_size: str, device: Optional[str], fp16: Optional[bool]) -> Dict[str, Any]:
    if not HAS_WHISPER:
        raise RuntimeError("openai-whisper kurulu değil: pip install -U openai-whisper")
    model = whisper_local.load_model(model_size, device=device or ("cuda" if _cuda_available() else "cpu"))
    res = model.transcribe(str(path), language=_norm_lang(language), fp16=(fp16 if fp16 is not None else _cuda_available()))
    segments = [
        {"start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
        for s in res.get("segments", [])
    ]
    return {"text": res.get("text", "").strip(), "language": res.get("language", language), "backend": "whisper", "segments": segments}

# -----------------------------
# Faster-Whisper (local, hızlı)
# -----------------------------

def _stt_faster_whisper(path: Path, language: str, model_size: str, device: Optional[str], **kwargs) -> Dict[str, Any]:
    if not HAS_FASTER:
        raise RuntimeError("faster-whisper kurulu değil: pip install faster-whisper")
    compute_type = kwargs.get("compute_type", "float16" if _cuda_available() else "int8")
    model = FasterWhisper(model_size, device=device or ("cuda" if _cuda_available() else "cpu"), compute_type=compute_type)
    seg_iter, info = model.transcribe(str(path), language=_norm_lang(language))
    segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    for s in seg_iter:
        segments.append({"start": s.start, "end": s.end, "text": s.text})
        text_parts.append(s.text)
    return {"text": " ".join(text_parts).strip(), "language": info.language or language, "backend": "faster_whisper", "segments": segments}

# -----------------------------
# Vosk (offline)
# -----------------------------

def _stt_vosk(path: Path, language: str, model_path: Optional[str] = None, sample_rate: int = 16000, **kwargs) -> Dict[str, Any]:
    if not HAS_VOSK:
        raise RuntimeError("vosk kurulu değil: pip install vosk")
    # Model indirme/konum: https://alphacephei.com/vosk/models (Türkçe modeli gerekli)
    model_path = model_path or os.getenv("VOSK_MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        raise RuntimeError("Vosk Türkçe modeli bulunamadı. VOSK_MODEL_PATH ayarla.")
    model = vosk.Model(model_path)

    import wave
    import json as _json

    # wav bekliyor, gerekirse dönüştürme kullanıcıya bırakıldı
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
            raise RuntimeError("Vosk için 16kHz, 16-bit, mono WAV dosyası gereklidir.")
        rec = vosk.KaldiRecognizer(model, sample_rate)
        segments: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                j = _json.loads(rec.Result())
                if j.get("text"):
                    segments.append({"text": j["text"]})
                    text_parts.append(j["text"])
        final = _json.loads(rec.FinalResult())
        if final.get("text"):
            segments.append({"text": final["text"]})
            text_parts.append(final["text"])
    return {"text": " ".join(text_parts).strip(), "language": _norm_lang(language), "backend": "vosk", "segments": segments}

# -----------------------------
# Google Cloud Speech-to-Text
# -----------------------------

def _stt_google(path: Path, language: str, **kwargs) -> Dict[str, Any]:
    if not HAS_GOOGLE:
        raise RuntimeError("google-cloud-speech kurulu değil: pip install google-cloud-speech")
    client = google_speech.SpeechClient()

    # FLAC/WAV önerilir; diğer formatlar için encoding belirtmek gerekir
    language_code = _lang_google(language)
    with open(path, "rb") as f:
        content = f.read()

    audio = google_speech.RecognitionAudio(content=content)
    config = google_speech.RecognitionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
        model=kwargs.get("model", "default"),
        use_enhanced=kwargs.get("use_enhanced", True),
    )
    response = client.recognize(config=config, audio=audio)
    texts: List[str] = []
    for result in response.results:
        alternative = result.alternatives[0]
        texts.append(alternative.transcript)
    return {"text": " ".join(texts).strip(), "language": language_code, "backend": "google", "segments": []}

# -----------------------------
# Azure Speech-to-Text
# -----------------------------

def _stt_azure(path: Path, language: str, **kwargs) -> Dict[str, Any]:
    if not HAS_AZURE:
        raise RuntimeError("azure-cognitiveservices-speech kurulu değil: pip install azure-cognitiveservices-speech")
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY ve AZURE_SPEECH_REGION gerekli")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = _lang_azure(language)
    audio_cfg = speechsdk.audio.AudioConfig(filename=str(path))

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_cfg)
    res = recognizer.recognize_once()

    if res.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = res.text
    elif res.reason == speechsdk.ResultReason.NoMatch:
        text = ""
    else:
        raise RuntimeError(f"Azure STT hata: {res.reason}")

    return {"text": (text or '').strip(), "language": speech_config.speech_recognition_language, "backend": "azure", "segments": []}

# -----------------------------
# OpenAI Whisper API (cloud)
# -----------------------------

def _stt_openai_api(path: Path, language: str, **kwargs) -> Dict[str, Any]:
    if not HAS_OPENAI:
        raise RuntimeError("openai paketi kurulu değil: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY gerekli (Whisper API için)")
    openai.api_key = api_key

    # Not: Yeni OpenAI SDK'larında Audio.transcriptions değişiklik gösterebilir.
    # Aşağıdaki kullanım OpenAI'nin eski python sdk'sına göredir;
    # gerekli görüldüğünde güncellenmelidir.
    with open(path, "rb") as f:
        resp = openai.Audio.transcriptions.create(
            model=kwargs.get("model", "whisper-1"),
            file=f,
            language=_norm_lang(language)
        )
    # resp.text tipik dönüş
    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else "")
    return {"text": (text or '').strip(), "language": _norm_lang(language), "backend": "openai_api", "segments": []}

# -----------------------------
# Helpers
# -----------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _norm_lang(lang: str) -> str:
    lang = (lang or "tr").strip()
    if lang.lower() in ("tr", "tr-tr", "turkish"):
        return "tr"
    return lang


def _lang_google(lang: str) -> str:
    # Google genelde BCP-47 bekler
    l = _norm_lang(lang)
    return "tr-TR" if l == "tr" else l


def _lang_azure(lang: str) -> str:
    l = _norm_lang(lang)
    return "tr-TR" if l == "tr" else l