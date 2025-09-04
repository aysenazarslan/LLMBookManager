"""
tts_engine.py
-------------
Tek sorumluluk: Metni sese (TTS) dönüştürmek.

Desteklenen sağlayıcılar (backend):
 - gTTS       (hızlı, ücretsiz; MP3)
 - Google     (Google Cloud Text-to-Speech; MP3/WAV/OGG; üretim için önerilir)
 - Azure      (Azure Cognitive Services Speech; WAV)
 - Coqui      (Local TTS; WAV)

Ortam değişkenleri:
  BACKEND_TTS           : varsayılan backend (gtts|google|azure|coqui) [default: gtts]
  TTS_OUTPUT_DIR        : çıktı klasörü [default: ./data/tts]

  # Google
  GOOGLE_APPLICATION_CREDENTIALS : service account json yolu
  GOOGLE_TTS_AUDIO_ENCODING      : MP3|LINEAR16|OGG_OPUS [default: MP3]

  # Azure
  AZURE_SPEECH_KEY      : Azure Speech anahtarı
  AZURE_SPEECH_REGION   : ör. westeurope
  AZURE_SPEECH_FORMAT   : Riff16Khz16BitMonoPcm | Raw24Khz16BitMonoPcm vs. [default: Riff16Khz16BitMonoPcm]

  # Coqui
  COQUI_MODEL_ID        : ör. "tts_models/trk/ek/ek"

Örnek kullanım:
  from tts_engine import synthesize_tts
  path = synthesize_tts("Merhaba dünya", voice="tr-TR-Standard-A", backend="google")

Not:
  - Fonksiyon, üretilen ses dosyasının tam yolunu döndürür.
  - Hata durumunda Exception fırlatır.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import os
import time
import uuid

# İsteğe bağlı bağımlılıklar: import hatalarını yumuşat
try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    from google.cloud import texttospeech as google_tts
    HAS_GOOGLE = True
except Exception:
    HAS_GOOGLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE = True
except Exception:
    HAS_AZURE = False

try:
    from TTS.api import TTS as CoquiTTS
    HAS_COQUI = True
except Exception:
    HAS_COQUI = False

DEFAULT_BACKEND = os.getenv("BACKEND_TTS", "gtts").lower()
OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "data/tts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _unique_name(prefix: str, ext: str) -> Path:
    ts = int(time.time())
    return OUTPUT_DIR / f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"


# -----------------------------
# Public API
# -----------------------------

def synthesize_tts(text: str,
                   voice: str = "tr-TR-Standard-A",
                   backend: Optional[str] = None,
                   speaking_rate: float = 1.0,
                   pitch: float = 0.0,
                   volume_gain_db: float = 0.0,
                   audio_format: Optional[str] = None) -> Path:
    """Metinden ses üret ve dosya yolunu döndür.

    Args:
      text: Türkçe metin
      voice: Sağlayıcıya göre ses kimliği (gTTS için dil kodu kullanılır: 'tr')
      backend: 'gtts' | 'google' | 'azure' | 'coqui'
      speaking_rate: Konuşma hızı (1.0 = normal)
      pitch: Ton (yarım ton aralığına göre; sağlayıcıya bağlı)
      volume_gain_db: Ses şiddeti (dB)
      audio_format: Çıkış formatı (google: MP3|LINEAR16|OGG_OPUS; azure: format string)
    """
    if not text or not text.strip():
        raise ValueError("Boş metin verildi")

    backend = (backend or DEFAULT_BACKEND).lower()

    if backend == "gtts":
        return _tts_gtts(text, lang=voice.split("-")[0] if "-" in voice else "tr")
    elif backend == "google":
        return _tts_google(text, voice=voice, speaking_rate=speaking_rate, pitch=pitch, volume_gain_db=volume_gain_db, audio_format=audio_format)
    elif backend == "azure":
        return _tts_azure(text, voice=voice, speaking_rate=speaking_rate, pitch=pitch, audio_format=audio_format)
    elif backend == "coqui":
        return _tts_coqui(text)
    else:
        raise ValueError(f"Bilinmeyen backend: {backend}")


# -----------------------------
# gTTS
# -----------------------------

def _tts_gtts(text: str, lang: str = "tr") -> Path:
    if not HAS_GTTS:
        raise RuntimeError("gTTS kurulu değil: pip install gTTS")
    tts = gTTS(text=text, lang=lang)
    out = _unique_name("tts_gtts", "mp3")
    tts.save(str(out))
    return out


# -----------------------------
# Google Cloud TTS
# -----------------------------

def _tts_google(text: str, voice: str, speaking_rate: float, pitch: float, volume_gain_db: float, audio_format: Optional[str]) -> Path:
    if not HAS_GOOGLE:
        raise RuntimeError("google-cloud-texttospeech kurulu değil: pip install google-cloud-texttospeech")
    client = google_tts.TextToSpeechClient()

    # Ses seçimi: ör. tr-TR-Standard-A / tr-TR-Wavenet-A
    language_code = voice.split("-")[:2]
    language_code = "-".join(language_code) if language_code else "tr-TR"

    input_text = google_tts.SynthesisInput(text=text)
    voice_sel = google_tts.VoiceSelectionParams(
        language_code=language_code,
        name=voice
    )

    encoding = (audio_format or os.getenv("GOOGLE_TTS_AUDIO_ENCODING", "MP3")).upper()
    if encoding == "LINEAR16":
        audio_cfg = google_tts.AudioConfig(
            audio_encoding=google_tts.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_gain_db,
        )
        ext = "wav"
    elif encoding == "OGG_OPUS":
        audio_cfg = google_tts.AudioConfig(
            audio_encoding=google_tts.AudioEncoding.OGG_OPUS,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_gain_db,
        )
        ext = "ogg"
    else:
        audio_cfg = google_tts.AudioConfig(
            audio_encoding=google_tts.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_gain_db,
        )
        ext = "mp3"

    response = client.synthesize_speech(input=input_text, voice=voice_sel, audio_config=audio_cfg)
    out = _unique_name("tts_google", ext)
    with open(out, "wb") as f:
        f.write(response.audio_content)
    return out


# -----------------------------
# Azure Cognitive Services Speech
# -----------------------------

def _tts_azure(text: str, voice: str, speaking_rate: float, pitch: float, audio_format: Optional[str]) -> Path:
    if not HAS_AZURE:
        raise RuntimeError("azure-cognitiveservices-speech kurulu değil: pip install azure-cognitiveservices-speech")
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY ve AZURE_SPEECH_REGION gerekli")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice  # ör. tr-TR-AhmetNeural
    fmt = audio_format or os.getenv("AZURE_SPEECH_FORMAT", "Riff16Khz16BitMonoPcm")
    speech_config.set_speech_synthesis_output_format(getattr(speechsdk.SpeechSynthesisOutputFormat, fmt))

    out = _unique_name("tts_azure", "wav")
    audio_cfg = speechsdk.audio.AudioOutputConfig(filename=str(out))

    # SSML ile hız/ton kontrolü
    # <prosody rate="+10%" pitch="+0st"> ... </prosody>
    rate_pct = int((speaking_rate - 1.0) * 100)
    pitch_st = int(pitch)  # basit dönüşüm
    ssml = f"""
    <speak version='1.0' xml:lang='tr-TR'>
      <voice name='{voice}'>
        <prosody rate='{rate_pct:+d}%' pitch='{pitch_st:+d}st'>{text}</prosody>
      </voice>
    </speak>
    """.strip()

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_cfg)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS başarısız: {result.reason}")
    return out


# -----------------------------
# Coqui TTS (local)
# -----------------------------
_COQUI_INSTANCE = None

def _get_coqui() -> CoquiTTS:
    global _COQUI_INSTANCE
    if _COQUI_INSTANCE is not None:
        return _COQUI_INSTANCE
    if not HAS_COQUI:
        raise RuntimeError("coqui TTS kurulu değil: pip install TTS")
    model_id = os.getenv("COQUI_MODEL_ID", "tts_models/trk/ek/ek")  # Türkçe örnek (varsa)
    _COQUI_INSTANCE = CoquiTTS(model_id)
    return _COQUI_INSTANCE


def _tts_coqui(text: str) -> Path:
    tts = _get_coqui()
    out = _unique_name("tts_coqui", "wav")
    # Coqui bazı modellerde 'speaker' ve 'language' parametreleri isteyebilir
    tts.tts_to_file(text=text, file_path=str(out))
    return out