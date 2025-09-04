# app/core/llm_responder.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os

# --- OpenAI (0.28.x için) ---
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# --- Local (opsiyonel) ---
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

DEFAULT_SYSTEM = (
    "Türkçe konuşan, kısa ve kaynaklara dayalı yanıt veren bir asistansın. "
    "Cevabını bağlamdaki bilgiden emin olmadığın konularda genişletme; 'bağlam yetersiz' de."
)

class LLMResponder:
    def __init__(self,
                 gen_model_id: Optional[str] = None,
                 openai_api_base: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 hf_local_model: Optional[str] = None) -> None:
        # Model seçimi
        self.gen_model_id = gen_model_id or os.getenv("GEN_MODEL_ID", "gpt-3.5-turbo")

        # Anahtarları oku
        or_key = os.getenv("OPENROUTER_API_KEY")
        oa_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Base URL override (opsiyonel)
        api_base_env = os.getenv("OPENAI_API_BASE")

        # Öncelik: OpenRouter → sonra OpenAI
        if or_key:
            self.api_base = api_base_env or "https://openrouter.ai/api/v1"
            self.api_key = or_key
        else:
            self.api_base = api_base_env or "https://api.openai.com/v1"
            self.api_key = oa_key

        self.hf_local_model = hf_local_model or os.getenv("HF_LOCAL_MODEL")
        self._pipe = None

        # Eski SDK (0.28) ayarları
        if HAS_OPENAI and self.api_key:
            openai.api_base = self.api_base
            openai.api_key = self.api_key
            # openai.proxy = ...  # gerekiyorsa

    def generate(self,
                 question: str,
                 contexts: List[str],
                 source_indices: Optional[List[int]] = None,
                 system_prompt: str = DEFAULT_SYSTEM,
                 style: Optional[Dict[str, Any]] = None,
                 provider: str = "openai",
                 add_citations: bool = True,
                 budget_chars: int = 6000) -> Tuple[str, Dict[str, Any]]:
        style = style or {"temperature": 0.4, "max_tokens": 512}

        pruned_ctx, used_idx = self._prune_context(contexts, source_indices or [], budget_chars)
        prompt = self._build_prompt(question, pruned_ctx)

        if provider == "local":
            text = self._generate_local(prompt, style)
        else:
            text = self._generate_openai(prompt, system_prompt, style)

        if add_citations and used_idx:
            cites = " ".join(f"[{i}]" for i in used_idx)
            text = f"{text}\n\nKaynaklar: {cites}"

        meta = {
            "model": self.gen_model_id if provider != "local" else (self.hf_local_model or "local"),
            "provider": provider,
            "used_indices": used_idx,
        }
        return text, meta

    # ---- helpers ----
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx = "\n---\n".join(contexts)
        return (
            "Aşağıdaki bağlam sadece yardımcı bilgidir. Sadece bağlamdaki bilgiden emin olduğun kısımlara dayanarak cevap ver. "
            "Emin değilsen 'bağlam yetersiz' de.\n\n"
            f"Soru: {question}\n\nBağlam:\n{ctx}\n\nCevap (Türkçe):"
        )

    def _prune_context(self, contexts: List[str], idxs: List[int], budget_chars: int) -> Tuple[List[str], List[int]]:
        if not contexts:
            return [], []
        out_ctx, out_idx, total = [], [], 0
        for c, i in zip(contexts, idxs if idxs else range(len(contexts))):
            c = c.strip()
            if total + len(c) > budget_chars and out_ctx:
                break
            out_ctx.append(c)
            out_idx.append(i)
            total += len(c)
        return out_ctx, out_idx

    # ---- OpenAI / OpenRouter (SDK 0.28) ----
    def _generate_openai(self, prompt: str, system_prompt: str, style: Dict[str, Any]) -> str:
        if not (HAS_OPENAI and self.api_key):
            return "[LLM yapılandırılmadı] OPENROUTER_API_KEY veya OPENAI_API_KEY ayarla."
        try:
            resp = openai.ChatCompletion.create(
                model=self.gen_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=style.get("temperature", 0.4),
                max_tokens=style.get("max_tokens", 512),
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            return f"[LLM hata]: {e}"

    # ---- Local (opsiyonel) ----
    def _lazy_local_pipe(self):
        if not HAS_TRANSFORMERS:
            return None
        if self._pipe is not None:
            return self._pipe
        model_id = self.hf_local_model or self.gen_model_id
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto"
        )
        self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
        return self._pipe

    def _generate_local(self, prompt: str, style: Dict[str, Any]) -> str:
        pipe = self._lazy_local_pipe()
        if pipe is None:
            return "[Local LLM hazır değil] transformers kurulu mu ve model erişilebilir mi?"
        out = pipe(
            prompt,
            max_new_tokens=style.get("max_tokens", 512),
            do_sample=True,
            temperature=style.get("temperature", 0.6),
        )
        text = out[0]["generated_text"]
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()