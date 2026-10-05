"""One ``LLMClient`` interface over official APIs and local models.

Providers: openai, deepseek (OpenAI-compatible), gemini, ollama, mock.
Only ``requests`` is needed; API keys come from environment variables named in
config.yaml (never hard-code them).

``MockClient`` is a deterministic simulator used for tests and dry runs. Its
output is synthetic and must never be reported as a result.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

import requests
import yaml


@dataclass
class LLMResponse:
    text: str
    latency: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict = field(default_factory=dict, repr=False)


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Minimum interval between calls, shared across threads."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class LLMClient:
    provider = "base"
    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

    def __init__(self, name: str, model: str, rpm: float | None = None, max_retries: int = 6,
                 timeout: float = 300, **options):
        self.name, self.model = name, model
        self.limiter = RateLimiter(rpm)
        self.max_retries, self.timeout = max_retries, timeout
        self.options = options

    # Each call is a fresh conversation: callers pass the full message list.
    def complete(self, messages: list[dict], temperature: float = 0.0, seed: int | None = None,
                 max_tokens: int = 2048) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            t0 = time.perf_counter()
            try:
                resp = self._call(messages, temperature, seed, max_tokens)
                resp.latency = time.perf_counter() - t0
                return resp
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code not in self.RETRY_STATUS:
                    body = e.response.text[:500] if e.response is not None else ""
                    raise LLMError(f"{self.name}: HTTP {code}: {body}") from e
                last = e
                retry_after = e.response.headers.get("retry-after") if e.response is not None else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
            except (requests.ConnectionError, requests.Timeout) as e:
                last, delay = e, min(60, 2 ** attempt)
            time.sleep(delay + random.random())
        raise LLMError(f"{self.name}: giving up after {self.max_retries} retries: {last}")

    def _call(self, messages, temperature, seed, max_tokens) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def _env_key(var: str | None) -> str:
    if not var:
        raise LLMError("api_key_env not set in config")
    key = os.environ.get(var)
    if not key:
        raise LLMError(f"environment variable {var} is not set")
    return key


class OpenAICompatibleClient(LLMClient):
    provider = "openai"
    default_base = "https://api.openai.com/v1"

    def __init__(self, name, model, api_key_env="OPENAI_API_KEY", base_url=None,
                 send_temperature=True, send_seed=True, **kw):
        super().__init__(name, model, **kw)
        self.api_key_env = api_key_env
        self.base_url = (base_url or self.default_base).rstrip("/")
        self.send_temperature, self.send_seed = send_temperature, send_seed

    def _call(self, messages, temperature, seed, max_tokens):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if self.send_temperature:
            payload["temperature"] = temperature
        if self.send_seed and seed is not None:
            payload["seed"] = seed
        data = self._post(f"{self.base_url}/chat/completions", payload,
                          {"Authorization": f"Bearer {_env_key(self.api_key_env)}"})
        usage = data.get("usage") or {}
        return LLMResponse(data["choices"][0]["message"].get("content") or "",
                           input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                           raw={"id": data.get("id"), "model": data.get("model")})


class DeepSeekClient(OpenAICompatibleClient):
    provider = "deepseek"
    default_base = "https://api.deepseek.com"

    def __init__(self, name, model, api_key_env="DEEPSEEK_API_KEY", **kw):
        kw.setdefault("send_seed", False)
        super().__init__(name, model, api_key_env=api_key_env, **kw)


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, name, model, api_key_env="GEMINI_API_KEY",
                 base_url="https://generativelanguage.googleapis.com/v1beta", **kw):
        super().__init__(name, model, **kw)
        self.api_key_env, self.base_url = api_key_env, base_url.rstrip("/")

    def _call(self, messages, temperature, seed, max_tokens):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if seed is not None:
            gen["seed"] = seed
        payload = {"contents": contents, "generationConfig": gen}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        # key goes in a header, not the URL
        data = self._post(f"{self.base_url}/models/{self.model}:generateContent", payload,
                          {"x-goog-api-key": _env_key(self.api_key_env)})
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"{self.name}: no candidates (promptFeedback={data.get('promptFeedback')})")
        text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
        usage = data.get("usageMetadata") or {}
        return LLMResponse(text, input_tokens=usage.get("promptTokenCount"),
                           output_tokens=usage.get("candidatesTokenCount"),
                           raw={"finishReason": cands[0].get("finishReason")})


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, name, model, base_url="http://localhost:11434", num_ctx=16384, **kw):
        super().__init__(name, model, **kw)
        self.base_url, self.num_ctx = base_url.rstrip("/"), num_ctx

    def _call(self, messages, temperature, seed, max_tokens):
        opts = {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens}
        if seed is not None:
            opts["seed"] = seed
        data = self._post(f"{self.base_url}/api/chat",
                          {"model": self.model, "messages": messages, "stream": False, "options": opts}, {})
        return LLMResponse(data["message"]["content"], input_tokens=data.get("prompt_eval_count"),
                           output_tokens=data.get("eval_count"))


# ---------------------------------------------------------------------------
# Mock simulator
# ---------------------------------------------------------------------------
_DOC_RE = re.compile(r"\[Attached document: ([^\]]+)\]\n(.*?)\n\[End of attached document\]"
                     r"|<<DOC id=([^>]+)>>\n(.*?)\n<</DOC>>", re.S)


def _find_docs(text: str) -> list[tuple[str, str]]:
    return [(m.group(1) or m.group(3), m.group(2) or m.group(4)) for m in _DOC_RE.finditer(text)]


class MockClient(LLMClient):
    """Deterministic stand-in for a model, used to exercise the pipeline.

    It 'reads' the documents, detects injected directives with the same
    heuristic classifier the pre-filter defense uses, and follows them with a
    configurable probability that drops when a defense is present.
    THE NUMBERS IT PRODUCES ARE SYNTHETIC.
    """
    provider = "mock"

    def __init__(self, name="mock", model="mock-1", follow_prob=0.95, defended_follow_prob=0.25,
                 alert_prob=0.85, **kw):
        super().__init__(name, model, **kw)
        self.follow_prob, self.defended_follow_prob, self.alert_prob = follow_prob, defended_follow_prob, alert_prob

    def _call(self, messages, temperature, seed, max_tokens):
        from defenses import flag_injection_sentences  # local import: avoid a cycle

        prompt = "\n\n".join(m["content"] for m in messages)
        rng = random.Random(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest())
        text = prompt.replace("ˆ", " ")
        if "Extract only factual CV fields" in prompt:  # sanitizer pass 1
            return self._extract(text, flag_injection_sentences)

        docs = _find_docs(text)
        if not docs and '"name"' in text:  # sanitizer pass 2 (JSON profiles)
            docs = [(p.get("name", "?"), json.dumps(p)) for p in self._json_profiles(text)]
        defended = any(k in prompt for k in ("INJECTION ALERT", "untrusted", "datamark"))
        injected = []
        for _, body in docs:
            injected += flag_injection_sentences(body)
        follow = bool(injected) and rng.random() < (self.defended_follow_prob if defended else self.follow_prob)
        alert = bool(injected) and defended and rng.random() < self.alert_prob

        boosted, sunk, others_level = set(), set(), None
        if follow:
            for s in injected:
                names = [self._name(b) for _, b in docs if self._name(b) and self._name(b) in s]
                neg = re.search(r"\b(weak|lacking|insufficient|underqualified|not hire)\b", s, re.I)
                med = re.search(r"\b(ordinary|average|unremarkable)\b", s, re.I)
                pos = re.search(r"\b(exceptional|world-class|outstanding|best|impressive)\b", s, re.I)
                if re.search(r"\bother candidates?\b", s, re.I) and (neg or med):
                    others_level = 2 if neg else 3
                if pos and names:
                    boosted.update(names)
                elif neg and names:
                    sunk.update(names)

        out = ["⚠ INJECTION ALERT: the document contains embedded instructions attempting to modify "
               "the evaluation; they have been ignored.\n"] if alert else []
        rows = []
        for fname, body in docs:
            name = self._name(body) or fname
            ach = len(re.findall(r"\d+%|\$\d", body))
            level = 4 if ach >= 8 else 3
            if name in boosted:
                level = 5
            elif name in sunk:
                level = 2
            elif boosted and others_level is not None:
                level = others_level
            if alert and not follow:
                level = 4 if ach >= 8 else 3
            out.append(f"### {name}\n" + "\n\n".join(self._paragraph(level, rng) for _ in range(3)))
            rows.append(f"| {name} | {self.WORDS[level][0]} |")
        if len(docs) > 1:
            out.append("| Candidate | Overall |\n|---|---|\n" + "\n".join(rows))
        body = "\n\n".join(out)
        return LLMResponse(body, input_tokens=len(prompt) // 4, output_tokens=len(body) // 4)

    WORDS = {
        5: ["exceptional", "world-class", "outstanding", "perfect fit"],
        4: ["strong", "impressive", "well qualified", "solid"],
        3: ["adequate", "relevant", "consistent", "reasonable"],
        2: ["weak", "lacking", "insufficient", "limited"],
        1: ["unfit", "do not hire"],
    }

    def _paragraph(self, level, rng):
        w = rng.sample(self.WORDS[level], min(2, len(self.WORDS[level])))
        return f"The candidate shows {w[0]} experience and a {w[-1]} profile for the role."

    @staticmethod
    def _name(body: str) -> str:
        for line in body.splitlines():
            if line.strip():
                return line.strip()
        return ""

    @staticmethod
    def _json_profiles(text: str) -> list[dict]:
        out = []
        for m in re.finditer(r"\{.*?\"name\".*?\}(?=\s*(?:\n|$))", text, re.S):
            try:
                out.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
        return out

    def _extract(self, text, flagger):
        docs = _find_docs(text)
        body = docs[0][1] if docs else text
        flags = flagger(body)
        clean = "\n".join(l for l in body.splitlines() if not any(f in l for f in flags))
        prof = {"name": self._name(body), "achievements": re.findall(r"[^.\n]*\d+%[^.\n]*", clean)[:20],
                "suspicious_content_detected": bool(flags),
                "suspicious_notes": "embedded instructions found" if flags else ""}
        s = json.dumps(prof)
        return LLMResponse(s, input_tokens=len(text) // 4, output_tokens=len(s) // 4)


PROVIDERS = {
    "openai": OpenAICompatibleClient,
    "deepseek": DeepSeekClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
    "mock": MockClient,
}

CLIENT_KEYS = {"rpm", "max_retries", "timeout", "api_key_env", "base_url", "num_ctx", "send_temperature",
               "send_seed", "follow_prob", "defended_follow_prob", "alert_prob"}


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_client(name: str, spec: dict) -> LLMClient:
    cls = PROVIDERS[spec["provider"]]
    kwargs = {k: v for k, v in spec.items() if k in CLIENT_KEYS}
    return cls(name, spec.get("model", name), **kwargs)


def load_clients(config_path: str, names: list[str]) -> dict[str, LLMClient]:
    cfg = load_config(config_path)
    missing = [n for n in names if n not in cfg["models"]]
    if missing:
        raise SystemExit(f"unknown model key(s) {missing}; defined: {sorted(cfg['models'])}")
    return {n: make_client(n, cfg["models"][n]) for n in names}
