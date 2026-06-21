"""
Unified zero-shot VLM "judge" client for full-reference video-frame quality
assessment across three providers: Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google).

Each judge receives a (reference CROP, distorted CROP) pair and returns structured JSON:
    {"score": 0-100, "artifact": <label>, "justification": <str>}

IMPORTANT (reviewer-proofing):
 * We send NATIVE-RESOLUTION CROPS (e.g. 512x512), NOT downscaled full frames, so the
   provider-side image downsampling does not destroy the fine texture-blur we measure.
 * The PRIMARY prompt (PROMPTS["frozen"]) is FROZEN before any correlation is computed
   (no tuning on the eval set). Alternates exist ONLY for the prompt-sensitivity ablation.
 * Each Verdict carries the exact model version string returned by the API (logged for
   reproducibility — API models drift over time).

Env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

NOTE (Anthropic): claude-opus-4-8 has high-res vision + structured outputs and removed
`temperature` (sending it 400s) — so we do NOT pass temperature for Claude. For
OpenAI/Gemini we set temperature=0.
"""

import base64
import json
import mimetypes
import os
from dataclasses import dataclass

ARTIFACT_LABELS = ["blocking", "texture-blur", "color-shift", "ringing", "none"]

SYSTEM_PROMPT = (
    "You are an expert video-quality assessor. You will be shown a REFERENCE crop and a "
    "RECONSTRUCTED (compressed) crop taken from the SAME location of a video frame. Judge "
    "the perceptual quality of the reconstruction RELATIVE to the reference, as a human "
    "viewer would. Neural codecs tend to blur fine texture without blocking; traditional "
    "codecs tend to produce blocking and ringing. Identify the DOMINANT artifact."
)

# --- frozen primary instruction (do NOT edit after results are computed) ---
_INSTRUCTION = (
    "Compare the two crops (first = reference, second = reconstructed). Return JSON only:\n"
    "  score: integer 0-100 (100 = perceptually identical to reference)\n"
    f"  artifact: one of {ARTIFACT_LABELS} (dominant degradation; 'none' if negligible)\n"
    "  justification: one short sentence naming what degraded and where.\n"
)

# Prompt variants. "frozen" is the primary; others are for the prompt-sensitivity ablation.
# "anchored" operationalizes Q-Align's discrete-level insight via the ITU-R BT.500 DSIS
# 5-grade impairment scale (the scale subjective MOS is collected on), mapped onto 0-100.
PROMPTS = {
    "frozen": _INSTRUCTION,
    "terse": ("Rate reconstruction vs reference. JSON only: score 0-100, "
              f"artifact in {ARTIFACT_LABELS}, justification (one sentence)."),
    "detailed": (_INSTRUCTION + "Focus on fine texture, edges, and color. Score 100 only "
                 "if you cannot tell the crops apart at normal viewing distance."),
    "anchored": (
        "Compare the two crops (first = reference, second = reconstructed). First note the "
        "visible differences, then rate the IMPAIRMENT of the reconstruction relative to the "
        "reference on this 5-grade scale (ITU-R BT.500): "
        "100 = imperceptible; 80 = perceptible but not annoying; 60 = slightly annoying; "
        "40 = annoying; 20 = very annoying (interpolate as needed). Return JSON only:\n"
        "  score: integer 0-100 on that impairment scale\n"
        f"  artifact: one of {ARTIFACT_LABELS} (dominant degradation; 'none' if negligible)\n"
        "  justification: one short sentence naming what degraded and where.\n"
    ),
    # Artifact-focused pass (for two-pass mode). Deliberately NOT anchored to impairment
    # magnitude — that anchoring collapsed every label to 'texture-blur'. Gives explicit
    # discriminative definitions and forbids defaulting to blur.
    "describe": (
        "Compare the two crops (first = reference, second = reconstructed). In one sentence, "
        "describe the visible difference. Then classify the SINGLE most salient compression "
        "artifact using these definitions:\n"
        "  blocking = visible blocky grid / tiling, along edges or in flat regions;\n"
        "  ringing = halos, echoes or oscillations next to sharp edges;\n"
        "  color-shift = a change in hue or saturation;\n"
        "  texture-blur = loss of fine texture or high-frequency detail;\n"
        "  none = no perceptible degradation.\n"
        "Choose the artifact that is MOST visually salient. Do NOT default to texture-blur "
        "if a structured artifact (blocking / ringing / color-shift) is present. Return JSON "
        "only:\n"
        "  score: integer 0-100 (100 = identical) -- rough, will be ignored\n"
        f"  artifact: one of {ARTIFACT_LABELS}\n"
        "  justification: one sentence naming the artifact and where it appears.\n"
    ),
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "artifact": {"type": "string", "enum": ARTIFACT_LABELS},
        "justification": {"type": "string"},
    },
    "required": ["score", "artifact", "justification"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    score: float
    artifact: str
    justification: str
    model_version: str
    raw: str


def _b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def _media_type(path):
    return mimetypes.guess_type(path)[0] or "image/png"

def _parse(text, model_version):
    if not text:
        raise ValueError("empty response content from API")
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        s = s[4:].strip() if s.lstrip().startswith("json") else s.strip()
    start, end = s.find("{"), s.rfind("}")
    obj = json.loads(s[start:end + 1])
    return Verdict(score=float(obj["score"]), artifact=str(obj["artifact"]),
                   justification=str(obj.get("justification", "")),
                   model_version=model_version, raw=text)


# ---- Claude (Anthropic) ----
_anthropic_client = None
CLAUDE_MODEL = os.environ.get("PAPER51_CLAUDE_MODEL", "claude-opus-4-8")

def judge_claude(ref_path, dist_path, prompt="frozen"):
    global _anthropic_client
    import anthropic
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    content = [
        {"type": "text", "text": "REFERENCE crop:"},
        {"type": "image", "source": {"type": "base64", "media_type": _media_type(ref_path), "data": _b64(ref_path)}},
        {"type": "text", "text": "RECONSTRUCTED crop:"},
        {"type": "image", "source": {"type": "base64", "media_type": _media_type(dist_path), "data": _b64(dist_path)}},
        {"type": "text", "text": PROMPTS[prompt]},
    ]
    resp = _anthropic_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return _parse(text, getattr(resp, "model", CLAUDE_MODEL))


# ---- GPT (OpenAI) ----
_openai_client = None
GPT_MODEL = os.environ.get("PAPER51_GPT_MODEL", "gpt-4o")

def judge_gpt(ref_path, dist_path, prompt="frozen"):
    global _openai_client
    from openai import OpenAI
    if _openai_client is None:
        _openai_client = OpenAI()
    def uri(p): return f"data:{_media_type(p)};base64,{_b64(p)}"
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "REFERENCE crop:"},
            {"type": "image_url", "image_url": {"url": uri(ref_path), "detail": "high"}},
            {"type": "text", "text": "RECONSTRUCTED crop:"},
            {"type": "image_url", "image_url": {"url": uri(dist_path), "detail": "high"}},
            {"type": "text", "text": PROMPTS[prompt]},
        ]},
    ]
    kw = dict(model=GPT_MODEL, response_format={"type": "json_object"}, messages=msgs)
    legacy = GPT_MODEL.startswith(("gpt-4o", "gpt-4.1", "chatgpt-4o"))
    if legacy:
        kw["temperature"] = 0; kw["max_tokens"] = 1024
    else:  # gpt-5.x / o-series: no custom temperature, use max_completion_tokens
        kw["max_completion_tokens"] = 4096
    resp = _openai_client.chat.completions.create(**kw)
    return _parse(resp.choices[0].message.content, getattr(resp, "model", GPT_MODEL))


# ---- Gemini (Google) ----
_gemini_client = None
GEMINI_MODEL = os.environ.get("PAPER51_GEMINI_MODEL", "gemini-flash-latest")

def judge_gemini(ref_path, dist_path, prompt="frozen"):
    # Direct REST (matches AI Studio quickstart); robust to the new "AQ." key
    # format and avoids SDK auth/routing quirks. Uses X-goog-api-key header.
    import os, urllib.request
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [
            {"text": SYSTEM_PROMPT + "\n\nREFERENCE crop:"},
            {"inline_data": {"mime_type": _media_type(ref_path), "data": _b64(ref_path)}},
            {"text": "RECONSTRUCTED crop:"},
            {"inline_data": {"mime_type": _media_type(dist_path), "data": _b64(dist_path)}},
            {"text": PROMPTS[prompt]},
        ]}],
        "generationConfig": {"temperature": 0, "response_mime_type": "application/json"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "X-goog-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode("utf-8"))
    text = out["candidates"][0]["content"]["parts"][0]["text"]
    return _parse(text, out.get("modelVersion", GEMINI_MODEL))


JUDGES = {"claude": judge_claude, "gpt": judge_gpt, "gemini": judge_gemini}


def judge_twopass(fn, ref_path, dist_path):
    """Two API calls per pair: 'describe' yields the artifact label + justification
    (free of the impairment anchoring that collapsed every label to texture-blur),
    'anchored' yields the ITU-BT.500 impairment score. Score from the anchored call,
    artifact/justification from the describe call."""
    desc = fn(ref_path, dist_path, prompt="describe")
    sc = fn(ref_path, dist_path, prompt="anchored")
    return Verdict(score=sc.score, artifact=desc.artifact,
                   justification=desc.justification, model_version=sc.model_version,
                   raw=json.dumps({"anchored": sc.raw, "describe": desc.raw}))
