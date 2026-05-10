#!/usr/bin/env python3
"""
Image -> (1) LaC-style dual-prompt hazard inference, (2) DFM2-style drift-cause inference.
----------------------------------------------------------------------------

This script provides two modular pipelines you can run on a single image:

1) LaC (paper-style dual prompting)
   - Hazard Reasoner: structured JSON with description, object list, hazard reasoning, hazardous objects.
   - Emotion Evaluator: anxiety score per hazardous object (paper uses an integer scale 1..3).

2) DFM2 (based on `scripts/vlm_node.py`)
   - Given narration text + image, return top-k "drift causes" with scores (0..1),
     including the special fallback: [{"name":"unclear_cause","score":0.0}]

Outputs are always JSON to stdout.

References:
  - LaC paper in this repo: /home/navin/resilience/LaC.pdf
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage


# ----------------------------
# JSON parsing helpers
# ----------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def _strip_fences(s: str) -> str:
    return _FENCE_RE.sub("", s.strip()).strip()


def _extract_first_json(s: str) -> Any:
    """
    Best-effort JSON extraction from a model response that *should* be JSON-only.
    Handles fenced JSON and extra text by taking the outermost {} or [] span.
    """
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Prefer array, else object.
    candidates: List[Tuple[int, int]] = []
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        try:
            start = s.index(open_ch)
            end = s.rindex(close_ch) + 1
            candidates.append((start, end))
        except ValueError:
            continue

    if not candidates:
        raise json.JSONDecodeError("No JSON object/array found", s, 0)

    start, end = min(candidates, key=lambda t: t[0])
    return json.loads(s[start:end])


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Image encoding
# ----------------------------


def load_bgr_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return img


def encode_image_b64_png(bgr_img: np.ndarray, max_dim: int = 768) -> str:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    pil.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ----------------------------
# VLM provider abstraction
# ----------------------------


class VLMError(RuntimeError):
    pass


class VLMClient:
    def complete_with_image(self, *, image_b64_png: str, system: str, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class ClaudeClient(VLMClient):
    def __init__(self, api_key: str, model: str):
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:
            raise VLMError("anthropic SDK not installed. `pip install anthropic`") from e
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete_with_image(self, *, image_b64_png: str, system: str, prompt: str, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": image_b64_png},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.content[0].text.strip()


class OpenAIClient(VLMClient):
    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise VLMError("openai SDK not installed. `pip install openai`") from e
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete_with_image(self, *, image_b64_png: str, system: str, prompt: str, max_tokens: int) -> str:
        data_url = f"data:image/png;base64,{image_b64_png}"
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


def make_vlm_client(provider: str, model: str) -> VLMClient:
    provider_l = provider.strip().lower()
    if provider_l == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise VLMError("Missing ANTHROPIC_API_KEY in environment")
        return ClaudeClient(api_key=api_key, model=model)
    if provider_l == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VLMError("Missing OPENAI_API_KEY in environment")
        return OpenAIClient(api_key=api_key, model=model)
    raise VLMError(f"Unknown provider: {provider!r} (use 'claude' or 'openai')")


# ----------------------------
# LaC: dual-prompt pipeline
# ----------------------------


LAC_SYSTEM_HAZARD_REASONER = (
    "You are a safety-aware flying drone robot navigation expert acting as a Hazard Reasoner. "
    "Given an image from the robot's viewpoint, you produce a structured hazard analysis in JSON."
)

LAC_SYSTEM_EMOTION_EVAL = (
    "You are a safety-aware flying drone robot navigation expert acting as an Emotion Evaluator. "
    "Given hazard reasoning + hazardous objects and the image, you assign anxiety scores for navigation."
)


LAC_HAZARD_REASONER_PROMPT = """\
Analyze the image from a flying drone robot's viewpoint.

Return ONLY a JSON object with EXACTLY these keys:
- "textual_description": string (a concise but specific description of the scene)
- "object_list": array of strings (objects/areas you notice, including relevant regions like corners/blind spots)
- "hazard_reasoning": array of strings (step-by-step reasoning linking observations to potential hazards)
- "hazardous_objects": array of strings (a refined list of the most important hazards to map)

Rules:
- Output MUST be only valid JSON (no markdown, no backticks).
- "hazardous_objects" should list as many objects/regions as necessary, but ONLY include items you are highly confident pose a genuine navigational hazard. Do not list safe or irrelevant background objects; each entry should be a short noun phrase (e.g., "closed door", "wet floor", "blind corner").
- Include BOTH objects and risky regions when applicable (e.g., corners/blind spots).
"""


def lac_emotion_evaluator_prompt(hazard_reasoning: Sequence[str], hazardous_objects: Sequence[str]) -> str:
    rt = "\n".join(f"- {s}" for s in hazard_reasoning)
    lt = "\n".join(f"- {s}" for s in hazardous_objects)
    return f"""\
You will assign an anxiety score to each hazardous object/region for safe navigation.

Inputs:
HAZARD_REASONING (Rt):
{rt}

HAZARDOUS_OBJECTS (Lt):
{lt}

Scoring:
- anxiety: integer in {{1,2,3}}
  1 = minor risk, 2 = moderate risk, 3 = high risk

Return ONLY a JSON array with one entry per hazard in the same order as Lt.
Each entry must be: {{"object": string, "anxiety": int}}

Rules:
- Output MUST be only valid JSON (no markdown, no backticks).
"""


@dataclass(frozen=True)
class LaCResult:
    object: str
    anxiety: int  # 1..3
    score: float  # 0..1 (normalized)


def run_lac_dual_prompt(
    vlm: VLMClient,
    *,
    image_b64_png: str,
    max_tokens: int = 900,
) -> Tuple[Dict[str, Any], List[LaCResult]]:
    raw1 = vlm.complete_with_image(
        image_b64_png=image_b64_png,
        system=LAC_SYSTEM_HAZARD_REASONER,
        prompt=LAC_HAZARD_REASONER_PROMPT,
        max_tokens=max_tokens,
    )
    j1 = _extract_first_json(raw1)
    if not isinstance(j1, dict):
        raise VLMError("LaC hazard reasoner did not return a JSON object")

    hazard_reasoning = j1.get("hazard_reasoning", [])
    hazardous_objects = j1.get("hazardous_objects", [])
    if not isinstance(hazard_reasoning, list):
        hazard_reasoning = []
    if not isinstance(hazardous_objects, list):
        hazardous_objects = []

    hazardous_objects = [str(x).strip() for x in hazardous_objects if str(x).strip()]
    if not hazardous_objects:
        return j1, []

    raw2 = vlm.complete_with_image(
        image_b64_png=image_b64_png,
        system=LAC_SYSTEM_EMOTION_EVAL,
        prompt=lac_emotion_evaluator_prompt(
            hazard_reasoning=[str(x) for x in hazard_reasoning if str(x).strip()],
            hazardous_objects=hazardous_objects,
        ),
        max_tokens=600,
    )
    j2 = _extract_first_json(raw2)
    if not isinstance(j2, list):
        raise VLMError("LaC emotion evaluator did not return a JSON array")

    out: List[LaCResult] = []
    for i, item in enumerate(j2[: len(hazardous_objects)]):
        if not isinstance(item, dict):
            continue
        obj = str(item.get("object", hazardous_objects[i])).strip()
        try:
            anxiety_i = int(item.get("anxiety", 2))
        except Exception:
            anxiety_i = 2
        anxiety_i = int(_clamp(float(anxiety_i), 1.0, 3.0))
        out.append(LaCResult(object=obj, anxiety=anxiety_i, score=anxiety_i / 3.0))

    out.sort(key=lambda r: r.score, reverse=True)
    return j1, out


# ----------------------------
# DFM2: drift-cause (vlm_node-inspired) pipeline
# ----------------------------


@dataclass(frozen=True)
class DFM2Cause:
    name: str
    score: float  # 0..1


@dataclass(frozen=True)
class NarrationTemplate:
    """
    Easy interface to swap narration text or base prompt.
    """

    base_prompt: str = "I am a drone, after 1s"

    def render(self, narration_text: str) -> str:
        narration_text = narration_text.strip()
        if narration_text:
            return f"{self.base_prompt} {narration_text}"
        return self.base_prompt


def dfm2_prompt(
    *,
    narration_text: str,
    existing_causes: Optional[Sequence[str]] = None,
    template: NarrationTemplate = NarrationTemplate(),
    top_k: int = 4,
) -> str:
    # Keep the core behavior from `scripts/vlm_node.py`, but tighten it:
    # - emphasize "very close / direct interaction" and strict JSON-only output
    # - ask for visually-grounded descriptions (color/shape/spatial)
    existing_causes = list(existing_causes or [])
    history_section = ""
    if existing_causes:
        history_section = (
            "\n\nPREVIOUSLY IDENTIFIED CAUSES:\n"
            + "\n".join([f"- {c}" for c in existing_causes])
            + "\n\nIMPORTANT: If you think the current drift was caused by the SAME object as one of the "
            + "previously identified causes above, you MUST return the EXACT SAME name/description from the list above."
        )

    full_narration = template.render(narration_text)
    return f"""\
{full_narration}

Task:
We are viewing the onboard footage from a flying drone robot. Just after this picture was taken, 2s later the drone deviated from the intended path. 
due to unforseen distubrance caused by object in the scene. Looking at this image, identify ONLY the specific physical object(s) you are highly confident most likely caused the drift (VERY CLOSE in the image).
Focus on objects that could directly or indirectly contact the drone and cause it to deviate from the intended path.
If you are not confident about any object, do not guess; instead use the "unclear_cause" option described below.

Return ONLY the specific object(s) you are highly confident directly caused the drift; this will typically be just 1 or 2 items. For each returned object, provide a confidence score (0.0 to 1.0), sorted descending by score.
Each "name" must be a short, visually grounded description (include color/shape and spatial relation if helpful).

{history_section}

SPECIAL OPTION - UNCLEAR CAUSE:
If the cause is unclear, direct interaction couldn't be found, or the image has insufficient information, return,
please do not return random obejcts from the scene that do not have a clear reasoning behind the interaction:
[{{"name": "unclear_cause", "score": 0.0}}]

RESPONSE FORMAT RULES:
- Output MUST be ONLY a JSON array.
- Only include the object(s) you are highly confident directly caused the drift; if no object clearly caused the drift, use "unclear_cause" as described above.
- Each entry must have: "name" (string) and "score" (float between 0.0 and 1.0).
- If returning "unclear_cause", it must be the ONLY entry with score 0.0.
- No markdown. No backticks. No explanation.
"""


def run_dfm2(
    vlm: VLMClient,
    *,
    image_b64_png: str,
    narration_text: str,
    existing_causes: Optional[Sequence[str]],
    template: NarrationTemplate,
    top_k: int = 4,
    max_tokens: int = 500,
) -> List[DFM2Cause]:
    raw = vlm.complete_with_image(
        image_b64_png=image_b64_png,
        system=(
            "You are a drone flight dynamics expert. "
            "Analyze the image and identify objects that could cause drift from intended paths. "
            "Always respond with valid JSON only."
        ),
        prompt=dfm2_prompt(
            narration_text=narration_text,
            existing_causes=existing_causes,
            template=template,
            top_k=top_k,
        ),
        max_tokens=max_tokens,
    )
    j = _extract_first_json(raw)
    if not isinstance(j, list):
        raise VLMError("DFM2 did not return a JSON array")

    out: List[DFM2Cause] = []
    for item in j[:top_k]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            score = 0.0
        score = float(_clamp(score, 0.0, 1.0))
        out.append(DFM2Cause(name=name, score=score))

    # Enforce unclear_cause rule
    if out and out[0].name == "unclear_cause":
        return [DFM2Cause(name="unclear_cause", score=0.0)]

    out.sort(key=lambda c: c.score, reverse=True)
    return out


# ----------------------------
# CLI / main
# ----------------------------


def _load_existing_causes_json(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    raise ValueError("existing causes JSON must be a list of strings")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run LaC dual-prompt + DFM2 drift-cause prompting on an image.")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--provider", default="openai", choices=["openai", "claude"], help="VLM provider")
    p.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name (provider-specific). For Claude, use env CLAUDE_MODEL or pass explicitly.",
    )
    p.add_argument("--run", default="both", choices=["lac", "dfm2", "both"], help="Which pipeline(s) to run")

    # DFM2 / narration knobs
    p.add_argument("--narration", default="", help="Narration text for DFM2 prompt (can be empty)")
    p.add_argument("--base-prompt", default="I am a drone, after 1s", help="Base prompt prefix for narration template")
    p.add_argument("--existing-causes-json", default=None, help="Path to JSON list of previously identified causes")
    p.add_argument("--top-k", type=int, default=4, help="Top-k objects for DFM2")

    args = p.parse_args(list(argv) if argv is not None else None)

    bgr = load_bgr_image(args.image)
    image_b64 = encode_image_b64_png(bgr)

    # Pick default Claude model from env if provider is claude and user didn't override.
    model = args.model
    if args.provider == "claude" and (not model or model == "gpt-4o-mini"):
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

    vlm = make_vlm_client(args.provider, model)

    result: Dict[str, Any] = {
        "image": os.path.abspath(args.image),
        "provider": args.provider,
        "model": model,
        "pipelines": {},
    }

    if args.run in ("lac", "both"):
        j1, lac_out = run_lac_dual_prompt(vlm, image_b64_png=image_b64)
        result["pipelines"]["lac"] = {
            "hazard_reasoner": j1,
            "objects": [{"object": r.object, "anxiety": r.anxiety, "score": r.score} for r in lac_out],
        }

    if args.run in ("dfm2", "both"):
        existing = _load_existing_causes_json(args.existing_causes_json)
        template = NarrationTemplate(base_prompt=args.base_prompt)
        causes = run_dfm2(
            vlm,
            image_b64_png=image_b64,
            narration_text=args.narration,
            existing_causes=existing,
            template=template,
            top_k=max(1, min(10, int(args.top_k))),
        )
        result["pipelines"]["dfm2"] = {
            "narration": template.render(args.narration),
            "objects": [{"name": c.name, "score": c.score} for c in causes],
        }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

