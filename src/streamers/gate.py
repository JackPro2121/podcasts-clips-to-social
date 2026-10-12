"""V2 Streamer Publish Gate.

The single authority that decides whether a rendered clip may be published.

Why this module exists
----------------------
Pipeline V1 (main.py) has an AST-pinned publish gate wiring verdict.verify →
repair.render_with_repair → BufferPublisher. V2 was built alongside it and
inherited none of that: pipeline.py:318-325 reaches Buffer with nothing between
render_viral_clip and the dispatch call.

This module closes that gap. It composes the existing public verification
primitives (src.verification.verdict.verify) behind a single V2-scoped function
that pipeline.py calls instead of invoking Buffer directly.

V1 contract
-----------
This module does NOT touch or import anything from:
    main.py, src/face_tracker.py, src/viral_detector.py
It composes only from src/verification/ (public API) and src/streamers/spec.py.

Usage
-----
    from src.streamers.gate import verify_clip, GateResult

    result = verify_clip(rendered_path, candidate=vm, run_id=run_id)
    if result.passed:
        publisher.schedule_clip(...)
    else:
        log_blocked(result)
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.streamers.spec import (
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_MIN_CLIP_DURATION_S,
    STREAMER_OUTPUT_HEIGHT,
    STREAMER_OUTPUT_WIDTH,
)

# Lazy import: do not fail at module load if ffmpeg is absent in test env
def _import_verdict():
    from src.verification import verdict as _v
    return _v


@dataclass
class GateResult:
    """Outcome of the V2 publish gate for one rendered clip."""

    path: str
    passed: bool = False
    blocking_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verdict_path: Optional[Path] = None
    duration_s: float = 0.0
    dimensions: Dict[str, int] = field(default_factory=dict)
    repair_attempted: bool = False
    error: Optional[str] = None

    def summary_line(self) -> str:
        if self.passed:
            return (
                f"[GATE ✓] {Path(self.path).name} passed "
                f"({self.duration_s:.1f}s, "
                f"{self.dimensions.get('width', '?')}x{self.dimensions.get('height', '?')})"
            )
        codes = ", ".join(self.blocking_codes) if self.blocking_codes else "unknown"
        return (
            f"[GATE ✗] {Path(self.path).name} BLOCKED — {codes}"
        )


def verify_clip(
    path: Path,
    *,
    run_id: str = "",
    runs_dir: Optional[Path] = None,
    min_duration_s: float = STREAMER_MIN_CLIP_DURATION_S,
    max_duration_s: float = STREAMER_MAX_CLIP_DURATION_S,
) -> GateResult:
    """Run every quality check against a rendered V2 clip.

    Parameters
    ----------
    path:
        Absolute path to the rendered MP4 clip.
    run_id:
        Identifier for the current pipeline run (used in verdict.json path).
    runs_dir:
        Override directory for verdict.json output. Defaults to
        ``data/runs/<run_id>/``.
    min_duration_s / max_duration_s:
        Duration contract for this clip (defaults to V2 spec: 30–45 s).

    Returns
    -------
    GateResult:
        .passed is True iff the clip meets every blocking criterion.
        .verdict_path points to the emitted verdict.json (always written).
    """
    result = GateResult(path=str(path))

    if not path.exists():
        result.error = f"Rendered clip not found: {path}"
        result.blocking_codes.append("output_missing")
        print(f"  [gate] ✗ {result.error}")
        return result

    # ------------------------------------------------------------------
    # 1. Run the full pixel-level verifier
    # ------------------------------------------------------------------
    verdict_mod = None
    raw_verdict = None
    try:
        verdict_mod = _import_verdict()
        raw_verdict = verdict_mod.verify(
            path,
            expected_width=STREAMER_OUTPUT_WIDTH,
            expected_height=STREAMER_OUTPUT_HEIGHT,
            with_mirror=False,   # Streamer clips: mirror check off (fast cam flips)
            with_captions=True,
            with_sway=True,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        result.duration_s = raw_verdict.duration_s
        result.dimensions = raw_verdict.dimensions or {}
    except Exception as exc:  # pragma: no cover
        result.error = f"verdict.verify raised: {exc}"
        result.blocking_codes.append("verifier_error")
        print(f"  [gate] ✗ verifier exception — {exc}")
        traceback.print_exc()
        return result

    # ------------------------------------------------------------------
    # 2. Enforce V2-specific duration contract
    #    (verdict.py's own CONTRACT_MAX is 55s; we need 45s for V2)
    # ------------------------------------------------------------------
    dur = raw_verdict.duration_s
    duration_ok = (min_duration_s - 0.15) <= dur <= (max_duration_s + 0.15)
    if not duration_ok:
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_duration_out_of_contract",
                severity="error",
                message=(
                    f"V2 duration {dur:.2f}s outside [{min_duration_s:.0f}s, "
                    f"{max_duration_s:.0f}s] contract"
                ),
                spec_clause="streamers/spec.py STREAMER_MAX_CLIP_DURATION_S",
                metrics={"duration_s": round(dur, 3)},
            )
        )

    # ------------------------------------------------------------------
    # 3. Collect findings
    # ------------------------------------------------------------------
    blocking = [f.code for f in raw_verdict.findings if f.blocks]
    warnings = [f.code for f in raw_verdict.findings if f.severity == "warning"]

    result.blocking_codes = blocking
    result.warnings = warnings
    result.passed = len(blocking) == 0

    # ------------------------------------------------------------------
    # 4. Write verdict.json
    # ------------------------------------------------------------------
    try:
        if runs_dir is None:
            runs_dir = Path("data") / "runs" / (run_id or f"run_{int(time.time())}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        verdict_file = runs_dir / f"{path.stem}_verdict.json"

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "pipeline": "v2_streamer",
            "clip": path.name,
            "passed": result.passed,
            "duration_s": round(dur, 3),
            "dimensions": result.dimensions,
            "blocking_codes": blocking,
            "warnings": warnings,
            "findings": [f.as_dict() for f in raw_verdict.findings],
            "metrics": raw_verdict.metrics,
        }
        verdict_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result.verdict_path = verdict_file
    except Exception as exc:  # pragma: no cover
        print(f"  [gate] ⚠ Could not write verdict.json: {exc}")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print(f"  {result.summary_line()}")
    if warnings:
        print(f"  [gate] ⚠ Warnings: {', '.join(warnings)}")
    if blocking:
        print(f"  [gate] ✗ Blocking codes: {', '.join(blocking)}")

    return result
