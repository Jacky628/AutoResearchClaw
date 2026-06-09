"""Dataset-source existence verification (P8).

Stage 9's design LLM cannot know which HuggingFace dataset ids actually exist,
so it sometimes invents non-existent ones (e.g. `wuchy143/deepcad`). P7 forbids
inventing ids but cannot supply real ones. This module closes the loop: it
verifies a declared dataset source against the HuggingFace API (metadata only —
no download), and when an id is MISSING it can search HF for real candidates so
the design can re-pick a genuine source.

All network calls are fail-soft (short timeout, never raise) — if HF is
unreachable a source is reported UNVERIFIABLE, which never blocks the pipeline.
Uses urllib (stdlib) — no extra dependency.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Verdict statuses
VERIFIED = "VERIFIED"          # exists, not gated, has loadable data files
GATED = "GATED"               # exists but needs auth → cannot auto-download
MISSING = "MISSING"           # 404/401 — does not exist (or private)
UNVERIFIABLE = "UNVERIFIABLE"  # network failure / non-HF source — can't tell

_LOADABLE_EXT = (".parquet", ".jsonl", ".json", ".csv", ".arrow", ".txt", ".tsv")
# A bare "user/name" HF id token (used to extract from free-text source strings).
_HF_ID_RE = re.compile(r"[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]+")
_HF_API = "https://huggingface.co/api/datasets"
_UA = {"User-Agent": "researchclaw-dataset-verify/1.0"}


# ---------------------------------------------------------------------------
# source classification + id extraction
# ---------------------------------------------------------------------------

def classify_source(src: str) -> str:
    """Return 'hf' | 'url' | 'github' | 'unknown' for a declared source string."""
    s = (src or "").strip().lower()
    if not s:
        return "unknown"
    if "huggingface.co/datasets" in s or s.startswith("hf:") or "load_dataset(" in s:
        return "hf"
    if "github.com" in s:
        return "github"
    if s.startswith(("http://", "https://")):
        return "url"
    # bare "user/name" with no scheme and no path-y leading slash → treat as HF id
    if _HF_ID_RE.fullmatch(src.strip()) or (("/" in src) and not src.strip().startswith("/")
                                            and " " not in src.strip()):
        return "hf"
    # free text that mentions a user/name token → still try HF
    if _HF_ID_RE.search(src or ""):
        return "hf"
    return "unknown"


def extract_hf_id(src: str) -> str | None:
    """Best-effort extraction of a `user/name` HF dataset id from *src*."""
    if not src:
        return None
    # huggingface.co/datasets/<id>
    m = re.search(r"huggingface\.co/datasets/([A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]+)", src)
    if m:
        return m.group(1)
    # hf:<id>
    m = re.search(r"hf:([A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]+)", src)
    if m:
        return m.group(1)
    # load_dataset('<id>')
    m = re.search(r"load_dataset\(\s*['\"]([^'\"]+)['\"]", src)
    if m and "/" in m.group(1):
        return m.group(1)
    # first bare user/name token, skip obvious filesystem paths
    for cand in _HF_ID_RE.findall(src):
        if cand.lower().split("/")[0] in ("workspace", "data", "tmp", "opt", "root", "home"):
            continue
        return cand
    return None


# ---------------------------------------------------------------------------
# HF API (fail-soft)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: float):
    """GET *url* → (status_code, parsed_json). Never raises.

    Returns (code, obj). On HTTP error code is the status (e.g. 401/404) and
    obj is None. On network/parse failure returns (None, None).
    """
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as exc:  # noqa: BLE001 — network/timeout/parse → unverifiable
        logger.debug("HF GET failed (%s): %s", url, exc)
        return None, None


@dataclass
class HFDatasetInfo:
    exists: bool = False
    gated: bool = False
    formats: tuple[str, ...] = ()
    downloads: int = 0
    http_status: int | None = None
    error: str = ""


def verify_hf_dataset(hf_id: str, *, timeout: float = 8.0) -> HFDatasetInfo:
    """Query the HF datasets API for *hf_id* (metadata only)."""
    code, obj = _http_get_json(f"{_HF_API}/{urllib.parse.quote(hf_id)}", timeout)
    if obj is None:
        # 401/404 → does not exist (or private); None code → network error
        return HFDatasetInfo(exists=False, http_status=code,
                             error="not found" if code in (401, 404) else "unreachable")
    gated_raw = obj.get("gated", False)
    gated = bool(gated_raw) and gated_raw is not False  # HF: false | "auto" | "manual"
    exts = {
        "." + s["rfilename"].rsplit(".", 1)[-1].lower()
        for s in obj.get("siblings", []) or []
        if isinstance(s, dict) and "." in s.get("rfilename", "")
    }
    formats = tuple(sorted(e for e in exts if e in _LOADABLE_EXT))
    return HFDatasetInfo(
        exists=True, gated=gated, formats=formats,
        downloads=int(obj.get("downloads") or 0), http_status=code or 200,
    )


def search_hf_datasets(query: str, *, limit: int = 8, timeout: float = 8.0) -> list[dict]:
    """Search HF for real datasets matching *query*. Returns [] on failure."""
    q = urllib.parse.urlencode({"search": query, "limit": limit})
    code, obj = _http_get_json(f"{_HF_API}?{q}", timeout)
    if not isinstance(obj, list):
        return []
    out = []
    for x in obj:
        if isinstance(x, dict) and x.get("id"):
            g = x.get("gated", False)
            out.append({"id": x["id"], "downloads": int(x.get("downloads") or 0),
                        "gated": bool(g) and g is not False})
    return out


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

@dataclass
class SourceVerdict:
    source: str
    kind: str = "unknown"
    hf_id: str | None = None
    status: str = UNVERIFIABLE
    detail: str = ""
    downloads: int = 0
    formats: tuple[str, ...] = ()
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "kind": self.kind, "hf_id": self.hf_id,
            "status": self.status, "detail": self.detail,
            "downloads": self.downloads, "formats": list(self.formats),
            "candidates": self.candidates,
        }


def verify_source(
    src: str, *, online: bool = True, timeout: float = 8.0,
    search_query: str | None = None, search_on_missing: bool = True,
) -> SourceVerdict:
    """Verify one declared dataset source. Non-HF / offline → UNVERIFIABLE."""
    kind = classify_source(src)
    v = SourceVerdict(source=src, kind=kind)
    if not online:
        v.detail = "offline (verification skipped)"
        return v
    if kind != "hf":
        v.detail = f"non-HF source ({kind}) — reachability not verified"
        return v
    hf_id = extract_hf_id(src)
    v.hf_id = hf_id
    if not hf_id:
        v.status = UNVERIFIABLE
        v.detail = "could not extract a HuggingFace id"
        return v
    info = verify_hf_dataset(hf_id, timeout=timeout)
    if not info.exists:
        v.status = MISSING if info.http_status in (401, 404) else UNVERIFIABLE
        v.detail = info.error or "not found"
        if v.status == MISSING and search_on_missing:
            v.candidates = search_hf_datasets(search_query or hf_id.split("/")[-1],
                                              timeout=timeout)
        return v
    v.downloads, v.formats = info.downloads, info.formats
    if info.gated:
        v.status, v.detail = GATED, "exists but gated (auth required for download)"
    elif not info.formats:
        v.status, v.detail = UNVERIFIABLE, "exists but no loadable data files found"
    else:
        v.status = VERIFIED
        v.detail = f"exists, downloads={info.downloads}, formats={','.join(info.formats)}"
    return v


def verify_manifest_datasets(manifest, *, online: bool = True) -> dict:
    """Verify every dataset declared in an EnvironmentManifest.

    Returns {dataset_name: SourceVerdict.to_dict()}.
    """
    out: dict[str, dict] = {}
    for ds in getattr(manifest, "datasets", ()) or ():
        name = getattr(ds, "name", "") or ""
        source = getattr(ds, "source", "") or ""
        verdict = verify_source(source or name, online=online, search_query=name or None)
        out[name or source] = verdict.to_dict()
    return out
