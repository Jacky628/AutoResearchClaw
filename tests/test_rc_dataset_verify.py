"""Dataset-source verification (P8). All network is mocked — no real HF calls."""

from __future__ import annotations

import researchclaw.pipeline.dataset_verify as dv
from researchclaw.pipeline.dataset_verify import GATED, MISSING, VERIFIED
from researchclaw.pipeline.environment import parse_manifest


# -- classify / extract -------------------------------------------------------

class TestClassifyExtract:
    def test_classify(self):
        assert dv.classify_source("wanhin/DEEPCAD-completion-sft") == "hf"
        assert dv.classify_source("https://huggingface.co/datasets/x/y") == "hf"
        assert dv.classify_source("hf:foo/bar") == "hf"
        assert dv.classify_source("https://github.com/ChrisWu1997/DeepCAD") == "github"
        assert dv.classify_source("https://example.com/data.zip") == "url"
        assert dv.classify_source("") == "unknown"

    def test_extract_id_from_freetext(self):
        s = "wanhin/DEEPCAD-completion-sft (HuggingFace; load via datasets.load_dataset('wanhin/DEEPCAD-completion-sft'))"
        assert dv.extract_hf_id(s) == "wanhin/DEEPCAD-completion-sft"
        assert dv.extract_hf_id("huggingface.co/datasets/aa/bb") == "aa/bb"
        assert dv.extract_hf_id("hf:cc/dd") == "cc/dd"
        # filesystem path should not be mistaken for an id
        assert dv.extract_hf_id("/workspace/data/deepcad") is None


# -- verify_hf_dataset (mock _http_get_json) ----------------------------------

def _patch_get(monkeypatch, mapping):
    """mapping: url-substring -> (code, obj)."""
    def fake(url, timeout):
        for key, val in mapping.items():
            if key in url:
                return val
        return (None, None)
    monkeypatch.setattr(dv, "_http_get_json", fake)


def test_verify_existing_non_gated_with_parquet(monkeypatch):
    _patch_get(monkeypatch, {"/wanhin/DEEPCAD-completion-sft": (200, {
        "gated": False, "downloads": 30,
        "siblings": [{"rfilename": "data/train-00000.parquet"}, {"rfilename": "README.md"}],
    })})
    info = dv.verify_hf_dataset("wanhin/DEEPCAD-completion-sft")
    assert info.exists and not info.gated and ".parquet" in info.formats and info.downloads == 30


def test_verify_missing_401(monkeypatch):
    _patch_get(monkeypatch, {"/wuchy143/deepcad": (401, None)})
    info = dv.verify_hf_dataset("wuchy143/deepcad")
    assert not info.exists and info.http_status == 401


def test_verify_source_missing_triggers_search(monkeypatch):
    _patch_get(monkeypatch, {"/wuchy143/deepcad": (401, None)})
    monkeypatch.setattr(dv, "search_hf_datasets",
                        lambda q, **k: [{"id": "wanhin/DEEPCAD-completion-sft", "downloads": 30, "gated": False}])
    v = dv.verify_source("wuchy143/deepcad", search_query="DeepCAD")
    assert v.status == MISSING
    assert v.candidates and v.candidates[0]["id"] == "wanhin/DEEPCAD-completion-sft"


def test_verify_source_verified(monkeypatch):
    monkeypatch.setattr(dv, "verify_hf_dataset",
                        lambda i, **k: dv.HFDatasetInfo(exists=True, gated=False,
                                                        formats=(".parquet",), downloads=30))
    v = dv.verify_source("foo/bar")
    assert v.status == VERIFIED and v.downloads == 30


def test_verify_source_gated(monkeypatch):
    monkeypatch.setattr(dv, "verify_hf_dataset",
                        lambda i, **k: dv.HFDatasetInfo(exists=True, gated=True, formats=(".parquet",)))
    assert dv.verify_source("foo/bar").status == GATED


def test_offline_is_unverifiable(monkeypatch):
    v = dv.verify_source("a/b", online=False)
    assert v.status == "UNVERIFIABLE"


def test_non_hf_source_unverifiable():
    v = dv.verify_source("https://github.com/ChrisWu1997/DeepCAD")
    assert v.status == "UNVERIFIABLE" and v.kind == "github"


def test_verify_manifest_datasets(monkeypatch):
    monkeypatch.setattr(dv, "verify_hf_dataset",
                        lambda i, **k: dv.HFDatasetInfo(exists=True, gated=False, formats=(".parquet",), downloads=5))
    m = parse_manifest({"environment": {"datasets": [
        {"name": "DeepCAD", "source": "wanhin/DEEPCAD-completion-sft"}]}})
    out = dv.verify_manifest_datasets(m, online=True)
    assert out["DeepCAD"]["status"] == VERIFIED
