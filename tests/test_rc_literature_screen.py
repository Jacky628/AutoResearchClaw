"""Stage 5 literature screen robustness.

Regression tests for the json-truncation + input-starvation bug that made the
LLM screen silently fall back to a keyword-only top-N (dropping high-citation
foundational papers like DeepCAD). The fix screens via compact id-referenced
I/O and dual-criteria (keyword + citation) candidate recall.
"""

from __future__ import annotations

import json

from researchclaw.pipeline.stage_impls._literature import (
    _compact_for_screen,
    _join_screen_shortlist,
    _prep_extract_shortlist,
    _select_screen_candidates,
)


def _row(pid: str, title: str, kw: int, cite: int, **extra):
    d = {
        "paper_id": pid,
        "title": title,
        "keyword_overlap": kw,
        "citation_count": cite,
        "doi": f"10.x/{pid}",
        "cite_key": pid,
        "abstract": "x" * 50,
    }
    d.update(extra)
    return d


class TestScreenCandidateSelection:
    def test_high_citation_low_keyword_paper_is_recalled(self) -> None:
        # Many recent high-keyword low-citation papers crowd out a foundational
        # paper under pure keyword ranking; the citation quota must rescue it.
        rows = [_row(f"r{i}", f"recent {i}", kw=8, cite=1) for i in range(80)]
        rows.append(_row("deepcad", "DeepCAD foundational", kw=3, cite=310))
        sel = _select_screen_candidates(rows, kw_quota=80, cite_quota=40)
        keys = {r["paper_id"] for r in sel}
        assert "deepcad" in keys

    def test_pure_keyword_topN_would_drop_foundational(self) -> None:
        # Sanity: confirm the OLD behaviour (keyword top-N) excludes it, so the
        # test above is meaningful.
        rows = [_row(f"r{i}", f"recent {i}", kw=8, cite=1) for i in range(80)]
        rows.append(_row("deepcad", "DeepCAD foundational", kw=3, cite=310))
        by_kw = sorted(rows, key=lambda r: -r["keyword_overlap"])[:15]
        assert "deepcad" not in {r["paper_id"] for r in by_kw}


class TestCompactView:
    def test_compact_view_drops_abstract_and_keeps_stable_id(self) -> None:
        rows = [_row("p1", "T1", kw=5, cite=10, abstract="A" * 2000)]
        text, mapping = _compact_for_screen(rows)
        assert "p1" in mapping
        obj = json.loads(text.splitlines()[0])
        assert obj["id"] == "p1"
        assert "abstract" not in obj  # not echoed to the LLM -> no input bloat
        assert obj["title"] == "T1"

    def test_compact_view_assigns_id_when_paper_id_missing(self) -> None:
        rows = [{"title": "no id paper", "keyword_overlap": 2, "citation_count": 0}]
        text, mapping = _compact_for_screen(rows)
        obj = json.loads(text.splitlines()[0])
        assert obj["id"] in mapping
        assert obj["title"] == "no id paper"


class TestScreenJoin:
    def test_join_reconstructs_full_records(self) -> None:
        rows = [_row("p1", "T1", kw=5, cite=10), _row("p2", "T2", kw=4, cite=5)]
        _, mapping = _compact_for_screen(rows)
        llm = [
            {"id": "p1", "relevance_score": 0.9, "quality_score": 0.8,
             "keep_reason": "core"},
        ]
        out = _join_screen_shortlist(llm, mapping)
        assert len(out) == 1
        assert out[0]["doi"] == "10.x/p1"          # original field preserved
        assert out[0]["relevance_score"] == 0.9
        assert out[0]["keep_reason"] == "core"

    def test_join_skips_unknown_ids(self) -> None:
        rows = [_row("p1", "T1", kw=5, cite=10)]
        _, mapping = _compact_for_screen(rows)
        out = _join_screen_shortlist(
            [{"id": "ghost", "relevance_score": 1.0}], mapping
        )
        assert out == []

    def test_join_falls_back_to_title_match(self) -> None:
        # Some models echo the title instead of the id; still recover the row.
        rows = [_row("p1", "Unique Title Here", kw=5, cite=10)]
        _, mapping = _compact_for_screen(rows)
        out = _join_screen_shortlist(
            [{"title": "Unique Title Here", "relevance_score": 0.7}], mapping
        )
        assert len(out) == 1
        assert out[0]["paper_id"] == "p1"


class TestKnowledgeExtractPrep:
    """Stage 6 knowledge-card extraction must bound its LLM workload so the
    cards JSON doesn't truncate (which silently produced template cards)."""

    def _sl(self, n, abstract_len=2000):
        return [
            {
                "cite_key": f"k{i}",
                "title": f"Paper {i}",
                "year": 2020 + (i % 5),
                "venue": "V",
                "abstract": "A" * abstract_len,
                "relevance_score": round(0.5 + i / 100, 3),
            }
            for i in range(n)
        ]

    def test_selects_top_n_by_relevance(self) -> None:
        rows = self._sl(40)
        text, selected = _prep_extract_shortlist(rows, max_papers=15)
        assert len(selected) == 15
        # highest-relevance paper (k39) kept, lowest (k0) dropped
        keys = {r["cite_key"] for r in selected}
        assert "k39" in keys
        assert "k0" not in keys

    def test_truncates_abstracts_and_preserves_cite_key(self) -> None:
        rows = self._sl(3, abstract_len=5000)
        text, _ = _prep_extract_shortlist(rows, max_papers=15, abstract_chars=400)
        first = json.loads(text.splitlines()[0])
        assert len(first["abstract"]) <= 410  # truncated (+ ellipsis)
        assert first["cite_key"].startswith("k")

    def test_output_is_bounded_for_large_shortlist(self) -> None:
        # 25 papers with long abstracts must not blow up the LLM input.
        rows = self._sl(25, abstract_len=3000)
        text, selected = _prep_extract_shortlist(rows)
        assert len(selected) <= 15
        assert len(text) < 12_000  # compact enough to leave output headroom


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def chat(self, messages, system=None, json_mode=False, max_tokens=None,
             strip_thinking=True, **_kw):
        return _FakeResp(self._payload)


class TestKnowledgeExtractIdempotent:
    """Re-running Stage 6 (e.g. via --from-stage) must not leave stale cards
    from a prior template-fallback run mixed in with fresh ones."""

    def test_rerun_clears_stale_cards(self, tmp_path) -> None:
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig
        from researchclaw.pipeline.stage_impls._literature import (
            _execute_knowledge_extract,
        )

        run_dir = tmp_path
        (run_dir / "stage-05").mkdir(parents=True)
        (run_dir / "stage-05" / "shortlist.jsonl").write_text(
            json.dumps({
                "cite_key": "a2021", "title": "Paper A",
                "abstract": "x", "relevance_score": 0.9,
            }) + "\n",
            encoding="utf-8",
        )
        stage_dir = run_dir / "stage-06"
        (stage_dir / "cards").mkdir(parents=True)
        # stale template card left by an earlier failed run
        (stage_dir / "cards" / "card-1.md").write_text(
            "# stale\n## Method\nTemplate method summary\n", encoding="utf-8"
        )

        cfg = RCConfig.load(
            "config.researchclaw.example.yaml", check_paths=False
        )
        fake = _FakeLLM(json.dumps({"cards": [{
            "card_id": "a2021", "title": "Paper A", "cite_key": "a2021",
            "problem": "p", "method": "a real method", "data": "d",
            "metrics": "m", "findings": "f", "limitations": "l",
            "citation": "u",
        }]}))

        _execute_knowledge_extract(
            stage_dir, run_dir, cfg, AdapterBundle(), llm=fake
        )

        files = {p.name for p in (stage_dir / "cards").glob("*.md")}
        assert "card-1.md" not in files          # stale removed
        assert "a2021.md" in files               # fresh real card written
        body = (stage_dir / "cards" / "a2021.md").read_text(encoding="utf-8")
        assert "Template method summary" not in body
        assert "a real method" in body
