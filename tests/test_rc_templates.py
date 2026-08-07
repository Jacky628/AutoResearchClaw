"""Unit tests for researchclaw.templates — conference templates + MD→LaTeX converter."""

from __future__ import annotations

import re
import threading

import pytest

from researchclaw.templates.conference import (
    CONFERENCE_REGISTRY,
    ConferenceTemplate,
    get_template,
    list_conferences,
    NEURIPS_2024,
    NEURIPS_2025,
    ICLR_2025,
    ICLR_2026,
    ICML_2025,
    ICML_2026,
)
from researchclaw.templates.converter import (
    markdown_to_latex,
    _parse_sections,
    _extract_title,
    _extract_abstract,
    _convert_inline,
    _escape_latex,
    _escape_algo_line,
    _render_code_block,
    _build_body,
    _render_table,
    _parse_table_row,
    _parse_alignments,
    _render_itemize,
    _render_enumerate,
    _reset_render_counters,
    _next_table_num,
    _next_figure_num,
    _resolve_markdown_footnotes,
    _FN_OPEN,
    _FN_CLOSE,
    _visual_len,
    _cell_breakable,
    _longest_unbreakable,
    TABLE_FONT,
    check_paper_completeness,  # noqa: F401
)


# =====================================================================
# conference.py tests
# =====================================================================


class TestConferenceTemplate:
    """Tests for ConferenceTemplate dataclass."""

    def test_neurips_basic_fields(self) -> None:
        t = NEURIPS_2024
        assert t.name == "neurips_2024"
        assert t.display_name == "NeurIPS 2024"
        assert t.year == 2024
        assert t.document_class == "article"
        assert t.style_package == "neurips_2024"
        assert t.columns == 1
        assert t.author_format == "neurips"
        assert t.bib_style == "plainnat"

    def test_iclr_basic_fields(self) -> None:
        t = ICLR_2025
        assert t.name == "iclr_2025"
        assert t.year == 2025
        assert t.style_package == "iclr2025_conference"
        assert t.bib_style == "iclr2025_conference"
        assert t.columns == 1
        assert t.author_format == "iclr"

    def test_icml_basic_fields(self) -> None:
        t = ICML_2025
        assert t.name == "icml_2025"
        assert t.year == 2025
        assert t.style_package == "icml2025"
        assert t.columns == 2
        assert t.author_format == "icml"
        assert t.bib_style == "icml2025"

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            NEURIPS_2024.name = "hacked"  # type: ignore[misc]


class TestRenderPreamble:
    """Tests for ConferenceTemplate.render_preamble()."""

    def test_neurips_preamble_structure(self) -> None:
        tex = NEURIPS_2024.render_preamble("My Title", "J. Doe", "An abstract.")
        assert r"\documentclass{article}" in tex
        assert r"\usepackage[preprint]{neurips_2024}" in tex
        assert r"\title{My Title}" in tex
        assert r"\author{J. Doe}" in tex
        assert r"\begin{abstract}" in tex
        assert "An abstract." in tex
        assert r"\end{abstract}" in tex
        assert r"\begin{document}" in tex
        assert r"\maketitle" in tex

    def test_iclr_preamble_no_options(self) -> None:
        tex = ICLR_2025.render_preamble("Title", "Author", "Abstract")
        assert r"\documentclass{article}" in tex  # no options
        assert r"\usepackage{iclr2025_conference}" in tex

    def test_icml_author_block(self) -> None:
        tex = ICML_2025.render_preamble("Title", "Alice", "Abstract")
        assert r"\begin{icmlauthorlist}" in tex
        assert r"\icmlauthor{Alice}{aff1}" in tex
        assert r"\end{icmlauthorlist}" in tex
        assert r"\icmlaffiliation{aff1}{Affiliation}" in tex

    def test_icml_preamble_extra(self) -> None:
        tex = ICML_2025.render_preamble("Title", "Author", "Abstract")
        assert r"\icmltitlerunning{Title}" in tex


class TestRenderFooter:
    """Tests for ConferenceTemplate.render_footer()."""

    def test_neurips_footer(self) -> None:
        tex = NEURIPS_2024.render_footer("refs")
        assert r"\bibliographystyle{plainnat}" in tex
        assert r"\bibliography{refs}" in tex
        assert r"\end{document}" in tex

    def test_icml_footer(self) -> None:
        tex = ICML_2025.render_footer()
        assert r"\bibliographystyle{icml2025}" in tex
        assert r"\bibliography{references}" in tex

    def test_default_bib_file(self) -> None:
        tex = NEURIPS_2024.render_footer()
        assert r"\bibliography{references}" in tex


class TestGetTemplate:
    """Tests for get_template() lookup."""

    def test_full_name(self) -> None:
        assert get_template("neurips_2024") is NEURIPS_2024

    def test_short_alias(self) -> None:
        assert get_template("neurips") is NEURIPS_2025
        assert get_template("iclr") is ICLR_2026
        assert get_template("icml") is ICML_2026

    def test_case_insensitive(self) -> None:
        assert get_template("NeurIPS") is NEURIPS_2025
        assert get_template("ICML_2026") is ICML_2026

    def test_dash_and_space_normalization(self) -> None:
        assert get_template("neurips-2025") is NEURIPS_2025
        assert get_template("icml 2026") is ICML_2026

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown conference"):
            get_template("aaai_2025")


class TestListConferences:
    """Tests for list_conferences()."""

    def test_returns_canonical_names(self) -> None:
        names = list_conferences()
        assert "neurips_2025" in names
        assert "iclr_2026" in names
        assert "icml_2026" in names
        # HEP-phenomenology templates (physics-collider branch)
        assert "jhep" in names
        assert "prd" in names
        assert "prl" in names
        assert "prx" in names
        assert "epjc" in names
        # Should be deduplicated — no aliases
        # 6 ML conferences + 1 generic + 5 HEP = 12
        assert len(names) == 12

    def test_sorted(self) -> None:
        names = list_conferences()
        assert names == sorted(names)


class TestConferenceRegistry:
    """Tests for CONFERENCE_REGISTRY dict."""

    def test_all_aliases_resolve(self) -> None:
        for key, tpl in CONFERENCE_REGISTRY.items():
            assert isinstance(tpl, ConferenceTemplate)
            assert tpl.name  # not empty


# =====================================================================
# converter.py tests
# =====================================================================


class TestParseSections:
    """Tests for _parse_sections()."""

    def test_empty(self) -> None:
        sections = _parse_sections("")
        assert len(sections) == 1
        assert sections[0].level == 1
        assert sections[0].body == ""

    def test_single_heading(self) -> None:
        md = "# Introduction\nHello world"
        sections = _parse_sections(md)
        assert len(sections) == 1
        assert sections[0].level == 1
        assert sections[0].heading == "Introduction"
        assert "Hello world" in sections[0].body

    def test_multiple_headings(self) -> None:
        md = "# Title\nfoo\n## Method\nbar\n### Details\nbaz"
        sections = _parse_sections(md)
        assert len(sections) == 3
        assert sections[0].heading == "Title"
        assert sections[1].heading == "Method"
        assert sections[2].heading == "Details"

    def test_preamble_before_heading(self) -> None:
        md = "Some text before\n\n# First\nBody"
        sections = _parse_sections(md)
        assert len(sections) == 2
        assert sections[0].level == 0
        assert "Some text before" in sections[0].body

    def test_heading_lower(self) -> None:
        md = "# Abstract\nContent"
        sections = _parse_sections(md)
        assert sections[0].heading_lower == "abstract"


class TestExtractTitle:
    """Tests for _extract_title()."""

    def test_bold_title_after_heading(self) -> None:
        md = "# Title\n**My Paper**\n\n# Abstract\nblah"
        sections = _parse_sections(md)
        assert _extract_title(sections, md) == "My Paper"

    def test_first_non_meta_h1(self) -> None:
        md = "# Introduction\nSome text"
        sections = _parse_sections(md)
        assert _extract_title(sections, md) == "Introduction"

    def test_fallback(self) -> None:
        sections = _parse_sections("")
        assert _extract_title(sections, "") == "Untitled Paper"


class TestExtractAbstract:
    """Tests for _extract_abstract()."""

    def test_from_h1(self) -> None:
        md = "# Abstract\nThis is the abstract.\n\n# Intro\nBody"
        sections = _parse_sections(md)
        assert "This is the abstract." in _extract_abstract(sections)

    def test_from_h2(self) -> None:
        md = "# Title\nfoo\n## Abstract\nAbstract text.\n## Intro"
        sections = _parse_sections(md)
        assert "Abstract text." in _extract_abstract(sections)

    def test_missing_abstract(self) -> None:
        md = "# Introduction\nNo abstract here"
        sections = _parse_sections(md)
        assert _extract_abstract(sections) == ""


class TestConvertInline:
    """Tests for _convert_inline()."""

    def test_bold(self) -> None:
        assert r"\textbf{bold}" in _convert_inline("**bold**")

    def test_italic(self) -> None:
        assert r"\textit{italic}" in _convert_inline("*italic*")

    def test_inline_code(self) -> None:
        assert r"\texttt{code}" in _convert_inline("`code`")

    def test_link(self) -> None:
        result = _convert_inline("[text](http://example.com)")
        assert r"\href{http://example.com}{text}" in result

    def test_special_chars_escaped(self) -> None:
        result = _convert_inline("100% done & 5# items")
        assert r"100\% done \& 5\# items" in result

    def test_math_preserved(self) -> None:
        result = _convert_inline(r"where \(x + y\) is given")
        assert r"\(x + y\)" in result

    def test_cite_preserved(self) -> None:
        result = _convert_inline(r"as shown by \cite{doe2024}")
        assert r"\cite{doe2024}" in result

    def test_dollar_math_preserved(self) -> None:
        result = _convert_inline("the value $x^2$ is")
        assert "$x^2$" in result

    def test_pre_escaped_underscore_not_doubled(self) -> None:
        """BUG-182: LLM pre-escapes underscores → must NOT double-escape to \\\\_."""
        result = _convert_inline(r"RawObservation\_PPO\_WithNorm")
        assert r"\\_" not in result, f"Double-escaped: {result}"
        assert r"\_" in result

    def test_pre_escaped_underscore_near_math(self) -> None:
        """BUG-182: Pre-escaped underscore adjacent to math must not break."""
        result = _convert_inline(
            r"RawObs\_PPO. Statistics \(\mu_t\) are given"
        )
        assert r"\\_" not in result
        assert r"\_" in result
        assert r"\(\mu_t\)" in result

    def test_pre_escaped_hash_not_doubled(self) -> None:
        """BUG-182: Pre-escaped hash should not be double-escaped."""
        result = _convert_inline(r"Section \#3 details")
        assert r"\\#" not in result
        assert r"\#" in result


class TestEscapeLatex:
    """Tests for _escape_latex()."""

    def test_special_chars(self) -> None:
        assert r"\#" in _escape_latex("#")
        assert r"\%" in _escape_latex("%")
        assert r"\&" in _escape_latex("&")
        assert r"\_" in _escape_latex("_")

    def test_math_not_escaped(self) -> None:
        result = _escape_latex(r"value \(x_1\) here")
        assert r"\(x_1\)" in result  # underscore inside math preserved


class TestBuildBody:
    """Tests for _build_body()."""

    def test_skips_title_and_abstract(self) -> None:
        md = "# Title\nfoo\n# Abstract\nbar\n# Introduction\nbaz"
        sections = _parse_sections(md)
        body = _build_body(sections)
        assert r"\section{Introduction}" in body
        assert "baz" in body
        # Title and abstract should not appear as sections
        assert r"\section{Title}" not in body
        assert r"\section{Abstract}" not in body

    def test_subsection_promoted_when_all_h2(self) -> None:
        """T1.3: When all body sections are H2, they should be promoted to \\section."""
        md = "## Method\ntext"
        sections = _parse_sections(md)
        body = _build_body(sections)
        # All-H2 document → auto-promoted to \section
        assert r"\section{Method}" in body

    def test_h2_promoted_under_h1_title(self) -> None:
        """When title occupies H1, H2 body sections promote to \\section."""
        md = "# My Paper\ntitle body\n## Method\ntext"
        sections = _parse_sections(md)
        body = _build_body(sections, title="My Paper")
        assert r"\section{Method}" in body

    def test_subsubsection(self) -> None:
        md = "## Intro\nintro\n### Details\ntext"
        sections = _parse_sections(md)
        body = _build_body(sections)
        # H2 promoted to \section, H3 promoted to \subsection
        assert r"\subsection{Details}" in body


class TestListRendering:
    """Tests for bullet and numbered list rendering."""

    def test_bullet_list(self) -> None:
        items = ["First item", "Second item"]
        result = _render_itemize(items)
        assert r"\begin{itemize}" in result
        assert r"\item First item" in result
        assert r"\item Second item" in result
        assert r"\end{itemize}" in result

    def test_numbered_list(self) -> None:
        items = ["Step one", "Step two"]
        result = _render_enumerate(items)
        assert r"\begin{enumerate}" in result
        assert r"\item Step one" in result
        assert r"\end{enumerate}" in result


class TestTableRendering:
    """Tests for Markdown table → LaTeX tabular conversion."""

    def test_parse_table_row(self) -> None:
        assert _parse_table_row("| a | b | c |") == ["a", "b", "c"]

    def test_parse_alignments(self) -> None:
        assert _parse_alignments("| --- | :---: | ---: |", 3) == ["l", "c", "r"]

    def test_render_simple_table(self) -> None:
        lines = [
            "| Name | Value |",
            "| --- | --- |",
            "| A | 1 |",
            "| B | 2 |",
        ]
        result = _render_table(lines)
        assert r"\begin{table}" in result
        assert r"\begin{tabular}{ll}" in result
        assert r"\toprule" in result
        assert r"\textbf{Name}" in result
        assert r"\midrule" in result
        assert r"\bottomrule" in result
        assert r"\end{tabular}" in result
        assert r"\end{table}" in result

    def test_render_counters_are_thread_local(self) -> None:
        results: list[tuple[int, int, int]] = []
        lock = threading.Lock()

        def worker() -> None:
            _reset_render_counters()
            value = (_next_table_num(), _next_table_num(), _next_figure_num())
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results == [(1, 2, 1)] * 4


# =====================================================================
# markdown_to_latex integration tests
# =====================================================================


class TestMarkdownToLatex:
    """Integration tests for the full conversion pipeline."""

    SAMPLE_MD = (
        "# Title\n"
        "**My Great Paper**\n\n"
        "# Abstract\n"
        "This is the abstract.\n\n"
        "# Introduction\n"
        "We study the problem of RL.\n\n"
        "## Related Work\n"
        "Prior work includes **many** approaches.\n\n"
        "# Method\n"
        "Our method uses \\(f(x) = x^2\\) as the objective.\n\n"
        "# Results\n"
        "- Result 1\n"
        "- Result 2\n\n"
        "# Conclusion\n"
        "We conclude.\n\n"
        "# References\n"
        "1. Doe et al. (2024)\n"
    )

    def test_neurips_full(self) -> None:
        tex = markdown_to_latex(self.SAMPLE_MD, NEURIPS_2024)
        assert r"\documentclass{article}" in tex
        assert r"\usepackage[preprint]{neurips_2024}" in tex
        assert r"\title{My Great Paper}" in tex
        assert r"\begin{abstract}" in tex
        assert "This is the abstract." in tex
        assert r"\section{Introduction}" in tex
        assert r"\subsection{Related Work}" in tex
        assert r"\section{Method}" in tex
        assert r"\begin{itemize}" in tex
        assert r"\bibliographystyle{plainnat}" in tex
        assert r"\end{document}" in tex

    def test_iclr_full(self) -> None:
        tex = markdown_to_latex(self.SAMPLE_MD, ICLR_2025)
        assert r"\usepackage{iclr2025_conference}" in tex
        assert r"\bibliographystyle{iclr2025_conference}" in tex

    def test_icml_full(self) -> None:
        tex = markdown_to_latex(self.SAMPLE_MD, ICML_2025, authors="Alice")
        assert r"\begin{icmlauthorlist}" in tex
        assert r"\icmlauthor{Alice}{aff1}" in tex
        assert r"\bibliographystyle{icml2025}" in tex

    def test_custom_title_override(self) -> None:
        tex = markdown_to_latex(
            "# Abstract\nblah\n# Intro\nbody",
            NEURIPS_2024,
            title="Override Title",
        )
        assert r"\title{Override Title}" in tex

    def test_custom_authors(self) -> None:
        tex = markdown_to_latex(self.SAMPLE_MD, NEURIPS_2024, authors="Jane Doe")
        assert r"\author{Jane Doe}" in tex

    def test_custom_bib_file(self) -> None:
        tex = markdown_to_latex(self.SAMPLE_MD, NEURIPS_2024, bib_file="my_refs")
        assert r"\bibliography{my_refs}" in tex

    def test_math_preserved_in_output(self) -> None:
        md = "# Abstract\nabs\n# Method\n\\(f(x)\\) and \\[E = mc^2\\]"
        tex = markdown_to_latex(md, NEURIPS_2024, title="T")
        assert r"\(f(x)\)" in tex
        assert r"\[E = mc^2\]" in tex

    def test_empty_paper(self) -> None:
        tex = markdown_to_latex("", NEURIPS_2024, title="Empty")
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex

    def test_display_math_block(self) -> None:
        md = "# Abstract\nabs\n# Method\n\\[\nx = y + z\n\\]"
        tex = markdown_to_latex(md, NEURIPS_2024, title="T")
        assert "x = y + z" in tex

    def test_code_block(self) -> None:
        md = "# Abstract\nabs\n# Method\n```python\nprint('hello')\n```"
        tex = markdown_to_latex(md, NEURIPS_2024, title="T")
        assert r"\begin{verbatim}" in tex
        assert "print('hello')" in tex
        assert r"\end{verbatim}" in tex

    def test_table_in_paper(self) -> None:
        md = (
            "# Abstract\nabs\n"
            "# Results\n"
            "| Model | Score |\n"
            "| --- | --- |\n"
            "| Ours | 95.0 |\n"
        )
        tex = markdown_to_latex(md, NEURIPS_2024, title="T")
        assert r"\begin{tabular}" in tex
        assert r"\textbf{Model}" in tex


# =====================================================================
# ExportConfig tests
# =====================================================================


class TestExportConfig:
    """Tests for ExportConfig in config.py."""

    def test_default_values(self) -> None:
        from researchclaw.config import ExportConfig

        ec = ExportConfig()
        assert ec.target_conference == "neurips_2025"
        assert ec.authors == "Anonymous"
        assert ec.bib_file == "references"

    def test_frozen(self) -> None:
        from researchclaw.config import ExportConfig

        ec = ExportConfig()
        with pytest.raises(AttributeError):
            ec.target_conference = "icml"  # type: ignore[misc]

    def test_rcconfig_has_export(self) -> None:
        from researchclaw.config import RCConfig

        cfg = RCConfig.load("config.researchclaw.example.yaml", check_paths=False)
        assert hasattr(cfg, "export")
        assert cfg.export.target_conference == "neurips_2025"

    def test_rcconfig_export_from_dict(self) -> None:
        from researchclaw.config import RCConfig
        import yaml
        from pathlib import Path

        data = yaml.safe_load(Path("config.researchclaw.example.yaml").read_text(encoding="utf-8"))
        data["export"] = {
            "target_conference": "icml_2025",
            "authors": "Test Author",
            "bib_file": "mybib",
        }
        cfg = RCConfig.from_dict(data, check_paths=False)
        assert cfg.export.target_conference == "icml_2025"
        assert cfg.export.authors == "Test Author"
        assert cfg.export.bib_file == "mybib"


# =====================================================================
# hitl_required_stages validation update test
# =====================================================================


class TestHitlStageValidation:
    """Test that hitl_required_stages now accepts up to stage 23."""

    def test_stage_23_valid(self) -> None:
        from researchclaw.config import validate_config
        import yaml
        from pathlib import Path

        data = yaml.safe_load(Path("config.researchclaw.example.yaml").read_text(encoding="utf-8"))
        data.setdefault("security", {})["hitl_required_stages"] = [1, 22, 23]
        result = validate_config(data, check_paths=False)
        assert result.ok, f"Errors: {result.errors}"

    def test_get_style_files_returns_bundled_sty(self) -> None:
        """Each conference template bundles at least one .sty file."""
        for name in ["neurips_2025", "neurips_2024", "iclr_2026", "iclr_2025", "icml_2026", "icml_2025"]:
            tpl = get_template(name)
            files = tpl.get_style_files()
            assert len(files) >= 1, f"No style files for {name}"
            sty_names = [f.name for f in files]
            assert any(f.endswith(".sty") for f in sty_names), f"No .sty file for {name}"

    def test_iclr_icml_have_bst_files(self) -> None:
        """ICLR and ICML templates bundle custom .bst files."""
        for name in ["iclr_2026", "iclr_2025", "icml_2026", "icml_2025"]:
            tpl = get_template(name)
            files = tpl.get_style_files()
            bst_names = [f.name for f in files if f.suffix == ".bst"]
            assert len(bst_names) >= 1, f"No .bst file for {name}"

    def test_stage_24_invalid(self) -> None:
        from researchclaw.config import validate_config
        import yaml
        from pathlib import Path

        data = yaml.safe_load(Path("config.researchclaw.example.yaml").read_text(encoding="utf-8"))
        data.setdefault("security", {})["hitl_required_stages"] = [24]
        result = validate_config(data, check_paths=False)
        assert not result.ok
        assert any("24" in e for e in result.errors)


# =====================================================================
# check_paper_completeness — section word count + bullet density checks
# =====================================================================


class TestCompletenessWordCountAndBullets:
    """Tests for new per-section word count and bullet density checks."""

    @staticmethod
    def _make_sections(section_specs: list[tuple[str, int, bool]]) -> list:
        """Build _Section objects from (heading, word_count, use_bullets) specs."""
        results = []
        for heading, wc, bullets in section_specs:
            if bullets:
                lines = [f"- Point number {i}" for i in range(wc // 3)]
                body = "\n".join(lines)
            else:
                body = " ".join(["word"] * wc)
            results.append(
                type("_Section", (), {
                    "level": 1,
                    "heading": heading,
                    "heading_lower": heading.lower(),
                    "body": body,
                })()
            )
        return results

    def test_completeness_section_word_count_short(self) -> None:
        """A Method section with only 100 words triggers a warning."""
        secs = self._make_sections([
            ("Title", 5, False),
            ("Abstract", 200, False),
            ("Introduction", 900, False),
            ("Related Work", 700, False),
            ("Method", 100, False),
            ("Experiments", 1000, False),
            ("Results", 700, False),
            ("Conclusion", 250, False),
        ])
        warns = check_paper_completeness(secs)
        method_warns = [w for w in warns if "Method" in w and "words" in w]
        assert len(method_warns) >= 1, f"Expected word count warning, got: {warns}"

    def test_completeness_bullet_density(self) -> None:
        """A Method section full of bullet points triggers a warning."""
        secs = self._make_sections([
            ("Title", 5, False),
            ("Abstract", 200, False),
            ("Introduction", 900, False),
            ("Related Work", 700, False),
            ("Method", 300, True),
            ("Experiments", 1000, False),
            ("Results", 700, False),
            ("Conclusion", 250, False),
        ])
        warns = check_paper_completeness(secs)
        bullet_warns = [w for w in warns if "bullet" in w.lower() and "Method" in w]
        assert len(bullet_warns) >= 1, f"Expected bullet warning, got: {warns}"


# =====================================================================
# BUG-177: Algorithm pseudocode escaping tests
# =====================================================================


class TestAlgorithmEscaping:
    """Tests for _escape_algo_line and algorithm rendering in _render_code_block."""

    def test_escape_underscore(self) -> None:
        assert r"psi\_1" in _escape_algo_line("psi_1")

    def test_escape_hash_comment(self) -> None:
        result = _escape_algo_line("x = y  # update rule")
        assert r"\COMMENT{update rule}" in result
        assert "x = y" in result

    def test_fullline_hash_comment(self) -> None:
        result = _escape_algo_line("# Initialize buffer")
        assert result == r"\COMMENT{Initialize buffer}"

    def test_escape_percent(self) -> None:
        assert r"\%" in _escape_algo_line("accuracy 95%")

    def test_escape_ampersand(self) -> None:
        assert r"\&" in _escape_algo_line("x & y")

    def test_preserve_latex_commands(self) -> None:
        result = _escape_algo_line(r"Set $x = \alpha$ and update")
        assert r"$x = \alpha$" in result

    def test_render_code_block_algo_escapes(self) -> None:
        code = (
            "Initialize theta_1, theta_2\n"
            "for t = 1 to T do\n"
            "  Sample batch B  # prioritized\n"
        )
        result = _render_code_block("algorithm", code)
        assert r"\begin{algorithm}" in result
        assert r"\begin{algorithmic}" in result
        assert r"theta\_1" in result
        assert r"\COMMENT{prioritized}" in result

    def test_render_code_block_verbatim_no_escape(self) -> None:
        """Non-algorithm code blocks should use verbatim (no escaping)."""
        code = "x_1 = y_2  # comment"
        result = _render_code_block("python", code)
        assert r"\begin{verbatim}" in result
        assert "x_1" in result  # NOT escaped in verbatim


# =====================================================================
# Markdown footnotes
# =====================================================================


_FN_SPAN_RE = re.compile(
    re.escape(_FN_OPEN) + r".*?" + re.escape(_FN_CLOSE), re.DOTALL
)


def _outside_footnotes(text: str) -> str:
    """The document with every inlined footnote body removed.

    Assertions go through this rather than testing the whole string: under the
    old DOTALL body capture the swallowed block was still *present*, just
    buried inside the footnote, so a plain ``in`` check passed on a corrupted
    document.
    """
    return _FN_SPAN_RE.sub("", text)


class TestMarkdownFootnotes:
    """Tests for _resolve_markdown_footnotes() and its LaTeX output.

    The definition-swallowing cases below are the reason this feature is
    tested at all: with a DOTALL body capture, a definition that is not
    followed by a blank line absorbs the next block of the paper into the
    footnote and deletes it from the body, and nothing downstream errors.
    """

    def test_reference_becomes_footnote(self) -> None:
        tex = markdown_to_latex(
            "## Method\n\nA claim.[^a]\n\n[^a]: The note body.\n",
            NEURIPS_2025,
            title="T",
        )
        assert r"\footnote{The note body.}" in tex
        assert "[^a]" not in tex
        assert "XFOOTNOTE" not in tex

    def test_definition_followed_by_heading_keeps_the_heading(self) -> None:
        out = _resolve_markdown_footnotes(
            "A claim.[^1]\n\n[^1]: See Appendix A.\n## Results\n\nBody.\n"
        )
        assert "## Results" in _outside_footnotes(out)
        assert "Body." in _outside_footnotes(out)
        assert f"{_FN_OPEN}See Appendix A.{_FN_CLOSE}" in out

    def test_definition_followed_by_prose_keeps_the_prose(self) -> None:
        out = _resolve_markdown_footnotes(
            "A claim.[^1]\n[^1]: See Appendix A.\nThe sentence carrying the claim.\n"
        )
        assert "The sentence carrying the claim." in _outside_footnotes(out)
        assert f"{_FN_OPEN}See Appendix A.{_FN_CLOSE}" in out

    @pytest.mark.parametrize(
        "trailer,survivor",
        [
            ("- item one\n- item two\n", "- item one"),
            ("| A | B |\n|---|---|\n| 1 | 2 |\n", "| A | B |"),
            ("```python\nx = 1\n```\n", "x = 1"),
        ],
    )
    def test_definition_followed_by_block_keeps_the_block(
        self, trailer: str, survivor: str
    ) -> None:
        out = _resolve_markdown_footnotes(f"A claim.[^a]\n[^a]: Note.\n{trailer}")
        assert survivor in _outside_footnotes(out)
        assert f"{_FN_OPEN}Note.{_FN_CLOSE}" in out

    def test_crlf_input_does_not_swallow_the_document(self) -> None:
        out = _resolve_markdown_footnotes(
            "A claim.[^a].\r\n\r\n[^a]: Note.\r\n\r\nThe next paragraph.\r\n"
        )
        assert "The next paragraph." in _outside_footnotes(out)
        assert f"{_FN_OPEN}Note.{_FN_CLOSE}" in out

    def test_indented_continuation_joins_the_body(self) -> None:
        out = _resolve_markdown_footnotes(
            "A claim.[^a]\n\n[^a]: First line\n    continued here.\n\nNext.\n"
        )
        assert f"{_FN_OPEN}First line continued here.{_FN_CLOSE}" in out
        assert "Next." in _outside_footnotes(out)

    @pytest.mark.parametrize(
        "markdown",
        [
            "## Method[^a]\n\nBody.\n\n[^a]: Note.\n",
            "## R\n\n| Model[^a] | Acc |\n|---|---|\n| M | 0.9 |\n\n[^a]: Note.\n",
            "## R\n\n![A caption[^a]](figures/f)\n\nBody.\n\n[^a]: Note.\n",
        ],
    )
    def test_unsafe_positions_emit_no_footnote(self, markdown: str) -> None:
        """LaTeX takes no \\footnote in a heading, a table cell or a caption.

        The old behaviour put one there anyway: inside \\caption{} that is a
        fatal pdflatex error, and inside \\section{} it tears the heading apart
        and corrupts the \\label slug derived from it.
        """
        tex = markdown_to_latex(markdown, NEURIPS_2025, title="T")
        assert r"\footnote" not in tex
        assert "XFOOTNOTE" not in tex
        assert not re.search(r"\\label\{[^}]*footnote", tex, re.IGNORECASE)

    def test_reference_inside_a_fenced_code_block_is_untouched(self) -> None:
        tex = markdown_to_latex(
            '## M\n\n```python\np = re.compile(r"[^a]")\n```\n\nA claim.[^a]\n\n[^a]: A note.\n',
            NEURIPS_2025,
            title="T",
        )
        assert r"\footnote{A note.}" in tex
        assert "[^a]" in tex  # the regex character class survives verbatim

    def test_body_ending_in_a_backslash_does_not_run_away(self) -> None:
        tex = markdown_to_latex(
            "A claim.[^a]\n\n[^a]: Body ending in a backslash \\\n",
            NEURIPS_2025,
            title="T",
        )
        body = tex.split(r"\footnote{", 1)[1]
        assert not body.startswith("}")
        assert r"\}" not in body[:80]

    def test_undefined_reference_is_left_as_written(self) -> None:
        out = _resolve_markdown_footnotes("Dangling[^zzz] ref.\n\n[^a]: unrelated.\n")
        assert "[^zzz]" in out

    def test_unreferenced_definition_is_kept(self) -> None:
        """Standard Markdown drops it; losing author text silently is worse."""
        out = _resolve_markdown_footnotes("Plain text.\n\n[^orphan]: Nobody points here.\n")
        assert "Nobody points here." in _outside_footnotes(out)
        assert _FN_OPEN not in out

    def test_document_without_footnotes_is_unchanged(self) -> None:
        markdown = "## M\n\nJust [cite2020key] and *text*.\n"
        assert _resolve_markdown_footnotes(markdown) == markdown


class TestHeadingNumberStripping:
    """LaTeX numbers headings itself, so a number left in the title text prints twice."""

    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("1. Introduction", "Introduction"),
            ("2.1 Related Work", "Related Work"),
            ("3.2.1 Details", "Details"),
            # appendix subsections carry a letter-prefixed number; leaving it in the
            # title produced "A.8  A.8 What the released hash manifest covers"
            ("A.8 What the released hash manifest covers", "What the released hash manifest covers"),
            ("B.10.2 Deep nesting", "Deep nesting"),
        ],
    )
    def test_manual_number_is_stripped(self, heading: str, expected: str) -> None:
        tex = markdown_to_latex(f"## {heading}\n\nBody.\n", NEURIPS_2025, title="T")
        assert f"\\section{{{expected}}}" in tex

    @pytest.mark.parametrize(
        "heading",
        ["A Study of Conditioning", "B. Results", "Introduction", "4-tag recall"],
    )
    def test_non_numbers_are_left_alone(self, heading: str) -> None:
        """A bare letter or a leading digit that is part of the words must survive."""
        tex = markdown_to_latex(f"## {heading}\n\nBody.\n", NEURIPS_2025, title="T")
        assert f"\\section{{{heading}}}" in tex


class TestTableTypography:
    """Every table in a document must be set at the same size.

    The previous behaviour wrapped wide tables in ``\\resizebox{\\columnwidth}{!}``,
    which scales a table to *exactly* the text width. The scale factor is then
    columnwidth/natural-width and differs per table: across one paper's fifteen
    tables the effective body size ran from 5.8pt to 10.8pt, narrow tables
    magnified above the surrounding prose and wide ones shrunk below legibility.
    """

    NARROW = "## R\n\n| A | B |\n|---|---:|\n| x | 1 |\n"
    WIDE_TEXT = (
        "## R\n\n| Method | Description of the approach, at some length | Acc |\n"
        "|---|---|---:|\n| M | a long explanatory sentence that would stretch it | 0.9 |\n"
    )
    MANY_NUMERIC = (
        "## R\n\n| Profile | a | b | c | d | e | f | g |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        "| CIRCLE | 0.111 | 0.222 | 0.333 | 0.444 | 0.555 | 0.666 | 0.777 |\n"
    )

    @pytest.mark.parametrize("name", ["NARROW", "WIDE_TEXT", "MANY_NUMERIC"])
    def test_every_table_carries_the_same_size(self, name: str) -> None:
        tex = markdown_to_latex(getattr(self, name), NEURIPS_2025, title="T")
        assert TABLE_FONT in tex

    @pytest.mark.parametrize("name", ["NARROW", "WIDE_TEXT", "MANY_NUMERIC"])
    def test_scale_to_fill_is_never_used(self, name: str) -> None:
        """\\resizebox{\\columnwidth}{!} magnifies a narrow table; max width does not."""
        tex = markdown_to_latex(getattr(self, name), NEURIPS_2025, title="T")
        assert r"\resizebox{\columnwidth}{!}" not in tex

    def test_long_text_column_wraps_instead_of_stretching(self) -> None:
        tex = markdown_to_latex(self.WIDE_TEXT, NEURIPS_2025, title="T")
        assert r"\begin{tabularx}{\columnwidth}" in tex
        assert "X" in re.search(r"\\begin\{tabularx\}\{\\columnwidth\}\{([^}]*)\}", tex).group(1)

    def test_narrow_table_is_left_alone(self) -> None:
        tex = markdown_to_latex(self.NARROW, NEURIPS_2025, title="T")
        block = re.search(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S).group(0)
        assert r"\begin{tabular}{lr}" in block
        # checked on the float, not the document: the preamble loads tabularx
        assert "adjustbox" not in block
        assert "tabularx" not in block

    def test_uncompressible_table_is_shrunk_not_magnified(self) -> None:
        """Numbers cannot wrap, so a table of them is scaled down — but only down."""
        tex = markdown_to_latex(self.MANY_NUMERIC, NEURIPS_2025, title="T")
        assert r"\adjustbox{max width=\columnwidth}" in tex

    def test_visual_len_counts_glyphs_not_markup(self) -> None:
        """Sizing columns by source length made maths look far too wide to wrap."""
        assert _visual_len(r"$5.9\times10^{-7}$") < len(r"$5.9\times10^{-7}$")
        assert _visual_len("CIRCLE+NGON+THIN") == len("CIRCLE+NGON+THIN")

    def test_compound_labels_become_breakable(self) -> None:
        assert r"\discretionary{}{}{}" in _cell_breakable("CIRCLE+NGON+THIN")

    @pytest.mark.parametrize("cell", ["TALL", r"$5.9\times10^{-7}$", "short words here"])
    def test_cells_that_need_no_break_are_untouched(self, cell: str) -> None:
        assert _cell_breakable(cell) == cell

    def test_maths_is_measured_as_one_unbreakable_run(self) -> None:
        """A cell of maths cannot break, so the column is sized by its longest atom."""
        cell = r"$5.9\times10^{-7}$ / $8.5\times10^{-15}$"
        assert _longest_unbreakable([cell]) == _visual_len(r"$8.5\times10^{-15}$")
        assert _longest_unbreakable([cell]) < _visual_len(cell)
