"""Environment manifest parsing + resolution (P1)."""

from __future__ import annotations

from researchclaw.pipeline.environment import (
    COMPUTE_INSUFFICIENT,
    COMPUTE_OK,
    COMPUTE_UNKNOWN,
    DS_CACHED,
    DS_WILL_DOWNLOAD,
    PKG_AVAILABLE,
    PKG_WILL_INSTALL,
    parse_manifest,
    pip_to_import,
    resolve_environment,
)


# -- pip → import mapping -----------------------------------------------------

class TestPipToImport:
    def test_known_aliases(self):
        assert pip_to_import("scikit-learn>=1.4") == "sklearn"
        assert pip_to_import("opencv-python") == "cv2"
        assert pip_to_import("Pillow") == "PIL"
        assert pip_to_import("PyYAML==6.0") == "yaml"
        assert pip_to_import("biopython") == "Bio"

    def test_default_dash_to_underscore(self):
        assert pip_to_import("stable-baselines3") == "stable_baselines3"
        assert pip_to_import("transformers>=4.40") == "transformers"

    def test_strips_extras_and_markers(self):
        assert pip_to_import("torch[cuda]>=2.1") == "torch"
        assert pip_to_import("foo ; python_version>'3.9'") == "foo"


# -- manifest parsing ---------------------------------------------------------

class TestParseManifest:
    def test_empty_when_no_block(self):
        m = parse_manifest({"objectives": ["o1"]})
        assert m.declared is False
        assert m.pip == ()

    def test_non_dict_plan(self):
        assert parse_manifest(None).declared is False
        assert parse_manifest(["a"]).declared is False

    def test_full_block(self):
        plan = {
            "environment": {
                "pip": ["transformers>=4.40", "peft", {"spec": "cadquery"}],
                "system": ["libGL1"],
                "datasets": [
                    {"name": "DeepCAD", "source": "hf://x", "size_gb": 2.1, "cache": "data/deepcad"},
                    "MNIST",
                ],
                "compute": {"gpu": "required", "min_vram_gb": 24, "gpus": 2},
                "est_setup_sec": 600,
            }
        }
        m = parse_manifest(plan)
        assert m.declared is True
        assert [p.pip_name for p in m.pip] == ["transformers", "peft", "cadquery"]
        assert m.import_names == ("transformers", "peft", "cadquery")
        assert m.system == ("libGL1",)
        assert m.datasets[0].name == "DeepCAD" and m.datasets[0].size_gb == 2.1
        assert m.datasets[1].name == "MNIST"
        assert m.compute.gpu == "required" and m.compute.min_vram_gb == 24 and m.compute.gpus == 2
        assert m.est_setup_sec == 600

    def test_tolerates_bad_types(self):
        m = parse_manifest({"environment": {"pip": "single-pkg", "datasets": None, "compute": "nope"}})
        assert [p.pip_name for p in m.pip] == ["single-pkg"]
        assert m.datasets == ()
        assert m.compute is None


# -- resolution ---------------------------------------------------------------

class TestResolveEnvironment:
    def _manifest(self):
        return parse_manifest({
            "environment": {
                "pip": ["transformers", "peft", "cadquery"],
                "datasets": [{"name": "DeepCAD", "size_gb": 2.0}],
                "compute": {"gpu": "required", "min_vram_gb": 24, "gpus": 2},
            }
        })

    def test_classifies_packages(self):
        installed = {"transformers": True, "peft": True, "cadquery": False}
        res = resolve_environment(self._manifest(), installed,
                                  hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2})
        by = {p.req.pip_name: p.status for p in res.packages}
        assert by["transformers"] == PKG_AVAILABLE
        assert by["cadquery"] == PKG_WILL_INSTALL
        assert res.needs_install == ["cadquery"]

    def test_dataset_cached_vs_download(self):
        res = resolve_environment(self._manifest(), {}, cached_datasets={"deepcad"})
        assert res.datasets[0].status == DS_CACHED
        res2 = resolve_environment(self._manifest(), {}, cached_datasets=set())
        assert res2.datasets[0].status == DS_WILL_DOWNLOAD
        assert res2.needs_download == ["DeepCAD"]

    def test_compute_ok_on_sufficient_hw(self):
        res = resolve_environment(
            self._manifest(), {"transformers": True, "peft": True, "cadquery": True},
            hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2},
            cached_datasets={"deepcad"},
        )
        assert res.compute_status == COMPUTE_OK
        assert res.runnable_as_is is True

    def test_compute_insufficient_no_gpu(self):
        res = resolve_environment(self._manifest(), {}, hw_profile={"has_gpu": False})
        assert res.compute_status == COMPUTE_INSUFFICIENT
        assert res.provisionable is False
        assert res.runnable_as_is is False

    def test_compute_insufficient_low_vram(self):
        res = resolve_environment(
            self._manifest(), {},
            hw_profile={"has_gpu": True, "vram_mb": 8192, "gpu_count": 1},
        )
        assert res.compute_status == COMPUTE_INSUFFICIENT
        assert "GB" in res.compute_detail

    def test_provisionable_when_only_install_needed(self):
        res = resolve_environment(
            self._manifest(), {"transformers": True, "peft": True, "cadquery": False},
            hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2},
            cached_datasets=set(),
        )
        assert res.runnable_as_is is False      # cadquery + DeepCAD missing
        assert res.provisionable is True        # but installable/downloadable, compute OK

    def test_undeclared_summary_and_unknown_compute(self):
        res = resolve_environment(parse_manifest({}), {})
        assert res.declared is False
        assert res.compute_status == COMPUTE_UNKNOWN
        assert "No `environment:`" in res.summary_md()

    def test_to_dict_roundtrip_keys(self):
        res = resolve_environment(self._manifest(), {"transformers": True},
                                  hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2})
        d = res.to_dict()
        assert set(d) >= {"declared", "runnable_as_is", "provisionable", "packages",
                          "datasets", "compute", "needs_install", "needs_download",
                          "needs_operator", "operator_setup"}

    def test_system_libs_need_operator_not_infeasible(self):
        # A scientifically-required tool with OS-level deps must NOT make the
        # design infeasible — it is provisionable via an operator install.
        m = parse_manifest({"environment": {
            "pip": ["cadquery"],
            "system": ["libgl1", "libglu1-mesa"],
            "compute": {"gpu": "required", "min_vram_gb": 24},
        }})
        res = resolve_environment(m, {"cadquery": False},
                                  hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2})
        assert res.needs_operator == ["libgl1", "libglu1-mesa"]
        assert res.provisionable is True          # operator can install → not infeasible
        assert res.runnable_as_is is False        # but needs action first
        cmds = res.operator_setup_lines()
        assert cmds == ["sudo apt-get install -y libgl1", "sudo apt-get install -y libglu1-mesa"]

    def test_no_system_libs_no_operator(self):
        res = resolve_environment(self._manifest(), {"transformers": True, "peft": True, "cadquery": True},
                                  hw_profile={"has_gpu": True, "vram_mb": 24576, "gpu_count": 2},
                                  cached_datasets={"deepcad"})
        assert res.needs_operator == []
        assert res.operator_setup_lines() == []
