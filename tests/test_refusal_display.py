"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the ``run_stage`` and ``obtain_upload`` helpers of
the install cell and the learner's Section 5 BYOD cell) against the real stage runner in a child process, as the
notebook does, with only ``google.colab``'s upload dialog stood in. For every path the cell must raise ``Refused`` (an
exception, so Run all stops at that cell), render as exactly one ``Refused: <file>: <reason>`` line through IPython's
``_render_traceback_`` hook, name the file the learner supplied, and print no ``Traceback``. The earlier
configuration's results are removed, so none can be read as this input's result. A genuine failure still raises
``RuntimeError`` with the stage's traceback. The content case is the hosted refusal run's input: Colab's own
``/content/sample_data/california_housing_test.csv``, a CSV without the ``id, source, target`` columns. CI's
dependencies suffice: the BYOD checks run before any model library is imported.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEM = "bart_summarization"
REFUSAL_CELL_ID = "bart_summarization-20"  # the hosted refusal run's halting cell: Section 5's BYOD cell
CALIFORNIA_HEADER = "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value\n"


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_refusal_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _tool("build_notebook")
TEMPLATE = _tool("notebook_template").TEMPLATE
CELLS = [c for c in BUILD.render(ROOT, TEMPLATE, "t")["cells"] if c["cell_type"] == "code"]


def _cell(needle: str) -> dict:
    found = [c for c in CELLS if needle in c["source"]]
    assert len(found) == 1, needle
    return found[0]


def _byod_cell(path: str) -> str:
    lines = []
    for line in _cell("USE_BYOD = False  # @param")["source"].split("\n"):
        if line.startswith("USE_BYOD = "):
            line = "USE_BYOD = True" + line[line.index("  #"):]
        elif line.startswith("BYOD_PATH = "):
            line = f"BYOD_PATH = {path!r}" + line[line.index("  #"):]
        lines.append(line)
    return "\n".join(lines)


def _kernel(tmp_path: Path) -> dict:
    """The notebook's own stage helpers, bound to a run directory that carries the package and the stage runner."""
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    (run / "outputs").mkdir()
    shutil.copytree(ROOT / "src", run / "src")
    shutil.copyfile(ROOT / "tools" / "tutorial_stages.py", run / TEMPLATE["stage_runner"])
    install = _cell("def run_stage(stage, *options):")["source"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    ns = {"ROOT": run, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path, "json": json, "subprocess": subprocess}
    exec(compile(install[install.index("REFUSED_EXIT = 3"):], "<install>", "exec"), ns)  # noqa: S102 - the notebook's own helpers
    return ns


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)  # noqa: S102 - the notebook's own cells
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown and "Error" not in shown.replace("Refused", ""), shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


def _with_earlier_results(ns: dict) -> Path:
    """An earlier accepted configuration's results on disk, as a learner who ran the notebook once would have them."""
    out, state = ns["ROOT"] / "outputs", ns["ROOT"] / "state"
    for name in (f"{STEM}_dataset_manifest.json", f"{STEM}_train.csv", f"{STEM}_evaluation_report.json", f"{STEM}_result.json", "runtime.json"):
        (out / name).write_text("{}", encoding="utf-8")
    (out / f"{STEM}_adapter").mkdir()
    for name in ("data.json", "splits.json", "generation.json", "adapt.json"):
        (state / name).write_text("{}", encoding="utf-8")
    return out


def test_the_refusal_cell_is_section_5s_byod_cell() -> None:
    assert _cell("USE_BYOD = False  # @param")["id"] == REFUSAL_CELL_ID


def test_colab_sample_csv_is_one_refused_line_at_the_byod_cell_and_clears_earlier_results(tmp_path: Path) -> None:
    """The hosted refusal run's input (``USE_BYOD=True;BYOD_PATH=/content/sample_data/california_housing_test.csv``):
    a CSV that exists in a fresh Colab runtime but has none of the dataset columns."""
    ns = _kernel(tmp_path)
    out = _with_earlier_results(ns)
    sample = tmp_path / "sample_data" / "california_housing_test.csv"
    sample.parent.mkdir()
    sample.write_text(CALIFORNIA_HEADER + "-122.05,37.37,27,3885,661,1537,606,6.6085,344700\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(sample)), "print('AFTER THE REFUSAL')"])
    line = _assert_refused(raised, shown, ran, "california_housing_test.csv")
    assert "missing columns ['id', 'source', 'target']" in line and "run the Section 5 cell again" in line
    assert "AFTER THE REFUSAL" not in shown
    assert sorted(p.name for p in out.iterdir()) == ["runtime.json"]
    assert sorted(p.name for p in (ns["ROOT"] / "state").iterdir()) == ["byod_incoming", "data.error.json"]
    error = json.loads((ns["ROOT"] / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "california_housing_test.csv"


@pytest.mark.parametrize(
    ("name", "text", "rule"),
    [
        ("hostname", "colab-runtime\n", "must be .csv, .json or .jsonl"),
        ("pairs.json", '{"id": "a"}', "must be an array of records"),
        ("pairs.jsonl", "not json\n", "Expecting value"),
        ("few.csv", "id,source,target\n" + "".join(f"r{i},source text number {i},target {i}\n" for i in range(10)), "at least 12 records with distinct sources"),
    ],
)
def test_byod_contract_violations_are_one_refused_line(tmp_path: Path, name: str, text: str, rule: str) -> None:
    ns = _kernel(tmp_path)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(path))])
    assert rule in _assert_refused(raised, shown, ran, name)


def test_missing_path_is_one_refused_line(tmp_path: Path) -> None:
    ns = _kernel(tmp_path)
    missing = str(tmp_path / "absent.csv")
    raised, shown, ran = _run_all(ns, [_byod_cell(missing)])
    assert "BYOD_PATH is not a file" in _assert_refused(raised, shown, ran, missing)


@pytest.mark.parametrize(
    ("uploads", "file"),
    [
        ({}, "(no file)"),
        ({"a.csv": b"id,source,target\n", "b.csv": b"id,source,target\n"}, "a.csv, b.csv"),
        ({"pairs.txt": b"text"}, "pairs.txt"),
    ],
)
def test_upload_dialog_refusals_are_one_refused_line(tmp_path: Path, uploads: dict, file: str) -> None:
    ns = _kernel(tmp_path)
    with _colab(uploads):
        raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, file)


def test_empty_path_outside_colab_is_one_refused_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing it raises ImportError
    ns = _kernel(tmp_path)
    raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, "BYOD_PATH")


def test_a_genuine_failure_is_still_an_error_with_its_traceback(tmp_path: Path) -> None:
    """A stage run out of order is a notebook failure, not a refusal: RuntimeError, and the stage's traceback shows."""
    ns = _kernel(tmp_path)
    raised, shown, ran = _run_all(ns, ["run_stage('evaluate')"])
    assert type(raised) is RuntimeError and "is missing" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown
