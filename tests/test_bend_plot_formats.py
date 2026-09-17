"""Render saved contribution measurements without importing JuTrack or recomputing physics."""
from __future__ import annotations

import copy

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from brilliance import accelerator_api as api
from brilliance.plotting import render_bend_radiation_contributions_matplotlib


@pytest.fixture
def evidence():
    # Deliberately not normalized: plotting must not change measured fractions.
    return {"rows": [
        {"bend": "B1", "I2_fraction": .1, "I5_fraction": .35, "I5_over_I2": .012},
        {"bend": "B2", "I2_fraction": .3, "I5_fraction": .05, "I5_over_I2": .0004},
    ]}


@pytest.mark.parametrize("suffix,header", [(".svg", b"<?xml"), (".png", b"\x89PNG"),
                                           (".pdf", b"%PDF"), (".PNG", b"\x89PNG")])
def test_public_plot_accepts_all_documented_formats_without_physics(tmp_path, monkeypatch, evidence, suffix, header):
    before = copy.deepcopy(evidence)

    def no_simulation(*args, **kwargs):
        pytest.fail("A rendering-only request must not recompute physics")

    monkeypatch.setattr(api, "jutrack", no_simulation)
    monkeypatch.setattr(api, "bend_radiation_contributions", no_simulation)
    open_figures = plt.get_fignums()
    output = tmp_path / "plots" / ("bend_contributions" + suffix)
    assert api.save_bend_radiation_contribution_plot(output, evidence) == output
    assert output.read_bytes().startswith(header)
    assert evidence == before
    assert plt.get_fignums() == open_figures


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_matplotlib_uses_exact_row_order_values_and_percentage_axis(tmp_path, monkeypatch, evidence, suffix):
    original = Figure.savefig
    observed = {}

    def inspect(figure, *args, **kwargs):
        axis = figure.axes[0]
        observed["heights"] = [[bar.get_height() for bar in group] for group in axis.containers]
        observed["names"] = [tick.get_text() for tick in axis.get_xticklabels()]
        observed["percent"] = axis.yaxis.get_major_formatter()(.1)
        observed["title"] = figure._suptitle.get_text()
        observed["zero_tick"] = 0.0 in axis.get_yticks()
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    api.save_bend_radiation_contribution_plot(tmp_path / ("measured" + suffix), evidence,
                                            title="Measured contributions", subtitle="Exact saved rows")
    assert observed["heights"] == [[.1, .3], [.35, .05]]
    assert observed["names"] == ["B1", "B2"]
    assert observed["percent"].startswith("10") and observed["percent"].endswith("%")
    assert observed["title"] == "Measured contributions\nExact saved rows"
    assert observed["zero_tick"]


@pytest.mark.parametrize("suffix", ["", ".jpg", ".txt"])
def test_invalid_extension_rejected_before_accessing_measurements(tmp_path, suffix):
    class UnreadableEvidence(dict):
        def get(self, *args, **kwargs):
            pytest.fail("Extension must be validated before reading measurements")

    output = tmp_path / "must_not_be_created" / ("plot" + suffix)
    with pytest.raises(ValueError, match=r"must end in .svg, .png, or .pdf"):
        api.save_bend_radiation_contribution_plot(output, UnreadableEvidence())
    assert not output.parent.exists()


@pytest.mark.parametrize("suffix", [".svg", ".png", ".pdf"])
def test_no_finite_measured_rows_fail_consistently_without_empty_figures(tmp_path, suffix):
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="no finite I2/I5 contribution rows"):
        api.save_bend_radiation_contribution_plot(tmp_path / ("empty" + suffix), {"rows": []})
    assert plt.get_fignums() == figures


def test_direct_matplotlib_renderer_rejects_unsupported_suffix(tmp_path, evidence):
    with pytest.raises(ValueError, match=".png or .pdf"):
        render_bend_radiation_contributions_matplotlib(evidence, tmp_path / "wrong.svg")


def test_save_failure_closes_figure_and_preserves_original_error(tmp_path, monkeypatch, evidence):
    figures = plt.get_fignums()
    failure = PermissionError("render destination is not writable")

    def cannot_save(*args, **kwargs):
        raise failure

    monkeypatch.setattr(Figure, "savefig", cannot_save)
    with pytest.raises(PermissionError) as caught:
        api.save_bend_radiation_contribution_plot(tmp_path / "blocked.png", evidence)
    assert caught.value is failure
    assert plt.get_fignums() == figures
