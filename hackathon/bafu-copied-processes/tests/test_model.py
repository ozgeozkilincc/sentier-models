"""Tests on small synthetic tables -- no sentier-inventory clone needed.

Run from the toolkit root:  python -m pytest tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import forecast_superstructure as fs  # noqa: E402
from geo import is_aggregate  # noqa: E402
from model import build_family, check_data_layout, check_roundtrip, compare_rows, reconstruct  # noqa: E402

RP = "Electricity, production mix"


def _process(pid, location):
    return {"process_id": pid, "reference_product": RP, "location": location,
            "reference_unit": "kWh", "reference_amount": 1.0, "process_type": "unit",
            "technology": None, "comment": None, "sector": "02-electricity"}


def _exchange(pid, flow, flow_name, amount, location, direction="input", unit="kWh"):
    return {"process_id": pid, "flow": flow, "flow_name": flow_name, "flow_type": "technosphere",
            "direction": direction, "amount": amount, "unit": unit, "location": location,
            "uncertainty_type": 2.0, "loc": np.nan, "scale": 0.1, "minimum": np.nan,
            "maximum": np.nan, "sector": "02-electricity"}


def _share_family():
    """Four members whose inputs sum to 1. CH has the awkward cases found in
    the real data: one technology linked to two locations (CH and RER), a
    fully duplicated exchange row, and an amount far below 1e-8."""
    P = pd.DataFrame([_process("ch", "CH"), _process("de", "DE"),
                      _process("fr", "FR"), _process("rer", "RER")])
    rows = [
        _exchange("ch", "hydro-ch", "hydro", 0.3, "CH"),
        _exchange("ch", "hydro-rer", "hydro", 0.3, "RER"),
        _exchange("ch", "nuclear-ch", "nuclear", 0.2, "CH"),
        _exchange("ch", "nuclear-ch", "nuclear", 0.2, "CH"),
        _exchange("ch", "coal-ch", "coal", 3e-12, "CH"),
        _exchange("de", "coal-de", "coal", 0.6, "DE"),
        _exchange("de", "wind-de", "wind", 0.4, "DE"),
        _exchange("fr", "nuclear-fr", "nuclear", 0.7, "FR"),
        _exchange("fr", "hydro-fr", "hydro", 0.3, "FR"),
        _exchange("rer", "coal-rer", "coal", 0.5, "RER"),
        _exchange("rer", "nuclear-rer", "nuclear", 0.5, "RER"),
    ]
    rows += [_exchange(pid, f"out-{pid}", RP, 1.0, loc, direction="output")
             for pid, loc in [("ch", "CH"), ("de", "DE"), ("fr", "FR"), ("rer", "RER")]]
    return P, pd.DataFrame(rows)


def test_roundtrip_is_lossless_with_repeated_slots_and_link_locations():
    P, E = _share_family()
    assert check_roundtrip(P, E, RP, verbose=False)


def test_values_key_is_unique_with_seq():
    P, E = _share_family()
    _, values, _ = build_family(P, E, RP)
    assert values.duplicated(["process_id", "slot_id"]).any()
    assert not values.duplicated(["process_id", "slot_id", "seq"]).any()


def test_link_location_survives_reconstruction():
    P, E = _share_family()
    structure, values, proc_meta = build_family(P, E, RP)
    rebuilt = reconstruct(structure, values, proc_meta, RP)
    assert set(rebuilt.loc[rebuilt["flow"] == "hydro-rer", "location"]) == {"RER"}


def test_compare_rows_catches_changes_below_1e_8():
    P, E = _share_family()
    original = E.drop(columns=["sector"]).reset_index(drop=True)
    changed = original.copy()
    changed.loc[changed["flow"] == "coal-ch", "amount"] = 5e-12
    assert compare_rows(original, original.copy(), verbose=False)
    assert not compare_rows(original, changed, verbose=False)


def test_activation_scores_count_total_misses_as_zero():
    assert fs._activation_scores({1, 2}, {3}) == (0.0, 0.0, 0.0)
    assert fs._activation_scores({1, 2}, set()) == (0.0, 0.0, 0.0)
    assert fs._activation_scores(set(), set()) == (1.0, 1.0, 1.0)
    assert fs._activation_scores({1, 2}, {2, 3}) == (0.5, 0.5, 0.5)


def test_member_slot_amounts_sum_repeated_slots():
    P, E = _share_family()
    structure, values, _ = build_family(P, E, RP)
    amounts = fs._member_slot_amounts(values)
    hydro = structure.loc[structure["flow_name"] == "hydro", "slot_id"].item()
    ch_hydro = amounts[(amounts["process_id"] == "ch") & (amounts["slot_id"] == hydro)]
    assert ch_hydro["amount"].item() == pytest.approx(0.6)


def test_sum_to_one_only_for_share_families():
    P, E = _share_family()
    structure, values, _ = build_family(P, E, RP)
    assert fs._is_share_family(structure, fs._member_slot_amounts(values))

    # Same unit throughout, but the inputs are quantities, not shares.
    E_qty = E.copy()
    E_qty.loc[E_qty["direction"] == "input", "amount"] *= 3
    structure, values, _ = build_family(P, E_qty, RP)
    assert not fs._is_share_family(structure, fs._member_slot_amounts(values))

    # Mixed input units.
    E_mixed = E.copy()
    E_mixed.loc[E_mixed["flow_name"] == "coal", "unit"] = "kg"
    structure, values, _ = build_family(P, E_mixed, RP)
    assert not fs._is_share_family(structure, fs._member_slot_amounts(values))


def test_solve_member_scores_against_summed_truth():
    P, E = _share_family()
    structure, values, proc_meta = build_family(P, E, RP)
    result = fs.solve_member(structure, values, proc_meta, "CH", anchor="global")
    assert result["status"] == "Optimal"
    assert result["mass_balance_applied"]
    hydro = structure.loc[structure["flow_name"] == "hydro", "slot_id"].item()
    real = result["comparison"].set_index("slot_id").loc[hydro, "real_amount"]
    assert real == pytest.approx(0.6)
    assert 0.0 <= result["baseline_f1"] <= 1.0


def test_regional_aggregates_are_not_scored():
    assert is_aggregate("RER") and is_aggregate("GLO") and is_aggregate("ENTSO-E")
    assert not is_aggregate("CH")


def test_check_data_layout_rejects_per_source_layout(tmp_path):
    (tmp_path / "data" / "bafu-2026-v1").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="c5aee15"):
        check_data_layout(str(tmp_path))
    (tmp_path / "data" / "99-obsolete").mkdir()
    check_data_layout(str(tmp_path))
