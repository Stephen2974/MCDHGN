"""IEC stays outside the sealed E0-E7 trainer unless its switch is on."""

import copy
import sys
from pathlib import Path

import pytest
import torch
import dgl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation_config import EXPERIMENTS
from iec import (
    SEALED_CODE_SHA256,
    CalibratedMCDHGN,
    IntrinsicEvidenceCalibration,
    assert_output_isolated,
    assert_sealed_code,
    e1_architecture,
    sha256_file,
)
from model import MCDHGN
import run_ablation as runner


ROOT = Path(__file__).resolve().parents[1]


def _graph(n=4):
    nodes = torch.arange(n)
    return dgl.graph((torch.cat([nodes, nodes]), torch.cat([nodes, (nodes + 1) % n])), num_nodes=n)


def test_sealed_files_and_e1_dict_are_unchanged():
    assert_sealed_code(ROOT)
    for name, expected in SEALED_CODE_SHA256.items():
        assert sha256_file(ROOT / name) == expected
    before = copy.deepcopy(EXPERIMENTS)
    on = e1_architecture("on")
    off = e1_architecture("off")
    assert EXPERIMENTS == before
    assert set(EXPERIMENTS) == {f"E{i}" for i in range(8)}
    assert "iec_mode" not in EXPERIMENTS["E1"]
    assert off == EXPERIMENTS["E1"]
    assert on["iec_mode"] == "on" and on["mutation_width"] == 16
    assert EXPERIMENTS["E3"] == dict(EXPERIMENTS["E1"], fusion_mode="cross")
    assert EXPERIMENTS["E2"] == dict(EXPERIMENTS["E1"], semantic_mode="node")


def test_output_guard_rejects_sealed_directories():
    allowed = ROOT / "results" / "iec_v1" / "unit"
    assert assert_output_isolated(ROOT, allowed) == allowed.resolve()
    with pytest.raises(ValueError):
        assert_output_isolated(ROOT, ROOT / "results" / "p0_legacy_D0" / "IEC")
    with pytest.raises(ValueError):
        assert_output_isolated(ROOT, ROOT / "results" / "ablation" / "IEC")
    with pytest.raises(ValueError):
        assert_output_isolated(ROOT, ROOT / "data_versions" / "p0_v1" / "iec")


def test_off_switch_matches_original_initialization_and_forward():
    graph = _graph()
    features = torch.randn(4, 5)
    torch.manual_seed(11)
    original = MCDHGN(2, 5, 4, 2, [2], 0, fusion_dim=3, **EXPERIMENTS["E1"])
    torch.manual_seed(11)
    switched = CalibratedMCDHGN(2, 5, 4, 2, [2], 0, fusion_dim=3, iec_mode="off", **EXPERIMENTS["E1"])
    assert list(original.state_dict()) == list(switched.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, switched.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(12)
    left = original( [graph, graph], features)
    torch.manual_seed(12)
    right = switched([graph, graph], features)
    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])


def test_on_switch_keeps_e1_weights_and_adds_only_iec():
    torch.manual_seed(13)
    original = MCDHGN(1, 4, 3, 2, [2], 0, fusion_dim=3, **EXPERIMENTS["E1"])
    torch.manual_seed(13)
    calibrated = CalibratedMCDHGN(
        1, 4, 3, 2, [2], 0, fusion_dim=3, iec_mode="on", mutation_width=2, **EXPERIMENTS["E1"]
    )
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, calibrated.state_dict()[key], rtol=0, atol=0)
    extra = [key for key in calibrated.state_dict() if key not in original.state_dict()]
    assert extra == ["iec.evidence.weight", "iec.evidence.bias", "iec.gate.weight", "iec.gate.bias"]
    graph = [_graph()]
    with pytest.raises(RuntimeError):
        calibrated(graph, torch.randn(4, 4))
    calibrated.bind_lengths(torch.tensor([20.0, 40.0, 80.0, 160.0]))
    encoded, logits, _, _ = calibrated(graph, torch.randn(4, 4))
    assert encoded.shape == (4, 6) and logits.shape == (4, 2)
    with pytest.raises(RuntimeError):
        CalibratedMCDHGN(1, 4, 3, 2, [2], 0, iec_mode="off").bind_lengths(torch.ones(4))


def test_length_changes_only_that_gene_and_gate_can_select_either_source():
    module = IntrinsicEvidenceCalibration(4, 2, mutation_width=1)
    with torch.no_grad():
        module.evidence.weight.zero_()
        module.evidence.bias.zero_()
        module.evidence.weight[0, 0] = 1
        module.gate.weight.zero_()
        module.gate.bias.fill_(-30)
    graph_h = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    raw = torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    short = torch.log(torch.tensor([2.0, 2.0])).reshape(-1, 1)
    mixed = torch.log(torch.tensor([2.0, 16.0])).reshape(-1, 1)
    closed_short = module(graph_h, raw, short)
    closed_mixed = module(graph_h, raw, mixed)
    torch.testing.assert_close(closed_short[0], closed_mixed[0])
    assert not torch.allclose(closed_short[1], closed_mixed[1])
    with torch.no_grad():
        module.gate.bias.fill_(30)
    opened = module(graph_h, raw, mixed)
    torch.testing.assert_close(opened, graph_h, atol=1e-5, rtol=1e-5)


def test_training_switch_restores_original_class():
    runner.dependencies()
    original = runner.MCDHGN
    from iec import use_training_model

    with use_training_model("on", lengths=torch.ones(3)):
        assert runner.MCDHGN is CalibratedMCDHGN
    assert runner.MCDHGN is original
    with use_training_model("off"):
        assert runner.MCDHGN is MCDHGN
    assert runner.MCDHGN is original


def test_smoke_training_is_isolated_and_switchable(tmp_path):
    output = ROOT / "results" / "iec_v1" / "_pytest_smoke"
    if output.exists():
        import shutil
        shutil.rmtree(output)
    script = ROOT / "scripts" / "run_iec.py"
    command = [
        sys.executable, str(script), "--iec", "on", "--smoke", "--cpu",
        "--output", str(output), "--seeds", "42", "--folds", "0",
    ]
    completed = __import__("subprocess").run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = output / "IEC" / "seed42" / "fold0" / "result.json"
    gate = output / "IEC" / "seed42" / "fold0" / "iec_gate.npz"
    assert result.is_file() and gate.is_file()
    import json
    saved = json.loads(result.read_text(encoding="utf-8"))
    assert saved["config"]["architecture"]["iec_mode"] == "on"
    assert saved["config"]["architecture"]["use_gatv2"] is True
    assert saved["config"]["protocol"] == "iec-v1-softmax-ap-checkpoint"
    assert not (ROOT / "results" / "p0_legacy_D0" / "IEC").exists()
    import shutil
    shutil.rmtree(output)
    del tmp_path
