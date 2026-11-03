"""CPU-only checks for launcher commands and process failure handling."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import time
import unittest
from unittest.mock import patch

import run_e3_parallel as launcher


class LauncherTests(unittest.TestCase):
    def make_baseline(self, project):
        source = project / "input.bin"
        source.write_bytes(b"frozen fixture")
        baseline = project / "results/p0_legacy_D0/E1/seed42"
        for name in ("ablation_config.py", "summarize_ablations.py"):
            shutil.copy2(launcher.ROOT / name, project / name)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fixture_config", project / "ablation_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = {name: i + 1 for i, name in enumerate(launcher.OPTIONS)}
        cfg.update(experiment="E1", seed=42, smoke=False, code_sha256={},
                   protocol="fixture", architecture=module.EXPERIMENTS["E1"])
        rows = {}
        for fold in range(5):
            dest = baseline / ("fold" + str(fold))
            dest.mkdir(parents=True)
            row = dict(config=dict(cfg, fold=fold), data_sha256={"input.bin": launcher.digest(source)},
                       environment={"fixture": True}, best_epoch=1, parameters=10,
                       train_seconds=1, validation=dict(ap=.8, auc=.85, accuracy=.8, macro_f1=.75))
            for name in ("result.json", "config.json"):
                (dest / name).write_text(json.dumps(row), encoding="utf-8")
            for name in ("history.jsonl", "validation_predictions.npz"):
                (dest / name).touch()
            rows[fold] = row
        return baseline, rows

    def test_baseline_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            baseline, _ = self.make_baseline(project)
            with patch.object(launcher, "ROOT", project):
                self.assertEqual(set(launcher.validate_baseline(baseline)), set(range(5)))
                (project / "input.bin").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    launcher.validate_baseline(baseline)

    def test_full_orchestration_selects_only_seed42_and_excludes_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            baseline, rows = self.make_baseline(project)
            # Other seeds/experiments exist alongside the selected baseline.
            for relative in ("E1/seed43/fold0", "E2/seed42/fold0"):
                sibling = project / "results/p0_legacy_D0" / relative
                sibling.mkdir(parents=True)
                (sibling / "result.json").write_text("{}", encoding="utf-8")
            output = project / "results/new_E3"
            real_execute = launcher.execute

            def fake_execute(argv, log, env):
                if Path(argv[1]).name == "summarize_ablations.py":
                    # Run the real summarizer, with exactly the formal and reference roots.
                    self.assertEqual(argv[2:4], [str(output / "reference_E1"), str(output / "train")])
                    real_execute(argv, log, env)
                else:
                    log.write_text("CPU test: GPU step mocked", encoding="utf-8")
                    if "--experiments" in argv:
                        self.assertEqual(argv[argv.index("--experiments") + 1], "E3")
                        self.assertEqual(argv[argv.index("--epochs") + 1], "2")
                        probe = output / "probe/E3/seed42/fold0"
                        probe.mkdir(parents=True)
                        (probe / "result.json").write_text("{}", encoding="utf-8")

            def fake_workers(commands, logs, env):
                self.assertEqual(len(commands), 3)
                for gpu, (command, log) in enumerate(zip(commands, logs)):
                    self.assertEqual(command[command.index("--devices") + 1], str(gpu))
                    log.write_text("CPU test: worker mocked", encoding="utf-8")
                for fold, baseline_row in rows.items():
                    row = copy.deepcopy(baseline_row)
                    row["config"]["experiment"] = "E3"
                    row["config"]["architecture"]["fusion_mode"] = "cross"
                    row["validation"]["ap"] += .01
                    dest = output / "train/E3/seed42" / ("fold" + str(fold))
                    dest.mkdir(parents=True)
                    (dest / "result.json").write_text(json.dumps(row), encoding="utf-8")

            with patch.object(launcher, "ROOT", project), \
                    patch.object(launcher, "DEFAULT_BASELINE", baseline), \
                    patch.object(sys, "argv", ["launcher", "--output", str(output)]), \
                    patch.object(launcher, "execute", side_effect=fake_execute), \
                    patch.object(launcher, "run_workers", side_effect=fake_workers):
                launcher.main()
            summary = launcher.read_json(output / "comparison/summary.json")
            self.assertEqual(set(summary["experiments"]), {"E1", "E3"})
            self.assertEqual(summary["experiments"]["E3"]["coverage"], {"42": list(range(5))})
            delta = summary["paired_contrasts"]["cross_view_with_global_weights"]["metrics"]["ap"]
            self.assertAlmostEqual(delta["mean_delta_pp"], 1.0)
            self.assertEqual(delta["wins"], 5)
            self.assertEqual(launcher.read_json(output / "run_status.json")["status"], "completed")
            self.assertTrue(output.with_name(output.name + "_feedback.tar.gz").is_file())

    def test_commands_preserve_e1_options_and_exclusive_gpu_assignment(self):
        cfg = {name: i + 1 for i, name in enumerate(launcher.OPTIONS)}
        cfg["seed"] = 42
        covered = []
        for gpu, folds in launcher.ASSIGNMENTS:
            covered.extend(folds)
            args = launcher.training_args({"config": cfg}, gpu, folds, Path("space in output"))
            self.assertEqual(args[args.index("--devices") + 1], str(gpu))
            self.assertEqual(args[args.index("--experiments") + 1], "E3")
            for key in launcher.OPTIONS:
                self.assertEqual(args[args.index("--" + key.replace("_", "-")) + 1], str(cfg[key]))
        self.assertEqual(sorted(covered), list(range(5)))
        probe = launcher.training_args({"config": cfg}, 0, (0,), Path("probe"), probe=True)
        self.assertEqual(probe[probe.index("--epochs") + 1], "2")
        self.assertEqual(probe[probe.index("--hidden-size") + 1], str(cfg["hidden_size"]))

    def test_workers_execute_and_capture_all_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = [Path(temp) / (str(i) + ".log") for i in range(3)]
            commands = [[sys.executable, "-c", "print('worker {}')".format(i)] for i in range(3)]
            real_sleep = time.sleep
            with patch.object(launcher.time, "sleep", side_effect=lambda _: real_sleep(0.02)):
                launcher.run_workers(commands, logs, os.environ.copy())
            for i, log in enumerate(logs):
                self.assertIn("worker " + str(i), log.read_text())

    def test_failure_stops_other_worker_and_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "should_not_exist"
            commands = [
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                [sys.executable, "-c", "import time; from pathlib import Path; time.sleep(20); Path({!r}).touch()".format(str(marker))],
            ]
            real_sleep = time.sleep
            started = time.monotonic()
            with patch.object(launcher.time, "sleep", side_effect=lambda _: real_sleep(0.02)):
                with self.assertRaisesRegex(ValueError, "worker failed"):
                    launcher.run_workers(commands, [Path(temp) / "fail.log", Path(temp) / "slow.log"], os.environ.copy())
            self.assertLess(time.monotonic() - started, 10)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
