"""CPU-only checks for the E1 learning-rate launcher."""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_e1_lr_sweep as launcher


class LauncherTests(unittest.TestCase):
    def make_baseline(self, project):
        source = project / "input.bin"
        source.write_bytes(b"frozen fixture")
        baseline = project / "results/p0_legacy_D0/E1/seed42"
        for name in ("ablation_config.py",):
            shutil.copy2(launcher.ROOT / name, project / name)
        import importlib.util
        spec = importlib.util.spec_from_file_location("fixture_config", project / "ablation_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = {name: index + 1 for index, name in enumerate(launcher.OPTIONS)}
        cfg.update(experiment="E1", seed=42, smoke=False, code_sha256={},
                   protocol="fixture", architecture=module.EXPERIMENTS["E1"])
        cfg["epochs"] = 1000
        cfg["lr"] = 0.01
        rows = {}
        for fold in range(5):
            dest = baseline / ("fold" + str(fold))
            dest.mkdir(parents=True)
            row = dict(config=dict(cfg, fold=fold), data_sha256={"input.bin": launcher.digest(source)},
                       environment={"fixture": True}, best_epoch=10 + fold, epochs_trained=100 + fold,
                       parameters=10, train_seconds=1,
                       validation=dict(ap=0.80 + fold / 1000, auc=0.85, macro_f1=0.75))
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

    def test_commands_cover_three_rates_without_test_or_scheduler(self):
        cfg = {name: index + 1 for index, name in enumerate(launcher.OPTIONS)}
        cfg["seed"] = 42
        cfg["lr"] = 0.01
        seen = []
        for lr_text in launcher.LEARNING_RATES:
            covered = []
            for gpu, folds in launcher.ASSIGNMENTS:
                covered.extend(folds)
                args = launcher.training_args({"config": cfg}, gpu, folds, Path("space in output"), lr_text)
                self.assertEqual(args[args.index("--experiments") + 1], "E1")
                self.assertEqual(args[args.index("--devices") + 1], str(gpu))
                self.assertEqual(args[args.index("--lr") + 1], lr_text)
                self.assertNotIn("--test-run", args)
                self.assertNotIn("--smoke", args)
                self.assertFalse(any("scheduler" in item.lower() for item in args))
                for key in launcher.OPTIONS:
                    if key == "lr":
                        continue
                    self.assertEqual(args[args.index("--" + key.replace("_", "-")) + 1], str(cfg[key]))
                seen.append((lr_text, tuple(folds), gpu))
            self.assertEqual(sorted(covered), list(range(5)))
        self.assertEqual(len(seen), 9)
        probe = launcher.training_args({"config": cfg}, 0, (0,), Path("probe"), "0.01", probe=True)
        self.assertEqual(probe[probe.index("--epochs") + 1], "2")
        self.assertEqual(probe[probe.index("--eval-every") + 1], "1")
        self.assertEqual(probe[probe.index("--patience") + 1], "0")
        self.assertEqual(probe[probe.index("--hidden-size") + 1], str(cfg["hidden_size"]))
        self.assertEqual(probe[probe.index("--experiments") + 1], "E1")

    def test_real_e1_arguments_round_trip(self):
        baseline = launcher.DEFAULT_BASELINE
        if not (baseline / "fold0" / "result.json").is_file():
            self.skipTest("real baseline absent")
        records = launcher.validate_baseline(baseline)
        reference = records[0][1]
        args = launcher.training_args(reference, 1, (1, 4), Path("out"), "0.005")
        for key in launcher.OPTIONS:
            text = args[args.index("--" + key.replace("_", "-")) + 1]
            expected = float("0.005") if key == "lr" else reference["config"][key]
            parsed = int(text) if key in launcher.INTEGER_OPTIONS else float(text)
            self.assertEqual(parsed, expected)

    def test_desktop_apps_do_not_block_and_python_training_does(self):
        rows = [
            r"12904, G:\soft\cursor\Cursor.exe, [N/A]",
            r"6656, C:\Windows\explorer.exe, [N/A]",
            r"9220, G:\LLLM\MCDHGN-main\.conda\python.exe, [N/A]",
        ]
        found = launcher.training_processes(rows)
        self.assertEqual(found, [rows[2]])
        self.assertEqual(launcher.training_processes(rows[:2]), [])

    def test_workers_execute_and_capture_all_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = [Path(temp) / (str(index) + ".log") for index in range(3)]
            commands = [[sys.executable, "-c", "print('worker %s')" % index] for index in range(3)]
            real_sleep = time.sleep
            with patch.object(launcher.time, "sleep", side_effect=lambda _: real_sleep(0.02)):
                launcher.run_workers(commands, logs, os.environ.copy())
            for index, log in enumerate(logs):
                self.assertIn("worker " + str(index), log.read_text(encoding="utf-8"))

    def test_failure_stops_other_worker_and_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "should_not_exist"
            commands = [
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                [sys.executable, "-c", "import time; from pathlib import Path; time.sleep(20); Path(%r).touch()" % str(marker)],
            ]
            real_sleep = time.sleep
            started = time.monotonic()
            with patch.object(launcher.time, "sleep", side_effect=lambda _: real_sleep(0.02)):
                with self.assertRaisesRegex(ValueError, "worker failed"):
                    launcher.run_workers(commands, [Path(temp) / "fail.log", Path(temp) / "slow.log"], os.environ.copy())
            self.assertLess(time.monotonic() - started, 10)
            self.assertFalse(marker.exists())

    def test_full_sweep_is_fresh_and_excludes_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            baseline, rows = self.make_baseline(project)
            output = project / "results" / "e1_lr_sweep_test"

            def fake_execute(argv, log, env):
                log.write_text("CPU test: step mocked\n", encoding="utf-8")
                if "--experiments" in argv:
                    self.assertEqual(argv[argv.index("--experiments") + 1], "E1")
                    self.assertEqual(argv[argv.index("--epochs") + 1], "2")
                    self.assertNotIn("--test-run", argv)
                    probe = output / "probe/E1/seed42/fold0"
                    probe.mkdir(parents=True)
                    (probe / "result.json").write_text("{}", encoding="utf-8")

            def fake_workers(commands, logs, env):
                self.assertEqual(len(commands), 3)
                rates = set()
                for gpu, (command, log) in enumerate(zip(commands, logs)):
                    self.assertEqual(command[command.index("--devices") + 1], str(gpu))
                    self.assertEqual(command[command.index("--experiments") + 1], "E1")
                    self.assertEqual(command[command.index("--epochs") + 1], "1000")
                    self.assertNotIn("--test-run", command)
                    lr_text = command[command.index("--lr") + 1]
                    rates.add(lr_text)
                    log.write_text("CPU test: worker mocked\n", encoding="utf-8")
                    folds = []
                    cursor = command.index("--folds") + 1
                    while cursor < len(command) and not command[cursor].startswith("--"):
                        folds.append(int(command[cursor]))
                        cursor += 1
                    run_root = Path(command[command.index("--output") + 1])
                    for fold in folds:
                        row = copy.deepcopy(rows[fold])
                        row["config"]["lr"] = float(lr_text)
                        row["validation"] = dict(row["validation"])
                        row["validation"]["ap"] = 0.90 - float(lr_text) + fold / 1000
                        if lr_text == "0.003" and fold == 2:
                            row["epochs_trained"] = row["config"]["epochs"]
                        dest = run_root / "E1" / "seed42" / ("fold" + str(fold))
                        dest.mkdir(parents=True)
                        (dest / "result.json").write_text(json.dumps(row), encoding="utf-8")
                        (dest / "config.json").write_text(json.dumps(row), encoding="utf-8")
                        (dest / "best.pt").write_bytes(b"checkpoint")
                        (dest / "history.jsonl").write_text("{}\n", encoding="utf-8")
                        (dest / "validation_predictions.npz").write_bytes(b"pred")
                self.assertEqual(len(rates), 1)

            calls = {"workers": 0}

            def counting_workers(commands, logs, env):
                calls["workers"] += 1
                fake_workers(commands, logs, env)

            with patch.object(launcher, "ROOT", project), \
                    patch.object(launcher, "DEFAULT_BASELINE", baseline), \
                    patch.object(sys, "argv", ["launcher", "--baseline", str(baseline), "--output", str(output)]), \
                    patch.object(launcher, "execute", side_effect=fake_execute), \
                    patch.object(launcher, "wait_for_idle_gpus", return_value=None), \
                    patch.object(launcher, "run_workers", side_effect=counting_workers):
                launcher.main()
            self.assertEqual(calls["workers"], 3)
            summary = launcher.read_json(output / "comparison/summary.json")
            self.assertEqual(summary["learning_rates"], ["0.01", "0.005", "0.003"])
            self.assertEqual(summary["epoch_cap"], [{
                "lr": "0.003", "fold": 2, "epochs_trained": rows[2]["config"]["epochs"],
                "best_epoch": rows[2]["best_epoch"], "epoch_limit": rows[2]["config"]["epochs"],
            }])
            self.assertEqual(summary["paired_vs_this_run_lr001"]["0.005"]["ap"]["wins"], 5)
            self.assertEqual(launcher.read_json(output / "run_status.json")["status"], "completed")
            self.assertTrue((output / "probe/E1/seed42/fold0/result.json").is_file())
            self.assertTrue(output.with_name(output.name + "_feedback.tar.gz").is_file())
            self.assertTrue((output / "train/lr0.01/E1/seed42/fold0/best.pt").is_file())
            with patch.object(launcher, "ROOT", project), \
                    patch.object(sys, "argv", ["launcher", "--baseline", str(baseline), "--output", str(output)]), \
                    patch.object(launcher, "execute", side_effect=fake_execute), \
                    patch.object(launcher, "wait_for_idle_gpus", return_value=None), \
                    patch.object(launcher, "run_workers", side_effect=counting_workers):
                with self.assertRaises(FileExistsError):
                    launcher.main()


if __name__ == "__main__":
    unittest.main()
