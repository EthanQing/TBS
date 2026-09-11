import csv
import time

from train_platform.domains.training.execution_paths import prepare_ultralytics_resume_output


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


def test_carries_best_and_prefers_checkpoint_history(tmp_path):
    source, target = tmp_path / "legacy", tmp_path / "current" / "output"
    source_best = source / "weights" / "best.pt"
    source_best.parent.mkdir(parents=True)
    source_best.write_bytes(b"historical best")
    _write_csv(source / "results.csv", [["epoch", "metric"], ["1", "0.1"], ["2", "99"], ["3", "100"]])
    source_csv = (source / "results.csv").read_bytes()

    prepare_ultralytics_resume_output(
        source, target, checkpoint_epoch=1, train_results={"epoch": [1, 2], "metric": [0.1, 0.2]}
    )

    assert (target / "weights" / "best.pt").read_bytes() == b"historical best"
    assert source_best.read_bytes() == b"historical best"
    assert (source / "results.csv").read_bytes() == source_csv
    assert _read_csv(target / "results.csv") == [["epoch", "metric"], ["1", "0.1"], ["2", "0.2"]]


def test_falls_back_to_source_csv_and_cuts_off_checkpoint_epoch(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_csv(
        source / "results.csv",
        [[" epoch", "time", "metric"], ["1", "1", "0.1"], ["2", "2", "0.2"], ["3", "3", "0.3"]],
    )
    source_csv = (source / "results.csv").read_bytes()

    prepare_ultralytics_resume_output(source, target, checkpoint_epoch=1, train_results={})

    assert _read_csv(target / "results.csv") == [
        [" epoch", "time", "metric"], ["1", "1", "0.1"], ["2", "2", "0.2"]
    ]
    assert (source / "results.csv").read_bytes() == source_csv


def test_retry_preserves_new_best_and_numeric_equivalent_csv(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "weights").mkdir(parents=True)
    (target / "weights").mkdir(parents=True)
    (source / "weights" / "best.pt").write_bytes(b"old")
    (target / "weights" / "best.pt").write_bytes(b"new")
    _write_csv(target / "results.csv", [["epoch", "metric"], ["1.0", "0.100000"]])
    before = (target / "results.csv").stat().st_mtime_ns

    prepare_ultralytics_resume_output(
        source, target, checkpoint_epoch=0, train_results={"epoch": [1], "metric": [0.1]}
    )

    assert (target / "weights" / "best.pt").read_bytes() == b"new"
    assert (target / "results.csv").stat().st_mtime_ns == before


def test_disjoint_target_history_is_not_merged(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_csv(target / "results.csv", [["epoch", "metric"], ["1", "0.1"]])

    prepare_ultralytics_resume_output(
        source, target, checkpoint_epoch=1, train_results={"epoch": [2], "metric": [None]}
    )

    assert _read_csv(target / "results.csv") == [["epoch", "metric"], ["2", ""]]


def test_conflicting_target_history_is_replaced(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_csv(target / "results.csv", [["epoch", "metric"], ["1", "9"]])

    prepare_ultralytics_resume_output(
        source, target, checkpoint_epoch=0, train_results={"epoch": [1], "metric": [0.1]}
    )

    assert _read_csv(target / "results.csv") == [["epoch", "metric"], ["1", "0.1"]]


def test_missing_source_history_only_filters_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_csv(target / "results.csv", [["epoch", "metric"], ["1", "0.1"], ["2", "0.2"], ["3", "0.3"]])

    prepare_ultralytics_resume_output(source, target, checkpoint_epoch=1, train_results=None)

    assert _read_csv(target / "results.csv") == [["epoch", "metric"], ["1", "0.1"], ["2", "0.2"]]


def test_same_output_is_noop(tmp_path):
    output = tmp_path / "output"
    _write_csv(output / "results.csv", [["epoch", "metric"], ["1", "0.1"], ["2", "0.2"]])
    before = (output / "results.csv").read_bytes()

    prepare_ultralytics_resume_output(output, output, checkpoint_epoch=0, train_results=None)

    assert (output / "results.csv").read_bytes() == before


def test_ultralytics_appends_after_imported_history(tmp_path):
    from ultralytics.engine.trainer import BaseTrainer

    source, target = tmp_path / "source", tmp_path / "target"
    prepare_ultralytics_resume_output(
        source,
        target,
        checkpoint_epoch=1,
        train_results={"epoch": [1, 2], "time": [0.5, 1.0], "metric": [0.1, 0.2]},
    )
    trainer = object.__new__(BaseTrainer)
    trainer.csv = target / "results.csv"
    trainer.epoch = 2
    trainer.train_time_start = time.time()

    trainer.save_metrics({"metric": 0.3})

    assert trainer.read_results_csv()["epoch"] == [1, 2, 3]
    assert trainer.read_results_csv()["metric"] == [0.1, 0.2, 0.3]
