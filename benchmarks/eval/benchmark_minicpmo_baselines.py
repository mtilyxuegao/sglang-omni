# SPDX-License-Identifier: Apache-2.0
"""Record and score the MiniCPM-o interruption and greeting baseline trials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import soundfile
from pydantic import JsonValue

from benchmarks.duplex.artifacts import add_server_identity_args, server_identity
from benchmarks.duplex.client import MAX_TIMEOUT_S
from benchmarks.duplex.minicpmo_scores import (
    PROFILE,
    Reply,
    greeting_table,
    official_replies,
    run_passed,
    serving_table,
    sglang_replies,
    stop_table,
)
from benchmarks.duplex.minicpmo_trials import (
    TRIALS_FILE,
    read_trial_inputs,
    write_inputs,
)
from benchmarks.eval import benchmark_duplex
from benchmarks.eval.benchmark_duplex_serving import DEFAULT_PLAYBACK_RESERVE_MS

logger = logging.getLogger(__name__)

RUN_NAME_PATTERN = re.compile(r"(?P<trial_name>.+)_r\d+")


def inputs(arguments: argparse.Namespace) -> int:
    trial_inputs = write_inputs(arguments.clips, arguments.output)
    print(f"Wrote {len(trial_inputs)} trial inputs to {arguments.output}")
    return 0


def set_aside_failed_run(run_dir: Path) -> None:
    """Rename a run that failed the protocol check, so the next record call records it again."""
    failed_indexes = [
        int(failed_dir.name.rsplit("-", 1)[1])
        for failed_dir in run_dir.parent.glob(f"{run_dir.name}.failed-*")
    ]
    failed_dir = run_dir.with_name(
        f"{run_dir.name}.failed-{max(failed_indexes, default=0) + 1}"
    )
    run_dir.rename(failed_dir)
    logger.warning(
        f"{run_dir.name} failed the protocol check; kept as {failed_dir.name} and left to record again"
    )


async def record_rounds(
    arguments: argparse.Namespace, server: dict[str, JsonValue]
) -> None:
    """Record each trial once per round; keep finished runs and set failed ones aside to record again."""
    trial_inputs = read_trial_inputs(arguments.inputs)
    longest_trial_s = max(trial_input.duration_s for trial_input in trial_inputs)
    if not longest_trial_s < arguments.timeout <= MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout must exceed the {longest_trial_s} s longest trial and be at most {MAX_TIMEOUT_S} s"
        )
    else:
        pass
    for round_index in range(
        arguments.first_round, arguments.first_round + arguments.rounds
    ):
        for trial_input in trial_inputs:
            run_dir = arguments.output / f"{trial_input.name}_r{round_index}"
            input_samples, _ = soundfile.read(
                str(arguments.inputs / f"{trial_input.name}.wav"), dtype="int16"
            )
            if (
                hashlib.sha256(input_samples.tobytes()).hexdigest()
                != trial_input.sha256
            ):
                raise ValueError(
                    f"{trial_input.name}.wav no longer matches {TRIALS_FILE}"
                )
            elif (run_dir / "report.json").exists() and not run_passed(run_dir):
                set_aside_failed_run(run_dir)
            else:
                pass
            if (run_dir / "report.json").exists():
                recorded_sha256 = json.loads((run_dir / "manifest.json").read_text())[
                    "input"
                ]["sha256"]
                if recorded_sha256 != trial_input.sha256:
                    raise ValueError(f"{run_dir} was recorded from another input")
                else:
                    logger.info(f"Keeping recorded {run_dir.name}")
            elif run_dir.exists():
                raise FileExistsError(
                    f"{run_dir} is incomplete; remove it to record again"
                )
            else:
                # note (Jisen Li): run reads only these fields of its command line.
                await benchmark_duplex.run(
                    argparse.Namespace(
                        url=arguments.url,
                        output=run_dir,
                        profile=PROFILE,
                        timeout=arguments.timeout,
                    ),
                    input_samples.tobytes(),
                    server,
                )
                if run_passed(run_dir):
                    logger.info(f"Recorded {run_dir.name}")
                else:
                    set_aside_failed_run(run_dir)


def record(arguments: argparse.Namespace) -> int:
    server = server_identity(
        arguments.url,
        revision=arguments.server_revision,
        model=arguments.model,
        model_revision=arguments.model_revision,
        runtime=arguments.runtime,
    )
    asyncio.run(record_rounds(arguments, server))
    return 0


def report(arguments: argparse.Namespace) -> int:
    trial_inputs = read_trial_inputs(arguments.inputs)
    trial_names = {trial_input.name for trial_input in trial_inputs}
    replies_by_trial: dict[str, list[list[Reply]]] = {}
    scored_run_dirs: list[Path] = []
    skipped_names: list[str] = []
    if arguments.runs is not None:
        for run_dir in sorted(arguments.runs.iterdir()):
            run_name_match = RUN_NAME_PATTERN.fullmatch(run_dir.name)
            if (
                run_name_match is None
                or run_name_match["trial_name"] not in trial_names
                or not (run_dir / "report.json").exists()
                or not run_passed(run_dir)
            ):
                skipped_names.append(run_dir.name)
            else:
                scored_run_dirs.append(run_dir)
                replies_by_trial.setdefault(run_name_match["trial_name"], []).append(
                    sglang_replies(run_dir)
                )
    else:
        for log_path in sorted(arguments.official.glob("*.jsonl")):
            run_name_match = RUN_NAME_PATTERN.fullmatch(log_path.stem)
            if (
                run_name_match is None
                or run_name_match["trial_name"] not in trial_names
            ):
                skipped_names.append(log_path.name)
            else:
                replies_by_trial.setdefault(run_name_match["trial_name"], []).append(
                    official_replies(log_path)
                )
    print(stop_table(trial_inputs, replies_by_trial))
    print()
    print(greeting_table(trial_inputs, replies_by_trial))
    if scored_run_dirs:
        print()
        print(serving_table(scored_run_dirs, DEFAULT_PLAYBACK_RESERVE_MS / 1000))
    else:
        pass
    if skipped_names:
        print()
        print(
            f"Skipped {len(skipped_names)} runs that are unfinished, failed or not in {TRIALS_FILE}: {', '.join(skipped_names)}"
        )
    else:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inputs_parser = commands.add_parser(
        "inputs", help="Compose every trial's input WAV from the spoken clips"
    )
    inputs_parser.add_argument(
        "--clips",
        type=Path,
        required=True,
        help="Directory of trimmed PCM16 mono 16 kHz clips named as in the README",
    )
    inputs_parser.add_argument(
        "--output", type=Path, required=True, help="New inputs directory"
    )
    inputs_parser.set_defaults(handler=inputs)

    record_parser = commands.add_parser(
        "record", help="Record every trial once per round, one session at a time"
    )
    record_parser.add_argument("--inputs", type=Path, required=True)
    record_parser.add_argument("--url", required=True, help="Native /v1/realtime URL")
    record_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Runs directory; finished runs are kept, so a stopped recording resumes",
    )
    add_server_identity_args(record_parser)
    record_parser.add_argument("--rounds", type=int, default=5)
    record_parser.add_argument("--first-round", type=int, default=1)
    record_parser.add_argument(
        "--timeout", type=float, default=90.0, help="Per-session deadline in seconds"
    )
    record_parser.set_defaults(handler=record)

    report_parser = commands.add_parser(
        "report", help="Print the stop, greeting and serving tables"
    )
    report_parser.add_argument("--inputs", type=Path, required=True)
    sources = report_parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--runs", type=Path, help="Runs directory written by record")
    sources.add_argument(
        "--official",
        type=Path,
        help="Logs written by benchmark_minicpmo_official_loop for one configuration",
    )
    report_parser.set_defaults(handler=report)

    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
