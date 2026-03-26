#!/usr/bin/env python3
"""
Simple SLURM job counter.

Counts queued and running jobs via `squeue`, and finished jobs via `sacct`.

Usage examples:
  - Single snapshot (default user, since today):
      python utils/slurm_job_counts.py

  - Specify user and include additional finished states:
      python utils/slurm_job_counts.py --user alice \
          --finished-states COMPLETED,FAILED,CANCELLED,TIMEOUT

  - Refresh every 30 seconds:
      python utils/slurm_job_counts.py --interval 30

Alternatively, you can use the shell `watch` command:
  watch -n 30 python utils/slurm_job_counts.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple


def _check_cmd_available(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def _run_command(command: List[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{command[0]} not found"


def _parse_array_count_from_jobid(job_id: str) -> int:
    """Estimate number of tasks represented by a job_id, handling array ranges.

    Examples of aggregated array IDs from squeue:
      - 12345_[1-100]
      - 12345_[1-100:2]
      - 12345_[1,3,5-10]
      - 12345_[1-100%10]   ("%10" throttle is ignored for counting)

    If no aggregated range is present, returns 1.
    """
    # Fast path: no bracket means a single job or a single array task like 12345_7
    if "[" not in job_id or "]" not in job_id:
        return 1

    try:
        inside = job_id.split("[", 1)[1].split("]", 1)[0]
    except Exception:
        return 1

    # Drop any throttle suffix like %10
    inside_no_throttle = inside.split("%", 1)[0]

    # Support comma-separated items: single index, or start-end[:step]
    total = 0
    for part in inside_no_throttle.split(","):
        token = part.strip()
        if not token:
            continue
        # Single number
        if token.isdigit():
            total += 1
            continue
        # Range with optional :step
        # Forms: start-end or start-end:step
        # All are integers.
        try:
            if ":" in token:
                range_part, step_part = token.split(":", 1)
                step = int(step_part)
            else:
                range_part, step = token, 1
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if step <= 0:
                step = 1
            if end < start:
                # Invalid or descending range; skip
                continue
            count = ((end - start) // step) + 1
            total += max(count, 0)
        except Exception:
            # Fallback if unexpected token
            total += 1
    return max(total, 1)


def _count_states_squeue(user: str) -> Counter:
    """Return counts of job states from squeue for the given user.

    This function accounts for SLURM job array aggregation by expanding
    ranges in job IDs like 12345_[1-100:2] to estimate the number of tasks.
    """
    if not _check_cmd_available("squeue"):
        return Counter()

    # Include job id and state so we can expand array ranges.
    cmd = [
        "squeue",
        "-u",
        user,
        "-h",  # no header
        "-o",
        "%i|%T",
    ]
    code, out, err = _run_command(cmd)
    if code != 0:
        return Counter()

    counts: Counter = Counter()
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        job_id, state = line.split("|", 1)
        job_id = job_id.strip()
        state = state.strip()
        tasks = _parse_array_count_from_jobid(job_id)
        counts[state] += tasks
    return counts


def _parse_sacct_lines(lines: Iterable[str]) -> Dict[str, str]:
    """Parse sacct -P output into a mapping of job_id_root -> state.

    sacct may output multiple rows per job (for job steps). Job IDs can be like
    12345, 12345.batch, 12345.0, etc. We reduce to the root part before any dot.
    We keep the last non-empty state seen for the root.
    """
    job_to_state: Dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id_raw, state = parts[0].strip(), parts[1].strip()
        job_id_root = job_id_raw.split(".")[0]
        if job_id_root:
            job_to_state[job_id_root] = state
    return job_to_state


def _count_finished_sacct(user: str, since: str, finished_states: Set[str]) -> Dict[str, int]:
    """Count finished jobs using sacct for the given user since a time point.

    By default, `since` can be 'today' (SLURM understands it) or a
    YYYY-MM-DD[THH:MM:SS] string. `finished_states` controls which terminal
    states are considered "finished".
    """
    if not _check_cmd_available("sacct"):
        return {state: 0 for state in finished_states}

    cmd = [
        "sacct",
        "-X",  # don't include jobs from other clusters
        "-P",  # pipe-delimited for robust parsing
        "-n",  # no header
        "-u",
        user,
        "-S",
        since,
        "--format",
        "JobID,State",
    ]
    code, out, err = _run_command(cmd)
    if code != 0:
        return {state: 0 for state in finished_states}

    job_to_state = _parse_sacct_lines(out.splitlines())
    
    counts = {state: 0 for state in finished_states}
    for state in job_to_state.values():
        # Handle cases where state has a suffix like 'CANCELLED by 12345' or 'FAILED (Non-zero exit code)'
        base_state = state.split()[0] if state else ""
        if base_state in finished_states:
            counts[base_state] += 1
            
    return counts


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count SLURM jobs for a user.")
    parser.add_argument(
        "--user",
        default=getpass.getuser(),
        help="Username to query (default: current user)",
    )
    parser.add_argument(
        "--since",
        default="today",
        help=(
            "Lower bound for finished jobs in sacct (default: 'today'). "
            "Accepts SLURM times like YYYY-MM-DD or 'now-2hours' depending on cluster."
        ),
    )
    parser.add_argument(
        "--finished-states",
        default="COMPLETED,FAILED",
        help=(
            "Comma-separated set of terminal states to count as finished. "
            "Default: COMPLETED,FAILED. Examples: COMPLETED,FAILED,CANCELLED,TIMEOUT"
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Refresh interval in seconds (0 = run once and exit).",
    )
    return parser.parse_args(argv)


def _print_counts(user: str, since: str, finished_states: Set[str]) -> None:
    squeue_counts = _count_states_squeue(user)
    queued = squeue_counts.get("PENDING", 0)
    running = squeue_counts.get("RUNNING", 0)

    finished_counts = _count_finished_sacct(user=user, since=since, finished_states=finished_states)
    
    # Calculate total finished for backward compatibility in the output string
    # or just show them individually
    completed = finished_counts.get("COMPLETED", 0)
    failed = finished_counts.get("FAILED", 0)
    
    # Format the counts to show
    counts_str = f"queued={queued}  running={running}  completed={completed}  failed={failed}"
    
    # Add any other states that were explicitly requested
    for state, count in finished_counts.items():
        if state not in ("COMPLETED", "FAILED") and count > 0:
            counts_str += f"  {state.lower()}={count}"

    print(
        f"{_now_str()}  user={user}  {counts_str}  "
        f"(since={since})"
    )


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    finished_states = {s.strip().upper() for s in ns.finished_states.split(",") if s.strip()}

    if ns.interval and ns.interval > 0:
        try:
            while True:
                _print_counts(user=ns.user, since=ns.since, finished_states=finished_states)
                time.sleep(ns.interval)
        except KeyboardInterrupt:
            return 0
    else:
        _print_counts(user=ns.user, since=ns.since, finished_states=finished_states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


