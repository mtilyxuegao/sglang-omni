# SPDX-License-Identifier: Apache-2.0
"""Gather window of batched session hooks: an idle stage waits briefly for more appends."""

import queue
import time

import pytest

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.session import SessionScheduler
from tests.unit_test.scheduling.test_session_batch import (
    RecordingHooks,
    SequentialHooks,
    message,
    opens,
)

WINDOW_SECONDS = 0.03


class WindowHooks(RecordingHooks):
    gather_window_ms = WINDOW_SECONDS * 1000


class ScriptedInbox:
    """Inbox whose messages arrive at scripted times on a clock the test advances."""

    def __init__(self, arrivals: list[tuple[float, IncomingMessage]]) -> None:
        self.now_seconds = 0.0
        self.arrivals = arrivals
        self.timeouts: list[float] = []

    def get(self, timeout: float) -> IncomingMessage:
        self.timeouts.append(timeout)
        if self.arrivals and self.arrivals[0][0] <= self.now_seconds + timeout:
            self.now_seconds = max(self.now_seconds, self.arrivals[0][0])
            return self.arrivals.pop(0)[1]
        else:
            self.now_seconds += timeout
            raise queue.Empty

    def get_nowait(self) -> IncomingMessage:
        if self.arrivals and self.arrivals[0][0] <= self.now_seconds:
            return self.arrivals.pop(0)[1]
        else:
            raise queue.Empty

    def empty(self) -> bool:
        return not self.arrivals


def run_batches(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: SessionScheduler,
    session_ids: str,
    arrivals: list[tuple[str, float]],
) -> list[list[str]]:
    """Open the sessions, then collect and run batches until every scripted arrival ran."""
    scripted: list[tuple[float, IncomingMessage]] = []
    for queued in opens(*session_ids):
        scheduler.register_operation(queued)
        scheduler.compute(queued.data)
    for request_id, seconds in arrivals:
        operation, _, session_id = request_id.rpartition("-")
        if operation:
            queued = message(request_id, operation, session_id)
        else:
            queued = message(request_id, "append", request_id[0])
        scheduler.register_operation(queued)
        scripted.append((seconds, queued))
    scheduler.inbox = inbox = ScriptedInbox(scripted)
    monkeypatch.setattr(time, "monotonic", lambda: inbox.now_seconds)
    batches: list[list[str]] = []
    while not inbox.empty():
        inbox.now_seconds = max(inbox.now_seconds, inbox.arrivals[0][0])
        batch = scheduler.collect_batch(inbox.get_nowait())
        scheduler.compute_batch([queued.data for queued in batch])
        batches.append([queued.request_id for queued in batch])
    return batches


@pytest.mark.parametrize(
    "hooks_class, session_ids, arrivals, batches, waits",
    [
        (WindowHooks, "abc", [("a0", 0), ("b0", 0.005)], [["a0", "b0"]], 2),
        (WindowHooks, "abc", [("a0", 0), ("b0", 0.05)], [["a0"], ["b0"]], 2),
        (WindowHooks, "ab", [("open-c", 0), ("a0", 0)], [["open-c", "a0"]], 0),
        (WindowHooks, "ab", [("close-b", 0), ("a0", 0)], [["close-b", "a0"]], 0),
        (WindowHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0", "b0"]], 1),
        (SequentialHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0"], ["b0"]], 0),
        (RecordingHooks, "ab", [("a0", 0), ("b0", 0.005)], [["a0"], ["b0"]], 0),
    ],
    ids=[
        "appends-inside-the-window-share-one-call",
        "append-after-the-window-runs-in-a-second-call",
        "open-is-not-held",
        "close-is-not-held",
        "wait-ends-once-every-open-session-has-an-append",
        "unbatched-hooks-never-wait",
        "zero-window-never-waits",
    ],
)
def test_idle_stage_gathers_appends_within_the_window(
    monkeypatch, hooks_class, session_ids, arrivals, batches, waits
):
    scheduler = SessionScheduler(hooks_class(), max_concurrency=1)
    assert run_batches(monkeypatch, scheduler, session_ids, arrivals) == batches
    assert len(scheduler.inbox.timeouts) == waits


def test_backlogged_stage_never_waits(monkeypatch):
    scheduler = SessionScheduler(WindowHooks())
    scheduler.is_backlogged = True
    arrivals = [("a0", 0), ("b0", 0), ("c0", 0.005)]
    batches = run_batches(monkeypatch, scheduler, "abc", arrivals)
    assert batches == [["a0", "b0"], ["c0"]]
    assert scheduler.inbox.timeouts == []
