"""Tests for orchestrator.work_queue — priority queue with status filtering."""

import pytest

from orchestrator.types import (
    ExpertKey,
    WorkItem,
    WorkOperation,
    WorkStatus,
)
from orchestrator.work_queue import WorkQueue


def _item(request_id: int = 1, layer: int = 0,
          op: WorkOperation = WorkOperation.ATTENTION,
          priority: float = 0.0,
          status: WorkStatus = WorkStatus.READY,
          **kwargs) -> WorkItem:
    return WorkItem(
        request_id=request_id, layer_idx=layer, operation=op,
        priority=priority, status=status, **kwargs,
    )


class TestInsertPop:
    def test_pop_highest_priority(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=1.0))
        q.insert(_item(request_id=2, layer=0, priority=3.0))
        q.insert(_item(request_id=3, layer=0, priority=2.0))
        item = q.pop()
        assert item is not None
        assert item.request_id == 2
        item = q.pop()
        assert item is not None
        assert item.request_id == 3
        item = q.pop()
        assert item is not None
        assert item.request_id == 1

    def test_fifo_tiebreak(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=5.0))
        q.insert(_item(request_id=2, layer=0, priority=5.0))
        q.insert(_item(request_id=3, layer=0, priority=5.0))
        assert q.pop().request_id == 1
        assert q.pop().request_id == 2
        assert q.pop().request_id == 3

    def test_pop_empty(self):
        q = WorkQueue()
        assert q.pop() is None

    def test_pop_n(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=1.0))
        q.insert(_item(request_id=2, layer=0, priority=3.0))
        q.insert(_item(request_id=3, layer=0, priority=2.0))
        items = q.pop_n(2)
        assert len(items) == 2
        assert items[0].request_id == 2
        assert items[1].request_id == 3

    def test_pop_n_fewer_available(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=1.0))
        items = q.pop_n(5)
        assert len(items) == 1


class TestStatusFiltering:
    def test_only_ready_items_popped(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=10.0,
                        status=WorkStatus.PENDING))
        q.insert(_item(request_id=2, layer=0, priority=1.0,
                        status=WorkStatus.READY))
        item = q.pop()
        assert item is not None
        assert item.request_id == 2

    def test_pending_not_popped(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=5.0,
                        status=WorkStatus.PENDING))
        assert q.pop() is None

    def test_dispatched_not_popped(self):
        q = WorkQueue()
        item = _item(request_id=1, layer=0, priority=5.0,
                     status=WorkStatus.READY)
        q.insert(item)
        q.update_status(item, WorkStatus.DISPATCHED)
        assert q.pop() is None


class TestUpdatePriority:
    def test_reorders_pop(self):
        q = WorkQueue()
        a = _item(request_id=1, layer=0, priority=1.0)
        b = _item(request_id=2, layer=0, priority=5.0)
        q.insert(a)
        q.insert(b)
        q.update_priority(a, 10.0)
        assert q.pop().request_id == 1
        assert q.pop().request_id == 2


class TestUpdateStatus:
    def test_pending_to_ready_makes_poppable(self):
        q = WorkQueue()
        item = _item(request_id=1, layer=0, priority=5.0,
                     status=WorkStatus.PENDING)
        q.insert(item)
        assert q.pop() is None
        q.update_status(item, WorkStatus.READY)
        popped = q.pop()
        assert popped is not None
        assert popped.request_id == 1

    def test_completed_removes_from_index(self):
        q = WorkQueue()
        item = _item(request_id=1, layer=0, priority=5.0)
        q.insert(item)
        assert len(q) == 1
        q.update_status(item, WorkStatus.COMPLETED)
        assert len(q) == 0
        assert q.get(1, 0, WorkOperation.ATTENTION) is None


class TestCancelByRequest:
    def test_cancels_all_items(self):
        q = WorkQueue()
        q.insert(_item(request_id=42, layer=0, op=WorkOperation.ATTENTION))
        q.insert(_item(request_id=42, layer=0, op=WorkOperation.GATING))
        q.insert(_item(request_id=42, layer=1, op=WorkOperation.ATTENTION))
        q.insert(_item(request_id=99, layer=0, op=WorkOperation.ATTENTION))
        count = q.cancel_by_request(42)
        assert count == 3
        assert len(q) == 1
        assert q.pop().request_id == 99

    def test_cancel_nonexistent(self):
        q = WorkQueue()
        assert q.cancel_by_request(999) == 0

    def test_cancelled_items_skipped_on_pop(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=10.0))
        q.insert(_item(request_id=2, layer=0, priority=1.0))
        q.cancel_by_request(1)
        item = q.pop()
        assert item is not None
        assert item.request_id == 2


class TestLookup:
    def test_get_existing(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=5, op=WorkOperation.EXPERT_FFN,
                        priority=3.0))
        item = q.get(1, 5, WorkOperation.EXPERT_FFN)
        assert item is not None
        assert item.priority == 3.0

    def test_get_nonexistent(self):
        q = WorkQueue()
        assert q.get(1, 0, WorkOperation.ATTENTION) is None

    def test_duplicate_raises(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0))
        with pytest.raises(ValueError, match="Duplicate"):
            q.insert(_item(request_id=1, layer=0))


class TestLen:
    def test_tracks_insertions(self):
        q = WorkQueue()
        assert len(q) == 0
        q.insert(_item(request_id=1, layer=0))
        assert len(q) == 1
        q.insert(_item(request_id=2, layer=0))
        assert len(q) == 2

    def test_pop_does_not_change_len(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0))
        q.pop()
        assert len(q) == 1

    def test_cancel_reduces_len(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0))
        q.cancel_by_request(1)
        assert len(q) == 0


class TestPendingItems:
    def test_returns_pending_and_waiting(self):
        q = WorkQueue()
        a = _item(request_id=1, layer=0, status=WorkStatus.PENDING)
        b = _item(request_id=2, layer=0, status=WorkStatus.WAITING_TRANSFER)
        c = _item(request_id=3, layer=0, status=WorkStatus.READY)
        d = _item(request_id=4, layer=0, status=WorkStatus.DISPATCHED)
        q.insert(a)
        q.insert(b)
        q.insert(c)
        q.insert(d)
        q.update_status(d, WorkStatus.DISPATCHED)
        pending = q.pending_items()
        ids = {item.request_id for item in pending}
        assert ids == {1, 2, 3}

    def test_by_status(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, status=WorkStatus.PENDING))
        q.insert(_item(request_id=2, layer=0, status=WorkStatus.READY))
        q.insert(_item(request_id=3, layer=0, status=WorkStatus.READY))
        ready = q.by_status(WorkStatus.READY)
        assert len(ready) == 2


class TestMixedOperations:
    def test_insert_pop_cancel_insert(self):
        q = WorkQueue()
        q.insert(_item(request_id=1, layer=0, priority=5.0))
        q.insert(_item(request_id=2, layer=0, priority=3.0))
        assert q.pop().request_id == 1

        q.cancel_by_request(2)
        assert q.pop() is None

        q.insert(_item(request_id=3, layer=0, priority=1.0))
        assert q.pop().request_id == 3
        assert len(q) == 2  # req 1 + req 3 still in index (pop doesn't remove)

    def test_full_lifecycle(self):
        q = WorkQueue()
        item = _item(request_id=1, layer=5, op=WorkOperation.EXPERT_FFN,
                     priority=2.0, status=WorkStatus.PENDING)
        q.insert(item)
        assert q.pop() is None

        q.update_status(item, WorkStatus.WAITING_TRANSFER)
        assert q.pop() is None

        q.update_status(item, WorkStatus.READY)
        popped = q.pop()
        assert popped is item

        q.update_status(item, WorkStatus.DISPATCHED)
        q.update_status(item, WorkStatus.COMPLETED)
        assert len(q) == 0
