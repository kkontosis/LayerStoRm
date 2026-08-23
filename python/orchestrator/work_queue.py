"""Priority work queue for the orchestrator main loop.

Heap-based priority queue with O(1) lookup by (request_id, layer_idx, operation)
and O(1) cancel-by-request. Items are heap-ordered by (-priority, insertion_seq)
for highest-priority-first with FIFO tiebreak. Only READY items are popped.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from orchestrator.types import WorkItem, WorkOperation, WorkStatus


class WorkQueue:

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, WorkItem]] = []
        self._index: dict[tuple[int, int, WorkOperation], WorkItem] = {}
        self._by_request: dict[int, list[WorkItem]] = defaultdict(list)
        self._seq = 0

    def insert(self, item: WorkItem) -> None:
        key = (item.request_id, item.layer_idx, item.operation)
        if key in self._index:
            raise ValueError(f"Duplicate work item: {key}")
        self._index[key] = item
        self._by_request[item.request_id].append(item)
        self._seq += 1
        heapq.heappush(self._heap, (-item.priority, self._seq, item))

    def pop(self) -> WorkItem | None:
        while self._heap:
            neg_pri, _seq, item = self._heap[0]
            key = (item.request_id, item.layer_idx, item.operation)
            if key not in self._index or item.status == WorkStatus.COMPLETED:
                heapq.heappop(self._heap)
                continue
            if -neg_pri != item.priority:
                heapq.heappop(self._heap)
                continue
            if item.status != WorkStatus.READY:
                heapq.heappop(self._heap)
                continue
            heapq.heappop(self._heap)
            return item
        return None

    def pop_n(self, n: int) -> list[WorkItem]:
        result: list[WorkItem] = []
        while len(result) < n:
            item = self.pop()
            if item is None:
                break
            result.append(item)
        return result

    def update_priority(self, item: WorkItem, priority: float) -> None:
        item.priority = priority
        self._seq += 1
        heapq.heappush(self._heap, (-priority, self._seq, item))

    def update_status(self, item: WorkItem, status: WorkStatus) -> None:
        item.status = status
        if status == WorkStatus.COMPLETED:
            key = (item.request_id, item.layer_idx, item.operation)
            self._index.pop(key, None)
        elif status == WorkStatus.READY:
            self._seq += 1
            heapq.heappush(self._heap, (-item.priority, self._seq, item))

    def cancel_by_request(self, request_id: int) -> int:
        items = self._by_request.pop(request_id, [])
        count = 0
        for item in items:
            key = (item.request_id, item.layer_idx, item.operation)
            if key in self._index:
                del self._index[key]
                item.status = WorkStatus.COMPLETED
                count += 1
        return count

    def pending_items(self) -> list[WorkItem]:
        return [item for item in self._index.values()
                if item.status < WorkStatus.DISPATCHED]

    def by_status(self, status: WorkStatus) -> list[WorkItem]:
        return [item for item in self._index.values()
                if item.status == status]

    def get(self, request_id: int, layer_idx: int,
            operation: WorkOperation) -> WorkItem | None:
        return self._index.get((request_id, layer_idx, operation))

    def __len__(self) -> int:
        return len(self._index)
