"""Tests for HTTP 409 server-side dedup handling in EngramClient.add_memory (R1-R4)."""
import time
from unittest.mock import Mock

import pytest
import requests

from consolidation.merge import (
    EngramClient,
    MergeExecutor,
    build_merge_op,
)
from consolidation.types import MemoryPoint

OLD = time.time() - 100 * 86400


def mp(idx, collection="engram_user"):
    return MemoryPoint(
        id=f"id-{idx}",
        collection=collection,
        vector=[1.0, 0.0],
        type="insight",
        content=f"content {idx}",
        importance=5.0,
        tags=[],
        source_type="reflection",
        created_at=OLD,
        metadata={},
    )


def make_op(cluster_id=0, collection="engram_user"):
    members = [mp(cluster_id * 10 + 0, collection), mp(cluster_id * 10 + 1, collection)]
    return build_merge_op(cluster_id, members, "auto_merge", "auto_merge", time.time())


def mock_response(status_code, json_body=None, text=""):
    r = Mock()
    r.status_code = status_code
    r.text = text
    if json_body is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = json_body

    def raise_for_status():
        if status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {status_code}")

    r.raise_for_status.side_effect = raise_for_status
    return r


class StubUndoLog:
    def build(self, ops):
        pass

    def write(self):
        pass

    def update_statuses(self, ops):
        pass


def test_add_memory_409_returns_existing_id(caplog):
    client = EngramClient("http://x")
    client.session.post = Mock(return_value=mock_response(
        409, {"status": "duplicate", "existing_id": "existing-abc", "similarity": 0.9876}))

    op = make_op()
    import logging
    with caplog.at_level(logging.INFO, logger="consolidation.merge"):
        result = client.add_memory("engram_user", op)

    assert result == "existing-abc"
    assert "existing-abc" in caplog.text
    assert "0.9876" in caplog.text


def test_add_memory_409_no_existing_id():
    client = EngramClient("http://x")
    client.session.post = Mock(return_value=mock_response(
        409, {"status": "duplicate"}))

    op = make_op()
    with pytest.raises(requests.exceptions.HTTPError):
        client.add_memory("engram_user", op)


def test_merge_executor_continues_on_dedup():
    client = EngramClient("http://x")

    responses = [
        mock_response(409, {"status": "duplicate", "existing_id": "existing-1", "similarity": 0.99}),
        mock_response(201, {"id": "fresh-2"}),
        mock_response(201, {"id": "fresh-3"}),
    ]
    client.session.post = Mock(side_effect=responses)
    client.session.get = Mock(return_value=mock_response(200, {"id": "any", "content": "x"}))
    client.session.delete = Mock(return_value=mock_response(204, {}))

    ops = [make_op(0), make_op(1), make_op(2)]
    executor = MergeExecutor(client, StubUndoLog(), verify=True)
    result = executor.run(ops)

    assert len(result.completed) == 3
    assert len(result.failed) == 0
    assert len(result.remaining) == 0
