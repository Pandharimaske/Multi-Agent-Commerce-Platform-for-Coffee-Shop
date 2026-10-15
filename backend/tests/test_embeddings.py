"""Tests for src/utils/embeddings.py (a fake client stands in for NVIDIA NIM; no network needed)."""
import math
from types import SimpleNamespace

import pytest

from src.utils.embeddings import NimEmbeddings


class FakeClient:
    """Mimics openai.OpenAI().embeddings.create and records every call."""

    def __init__(self, native_dim=8, reject_dimensions=False, fail_times=0):
        self.native_dim = native_dim
        self.reject_dimensions = reject_dimensions
        self.fail_times = fail_times
        self.calls = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("503 service unavailable")
        if "dimensions" in kwargs and self.reject_dimensions:
            raise RuntimeError("400 Bad Request: this model does not support the 'dimensions' parameter")
        dim = kwargs.get("dimensions") or self.native_dim
        # return items out of order on purpose, to prove the index is respected
        items = [SimpleNamespace(index=i, embedding=[float(i + 1)] * dim) for i, _ in enumerate(kwargs["input"])]
        return SimpleNamespace(data=list(reversed(items)))


def make(client, **kw):
    kw.setdefault("retry_delay", 0)
    return NimEmbeddings(model="m", client=client, **kw)


def test_query_and_document_input_types_differ():
    c = FakeClient()
    e = make(c)
    e.embed_query("hello")
    e.embed_documents(["a", "b"])
    assert c.calls[0]["extra_body"]["input_type"] == "query"
    assert c.calls[1]["extra_body"]["input_type"] == "passage"
    assert c.calls[0]["encoding_format"] == "float"


def test_documents_are_batched_and_order_is_preserved():
    c = FakeClient()
    e = make(c, batch_size=3)
    out = e.embed_documents([f"t{i}" for i in range(7)])
    assert len(out) == 7 and len(c.calls) == 3                      # 3 + 3 + 1
    assert [len(call["input"]) for call in c.calls] == [3, 3, 1]
    assert out[0][0] == 1.0 and out[1][0] == 2.0 and out[2][0] == 3.0  # index order restored within a batch


def test_empty_input():
    c = FakeClient()
    assert make(c).embed_documents([]) == [] and c.calls == []


def test_requested_dimensions_are_sent():
    c = FakeClient(native_dim=2048)
    out = make(c, dimensions=768, expected_dim=768).embed_query("x")
    assert c.calls[0]["dimensions"] == 768 and len(out) == 768


def test_model_that_rejects_dimensions_is_retried_without_it_and_remembered():
    c = FakeClient(native_dim=768, reject_dimensions=True)
    e = make(c, dimensions=768, expected_dim=768)
    assert len(e.embed_query("x")) == 768
    assert "dimensions" in c.calls[0] and "dimensions" not in c.calls[1]
    e.embed_query("y")
    assert "dimensions" not in c.calls[2]                            # not sent again


def test_wrong_size_raises_a_helpful_error():
    c = FakeClient(native_dim=1024, reject_dimensions=True)
    with pytest.raises(ValueError, match="1024 dimensions but the database expects 768"):
        make(c, dimensions=768, expected_dim=768).embed_query("x")


def test_truncation_is_opt_in_and_renormalises():
    c = FakeClient(native_dim=1024, reject_dimensions=True)
    out = make(c, expected_dim=768, allow_truncate=True).embed_query("x")
    assert len(out) == 768 and abs(math.sqrt(sum(v * v for v in out)) - 1.0) < 1e-9


def test_transient_errors_are_retried_then_raised():
    c = FakeClient(fail_times=2)
    assert len(make(c, max_retries=2).embed_query("x")) == 8          # 2 failures, 3rd attempt succeeds
    c2 = FakeClient(fail_times=5)
    with pytest.raises(RuntimeError):
        make(c2, max_retries=2).embed_query("x")
    assert len(c2.calls) == 3
