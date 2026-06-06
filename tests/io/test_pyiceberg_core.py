# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import sys
import threading
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

from pyiceberg.expressions import And, EqualTo, IsNull, StartsWith
from pyiceberg.io import FileIO, InputFile, OutputFile
from pyiceberg.io.pyiceberg_core import (
    _cast_batches,
    _limited_batches,
    _ShardedBatchStream,
    arrow_batch_reader_from_pyiceberg_core,
    can_read_projected_schema_with_pyiceberg_core,
    delete_file_to_pyiceberg_core,
    expression_to_pyiceberg_core,
    file_io_to_pyiceberg_core,
    file_scan_task_to_pyiceberg_core,
    plan_and_read_with_pyiceberg_core,
    schema_to_pyiceberg_core,
)
from pyiceberg.manifest import DataFile, DataFileContent
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import FileScanTask
from pyiceberg.table.name_mapping import MappedField, NameMapping
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import FloatType, IntegerType, NestedField, StringType


class CoreObject:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class CoreSchema(CoreObject):
    @classmethod
    def from_json(cls, schema_json: str) -> CoreSchema:
        return cls(schema_json=schema_json)


class CoreFileIO(CoreObject):
    @classmethod
    def from_props(cls, properties: dict[str, str]) -> CoreFileIO:
        return cls(properties=properties)


class CorePredicate(CoreObject):
    @staticmethod
    def always_true() -> CorePredicate:
        return CorePredicate(kind="always_true")

    @staticmethod
    def always_false() -> CorePredicate:
        return CorePredicate(kind="always_false")

    def and_(self, other: CorePredicate) -> CorePredicate:
        return CorePredicate(op="and", left=self, right=other)

    def or_(self, other: CorePredicate) -> CorePredicate:
        return CorePredicate(op="or", left=self, right=other)

    def negate(self) -> CorePredicate:
        return CorePredicate(op="not", child=self)


class CoreScanTable(CoreObject):
    """Fake ``pyiceberg_core.scan.Table`` that records planning args and emits planned tasks."""

    last_from_metadata_json: dict[str, Any] = {}
    last_plan_files: dict[str, Any] = {}

    @classmethod
    def from_metadata_json(cls, file_io: Any, identifier: Any, metadata_json: str, **kwargs: Any) -> CoreScanTable:
        CoreScanTable.last_from_metadata_json = {
            "file_io": file_io,
            "identifier": identifier,
            "metadata_json": metadata_json,
            **kwargs,
        }
        return cls()

    def plan_files(self, **kwargs: Any) -> list[CoreObject]:
        CoreScanTable.last_plan_files = kwargs
        # One planned task per selected field keeps the count deterministic and lets the read fake
        # echo back exactly what planning produced.
        return [CoreObject(planned=name) for name in kwargs["selected_fields"]]


class CoreReference(CoreObject):
    def _predicate(self, op: str, *args: Any) -> CorePredicate:
        return CorePredicate(op=op, name=self.args[0], args=args)

    def eq(self, value: Any) -> CorePredicate:
        return self._predicate("eq", value)

    def starts_with(self, value: Any) -> CorePredicate:
        return self._predicate("starts_with", value)

    def is_null(self) -> CorePredicate:
        return self._predicate("is_null")


class FakeFileIO(FileIO):
    def new_input(self, location: str) -> InputFile:
        raise NotImplementedError

    def new_output(self, location: str) -> OutputFile:
        raise NotImplementedError

    def delete(self, location: str | InputFile | OutputFile) -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def fake_pyiceberg_core(monkeypatch: pytest.MonkeyPatch) -> None:
    root = ModuleType("pyiceberg_core")

    schema: Any = ModuleType("pyiceberg_core.schema")
    schema.Schema = CoreSchema

    file_io: Any = ModuleType("pyiceberg_core.file_io")
    file_io.FileIO = CoreFileIO

    expression: Any = ModuleType("pyiceberg_core.expression")
    expression.Predicate = CorePredicate
    expression.Reference = CoreReference

    scan: Any = ModuleType("pyiceberg_core.scan")
    scan.DeleteFile = CoreObject
    scan.FileScanTask = CoreObject
    scan.Table = CoreScanTable  # planning tests drive from_metadata_json + plan_files through this
    scan.ArrowReader = CoreObject  # streaming tests override this with a batch-producing fake

    monkeypatch.setitem(sys.modules, "pyiceberg_core", root)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.schema", schema)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.file_io", file_io)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.expression", expression)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.scan", scan)


def test_schema_to_pyiceberg_core_feature_gates_old_core_wheels(monkeypatch: pytest.MonkeyPatch, simple_schema: Schema) -> None:
    monkeypatch.setitem(sys.modules, "pyiceberg_core", ModuleType("pyiceberg_core"))
    monkeypatch.delitem(sys.modules, "pyiceberg_core.schema", raising=False)

    with pytest.raises(NotImplementedError, match="does not expose native scan bindings"):
        schema_to_pyiceberg_core(simple_schema)


@pytest.fixture
def simple_schema() -> Schema:
    return Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType()),
        schema_id=3,
    )


def test_schema_to_pyiceberg_core_uses_lazy_core_schema_json(simple_schema: Schema) -> None:
    converted = schema_to_pyiceberg_core(simple_schema)

    assert converted.kwargs["schema_json"] == simple_schema.model_dump_json(by_alias=True, exclude_none=True)


def test_file_io_to_pyiceberg_core_uses_file_io_properties() -> None:
    converted = file_io_to_pyiceberg_core(FakeFileIO(properties={"s3.region": "us-east-1"}))

    assert converted.kwargs == {"properties": {"s3.region": "us-east-1"}}


def test_expression_to_pyiceberg_core_converts_expression_tree(simple_schema: Schema) -> None:
    converted = expression_to_pyiceberg_core(And(EqualTo("id", 34), StartsWith("data", "abc")), simple_schema)

    assert converted.kwargs["op"] == "and"
    assert converted.kwargs["left"].kwargs == {"op": "eq", "name": "id", "args": (34,)}
    assert converted.kwargs["right"].kwargs == {"op": "starts_with", "name": "data", "args": ("abc",)}


def test_expression_to_pyiceberg_core_converts_unary_expression(simple_schema: Schema) -> None:
    converted = expression_to_pyiceberg_core(IsNull("data"), simple_schema)

    assert converted.kwargs == {"op": "is_null", "name": "data", "args": ()}


def test_expression_to_pyiceberg_core_requires_schema_for_unbound_expression() -> None:
    with pytest.raises(NotImplementedError, match="without a Schema"):
        expression_to_pyiceberg_core(EqualTo("id", 34))


def test_expression_to_pyiceberg_core_raises_clear_error_for_unsupported_expression() -> None:
    from pyiceberg.expressions import IsNaN

    nan_schema = Schema(NestedField(1, "value", FloatType()))
    with pytest.raises(NotImplementedError, match="unsupported unary predicate"):
        expression_to_pyiceberg_core(IsNaN("value"), nan_schema)


def test_delete_file_to_pyiceberg_core_converts_delete_file_payload() -> None:
    delete_file = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path="s3://warehouse/table/delete.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=1,
        file_size_in_bytes=123,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    delete_file.spec_id = 7

    converted = delete_file_to_pyiceberg_core(delete_file)

    assert converted.args == ("s3://warehouse/table/delete.parquet", 123, "position-deletes")
    assert converted.kwargs == {"partition_spec_id": 7, "equality_ids": None}


def test_delete_file_to_pyiceberg_core_rejects_equality_deletes_until_parity_lands() -> None:
    delete_file = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES,
        file_path="s3://warehouse/table/eq-delete.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=1,
        file_size_in_bytes=123,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
        equality_ids=[1],
    )
    delete_file.spec_id = 7

    with pytest.raises(NotImplementedError, match="equality delete scan parity"):
        delete_file_to_pyiceberg_core(delete_file)


def test_file_scan_task_to_pyiceberg_core_converts_task_payload(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={1: 20},
        value_counts={1: 10},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"\x01\x00\x00\x00"},
        upper_bounds={1: b"\x0a\x00\x00\x00"},
    )
    data_file.spec_id = 7
    partition_spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"))
    name_mapping = NameMapping([MappedField(field_id=1, names=["id"])])

    task = FileScanTask(data_file, residual=EqualTo("id", 3))
    converted = file_scan_task_to_pyiceberg_core(
        task,
        simple_schema,
        partition_spec=partition_spec,
        name_mapping=name_mapping,
        case_sensitive=False,
    )

    assert converted.kwargs["data_file_path"] == "s3://warehouse/table/data.parquet"
    assert converted.kwargs["data_file_format"] == "parquet"
    assert converted.kwargs["file_size_in_bytes"] == 1234
    assert converted.kwargs["length"] == 1234
    assert converted.kwargs["record_count"] == 10
    assert converted.kwargs["partition_data"] == ["bucket-1"]
    assert converted.kwargs["partition_spec"] == partition_spec.model_dump_json(by_alias=True, exclude_none=True)
    assert converted.kwargs["name_mapping"] == name_mapping.model_dump_json(by_alias=True, exclude_none=True)
    assert converted.kwargs["case_sensitive"] is False
    assert set(converted.kwargs["project_field_ids"]) == {1, 2}
    assert converted.kwargs["predicate"].kwargs == {"op": "eq", "name": "id", "args": (3,)}


def test_file_scan_task_to_pyiceberg_core_adds_filter_only_field_to_read_projection(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={1: 20},
        value_counts={1: 10},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"\x01\x00\x00\x00"},
        upper_bounds={1: b"\x0a\x00\x00\x00"},
    )
    data_file.spec_id = 0
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=3)

    task = FileScanTask(data_file, residual=EqualTo("data", "abc"))
    converted = file_scan_task_to_pyiceberg_core(task, simple_schema, projected_schema=projected_schema)

    assert converted.kwargs["project_field_ids"] == [1, 2]


def test_can_read_projected_schema_with_pyiceberg_core_requires_filter_fields(simple_schema: Schema) -> None:
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=3)

    assert not can_read_projected_schema_with_pyiceberg_core(simple_schema, projected_schema, EqualTo("data", "abc"), True)
    assert can_read_projected_schema_with_pyiceberg_core(simple_schema, projected_schema, EqualTo("id", 1), True)


def test_file_scan_task_to_pyiceberg_core_requires_partition_spec_for_partitioned_task(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    data_file.spec_id = 7

    with pytest.raises(ValueError, match="partition_spec is required"):
        file_scan_task_to_pyiceberg_core(FileScanTask(data_file), simple_schema)


# --- streaming sharded reader -------------------------------------------------

_STREAM_SCHEMA = pa.schema([("id", pa.int64())])


def _batch(values: list[int]) -> pa.RecordBatch:
    return pa.record_batch({"id": pa.array(values, type=pa.int64())})


class FakeShardReader:
    """A stand-in for a native ``pyiceberg_core`` RecordBatchReader over one shard's tasks."""

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = list(batches)
        self._pos = 0
        self.schema = _STREAM_SCHEMA

    def read_next_batch(self) -> pa.RecordBatch:
        if self._pos >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._pos]
        self._pos += 1
        return batch

    # A real pyarrow.RecordBatchReader (what the single-shard fast path drains) is also iterable.
    def __iter__(self) -> FakeShardReader:
        return self

    def __next__(self) -> pa.RecordBatch:
        return self.read_next_batch()


def _drain(reader: pa.RecordBatchReader) -> tuple[int, int]:
    """Return (row_count, sum-of-id checksum) — the same parity signal used in the perf gate."""
    table = reader.read_all()
    rows = table.num_rows
    checksum = pa.compute.sum(table["id"]).as_py() or 0
    return rows, checksum


def test_sharded_batch_stream_preserves_all_rows_and_checksum() -> None:
    # Uneven shards exercise the "one shard finishes early" path of the fan-in.
    readers = [
        FakeShardReader([_batch([1, 2]), _batch([3])]),
        FakeShardReader([_batch([4, 5, 6])]),
        FakeShardReader([]),  # an empty shard must not stall the union
    ]
    expected_rows = 6
    expected_sum = 21

    stream = _ShardedBatchStream(readers)
    reader = pa.RecordBatchReader.from_batches(_STREAM_SCHEMA, stream)

    assert _drain(reader) == (expected_rows, expected_sum)


def test_arrow_batch_reader_streams_lazily_without_materializing(monkeypatch: pytest.MonkeyPatch) -> None:
    # The legacy implementation called read_all() per shard up front; assert the new path does not
    # pull any batch until the consumer asks for one (the streaming contract).
    pulled: list[int] = []

    class ObservingReader(FakeShardReader):
        def read_next_batch(self) -> pa.RecordBatch:
            batch = super().read_next_batch()
            pulled.append(batch.num_rows)
            return batch

    shard_readers = [ObservingReader([_batch([1]), _batch([2])]), ObservingReader([_batch([3]), _batch([4])])]

    class FakeArrowReader:
        def __init__(self, _file_io: Any, **_kwargs: Any) -> None:
            pass

        def read(self, _projection: Any, _tasks: list[Any]) -> ObservingReader:
            return shard_readers.pop(0)

    monkeypatch.setattr(sys.modules["pyiceberg_core.scan"], "ArrowReader", FakeArrowReader)
    monkeypatch.setenv("PYICEBERG_RUST_ARROW_SHARDS", "2")

    reader = _build_reader(monkeypatch, n_tasks=2)
    assert pulled == []  # construction must not drain anything

    first = reader.read_next_batch()
    assert first.num_rows == 1
    # Backpressure: at most one read per shard is outstanding, so the first pull never drains all 4.
    assert len(pulled) < 4

    rows, checksum = _drain(reader)
    assert (rows + first.num_rows, checksum + first["id"][0].as_py()) == (4, 1 + 2 + 3 + 4)


def test_sharded_batch_stream_bounds_in_flight_reads() -> None:
    # Each shard read parks on a gate; assert no more than one read per shard is ever outstanding,
    # so peak memory is bounded to one decoded batch per shard rather than the whole result. A
    # shard must not be asked for its next batch until its current one is consumed (backpressure).
    n_shards = 5
    release = threading.Event()
    started = threading.Semaphore(0)
    concurrent = 0
    peak = 0
    lock = threading.Lock()

    class GatedReader:
        def __init__(self) -> None:
            self.schema = _STREAM_SCHEMA
            self._remaining = 4

        def read_next_batch(self) -> pa.RecordBatch:
            nonlocal concurrent, peak
            if self._remaining <= 0:
                raise StopIteration
            self._remaining -= 1
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            started.release()
            release.wait(timeout=5)
            with lock:
                concurrent -= 1
            return _batch([1])

    stream = _ShardedBatchStream([GatedReader() for _ in range(n_shards)])

    consumer = threading.Thread(target=lambda: list(stream))
    consumer.start()
    # All shards start their first read and park. An (n_shards + 1)th concurrent start would mean a
    # shard was double-polled; assert it does not happen while the gate is closed.
    for _ in range(n_shards):
        assert started.acquire(timeout=5)
    assert not started.acquire(timeout=0.2), "a shard was polled twice before its batch was consumed"
    with lock:
        assert peak <= n_shards

    release.set()
    consumer.join(timeout=10)
    assert not consumer.is_alive()
    assert peak <= n_shards


def test_sharded_batch_stream_propagates_worker_exceptions() -> None:
    class BoomReader:
        def __init__(self) -> None:
            self.schema = _STREAM_SCHEMA

        def read_next_batch(self) -> pa.RecordBatch:
            raise RuntimeError("native decode failed")

    stream = _ShardedBatchStream([FakeShardReader([_batch([1])]), BoomReader()])
    reader = pa.RecordBatchReader.from_batches(_STREAM_SCHEMA, stream)

    with pytest.raises(RuntimeError, match="native decode failed"):
        reader.read_all()

    # The pool must be torn down (no leaked worker threads) once the error surfaced.
    assert stream._closed
    assert stream._pool._shutdown


def test_sharded_batch_stream_shuts_down_on_early_close() -> None:
    readers = [FakeShardReader([_batch([i]) for i in range(10)]) for _ in range(3)]
    stream = _ShardedBatchStream(readers)

    first = next(stream)
    assert first.num_rows == 1

    stream.close()
    assert stream._closed
    assert stream._pool._shutdown
    # close() is idempotent and a closed stream is exhausted.
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)


def test_sharded_batch_stream_shuts_down_on_garbage_collection() -> None:
    import gc
    import weakref

    readers = [FakeShardReader([_batch([i]) for i in range(50)]) for _ in range(3)]
    stream = _ShardedBatchStream(readers)
    pool = stream._pool

    next(stream)  # leave the pool and workers live, then abandon the stream without close()
    ref = weakref.ref(stream)
    del stream
    gc.collect()

    assert ref() is None  # no lingering references kept the stream (and its threads) alive
    assert pool._shutdown  # the finalizer tore the pool down


def test_arrow_batch_reader_single_shard_casts_to_target_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    # The native reader hands back int64; the projected schema is IntegerType, so even the
    # single-shard fast path must cast its output to the PyArrow target schema (here int32).
    native = FakeShardReader([_batch([7])])

    class FakeArrowReader:
        def __init__(self, _file_io: Any, **_kwargs: Any) -> None:
            pass

        def read(self, _projection: Any, _tasks: list[Any]) -> FakeShardReader:
            return native

    monkeypatch.setattr(sys.modules["pyiceberg_core.scan"], "ArrowReader", FakeArrowReader)
    monkeypatch.setenv("PYICEBERG_RUST_ARROW_SHARDS", "1")

    reader = _build_reader(monkeypatch, n_tasks=4)
    table = reader.read_all()

    assert table.column("id").to_pylist() == [7]
    assert table.schema.field("id").type == pa.int32()  # cast to the projected schema's type, not the native int64


def test_cast_batches_decodes_ree_and_matches_target_schema() -> None:
    import pyarrow.compute as pc

    ree = pc.run_end_encode(pa.array(["a", "a", "b"], pa.string()))
    batch = pa.record_batch({"id": pa.array([1, 2, 3], pa.int64()), "cat": ree})
    target = pa.schema([("id", pa.int32()), ("cat", pa.large_string())])

    class Source:
        def __init__(self, batches: list[pa.RecordBatch]) -> None:
            self._it = iter(batches)
            self.closed = False

        def __iter__(self) -> Source:
            return self

        def __next__(self) -> pa.RecordBatch:
            return next(self._it)

        def close(self) -> None:
            self.closed = True

    source = Source([batch])
    out = list(_cast_batches(source, target))

    assert len(out) == 1
    assert out[0].schema.field("id").type == pa.int32()  # native int64 widened/narrowed to the target
    assert out[0].schema.field("cat").type == pa.large_string()  # run-end-encoded column decoded then cast
    assert out[0].column("cat").to_pylist() == ["a", "a", "b"]
    assert source.closed


def test_limited_batches_truncates_to_limit_and_closes_source() -> None:
    class ClosableSource:
        def __init__(self, batches: list[pa.RecordBatch]) -> None:
            self._it = iter(batches)
            self.closed = False

        def __iter__(self) -> ClosableSource:
            return self

        def __next__(self) -> pa.RecordBatch:
            return next(self._it)

        def close(self) -> None:
            self.closed = True

    source = ClosableSource([_batch([1, 2, 3]), _batch([4, 5, 6]), _batch([7, 8, 9])])
    out = list(_limited_batches(source, 4))

    rows = [v for batch in out for v in batch["id"].to_pylist()]
    assert rows == [1, 2, 3, 4]  # the batch crossing the limit is sliced
    assert source.closed  # the underlying source is closed so a sharded scan stops decoding early


def test_arrow_batch_reader_applies_limit_across_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_batch_size: list[Any] = []
    shard_readers = [
        FakeShardReader([_batch([1, 2]), _batch([3, 4])]),
        FakeShardReader([_batch([5, 6]), _batch([7, 8])]),
    ]

    class FakeArrowReader:
        def __init__(self, _file_io: Any, **kwargs: Any) -> None:
            captured_batch_size.append(kwargs.get("batch_size"))

        def read(self, _projection: Any, _tasks: list[Any]) -> FakeShardReader:
            return shard_readers.pop(0)

    monkeypatch.setattr(sys.modules["pyiceberg_core.scan"], "ArrowReader", FakeArrowReader)
    monkeypatch.setenv("PYICEBERG_RUST_ARROW_SHARDS", "2")

    reader = _build_reader(monkeypatch, n_tasks=4, limit=3)
    table = reader.read_all()

    assert table.num_rows == 3  # the global limit is enforced across shards, not per shard
    assert captured_batch_size == [3, 3]  # batch size is capped to the limit so small limits don't over-decode


def _build_reader(monkeypatch: pytest.MonkeyPatch, n_tasks: int, limit: int | None = None) -> Any:
    """Drive ``arrow_batch_reader_from_pyiceberg_core`` with ``n_tasks`` trivial data-file tasks."""

    def _identity_task_conversion(task: Any, *_args: Any, **_kwargs: Any) -> Any:
        return task

    # The conversion + projection helpers are covered by their own tests; stub them so this test
    # focuses on the streaming fan-in rather than re-exercising payload conversion.
    monkeypatch.setattr("pyiceberg.io.pyiceberg_core.file_scan_task_to_pyiceberg_core", _identity_task_conversion)
    monkeypatch.setattr("pyiceberg.io.pyiceberg_core.schema_to_pyiceberg_core", lambda schema: schema)

    schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=0)

    tasks = []
    for _ in range(n_tasks):
        data_file = DataFile.from_args(
            content=DataFileContent.DATA,
            file_path="s3://warehouse/table/data.parquet",
            file_format="PARQUET",
            partition=Record(),
            record_count=1,
            file_size_in_bytes=1,
            column_sizes={},
            value_counts={},
            null_value_counts={},
            nan_value_counts={},
            lower_bounds={},
            upper_bounds={},
        )
        data_file.spec_id = 0
        tasks.append(FileScanTask(data_file))

    return arrow_batch_reader_from_pyiceberg_core(
        FakeFileIO(properties={}),
        tasks,
        schema,
        schema,
        {0: PartitionSpec(spec_id=0)},
        None,
        True,
        limit=limit,
    )


# --- native scan planning -----------------------------------------------------


class FakeTableMetadata:
    """Minimal stand-in: native planning only needs the metadata JSON and the table schema."""

    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def model_dump_json(self) -> str:
        return '{"format-version": 2}'

    def schema(self) -> Schema:
        return self._schema


def _planning_arrow_reader(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Wire a fake ArrowReader that records the projection + planned tasks and emits one batch."""

    class FakeArrowReader:
        def __init__(self, _file_io: Any, **kwargs: Any) -> None:
            captured["reader_kwargs"] = kwargs

        def read(self, projection: Any, tasks: list[Any]) -> FakeShardReader:
            captured["projection"] = projection
            captured["tasks"] = tasks
            return FakeShardReader([_batch([1, 2, 3])])

    monkeypatch.setattr(sys.modules["pyiceberg_core.scan"], "ArrowReader", FakeArrowReader)
    monkeypatch.setenv("PYICEBERG_RUST_ARROW_SHARDS", "1")


def test_plan_and_read_passes_projection_and_filter_to_native_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _planning_arrow_reader(monkeypatch, captured)

    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType()),
        schema_id=0,
    )
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=0)

    reader = plan_and_read_with_pyiceberg_core(
        FakeTableMetadata(schema),
        FakeFileIO(properties={"s3.region": "us-east-1"}),
        projected_schema,
        EqualTo("id", 5),
        ("ns", "t"),
        case_sensitive=False,
        snapshot_id=42,
    )
    table = reader.read_all()

    # The metadata JSON and a >=2-part identifier are handed to from_metadata_json verbatim.
    assert CoreScanTable.last_from_metadata_json["identifier"] == ["ns", "t"]
    assert CoreScanTable.last_from_metadata_json["metadata_json"] == '{"format-version": 2}'
    # plan_files receives the projection as field NAMES, the converted predicate, snapshot, sensitivity.
    plan = CoreScanTable.last_plan_files
    assert plan["selected_fields"] == ["id"]
    assert plan["snapshot_id"] == 42
    assert plan["case_sensitive"] is False
    assert plan["predicate"].kwargs == {"op": "eq", "name": "id", "args": (5,)}
    # The planned tasks (not python-built tasks) are what the reader consumes.
    assert [task.kwargs["planned"] for task in captured["tasks"]] == ["id"]
    # Output is cast to the projected schema's PyArrow type (IntegerType -> int32, not native int64).
    assert table.schema.field("id").type == pa.int32()
    assert table.column("id").to_pylist() == [1, 2, 3]


def test_plan_and_read_skips_predicate_for_always_true_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyiceberg.expressions import AlwaysTrue

    captured: dict[str, Any] = {}
    _planning_arrow_reader(monkeypatch, captured)

    schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=0)

    plan_and_read_with_pyiceberg_core(
        FakeTableMetadata(schema),
        FakeFileIO(properties={}),
        schema,
        AlwaysTrue(),
        ("ns", "t"),
    ).read_all()

    # An unfiltered scan must not pass a predicate at all (rather than always_true()).
    assert CoreScanTable.last_plan_files["predicate"] is None


def test_plan_and_read_falls_back_to_safe_identifier_for_short_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyiceberg.expressions import AlwaysTrue

    captured: dict[str, Any] = {}
    _planning_arrow_reader(monkeypatch, captured)

    schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=0)

    # A single-part (or None) identifier cannot name a table rust-side; a 2-part fallback is used.
    plan_and_read_with_pyiceberg_core(
        FakeTableMetadata(schema),
        FakeFileIO(properties={}),
        schema,
        AlwaysTrue(),
        ("only_one",),
    ).read_all()

    identifier = CoreScanTable.last_from_metadata_json["identifier"]
    assert len(identifier) >= 2


def test_plan_and_read_applies_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyiceberg.expressions import AlwaysTrue

    captured: dict[str, Any] = {}
    _planning_arrow_reader(monkeypatch, captured)

    schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=0)

    reader = plan_and_read_with_pyiceberg_core(
        FakeTableMetadata(schema),
        FakeFileIO(properties={}),
        schema,
        AlwaysTrue(),
        ("ns", "t"),
        limit=2,
    )
    table = reader.read_all()

    assert table.num_rows == 2  # the batch of 3 is truncated to the limit
    assert captured["reader_kwargs"]["batch_size"] == 2  # batch size capped to the limit
