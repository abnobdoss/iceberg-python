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
"""Internal adapters from PyIceberg models to pyiceberg-core objects."""

from __future__ import annotations

import importlib
import os
import threading
import weakref
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

from pyiceberg.expressions import (
    AlwaysFalse,
    AlwaysTrue,
    And,
    BooleanExpression,
    BoundPredicate,
    BoundTerm,
    BoundUnaryPredicate,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThan,
    LessThanOrEqual,
    LiteralPredicate,
    Not,
    NotEqualTo,
    NotIn,
    NotNull,
    NotStartsWith,
    Or,
    Reference,
    SetPredicate,
    StartsWith,
    UnaryPredicate,
    UnboundTerm,
)
from pyiceberg.io import FileIO
from pyiceberg.manifest import DataFile
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import FileScanTask
from pyiceberg.table.name_mapping import NameMapping
from pyiceberg.typedef import Record

if TYPE_CHECKING:
    from collections.abc import Iterable


def _core_module(name: str) -> Any:
    """Import a pyiceberg-core submodule lazily so PyIceberg can run without the Rust wheel."""
    try:
        return importlib.import_module(f"pyiceberg_core.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"pyiceberg_core.{name}":
            raise NotImplementedError(
                "The installed pyiceberg-core wheel does not expose native scan bindings. "
                "Install a pyiceberg-core build that includes schema, expression, file_io, and scan modules."
            ) from exc
        raise ModuleNotFoundError(
            'pyiceberg-core is required for native scan adapters. Install it with `pip install "pyiceberg[pyiceberg-core]"`.'
        ) from exc


def _model_json(value: Any) -> str:
    return value.model_dump_json(by_alias=True, exclude_none=True)


def schema_to_pyiceberg_core(schema: Schema) -> Any:
    """Convert a PyIceberg Schema to a pyiceberg-core schema object."""
    return _core_module("schema").Schema.from_json(_model_json(schema))


def file_io_to_pyiceberg_core(file_io: FileIO) -> Any:
    """Convert a PyIceberg FileIO to a pyiceberg-core FileIO-like object."""
    return _core_module("file_io").FileIO.from_props(dict(file_io.properties))


def _literal_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _term_name(term: UnboundTerm | BoundTerm) -> str:
    if isinstance(term, BoundTerm):
        ref = term.ref()
        return ref.field.name
    if isinstance(term, Reference):
        return term.name

    raise NotImplementedError(f"Cannot convert expression term {term!r} to pyiceberg_core")


def _term_field_id(term: UnboundTerm | BoundTerm, schema: Schema, case_sensitive: bool) -> int:
    if isinstance(term, BoundTerm):
        return term.ref().field.field_id
    if isinstance(term, Reference):
        return schema.find_field(term.name, case_sensitive=case_sensitive).field_id

    raise NotImplementedError(f"Cannot collect field id for expression term {term!r}")


def _expression_field_ids(expr: BooleanExpression, schema: Schema, case_sensitive: bool) -> set[int]:
    if isinstance(expr, (AlwaysTrue, AlwaysFalse)):
        return set()
    if isinstance(expr, (And, Or)):
        return _expression_field_ids(expr.left, schema, case_sensitive) | _expression_field_ids(
            expr.right, schema, case_sensitive
        )
    if isinstance(expr, Not):
        return _expression_field_ids(expr.child, schema, case_sensitive)
    if isinstance(expr, BoundPredicate):
        return {_term_field_id(expr.term, schema, case_sensitive)}
    if isinstance(expr, (UnaryPredicate, LiteralPredicate, SetPredicate)):
        return {_term_field_id(expr.term, schema, case_sensitive)}

    raise NotImplementedError(f"Cannot collect field ids for unsupported PyIceberg expression {expr!r}")


def _read_field_ids(
    projected_schema: Schema,
    residual: BooleanExpression,
    schema: Schema,
    case_sensitive: bool,
    project_field_ids: Iterable[int] | None,
) -> list[int]:
    ids = list(project_field_ids) if project_field_ids is not None else list(projected_schema.field_ids)
    seen = set(ids)
    for field_id in sorted(_expression_field_ids(residual, schema, case_sensitive)):
        if field_id not in seen:
            ids.append(field_id)
            seen.add(field_id)
    return ids


def can_read_projected_schema_with_pyiceberg_core(
    schema: Schema,
    projected_schema: Schema,
    row_filter: BooleanExpression,
    case_sensitive: bool,
) -> bool:
    """Return whether pyiceberg-core can read exactly the requested projection for this filter."""
    return _expression_field_ids(row_filter, schema, case_sensitive).issubset(projected_schema.field_ids)


_UNARY_METHODS: dict[type[BooleanExpression], str] = {
    IsNull: "is_null",
    NotNull: "is_not_null",
}

_LITERAL_METHODS: dict[type[BooleanExpression], str] = {
    EqualTo: "eq",
    NotEqualTo: "ne",
    LessThan: "lt",
    LessThanOrEqual: "lte",
    GreaterThan: "gt",
    GreaterThanOrEqual: "gte",
    StartsWith: "starts_with",
    NotStartsWith: "not_starts_with",
}

_SET_METHODS: dict[type[BooleanExpression], str] = {
    In: "is_in",
    NotIn: "is_not_in",
}


def expression_to_pyiceberg_core(
    expr: BooleanExpression,
    schema: Schema | None = None,
    case_sensitive: bool = True,
) -> Any:
    """Convert a PyIceberg BooleanExpression to a pyiceberg-core expression object."""
    expression = _core_module("expression")
    if isinstance(expr, AlwaysTrue):
        return expression.Predicate.always_true()
    if isinstance(expr, AlwaysFalse):
        return expression.Predicate.always_false()
    if isinstance(expr, And):
        return expression_to_pyiceberg_core(expr.left, schema, case_sensitive).and_(
            expression_to_pyiceberg_core(expr.right, schema, case_sensitive)
        )
    if isinstance(expr, Or):
        return expression_to_pyiceberg_core(expr.left, schema, case_sensitive).or_(
            expression_to_pyiceberg_core(expr.right, schema, case_sensitive)
        )
    if isinstance(expr, Not):
        return expression_to_pyiceberg_core(expr.child, schema, case_sensitive).negate()

    if isinstance(expr, BoundPredicate):
        return _bound_predicate_to_pyiceberg_core(expr)

    if isinstance(expr, (UnaryPredicate, LiteralPredicate, SetPredicate)):
        if schema is None:
            raise NotImplementedError(f"Cannot convert unbound expression {expr!r} without a Schema")
        return expression_to_pyiceberg_core(expr.bind(schema, case_sensitive=case_sensitive), schema, case_sensitive)

    raise NotImplementedError(f"Cannot convert unsupported PyIceberg expression {expr!r} to pyiceberg_core")


def _bound_predicate_to_pyiceberg_core(expr: BoundPredicate) -> Any:
    ref = _core_module("expression").Reference(_term_name(expr.term))

    if isinstance(expr, BoundUnaryPredicate):
        method = _UNARY_METHODS.get(expr.as_unbound)
        if method is None:
            raise NotImplementedError(f"Cannot convert unsupported unary predicate {expr!r} to pyiceberg_core")
        return getattr(ref, method)()

    if isinstance(expr, LiteralPredicate):
        raise NotImplementedError(f"Expected a bound literal predicate, got unbound predicate {expr!r}")

    if hasattr(expr, "literal"):
        method = _LITERAL_METHODS.get(expr.as_unbound)
        if method is None:
            raise NotImplementedError(f"Cannot convert unsupported literal predicate {expr!r} to pyiceberg_core")
        return getattr(ref, method)(_literal_value(expr.literal))

    if hasattr(expr, "literals"):
        method = _SET_METHODS.get(expr.as_unbound)
        if method is None:
            raise NotImplementedError(f"Cannot convert unsupported set predicate {expr!r} to pyiceberg_core")
        return getattr(ref, method)([_literal_value(lit) for lit in expr.literals])

    raise NotImplementedError(f"Cannot convert unsupported bound predicate {expr!r} to pyiceberg_core")


def _record_to_values(record: Record | None) -> list[Any] | None:
    if record is None:
        return None
    return [record[pos] for pos in range(len(record))]


def _file_format_value(data_file: DataFile) -> str:
    file_format = data_file.file_format
    return getattr(file_format, "value", file_format).lower()


def delete_file_to_pyiceberg_core(delete_file: DataFile) -> Any:
    """Convert a PyIceberg delete DataFile to a pyiceberg-core DeleteFile."""
    content = int(delete_file.content)
    if content == 1:
        file_type = "position-deletes"
    elif content == 2:
        raise NotImplementedError("pyiceberg-core equality delete scan parity is tracked separately")
    else:
        raise ValueError(f"Expected a delete file, got data file content {delete_file.content!r}")

    return _core_module("scan").DeleteFile(
        delete_file.file_path,
        delete_file.file_size_in_bytes,
        file_type,
        partition_spec_id=delete_file.spec_id,
        equality_ids=delete_file.equality_ids,
    )


def file_scan_task_to_pyiceberg_core(
    task: FileScanTask,
    schema: Schema,
    projected_schema: Schema | None = None,
    partition_spec: PartitionSpec | None = None,
    name_mapping: NameMapping | None = None,
    case_sensitive: bool = True,
    project_field_ids: Iterable[int] | None = None,
) -> Any:
    """Convert a PyIceberg FileScanTask to a pyiceberg-core file scan task object."""
    projected = projected_schema or schema
    field_ids = _read_field_ids(projected, task.residual, schema, case_sensitive, project_field_ids)
    file_size_in_bytes = task.file.file_size_in_bytes
    partition_data = _record_to_values(task.file.partition)
    if partition_data and partition_spec is None:
        raise ValueError("partition_spec is required when converting a partitioned FileScanTask")

    return _core_module("scan").FileScanTask(
        schema=schema_to_pyiceberg_core(schema),
        data_file_path=task.file.file_path,
        file_size_in_bytes=file_size_in_bytes,
        project_field_ids=field_ids,
        start=0,
        length=file_size_in_bytes,
        record_count=task.file.record_count,
        data_file_format=_file_format_value(task.file),
        predicate=expression_to_pyiceberg_core(task.residual, schema, case_sensitive),
        deletes=[delete_file_to_pyiceberg_core(delete_file) for delete_file in task.delete_files],
        partition_data=partition_data,
        partition_spec=_model_json(partition_spec) if partition_spec is not None else None,
        name_mapping=_model_json(name_mapping) if name_mapping is not None else None,
        case_sensitive=case_sensitive,
    )


# Rows per Arrow batch handed back from the native reader. The native default emits very small
# batches, and the sharded fan-in marshals every batch back through the GIL-holding consumer, so a
# tiny batch makes per-batch Python orchestration dominate the decode (measured ~3x slower than a
# whole-shard drain). A larger batch amortizes that handoff while keeping in-flight memory bounded
# to shards x batch (so the streaming contract still holds). Override with
# PYICEBERG_RUST_ARROW_BATCH_SIZE.
_DEFAULT_ARROW_BATCH_SIZE = 262144


def _reader_kwargs() -> dict[str, int]:
    batch_size = os.environ.get("PYICEBERG_RUST_ARROW_BATCH_SIZE")
    kwargs: dict[str, int] = {"batch_size": int(batch_size) if batch_size else _DEFAULT_ARROW_BATCH_SIZE}
    concurrency = os.environ.get("PYICEBERG_RUST_ARROW_FILE_CONCURRENCY")
    if concurrency:
        kwargs["data_file_concurrency_limit"] = int(concurrency)
    return kwargs


def _shard_count(n_tasks: int) -> int:
    """How many decode threads to fan out across.

    The native ArrowReader decodes a single stream on one core (it parallelizes I/O, not CPU
    decode), so a single-stream read of many files leaves the box idle. Sharding the file tasks
    across threads — each driving its own reader — recovers multi-core decode (the GIL is released
    during the C-stream drain). Default scales with cores, capped so tiny scans don't pay thread
    overhead; override with PYICEBERG_RUST_ARROW_SHARDS (1 disables sharding).
    """
    override = os.environ.get("PYICEBERG_RUST_ARROW_SHARDS")
    if override:
        return max(1, int(override))
    if n_tasks <= 1:
        return 1
    return max(1, min(n_tasks, (os.cpu_count() or 1)))


class _ShardedBatchStream:
    """Generator-backed, backpressured fan-in over several native shard readers.

    Each shard owns one ``pyiceberg_core`` ``RecordBatchReader`` and is drained sequentially (a
    stateful reader must not be polled concurrently), so at most one read per shard is ever in
    flight. A ``ThreadPoolExecutor`` pulls the *next* batch from every idle shard at once and the
    consumer is handed batches as they complete, so decode runs on up to ``n_shards`` cores (the
    GIL is released during the C-stream drain). Peak memory is bounded to at most one decoded
    batch per shard plus what the consumer holds — never the whole result — because a shard is not
    asked for its next batch until its current one has been handed out (backpressure: a slow
    consumer stalls the shards rather than buffering ahead).

    Batches are yielded as they complete (``FIRST_COMPLETED``); ordering across shards is not
    preserved, which is sound because the scan result is an unordered union of file tasks. Worker
    exceptions are re-raised to the consumer, and the pool is shut down on exhaustion, on an
    exception during iteration, on an explicit :meth:`close`, or — for a consumer that simply
    stops iterating and drops the reader — via the ``weakref`` finalizer at garbage collection.
    """

    def __init__(self, readers: list[Any]) -> None:
        self._readers = readers
        self._pool = ThreadPoolExecutor(max_workers=len(readers))
        # Shard indices whose reader is idle and not yet known to be exhausted.
        self._idle: list[int] = list(range(len(readers)))
        self._in_flight: dict[Future[Any], int] = {}
        self._closed = False
        self._lock = threading.Lock()
        # Shut the pool down if the consumer abandons the iterator without closing it.
        self._finalizer = weakref.finalize(self, self._shutdown, self._pool, self._readers)

    @staticmethod
    def _next_batch(reader: Any) -> Any | None:
        """Pull one batch from a shard reader, returning ``None`` when the shard is exhausted."""
        try:
            return reader.read_next_batch()
        except StopIteration:
            return None

    def _submit_idle(self) -> None:
        """Submit the next read for every idle, non-exhausted shard (one read per shard)."""
        while self._idle:
            shard = self._idle.pop()
            future = self._pool.submit(self._next_batch, self._readers[shard])
            self._in_flight[future] = shard

    def __iter__(self) -> _ShardedBatchStream:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            while True:
                self._submit_idle()
                if not self._in_flight:
                    # Every shard is exhausted: the union is complete.
                    self.close()
                    raise StopIteration
                done, _ = wait(self._in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    shard = self._in_flight.pop(future)
                    batch = future.result()  # re-raises any worker exception here
                    if batch is None:
                        continue  # shard exhausted; do not return it to the idle set
                    # Shard produced a batch: it may have more, so mark it idle again.
                    self._idle.append(shard)
                    return batch
        except StopIteration:
            raise
        except BaseException:
            # Worker exception, GeneratorExit, or KeyboardInterrupt: tear the workers down so an
            # aborted scan never leaves decode threads running.
            self.close()
            raise

    def close(self) -> None:
        """Cancel pending work and release shard readers; idempotent and consumer-safe."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._finalizer.detach()
        self._shutdown(self._pool, self._readers)

    @staticmethod
    def _shutdown(pool: ThreadPoolExecutor, readers: list[Any]) -> None:
        # cancel_futures drops queued reads; in-flight reads are joined so no thread outlives us.
        pool.shutdown(wait=True, cancel_futures=True)
        readers.clear()


def arrow_batch_reader_from_pyiceberg_core(
    file_io: FileIO,
    tasks: Iterable[FileScanTask],
    schema: Schema,
    projected_schema: Schema,
    partition_specs: dict[int, PartitionSpec],
    name_mapping: NameMapping | None,
    case_sensitive: bool = True,
) -> Any:
    """Read PyIceberg scan tasks through pyiceberg-core's ArrowReader as a streaming reader.

    Multi-file scans are sharded across a thread pool (see ``_shard_count``) so decode uses
    multiple cores; a single native reader over all files would decode on one core. Each shard
    drives its own native ``RecordBatchReader``; the returned ``pyarrow.RecordBatchReader`` pulls
    batches from the shards lazily (at most one decoded batch per shard in flight), so the whole
    result is never materialized and peak memory stays bounded. The single-file or single-shard
    case skips the fan-out entirely and returns the native reader directly.

    Worker-thread exceptions propagate to the consumer, and the shard threads are shut down when
    the reader is exhausted, closed early, or garbage collected.
    """
    core_tasks = [
        file_scan_task_to_pyiceberg_core(
            task,
            schema,
            projected_schema,
            partition_spec=partition_specs.get(task.file.spec_id),
            name_mapping=name_mapping,
            case_sensitive=case_sensitive,
            project_field_ids=list(projected_schema.field_ids),
        )
        for task in tasks
    ]

    core_projection = schema_to_pyiceberg_core(projected_schema)
    reader_kwargs = _reader_kwargs()

    def _read(shard_tasks: list[Any]) -> Any:
        reader = _core_module("scan").ArrowReader(file_io_to_pyiceberg_core(file_io), **reader_kwargs)
        return reader.read(core_projection, shard_tasks)

    shards = _shard_count(len(core_tasks))
    if shards <= 1 or len(core_tasks) <= 1:
        return _read(core_tasks)

    import pyarrow as pa

    groups = [g for g in (core_tasks[i::shards] for i in range(shards)) if g]
    readers = [_read(group) for group in groups]
    # Every native reader carries the same projected Arrow schema; use it to type the stream.
    arrow_schema = readers[0].schema
    stream = _ShardedBatchStream(readers)
    return pa.RecordBatchReader.from_batches(arrow_schema, stream)
