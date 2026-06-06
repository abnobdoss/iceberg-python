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
"""Tests for Arrow stream producer and consumer support."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.io.pyarrow import _coerce_arrow_input
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import IntegerType, NestedField, StringType

SCHEMA = Schema(
    NestedField(1, "id", IntegerType(), required=False),
    NestedField(2, "region", StringType(), required=False),
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32(), nullable=True),
        pa.field("region", pa.string(), nullable=True),
    ]
)
PARTITION_SPEC = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region"))


class _ArrowStreamWrapper:
    """A minimal third-party-style Arrow stream producer."""

    def __init__(self, data: pa.Table):
        self._data = data

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self._data.__arrow_c_stream__(requested_schema)


@pytest.fixture
def catalog(tmp_path: Path) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def _data(ids: list[int], regions: list[str]) -> pa.Table:
    return pa.table({"id": pa.array(ids, type=pa.int32()), "region": regions}, schema=ARROW_SCHEMA)


def _string_view_data(ids: list[int], regions: list[str]) -> pa.Table:
    if not hasattr(pa, "string_view"):
        pytest.skip("pyarrow does not support string_view")
    return pa.table(
        {"id": pa.array(ids, type=pa.int32()), "region": pa.array(regions, type=pa.string_view())},
        schema=pa.schema(
            [
                pa.field("id", pa.int32(), nullable=True),
                pa.field("region", pa.string_view(), nullable=True),
            ]
        ),
    )


def _rows(table: pa.Table) -> list[dict[str, Any]]:
    return sorted(table.to_pylist(), key=lambda row: row["id"])


def test_coerce_arrow_input() -> None:
    table = _data([1, 2, 3], ["us", "eu", "us"])

    assert _coerce_arrow_input(table) is table
    reader = table.to_reader()
    assert _coerce_arrow_input(reader) is reader

    coerced = _coerce_arrow_input(_ArrowStreamWrapper(table))
    assert isinstance(coerced, pa.RecordBatchReader)
    assert coerced.read_all().num_rows == 3

    with pytest.raises(ValueError, match="Expected pa.Table, pa.RecordBatchReader"):
        _coerce_arrow_input(object())


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(lambda data: data, id="table"),
        pytest.param(lambda data: data.to_reader(), id="reader"),
        pytest.param(lambda data: _ArrowStreamWrapper(data), id="stream"),
        pytest.param(
            lambda data: _ArrowStreamWrapper(pa.Table.from_batches(data.to_batches(max_chunksize=1))),
            id="multi_batch_stream",
        ),
    ],
)
def test_append_accepts_arrow_inputs(catalog: InMemoryCatalog, make_input: Callable[[pa.Table], object]) -> None:
    tbl = catalog.create_table("default.append", schema=SCHEMA)

    tbl.append(make_input(_data([1, 2, 3], ["us", "eu", "us"])))

    assert _rows(tbl.scan().to_arrow()) == _rows(_data([1, 2, 3], ["us", "eu", "us"]))


def test_overwrite_accepts_arrow_stream(catalog: InMemoryCatalog) -> None:
    tbl = catalog.create_table("default.overwrite_stream", schema=SCHEMA)
    tbl.append(_data([1, 2], ["us", "eu"]))

    tbl.overwrite(_ArrowStreamWrapper(_data([9], ["jp"])))

    assert _rows(tbl.scan().to_arrow()) == _rows(_data([9], ["jp"]))


def test_append_accepts_arrow_stream_with_string_view(catalog: InMemoryCatalog) -> None:
    tbl = catalog.create_table("default.append_string_view", schema=SCHEMA)

    tbl.append(_ArrowStreamWrapper(_string_view_data([10, 11], ["ca", "mx"])))

    assert _rows(tbl.scan().to_arrow()) == _rows(_data([10, 11], ["ca", "mx"]))


def test_append_table_to_partitioned_table_keeps_partitioned_write_path(catalog: InMemoryCatalog) -> None:
    tbl = catalog.create_table("default.append_partitioned", schema=SCHEMA, partition_spec=PARTITION_SPEC)

    tbl.append(_data([1, 2], ["us", "eu"]))

    assert tbl.scan().to_arrow().num_rows == 2


@pytest.mark.parametrize(
    "produce",
    [
        pytest.param(lambda tbl: tbl, id="table"),
        pytest.param(lambda tbl: tbl.scan(), id="scan"),
    ],
)
def test_supports_arrow_c_stream(catalog: InMemoryCatalog, produce: Callable[[Table], object]) -> None:
    tbl = catalog.create_table("default.stream", schema=SCHEMA)
    tbl.append(_data([1, 2, 3], ["us", "eu", "us"]))

    consumed = pa.table(produce(tbl))

    assert _rows(consumed) == _rows(tbl.scan().to_arrow())


def test_scan_arrow_c_stream_respects_filter_and_projection(catalog: InMemoryCatalog) -> None:
    tbl = catalog.create_table("default.scan_stream_filtered", schema=SCHEMA)
    tbl.append(_data([1, 2, 3], ["us", "eu", "us"]))

    scan = tbl.scan(row_filter="region == 'us'", selected_fields=("id",))
    consumed = pa.table(scan)

    assert consumed.column_names == ["id"]
    assert sorted(consumed.column("id").to_pylist()) == [1, 3]


def test_arrow_stream_roundtrip_scan_into_append(catalog: InMemoryCatalog) -> None:
    src = catalog.create_table("default.roundtrip_src", schema=SCHEMA)
    src.append(_data([1, 2, 3], ["us", "eu", "us"]))
    dst = catalog.create_table("default.roundtrip_dst", schema=SCHEMA)

    dst.append(src.scan())

    assert _rows(dst.scan().to_arrow()) == _rows(src.scan().to_arrow())


def test_scan_to_duckdb_registers_stream(catalog: InMemoryCatalog) -> None:
    pytest.importorskip("duckdb")
    tbl = catalog.create_table("default.duckdb_stream", schema=SCHEMA)
    tbl.append(_data([1, 2, 3], ["us", "eu", "us"]))

    con = tbl.scan().to_duckdb("iceberg_table")

    assert con.sql("select count(*) as count, sum(id) as total from iceberg_table").fetchall() == [(3, 6)]


def test_scan_to_polars_consumes_stream(catalog: InMemoryCatalog) -> None:
    pytest.importorskip("polars")
    tbl = catalog.create_table("default.polars_stream", schema=SCHEMA)
    tbl.append(_data([1, 2, 3], ["us", "eu", "us"]))

    result = tbl.scan(row_filter="region == 'us'", selected_fields=("id",)).to_polars()

    assert sorted(result["id"].to_list()) == [1, 3]
