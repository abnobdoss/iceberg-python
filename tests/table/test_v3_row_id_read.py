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

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.expressions import AlwaysTrue, GreaterThan
from pyiceberg.expressions.visitors import bind
from pyiceberg.io.pyarrow import PYARROW_PARQUET_FIELD_ID_KEY, PyArrowFileIO, _task_to_record_batches
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat, ManifestContent, ManifestEntry, ManifestEntryStatus
from pyiceberg.schema import ROW_ID_FIELD, ROW_ID_FIELD_ID, Schema
from pyiceberg.table import FileScanTask, _open_manifest
from pyiceberg.types import IntegerType, NestedField, StringType

SCHEMA = Schema(
    NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
    NestedField(field_id=2, name="name", field_type=StringType(), required=False),
)

ARROW_SCHEMA = pa.schema([
    pa.field("id", pa.int32(), nullable=True),
    pa.field("name", pa.string(), nullable=True),
])


def _batch(ids: list[int]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"id": row_id, "name": f"row-{row_id}"} for row_id in ids],
        schema=ARROW_SCHEMA,
    )


def _create_table(tmp_path: Path, format_version: str = "3"):
    catalog = InMemoryCatalog(f"row-id-read-{format_version}", warehouse=f"file://{tmp_path}")
    catalog.create_namespace("ns")
    table = catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": format_version})
    return catalog, table


def _data_sequence_numbers(table) -> set[int]:
    snapshot = table.metadata.current_snapshot()
    assert snapshot is not None

    sequence_numbers: set[int] = set()
    for manifest in snapshot.manifests(table.io):
        if manifest.content != ManifestContent.DATA:
            continue
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            assert entry.sequence_number is not None
            sequence_numbers.add(entry.sequence_number)

    return sequence_numbers


def _manifest_entry(status: ManifestEntryStatus, file_path: str, record_count: int) -> ManifestEntry:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=file_path,
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=record_count,
        file_size_in_bytes=1,
    )
    return ManifestEntry.from_args(status=status, data_file=data_file)


def test_v3_scan_select_row_id(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path)
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    result = table.scan().select("_row_id").to_arrow()

    assert result.column_names == ["_row_id"]
    assert result.column("_row_id").to_pylist() == [0, 1, 2]


def test_open_manifest_row_id_inheritance_counts_deleted_null_first_row_id_entries() -> None:
    deleted_entry = _manifest_entry(ManifestEntryStatus.DELETED, "deleted.parquet", 5)
    live_entry = _manifest_entry(ManifestEntryStatus.ADDED, "live.parquet", 2)

    class ManifestWithDeletedEntry:
        content = ManifestContent.DATA
        first_row_id = 100

        def __init__(self) -> None:
            self.discard_deleted: bool | None = None

        def fetch_manifest_entry(self, io, discard_deleted: bool = True):
            self.discard_deleted = discard_deleted
            entries = [deleted_entry, live_entry]
            if discard_deleted:
                return [entry for entry in entries if entry.status != ManifestEntryStatus.DELETED]
            return entries

    manifest = ManifestWithDeletedEntry()

    entries = _open_manifest(object(), manifest, lambda _: True, lambda _: True)

    assert manifest.discard_deleted is False
    assert [entry.data_file.file_path for entry in entries] == ["live.parquet"]
    assert deleted_entry.data_file.first_row_id == 100
    assert live_entry.data_file.first_row_id == 105


def test_v3_scan_row_ids_continue_across_appends(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path)
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")
    table.append(_batch([4, 5]))
    table = catalog.load_table("ns.t")

    rows = table.scan().select("id", "_row_id").to_arrow().to_pylist()

    assert {row["id"]: row["_row_id"] for row in rows} == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def test_v3_scan_projects_row_id_alongside_real_columns(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path)
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    plain = table.scan().select("id", "name").to_arrow()
    with_row_id = table.scan().select("id", "name", "_row_id").to_arrow()

    assert with_row_id.column_names == ["id", "name", "_row_id"]
    assert with_row_id.select(["id", "name"]).to_pylist() == plain.to_pylist()
    assert with_row_id.column("_row_id").to_pylist() == [0, 1, 2]


def test_v3_scan_select_last_updated_sequence_number(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path)
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    sequence_numbers = _data_sequence_numbers(table)
    assert len(sequence_numbers) == 1
    data_sequence_number = next(iter(sequence_numbers))

    result = table.scan().select("_last_updated_sequence_number").to_arrow()

    assert result.column_names == ["_last_updated_sequence_number"]
    assert result.column("_last_updated_sequence_number").to_pylist() == [data_sequence_number] * 3


def test_v2_scan_select_row_id_raises(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path, format_version="2")
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    with pytest.raises(ValueError, match="only available for v3"):
        table.scan().select("_row_id").to_arrow()


def test_v2_scan_select_last_updated_sequence_number_raises(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path, format_version="2")
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    with pytest.raises(ValueError, match="only available for v3"):
        table.scan().select("_last_updated_sequence_number").to_arrow()


def test_v3_select_star_does_not_include_row_id(tmp_path: Path) -> None:
    catalog, table = _create_table(tmp_path)
    table.append(_batch([1, 2, 3]))
    table = catalog.load_table("ns.t")

    result = table.scan().select("*").to_arrow()

    assert result.column_names == ["id", "name"]


def test_row_id_positions_survive_positional_deletes_and_filter(tmp_path: Path) -> None:
    arrow_schema = pa.schema((pa.field("id", pa.int32(), nullable=True, metadata={PYARROW_PARQUET_FIELD_ID_KEY: "1"}),))
    arrow_table = pa.table([pa.array([1, 2, 3, 4, 5], type=pa.int32())], schema=arrow_schema)
    file_path = str(tmp_path / "row-id-positional-filter.parquet")
    with pq.ParquetWriter(file_path, arrow_schema) as writer:
        writer.write_table(arrow_table)

    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=file_path,
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=len(arrow_table),
        file_size_in_bytes=22,
    )
    data_file.first_row_id = 10

    table_schema = Schema(NestedField(1, "id", IntegerType(), required=False))
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=False), ROW_ID_FIELD)
    positional_deletes = [pa.chunked_array([pa.array([1, 3], type=pa.int64())])]

    batches = list(
        _task_to_record_batches(
            PyArrowFileIO(),
            FileScanTask(data_file, data_sequence_number=7),
            bound_row_filter=bind(table_schema, GreaterThan("id", 2), case_sensitive=True),
            projected_schema=projected_schema,
            table_schema=table_schema,
            projected_field_ids={1, ROW_ID_FIELD_ID},
            positional_deletes=positional_deletes,
            case_sensitive=True,
        )
    )

    assert len(batches) == 1
    assert batches[0].to_pydict() == {"id": [3, 5], "_row_id": [12, 14]}


def test_row_id_null_when_first_row_id_missing(tmp_path: Path) -> None:
    arrow_schema = pa.schema((pa.field("id", pa.int32(), nullable=True, metadata={PYARROW_PARQUET_FIELD_ID_KEY: "1"}),))
    arrow_table = pa.table([pa.array([1, 2, 3, 4, 5], type=pa.int32())], schema=arrow_schema)
    file_path = str(tmp_path / "row-id-null-first-row-id.parquet")
    with pq.ParquetWriter(file_path, arrow_schema) as writer:
        writer.write_table(arrow_table)

    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=file_path,
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=len(arrow_table),
        file_size_in_bytes=22,
    )
    # Pre-upgrade snapshots have a null first_row_id; _row_id must read as null per the v3 spec.
    data_file.first_row_id = None

    table_schema = Schema(NestedField(1, "id", IntegerType(), required=False))
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=False), ROW_ID_FIELD)

    batches = list(
        _task_to_record_batches(
            PyArrowFileIO(),
            FileScanTask(data_file, data_sequence_number=7),
            bound_row_filter=AlwaysTrue(),
            projected_schema=projected_schema,
            table_schema=table_schema,
            projected_field_ids={1, ROW_ID_FIELD_ID},
            positional_deletes=None,
            case_sensitive=True,
        )
    )

    assert len(batches) == 1
    assert batches[0].to_pydict() == {"id": [1, 2, 3, 4, 5], "_row_id": [None, None, None, None, None]}
