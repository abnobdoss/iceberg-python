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

import itertools
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.conversions import from_bytes, to_bytes
from pyiceberg.exceptions import NamespaceAlreadyExistsError
from pyiceberg.io.pyarrow import PYARROW_PARQUET_FIELD_ID_KEY, write_position_delete_file
from pyiceberg.manifest import DataFile, DataFileContent, ManifestContent, ManifestFile
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation
from pyiceberg.types import LongType, NestedField, StringType

ICEBERG_SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "data", StringType(), required=True),
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("data", pa.string(), nullable=False),
    ]
)


def _arrow_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)


def _create_v2_table(catalog: Catalog, identifier: str) -> Table:
    try:
        catalog.create_namespace("default")
    except NamespaceAlreadyExistsError:
        pass
    return catalog.create_table(identifier=identifier, schema=ICEBERG_SCHEMA, properties={"format-version": "2"})


def _append_initial_rows(table: Table) -> DataFile:
    table.append(
        _arrow_table(
            [
                {"id": 1, "data": "a"},
                {"id": 2, "data": "b"},
                {"id": 3, "data": "c"},
                {"id": 4, "data": "d"},
            ]
        )
    )
    data_files = _current_data_files(table)
    assert len(data_files) == 1
    return data_files[0]


def _current_data_files(table: Table) -> list[DataFile]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return [
        entry.data_file
        for manifest in snapshot.manifests(io=table.io)
        if manifest.content == ManifestContent.DATA
        for entry in manifest.fetch_manifest_entry(io=table.io)
        if entry.data_file.content == DataFileContent.DATA
    ]


def _current_manifests_with_content(table: Table, content: ManifestContent) -> list[ManifestFile]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return [manifest for manifest in snapshot.manifests(io=table.io) if manifest.content == content]


def _read_delete_parquet(table: Table, delete_file: DataFile) -> tuple[pa.Schema, pa.Table]:
    with table.io.new_input(delete_file.file_path).open() as input_file:
        parquet_file = pq.ParquetFile(input_file)
        return parquet_file.schema_arrow, parquet_file.read()


def _rows(table: Table) -> list[dict[str, object]]:
    return sorted(table.scan().to_arrow().to_pylist(), key=lambda row: row["id"])


def _commit_delete_file(table: Table, delete_file: DataFile) -> None:
    with table.transaction() as tx:
        with tx.update_snapshot().row_delta() as row_delta:
            row_delta.append_delete_file(delete_file)


def test_write_position_delete_file(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file")
    data_file = _append_initial_rows(table)

    delete_file = write_position_delete_file(table.io, table.metadata, data_file, [0, 2])

    assert delete_file.content == DataFileContent.POSITION_DELETES
    assert delete_file.equality_ids is None
    assert delete_file.record_count == 2
    assert delete_file.partition == data_file.partition
    assert delete_file.spec_id == data_file.spec_id
    assert delete_file.lower_bounds[2147483546] == to_bytes(StringType(), data_file.file_path)
    assert delete_file.upper_bounds[2147483546] == to_bytes(StringType(), data_file.file_path)

    schema, rows = _read_delete_parquet(table, delete_file)
    assert schema.field("file_path").metadata == {PYARROW_PARQUET_FIELD_ID_KEY: b"2147483546"}
    assert schema.field("file_path").type == pa.string()
    assert schema.field("pos").metadata == {PYARROW_PARQUET_FIELD_ID_KEY: b"2147483545"}
    assert schema.field("pos").type == pa.int64()
    assert rows.column("pos").to_pylist() == [0, 2]
    assert rows.column("file_path").to_pylist() == [data_file.file_path, data_file.file_path]


def test_write_position_delete_file_empty_positions_raises(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_empty_positions")
    data_file = _append_initial_rows(table)

    with pytest.raises(ValueError, match="Cannot write an empty position-delete file"):
        write_position_delete_file(table.io, table.metadata, data_file, [])


def test_write_position_delete_file_negative_position_raises(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_negative_position")
    data_file = _append_initial_rows(table)

    with pytest.raises(ValueError, match="non-negative"):
        write_position_delete_file(table.io, table.metadata, data_file, [-1])


def test_write_position_delete_file_rejects_non_data_reference(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_rejects_non_data_reference")
    data_file = _append_initial_rows(table)
    delete_file = write_position_delete_file(table.io, table.metadata, data_file, [0])

    with pytest.raises(ValueError, match="referenced_data_file must be a DATA file"):
        write_position_delete_file(table.io, table.metadata, delete_file, [0])


def test_write_position_delete_file_falls_back_to_default_spec_id(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_falls_back_to_default_spec_id")
    data_file = _append_initial_rows(table)
    fresh_data_file = DataFile.from_args(
        _table_format_version=table.metadata.format_version,
        content=data_file.content,
        file_path=data_file.file_path,
        file_format=data_file.file_format,
        partition=data_file.partition,
        record_count=data_file.record_count,
        file_size_in_bytes=data_file.file_size_in_bytes,
        column_sizes=data_file.column_sizes,
        value_counts=data_file.value_counts,
        null_value_counts=data_file.null_value_counts,
        nan_value_counts=data_file.nan_value_counts,
        lower_bounds=data_file.lower_bounds,
        upper_bounds=data_file.upper_bounds,
        key_metadata=data_file.key_metadata,
        split_offsets=data_file.split_offsets,
        equality_ids=data_file.equality_ids,
        sort_order_id=data_file.sort_order_id,
    )

    delete_file = write_position_delete_file(table.io, table.metadata, fresh_data_file, [0])

    assert delete_file.spec_id == table.metadata.default_spec_id


def test_write_position_delete_file_dedupes_and_sorts_positions(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_dedupes")
    data_file = _append_initial_rows(table)

    delete_file = write_position_delete_file(table.io, table.metadata, data_file, [2, 0, 2, 1])

    assert delete_file.record_count == 3
    _, rows = _read_delete_parquet(table, delete_file)
    assert rows.column("pos").to_pylist() == [0, 1, 2]


def test_row_delta_position_delete_end_to_end(catalog: Catalog) -> None:
    identifier = "default.test_row_delta_position_delete_end_to_end"
    table = _create_v2_table(catalog, identifier)
    data_file = _append_initial_rows(table)
    before_data_paths = {data_file.file_path for data_file in _current_data_files(table)}

    delete_file = write_position_delete_file(table.io, table.metadata, data_file, [0, 2])
    _commit_delete_file(table, delete_file)

    assert _rows(table) == [{"id": 2, "data": "b"}, {"id": 4, "data": "d"}]
    assert {data_file.file_path for data_file in _current_data_files(table)} == before_data_paths
    assert len(_current_manifests_with_content(table, ManifestContent.DELETES)) == 1

    current_snapshot = table.current_snapshot()
    assert current_snapshot is not None
    assert current_snapshot.summary.operation == Operation.OVERWRITE

    reloaded = catalog.load_table(identifier)
    assert _rows(reloaded) == [{"id": 2, "data": "b"}, {"id": 4, "data": "d"}]


def test_write_position_delete_file_accepts_large_position(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_write_position_delete_file_accepts_large_position")
    data_file = _append_initial_rows(table)
    large_position = 2**31 + 5

    delete_file = write_position_delete_file(table.io, table.metadata, data_file, [large_position])

    assert delete_file.record_count == 1
    _, rows = _read_delete_parquet(table, delete_file)
    assert rows.column("pos").to_pylist() == [large_position]


def test_append_delete_file_rejects_data_file(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_append_delete_file_rejects_data_file")
    data_file = _append_initial_rows(table)

    with pytest.raises(ValueError, match="append_delete_file requires a delete file"):
        _commit_delete_file(table, data_file)


def test_position_delete_sequence_number_does_not_affect_later_appends(catalog: Catalog) -> None:
    table = _create_v2_table(catalog, "default.test_position_delete_sequence_number_scoping")
    data_file = _append_initial_rows(table)
    first_snapshot = table.current_snapshot()
    assert first_snapshot is not None
    first_sequence_number = first_snapshot.sequence_number

    delete_file = write_position_delete_file(
        table.io,
        table.metadata,
        data_file,
        [0],
        write_uuid=uuid.uuid4(),
        counter=itertools.count(0),
    )
    _commit_delete_file(table, delete_file)
    row_delta_snapshot = table.current_snapshot()
    assert row_delta_snapshot is not None
    row_delta_sequence_number = row_delta_snapshot.sequence_number

    table.append(_arrow_table([{"id": 5, "data": "e"}, {"id": 6, "data": "f"}]))
    later_append_snapshot = table.current_snapshot()
    assert later_append_snapshot is not None
    later_append_sequence_number = later_append_snapshot.sequence_number

    assert row_delta_sequence_number == first_sequence_number + 1
    assert later_append_sequence_number == row_delta_sequence_number + 1
    assert _rows(table) == [
        {"id": 2, "data": "b"},
        {"id": 3, "data": "c"},
        {"id": 4, "data": "d"},
        {"id": 5, "data": "e"},
        {"id": 6, "data": "f"},
    ]


def test_row_delta_targets_only_referenced_file_among_many_in_one_partition(catalog: Catalog) -> None:
    # Two separate data files live in the SAME (empty) partition. A position delete
    # referencing file_one must remove ONLY a row of file_one; file_two must be untouched.
    # This guards against silent mis-routing if the file_path bound were ever truncated
    # (the DeleteFileIndex falls back to partition routing when lower != upper, which would
    # match BOTH files in the same partition and delete the wrong row).
    identifier = "default.test_row_delta_multi_file_one_partition"
    table = _create_v2_table(catalog, identifier)

    table.append(_arrow_table([{"id": 1, "data": "a"}, {"id": 2, "data": "b"}]))
    table.append(_arrow_table([{"id": 3, "data": "c"}, {"id": 4, "data": "d"}]))

    data_files = _current_data_files(table)
    assert len(data_files) == 2
    # Confirm both files share the same (empty) partition, so routing cannot rely on partition.
    assert data_files[0].partition == data_files[1].partition

    # file_one is the data file whose id-column lower bound is 1 (rows id=1,2).
    file_one = next(df for df in data_files if from_bytes(LongType(), df.lower_bounds[1]) == 1)
    before_data_paths = {df.file_path for df in data_files}

    # Delete pos 0 of file_one only (its first row, id=1).
    delete_file = write_position_delete_file(table.io, table.metadata, file_one, [0])
    _commit_delete_file(table, delete_file)

    # id=1 (file_one pos 0) gone; id=3 (file_two pos 0) MUST survive.
    assert _rows(table) == [{"id": 2, "data": "b"}, {"id": 3, "data": "c"}, {"id": 4, "data": "d"}]
    assert {df.file_path for df in _current_data_files(table)} == before_data_paths
