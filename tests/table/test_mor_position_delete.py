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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.expressions import AlwaysTrue, EqualTo, In
from pyiceberg.manifest import DataFile, DataFileContent, ManifestContent
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table, TableProperties
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

ICEBERG_SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "category", StringType(), required=False),
    NestedField(3, "payload", StringType(), required=False),
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("category", pa.string(), nullable=True),
        pa.field("payload", pa.string(), nullable=True),
    ]
)
MOR_PROPERTIES = {
    TableProperties.FORMAT_VERSION: "2",
    TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ,
}


def _ensure_namespace(catalog: Catalog) -> None:
    if not catalog.namespace_exists("default"):
        catalog.create_namespace("default")


def _create_table(
    catalog: Catalog,
    name: str,
    properties: dict[str, str] | None = None,
    partition_spec: PartitionSpec | None = None,
) -> Table:
    _ensure_namespace(catalog)
    return catalog.create_table(
        f"default.{name}",
        ICEBERG_SCHEMA,
        partition_spec=partition_spec or PartitionSpec(),
        properties=properties or MOR_PROPERTIES,
    )


def _append_rows(table: Table, rows: list[tuple[int, str, str]]) -> None:
    table.append(
        pa.Table.from_pylist(
            [{"id": row[0], "category": row[1], "payload": row[2]} for row in rows],
            schema=ARROW_SCHEMA,
        )
    )


def _visible_rows(table: Table) -> list[tuple[int, str, str]]:
    rows = table.scan().to_arrow().sort_by([("id", "ascending")]).to_pylist()
    return [(row["id"], row["category"], row["payload"]) for row in rows]


def _content_files(table: Table, content: DataFileContent) -> list[DataFile]:
    snapshot = table.current_snapshot()
    if snapshot is None:
        return []

    return [
        entry.data_file
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io)
        if entry.data_file.content == content
    ]


def _data_file_paths(table: Table) -> list[str]:
    return sorted(file.file_path for file in _content_files(table, DataFileContent.DATA))


def _position_delete_files(table: Table) -> list[DataFile]:
    return _content_files(table, DataFileContent.POSITION_DELETES)


def _delete_manifests(table: Table):
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return [manifest for manifest in snapshot.manifests(table.io) if manifest.content == ManifestContent.DELETES]


def test_single_row_mor_delete(catalog: Catalog) -> None:
    table = _create_table(catalog, "single_row")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])
    data_paths_before = _data_file_paths(table)

    table.delete(EqualTo("id", 2))

    assert _visible_rows(table) == [(1, "a", "one"), (3, "a", "three")]
    assert _data_file_paths(table) == data_paths_before
    snapshot = table.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary.operation == Operation.OVERWRITE

    delete_files = _position_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 1
    assert len(_delete_manifests(table)) == 1

    with table.io.new_input(delete_files[0].file_path).open() as input_file:
        delete_schema = pq.read_schema(input_file)

    assert delete_schema.field("file_path").type == pa.string()
    assert not delete_schema.field("file_path").nullable
    assert delete_schema.field("file_path").metadata[b"PARQUET:field_id"] == b"2147483546"
    assert delete_schema.field("pos").type == pa.int64()
    assert not delete_schema.field("pos").nullable
    assert delete_schema.field("pos").metadata[b"PARQUET:field_id"] == b"2147483545"


def test_delete_file_path_bound_is_full_untruncated_path(catalog: Catalog) -> None:
    table = _create_table(catalog, "path_bound")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])

    table.delete(EqualTo("id", 2))

    delete_files = _position_delete_files(table)
    assert len(delete_files) == 1
    delete_file = delete_files[0]
    data_path = _data_file_paths(table)[0]

    path_field_id = 2147483546
    lower = delete_file.lower_bounds[path_field_id].decode("utf-8")
    upper = delete_file.upper_bounds[path_field_id].decode("utf-8")
    # The reader only routes through the exact-path delete index when lower == upper == the full
    # data-file path; a truncated string bound would silently fall back to the partition index.
    assert lower == upper == data_path


def test_multi_row_mor_delete(catalog: Catalog) -> None:
    table = _create_table(catalog, "multi_row")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three"), (4, "a", "four")])
    data_paths_before = _data_file_paths(table)

    table.delete(In("id", [2, 4]))

    assert _visible_rows(table) == [(1, "a", "one"), (3, "a", "three")]
    assert _data_file_paths(table) == data_paths_before
    delete_files = _position_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 2


def test_empty_match_warns_and_writes_nothing(catalog: Catalog) -> None:
    table = _create_table(catalog, "empty_match")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])
    snapshot_before = table.current_snapshot()
    assert snapshot_before is not None

    with pytest.warns(UserWarning, match="Delete operation did not match any records"):
        table.delete(EqualTo("id", 999))

    snapshot_after = table.current_snapshot()
    assert snapshot_after is not None
    assert snapshot_after.snapshot_id == snapshot_before.snapshot_id
    assert _position_delete_files(table) == []
    assert _delete_manifests(table) == []


def test_delete_all_rows_writes_full_delete_file_not_drop(catalog: Catalog) -> None:
    table = _create_table(catalog, "delete_all")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])
    data_paths_before = _data_file_paths(table)

    table.delete(AlwaysTrue())

    assert _data_file_paths(table) == data_paths_before
    delete_files = _position_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 3
    assert table.scan().to_arrow().num_rows == 0


def test_multiple_data_files_one_partition(catalog: Catalog) -> None:
    table = _create_table(catalog, "multiple_data_files_one_partition")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two")])
    _append_rows(table, [(3, "b", "three"), (4, "b", "four")])
    data_paths_before = _data_file_paths(table)
    assert len(data_paths_before) == 2

    table.delete(In("id", [2, 3]))

    assert _data_file_paths(table) == data_paths_before
    delete_files = _position_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 2

    with table.io.new_input(delete_files[0].file_path).open() as input_file:
        delete_table = pq.read_table(input_file)

    assert set(delete_table["file_path"].to_pylist()) == set(data_paths_before)
    assert len(set(delete_table["file_path"].to_pylist())) == 2
    assert sorted(delete_table["pos"].to_pylist()) == [0, 1]
    assert _visible_rows(table) == [(1, "a", "one"), (4, "b", "four")]


def test_partitioned_mor_delete_writes_delete_files_per_partition(catalog: Catalog) -> None:
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="category"))
    table = _create_table(catalog, "partitioned", partition_spec=spec)
    _append_rows(
        table,
        [(1, "a", "one"), (2, "a", "two"), (3, "b", "three"), (4, "b", "four"), (5, "c", "five")],
    )
    data_partitions = {file.partition for file in _content_files(table, DataFileContent.DATA)}
    data_paths_before = _data_file_paths(table)

    table.delete(In("id", [2, 4]))

    assert _visible_rows(table) == [(1, "a", "one"), (3, "b", "three"), (5, "c", "five")]
    assert _data_file_paths(table) == data_paths_before
    delete_files = _position_delete_files(table)
    assert len(delete_files) == 2
    assert {file.partition[0] for file in delete_files} == {"a", "b"}
    assert {file.partition for file in delete_files}.issubset(data_partitions)
    assert sorted(file.record_count for file in delete_files) == [1, 1]


def test_sequence_number_scoping_does_not_delete_later_appends(catalog: Catalog) -> None:
    properties = {**MOR_PROPERTIES, TableProperties.WRITE_TARGET_FILE_SIZE_BYTES: "1"}
    table = _create_table(catalog, "sequence_scoping", properties=properties)
    _append_rows(table, [(1, "delete", "a1"), (2, "keep", "a2"), (3, "delete", "a3")])

    table.delete(EqualTo("category", "delete"))
    delete_files_after_delete = _position_delete_files(table)
    assert len(delete_files_after_delete) == 1
    assert delete_files_after_delete[0].record_count == 2

    _append_rows(table, [(4, "delete", "b1"), (5, "keep", "b2")])

    assert _visible_rows(table) == [(2, "keep", "a2"), (4, "delete", "b1"), (5, "keep", "b2")]


def test_multiple_successive_mor_deletes(catalog: Catalog) -> None:
    table = _create_table(catalog, "successive")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three"), (4, "a", "four")])

    table.delete(EqualTo("id", 2))
    table.delete(In("id", [2, 4]))

    assert _visible_rows(table) == [(1, "a", "one"), (3, "a", "three")]
    delete_files = _position_delete_files(table)
    assert len(delete_files) == 2
    assert sorted(file.record_count for file in delete_files) == [1, 1]


def test_plain_scan_to_arrow_after_mor_delete(catalog: Catalog) -> None:
    table = _create_table(catalog, "plain_scan")
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])

    table.delete(EqualTo("id", 3))

    result = table.scan().to_arrow().sort_by([("id", "ascending")])
    assert result["id"].to_pylist() == [1, 2]
    assert result["payload"].to_pylist() == ["one", "two"]


def test_copy_on_write_default_still_rewrites_data_files(catalog: Catalog) -> None:
    table = _create_table(catalog, "cow_default", properties={TableProperties.FORMAT_VERSION: "2"})
    _append_rows(table, [(1, "a", "one"), (2, "a", "two"), (3, "a", "three")])
    data_paths_before = _data_file_paths(table)

    table.delete(EqualTo("id", 2))

    assert _visible_rows(table) == [(1, "a", "one"), (3, "a", "three")]
    assert _data_file_paths(table) != data_paths_before
    assert _position_delete_files(table) == []
