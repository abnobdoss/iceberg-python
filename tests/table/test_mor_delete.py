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
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError
from pyiceberg.expressions import EqualTo, In
from pyiceberg.manifest import DataFile, DataFileContent, ManifestContent, ManifestFile
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table, TableProperties
from pyiceberg.table.snapshots import Operation
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

ICEBERG_SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "data", StringType(), required=True),
)
PARTITIONED_SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=True),
    NestedField(2, "data", StringType(), required=True),
    NestedField(3, "category", StringType(), required=True),
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("data", pa.string(), nullable=False),
    ]
)
PARTITIONED_ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("data", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
    ]
)
PARTITION_SPEC = PartitionSpec(PartitionField(source_id=3, field_id=1000, transform=IdentityTransform(), name="category"))


def _arrow_table(rows: list[dict[str, object]], schema: pa.Schema = ARROW_SCHEMA) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _create_table(
    catalog: Catalog,
    identifier: str,
    *,
    format_version: int = 2,
    merge_on_read: bool = True,
    schema: Schema = ICEBERG_SCHEMA,
    partition_spec: PartitionSpec | None = None,
) -> Table:
    try:
        catalog.create_namespace("default")
    except NamespaceAlreadyExistsError:
        pass

    properties = {TableProperties.FORMAT_VERSION: str(format_version)}
    if merge_on_read:
        properties[TableProperties.DELETE_MODE] = TableProperties.DELETE_MODE_MERGE_ON_READ

    return catalog.create_table(
        identifier=identifier,
        schema=schema,
        partition_spec=partition_spec or PartitionSpec(),
        properties=properties,
    )


def _append_rows(table: Table, rows: list[dict[str, object]], schema: pa.Schema = ARROW_SCHEMA) -> None:
    table.append(_arrow_table(rows, schema=schema))


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


def _current_delete_files(table: Table) -> list[DataFile]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return [
        entry.data_file
        for manifest in snapshot.manifests(io=table.io)
        if manifest.content == ManifestContent.DELETES
        for entry in manifest.fetch_manifest_entry(io=table.io)
        if entry.data_file.content == DataFileContent.POSITION_DELETES
    ]


def _current_manifests_with_content(table: Table, content: ManifestContent) -> list[ManifestFile]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return [manifest for manifest in snapshot.manifests(io=table.io) if manifest.content == content]


def _data_paths(table: Table) -> set[str]:
    return {data_file.file_path for data_file in _current_data_files(table)}


def _rows(table: Table) -> list[dict[str, object]]:
    return sorted(table.scan().to_arrow().to_pylist(), key=lambda row: (row["id"], row.get("data", "")))


def test_mor_delete_basic_produces_position_delete_without_rewriting_data_files(catalog: Catalog) -> None:
    identifier = "default.test_mor_delete_basic"
    table = _create_table(catalog, identifier)
    _append_rows(
        table,
        [
            {"id": 1, "data": "a"},
            {"id": 2, "data": "b"},
            {"id": 3, "data": "c"},
        ],
    )
    before_data_paths = _data_paths(table)

    table.delete(EqualTo("id", 2))

    assert _rows(table) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]
    assert _data_paths(table) == before_data_paths
    assert len(_current_delete_files(table)) == 1
    assert len(_current_manifests_with_content(table, ManifestContent.DELETES)) == 1

    current_snapshot = table.current_snapshot()
    assert current_snapshot is not None
    assert current_snapshot.summary is not None
    assert current_snapshot.summary.operation == Operation.OVERWRITE

    reloaded = catalog.load_table(identifier)
    assert _rows(reloaded) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]


def test_mor_delete_multiple_rows_in_one_file(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_mor_delete_multiple_rows")
    _append_rows(
        table,
        [
            {"id": 1, "data": "a"},
            {"id": 2, "data": "b"},
            {"id": 3, "data": "c"},
            {"id": 4, "data": "d"},
        ],
    )

    table.delete(In("id", (2, 4)))

    assert _rows(table) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]
    delete_files = _current_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 2


def test_partitioned_mor_delete_writes_delete_file_per_affected_partition(catalog: Catalog) -> None:
    table = _create_table(
        catalog,
        "default.test_partitioned_mor_delete",
        schema=PARTITIONED_SCHEMA,
        partition_spec=PARTITION_SPEC,
    )
    _append_rows(
        table,
        [
            {"id": 1, "data": "a", "category": "alpha"},
            {"id": 2, "data": "b", "category": "alpha"},
            {"id": 3, "data": "c", "category": "beta"},
            {"id": 4, "data": "d", "category": "beta"},
            {"id": 5, "data": "e", "category": "gamma"},
        ],
        schema=PARTITIONED_ARROW_SCHEMA,
    )
    before_data_paths = _data_paths(table)

    table.delete(In("id", (2, 4)))

    assert _rows(table) == [
        {"id": 1, "data": "a", "category": "alpha"},
        {"id": 3, "data": "c", "category": "beta"},
        {"id": 5, "data": "e", "category": "gamma"},
    ]
    assert _data_paths(table) == before_data_paths

    delete_files = _current_delete_files(table)
    assert len(delete_files) == 2
    assert {delete_file.partition[0] for delete_file in delete_files} == {"alpha", "beta"}
    assert {delete_file.record_count for delete_file in delete_files} == {1}
    assert "gamma" not in {delete_file.partition[0] for delete_file in delete_files}


def test_mor_delete_sequence_number_scopes_delete_to_existing_data_files(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_mor_delete_sequence_number_scoping")
    _append_rows(
        table,
        [
            {"id": 1, "data": "old-a"},
            {"id": 2, "data": "old-b"},
            {"id": 3, "data": "old-c"},
        ],
    )

    table.delete(EqualTo("id", 2))
    delete_snapshot = table.current_snapshot()
    assert delete_snapshot is not None
    assert delete_snapshot.sequence_number is not None

    # The position-delete file's sequence number must equal the delete snapshot's sequence
    # number. This is what scopes the delete to pre-existing data files only: a positional
    # delete applies to data with a data-sequence-number <= the delete's sequence number.
    delete_entry_sequence_numbers = [
        entry.sequence_number
        for manifest in delete_snapshot.manifests(io=table.io)
        if manifest.content == ManifestContent.DELETES
        for entry in manifest.fetch_manifest_entry(io=table.io)
    ]
    assert delete_entry_sequence_numbers == [delete_snapshot.sequence_number]

    _append_rows(table, [{"id": 2, "data": "new-b"}, {"id": 4, "data": "new-d"}])

    assert _rows(table) == [
        {"id": 1, "data": "old-a"},
        {"id": 2, "data": "new-b"},
        {"id": 3, "data": "old-c"},
        {"id": 4, "data": "new-d"},
    ]
    assert len(_current_delete_files(table)) == 1

    append_snapshot = table.current_snapshot()
    assert append_snapshot is not None
    assert append_snapshot.sequence_number == delete_snapshot.sequence_number + 1
    # The newly appended data has a strictly higher data sequence number than the delete
    # file, so the re-inserted id=2 survives even though it shares the deleted value.


def test_successive_mor_deletes_do_not_reemit_already_deleted_positions(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_successive_mor_deletes")
    _append_rows(
        table,
        [
            {"id": 1, "data": "a"},
            {"id": 2, "data": "b"},
            {"id": 3, "data": "c"},
            {"id": 4, "data": "d"},
        ],
    )

    table.delete(EqualTo("id", 2))
    table.delete(In("id", (2, 4)))

    assert _rows(table) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]
    delete_files = _current_delete_files(table)
    assert len(delete_files) == 2
    assert sum(delete_file.record_count for delete_file in delete_files) == 2
    assert sorted(delete_file.record_count for delete_file in delete_files) == [1, 1]


def test_default_delete_mode_is_copy_on_write(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_default_delete_mode_is_copy_on_write", merge_on_read=False)
    _append_rows(
        table,
        [
            {"id": 1, "data": "a"},
            {"id": 2, "data": "b"},
            {"id": 3, "data": "c"},
        ],
    )
    before_data_paths = _data_paths(table)

    table.delete(EqualTo("id", 2))

    assert _rows(table) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]
    assert _data_paths(table) != before_data_paths
    assert _current_delete_files(table) == []


def test_mor_delete_on_v1_warns_and_falls_back_to_copy_on_write(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_mor_delete_v1_fallback", format_version=1)
    _append_rows(
        table,
        [
            {"id": 1, "data": "a"},
            {"id": 2, "data": "b"},
            {"id": 3, "data": "c"},
        ],
    )
    before_data_paths = _data_paths(table)

    with pytest.warns(UserWarning, match="Merge on read is not yet supported, falling back to copy-on-write"):
        table.delete(EqualTo("id", 2))

    assert _rows(table) == [{"id": 1, "data": "a"}, {"id": 3, "data": "c"}]
    assert _data_paths(table) != before_data_paths
    assert _current_delete_files(table) == []


def test_mor_delete_no_match_warns_and_does_not_create_snapshot(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_mor_delete_no_match")
    _append_rows(table, [{"id": 1, "data": "a"}, {"id": 2, "data": "b"}])
    before_snapshot = table.current_snapshot()
    assert before_snapshot is not None
    before_snapshot_count = len(table.snapshots())

    with pytest.warns(UserWarning, match="Delete operation did not match any records"):
        table.delete(EqualTo("id", 99))

    after_snapshot = table.current_snapshot()
    assert after_snapshot is not None
    assert after_snapshot.snapshot_id == before_snapshot.snapshot_id
    assert len(table.snapshots()) == before_snapshot_count
    assert _current_delete_files(table) == []


def test_mor_delete_with_user_column_named_like_internal_position(catalog: Catalog) -> None:
    # The MoR delete path appends a temporary position column to compute matching rows.
    # If a real table column shares that internal name, the path must not collide or
    # mistakenly read the user column as positions.
    schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "__pyiceberg_position", LongType(), required=True),
    )
    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("__pyiceberg_position", pa.int64(), nullable=False),
        ]
    )
    table = _create_table(catalog, "default.test_mor_delete_position_name_collision", schema=schema)
    _append_rows(
        table,
        [
            {"id": 1, "__pyiceberg_position": 100},
            {"id": 2, "__pyiceberg_position": 200},
            {"id": 3, "__pyiceberg_position": 300},
        ],
        schema=arrow_schema,
    )
    before_data_paths = _data_paths(table)

    table.delete(EqualTo("id", 2))

    assert sorted(table.scan().to_arrow().to_pylist(), key=lambda row: row["id"]) == [
        {"id": 1, "__pyiceberg_position": 100},
        {"id": 3, "__pyiceberg_position": 300},
    ]
    assert _data_paths(table) == before_data_paths
    assert len(_current_delete_files(table)) == 1


def test_mor_delete_position_alignment_across_multiple_record_batches(catalog: Catalog) -> None:
    # A position-delete `pos` is a GLOBAL 0-based physical row index into the data file.
    # When a single data file is read in multiple record batches, the running position
    # accounting must be global, not batch-local. Force small row groups so the file is
    # read in many batches, then delete a row well past the first batch and assert exactly
    # that row is gone. A batch-local position bug would silently delete the wrong row.
    table = _create_table(
        catalog,
        "default.test_mor_delete_multi_batch_alignment",
        # Small parquet row group + batch size so a single file spans many batches on read.
    )
    table = (
        table.transaction()
        .set_properties(
            {
                TableProperties.PARQUET_ROW_GROUP_LIMIT: "8",
                TableProperties.PARQUET_PAGE_ROW_LIMIT: "8",
            }
        )
        .commit_transaction()
    )

    rows = [{"id": i, "data": f"v{i}"} for i in range(200)]
    _append_rows(table, rows)
    data_files = _current_data_files(table)
    assert len(data_files) == 1  # single physical file; positions are global within it
    before_paths = _data_paths(table)

    # Guard the test's own premise: the single file must actually be read in multiple
    # batches, otherwise a batch-local position regression would never be exercised.
    batch_count = sum(1 for _ in table.scan().to_arrow_batch_reader())
    assert batch_count > 1, "expected the data file to be read in multiple record batches"

    # id=137 sits far past the first batch; its physical position equals 137.
    table.delete(EqualTo("id", 137))

    remaining_ids = sorted(row["id"] for row in table.scan().to_arrow().to_pylist())
    assert 137 not in remaining_ids
    assert remaining_ids == [i for i in range(200) if i != 137]
    # No rewrite, and exactly one matched position recorded.
    assert _data_paths(table) == before_paths
    delete_files = _current_delete_files(table)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 1
