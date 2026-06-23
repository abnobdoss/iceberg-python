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
"""Acceptance tests for v3 write gate + row-lineage assignment.

T1-foundation: exercises the real local write path end to end, asserting that
v3 row lineage (next-row-id / first-row-id / added-rows) is correct and
monotonic across multiple commits, that next-row-id never silently falls back
to None, and that the full v3 metadata round-trips through JSON.
"""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.manifest import ManifestContent
from pyiceberg.schema import Schema
from pyiceberg.table.metadata import TableMetadataUtil, TableMetadataV3
from pyiceberg.types import IntegerType, NestedField, StringType


@pytest.fixture
def v3_catalog(tmp_path: Path) -> Catalog:
    return InMemoryCatalog("t1", warehouse=f"file://{tmp_path}")


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
        [{"id": i, "name": f"row-{i}"} for i in ids],
        schema=ARROW_SCHEMA,
    )


def test_v3_table_creation_starts_next_row_id_at_zero(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    assert tbl.metadata.format_version == 3
    assert tbl.metadata.next_row_id == 0


def test_v3_append_twice_row_lineage_is_monotonic(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})

    assert tbl.metadata.next_row_id == 0
    assert tbl.metadata.next_row_id is not None

    # First append: 3 rows.
    tbl.append(_batch([1, 2, 3]))
    tbl = v3_catalog.load_table("ns.t")

    snap1 = tbl.metadata.current_snapshot()
    assert snap1 is not None
    assert snap1.first_row_id == 0, "first snapshot must start assigning at row id 0"
    assert snap1.added_rows == 3, "added_rows must reflect the 3 rows written, not None"
    assert tbl.metadata.next_row_id == 3, "next_row_id must advance by added rows"
    assert tbl.metadata.next_row_id is not None, "next_row_id must NEVER fall back to None on v3"

    # Second append: 2 rows.
    tbl.append(_batch([4, 5]))
    tbl = v3_catalog.load_table("ns.t")

    snaps = sorted(tbl.metadata.snapshots, key=lambda s: s.sequence_number or 0)
    snap2 = snaps[-1]
    assert snap2.first_row_id == 3, "second snapshot must start where the first left off"
    assert snap2.added_rows == 2
    assert tbl.metadata.next_row_id == 5, "next_row_id must be strictly monotonically increasing"

    # Monotonicity across all snapshots: each first_row_id + added_rows chains.
    running = 0
    for s in snaps:
        assert s.first_row_id is not None
        assert s.added_rows is not None
        assert s.first_row_id == running, f"gap/overlap in row-id assignment at snapshot {s.snapshot_id}"
        running += s.added_rows
    assert running == tbl.metadata.next_row_id


def test_v3_merge_append_does_not_double_count_existing_rows(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table(
        "ns.t",
        schema=SCHEMA,
        properties={
            "format-version": "3",
            "commit.manifest-merge.enabled": "true",
            "commit.manifest.min-count-to-merge": "1",
        },
    )

    for ids in ([1, 2, 3], [4, 5], [6, 7, 8, 9]):
        tbl.append(_batch(ids))
        tbl = v3_catalog.load_table("ns.t")

    snaps = sorted(tbl.metadata.snapshots, key=lambda s: s.sequence_number or 0)
    added_rows = [s.added_rows for s in snaps]

    assert tbl.metadata.next_row_id == 9
    assert added_rows == [3, 2, 4]
    assert all(rows is not None for rows in added_rows)
    assert sum(rows for rows in added_rows if rows is not None) == tbl.metadata.next_row_id


def test_v3_manifest_carries_first_row_id(v3_catalog: Catalog) -> None:
    """The data manifest in the manifest list must be assigned a first_row_id per spec."""
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    tbl.append(_batch([1, 2, 3]))
    tbl = v3_catalog.load_table("ns.t")

    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    manifests = snap.manifests(tbl.io)
    data_manifests = [m for m in manifests if m.content == ManifestContent.DATA]
    assert len(data_manifests) >= 1
    # The first (only) data manifest must carry the snapshot's first_row_id (0).
    assert data_manifests[0].first_row_id == 0


def test_v3_metadata_round_trips_through_json(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    tbl.append(_batch([1, 2, 3]))
    tbl.append(_batch([4, 5]))
    tbl = v3_catalog.load_table("ns.t")

    meta = tbl.metadata
    assert isinstance(meta, TableMetadataV3)

    dumped = meta.model_dump_json()
    reparsed = TableMetadataUtil.parse_raw(dumped)
    assert isinstance(reparsed, TableMetadataV3)
    assert reparsed.next_row_id == meta.next_row_id == 5
    # Round-trip equality of the model.
    assert reparsed == meta
    # The serialized JSON must include next-row-id.
    assert json.loads(dumped)["next-row-id"] == 5


def test_v3_merge_preserves_row_ids_after_delete_creates_gaps(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table(
        "ns.t",
        schema=SCHEMA,
        properties={
            "format-version": "3",
            "commit.manifest-merge.enabled": "true",
            "commit.manifest.min-count-to-merge": "2",
        },
    )

    tbl.append(_batch([1, 2, 3, 4, 5, 6]))
    tbl = v3_catalog.load_table("ns.t")
    tbl.append(_batch([7, 8]))
    tbl = v3_catalog.load_table("ns.t")

    tbl.delete(delete_filter="id in (2, 3)")
    tbl = v3_catalog.load_table("ns.t")

    tbl.append(_batch([9, 10]))
    tbl = v3_catalog.load_table("ns.t")
    tbl.append(_batch([11, 12]))
    tbl = v3_catalog.load_table("ns.t")

    distinct_rows_ever_appended = set(range(1, 13))
    expected_surviving_ids = distinct_rows_ever_appended - {2, 3}
    actual_ids = [row["id"] for row in tbl.scan().to_arrow().to_pylist()]

    assert tbl.metadata.next_row_id is not None
    assert tbl.metadata.next_row_id >= len(distinct_rows_ever_appended)
    assert all(snapshot.added_rows is not None for snapshot in tbl.metadata.snapshots)
    assert len(actual_ids) == len(expected_surviving_ids)
    assert set(actual_ids) == expected_surviving_ids
