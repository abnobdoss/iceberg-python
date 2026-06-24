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
from pyiceberg.manifest import DataFile, ManifestContent, ManifestEntryStatus
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.metadata import TableMetadataUtil, TableMetadataV3
from pyiceberg.table.snapshots import Operation, Snapshot, Summary
from pyiceberg.table.update import AddSnapshotUpdate, update_table_metadata
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


def test_v3_merge_append_does_not_double_count_existing_rows(
    v3_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merge-append must actually MERGE manifests and must NOT double-count existing rows.

    This is the #3070 double-count regression guard. We instrument the merge manager so the
    test fails if no manifest merge ever happens (the historical bug: v3 merging was silently
    disabled by a descending-vs-ascending ordering check, so this fix was dead code).
    """
    from pyiceberg.table.update import snapshot as snapshot_module

    merge_calls = {"count": 0}
    original_create = snapshot_module._ManifestMergeManager._create_manifest

    def _counting_create(self, spec_id, manifest_bin):  # type: ignore[no-untyped-def]
        merge_calls["count"] += 1
        return original_create(self, spec_id, manifest_bin)

    monkeypatch.setattr(snapshot_module._ManifestMergeManager, "_create_manifest", _counting_create)

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

    # The merge path MUST have run at least once; otherwise the double-count fix is dead code.
    assert merge_calls["count"] > 0, "v3 manifest merge never ran — merging is silently disabled"

    # The data manifests must have been compacted (3 appends -> fewer than 3 DATA manifests).
    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    data_manifests = [m for m in snap.manifests(tbl.io) if m.content == ManifestContent.DATA]
    assert len(data_manifests) < 3, "manifests were not actually merged"

    snaps = sorted(tbl.metadata.snapshots, key=lambda s: s.sequence_number or 0)
    added_rows = [s.added_rows for s in snaps]

    # next_row_id == 9 proves no double-counting (9 real rows, not 17).
    assert tbl.metadata.next_row_id == 9
    assert added_rows == [3, 2, 4]
    assert all(rows is not None for rows in added_rows)
    assert sum(rows for rows in added_rows if rows is not None) == tbl.metadata.next_row_id

    # The merged manifests must tile [0, 9) exactly, with each data file's row range coherent.
    assigned = sorted(
        ((m.first_row_id, m.existing_rows_count + m.added_rows_count) for m in data_manifests),
        key=lambda pair: pair[0],
    )
    cursor = 0
    for first_row_id, rows in assigned:
        assert first_row_id == cursor, "merged manifest row-id ranges have a gap/overlap"
        cursor += rows
    assert cursor == 9

    # The data is still fully readable and correct after merging.
    actual_ids = sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist())
    assert actual_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    tbl.append(_batch([]))
    tbl = v3_catalog.load_table("ns.t")

    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    merged_manifests = [m for m in snap.manifests(tbl.io) if m.content == ManifestContent.DATA and m.existing_files_count]
    assert len(merged_manifests) == 1
    explicit_ranges = [
        (
            entry.data_file.first_row_id,
            entry.data_file.record_count,
        )
        for manifest in merged_manifests
        for entry in manifest.fetch_manifest_entry(tbl.io, discard_deleted=True)
    ]
    assert all(first_row_id is not None for first_row_id, _ in explicit_ranges)
    explicit_ranges = sorted(explicit_ranges)
    cursor = 0
    for first_row_id, rows in explicit_ranges:
        assert first_row_id == cursor
        cursor += rows
    assert cursor == 9


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


def _data_file_row_ids(tbl: object) -> list[tuple[int | None, int | None, int]]:
    """Return (manifest.first_row_id, data_file.first_row_id (field 142), record_count) for DATA entries."""
    snap = tbl.metadata.current_snapshot()  # type: ignore[attr-defined]
    out: list[tuple[int | None, int | None, int]] = []
    for m in snap.manifests(tbl.io):  # type: ignore[attr-defined]
        if m.content != ManifestContent.DATA:
            continue
        for entry in m.fetch_manifest_entry(tbl.io, discard_deleted=True):  # type: ignore[attr-defined]
            out.append((m.first_row_id, entry.data_file.first_row_id, entry.data_file.record_count))
    return out


def _data_files(tbl: Table) -> list[DataFile]:
    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    return [
        entry.data_file
        for manifest in snap.manifests(tbl.io)
        if manifest.content == ManifestContent.DATA
        for entry in manifest.fetch_manifest_entry(tbl.io, discard_deleted=True)
    ]


def _append_data_files_in_one_manifest(tbl: Table, id_groups: list[list[int]]) -> list[DataFile]:
    import itertools
    import uuid

    from pyiceberg.io.pyarrow import _dataframe_to_data_files

    data_files: list[DataFile] = []
    with tbl.transaction() as txn:
        with txn.update_snapshot().fast_append() as fast_append:
            for ids in id_groups:
                for data_file in _dataframe_to_data_files(
                    io=tbl.io,
                    df=_batch(ids),
                    table_metadata=txn.table_metadata,
                    write_uuid=uuid.uuid4(),
                    counter=itertools.count(),
                ):
                    data_files.append(data_file)
                    fast_append.append_data_file(data_file)
    return data_files


def test_v3_overwrite_delete_in_shared_manifest_preserves_survivor_row_ids(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})

    _append_data_files_in_one_manifest(tbl, [[0, 1, 2], [100, 101, 102]])
    tbl = v3_catalog.load_table("ns.t")
    first_file = _data_files(tbl)[0]
    assert tbl.metadata.next_row_id == 6
    assert [m[0] for m in _data_file_row_ids(tbl)] == [0, 0]

    with tbl.transaction() as txn:
        with txn.update_snapshot().overwrite() as overwrite:
            overwrite.delete_data_file(first_file)
    tbl = v3_catalog.load_table("ns.t")

    assert tbl.metadata.next_row_id == 6
    overwrite_snap = tbl.metadata.current_snapshot()
    assert overwrite_snap is not None
    assert overwrite_snap.added_rows == 0

    surviving = _data_file_row_ids(tbl)
    assert len(surviving) == 1
    manifest_frid, datafile_frid, rows = surviving[0]
    assert rows == 3
    assert manifest_frid == 0
    assert datafile_frid == 3
    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [100, 101, 102]


def test_v3_overwrite_delete_fails_when_survivor_row_id_cannot_be_preserved(tmp_path: Path) -> None:
    """When a survivor cannot have its row id preserved, the v3 overwrite must fail loudly.

    Genuine un-preservable case: two data files share ONE manifest written while the table was
    still v2 (manifest first_row_id is null and the data files have no field-142). After
    upgrading to v3, deleting one file via the overwrite path would rewrite the manifest with
    the survivor whose absolute row id is unknown, so the writer must NOT silently renumber it.
    """
    cat = InMemoryCatalog("t1undp", warehouse=f"file://{tmp_path}")
    cat.create_namespace("ns")
    tbl = cat.create_table("ns.t", schema=SCHEMA, properties={"format-version": "2"})
    _append_data_files_in_one_manifest(tbl, [[0, 1, 2], [100, 101, 102]])
    tbl = cat.load_table("ns.t")

    with tbl.transaction() as txn:
        txn.upgrade_table_version(3)
    tbl = cat.load_table("ns.t")
    first_file = _data_files(tbl)[0]

    with pytest.raises(NotImplementedError, match="row lineage"):
        with tbl.transaction() as txn:
            with txn.update_snapshot().overwrite() as overwrite:
                overwrite.delete_data_file(first_file)


def test_v3_whole_file_delete_does_not_renumber_surviving_rows(v3_catalog: Catalog) -> None:
    """Copy-on-write whole-file delete must NEVER re-number the surviving rows.

    Two separate data files are written (row ids [0,1,2] and [3,4,5]). Deleting the whole
    first file (predicate aligned to a full file) must:
      - leave next_row_id unchanged (0 new rows assigned), and
      - keep the surviving file's row-id lineage intact (manifest first_row_id == 3, and the
        materialized data-file field-142 first_row_id == 3) — NOT renumbered to 0.
    This asserts on the _row_id (field 142 / manifest first_row_id), not the user `id` column.
    """
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})

    # Two separate files: ids 0,1,2 (row ids 0-2) then ids 10,11,12 (row ids 3-5).
    tbl.append(_batch([0, 1, 2]))
    tbl = v3_catalog.load_table("ns.t")
    tbl.append(_batch([10, 11, 12]))
    tbl = v3_catalog.load_table("ns.t")
    assert tbl.metadata.next_row_id == 6

    before = sorted(_data_file_row_ids(tbl), key=lambda triple: triple[0] or 0)
    # manifest first_row_ids should be 0 and 3.
    assert [triple[0] for triple in before] == [0, 3]

    # Delete the entire first file (all ids < 5). This is a metadata-only (whole-file) delete.
    tbl.delete(delete_filter="id < 5")
    tbl = v3_catalog.load_table("ns.t")

    # next_row_id must NOT advance — zero rows were newly assigned.
    assert tbl.metadata.next_row_id == 6, "whole-file delete must not assign new row ids"
    delete_snap = tbl.metadata.current_snapshot()
    assert delete_snap is not None
    assert delete_snap.added_rows == 0, "a delete must report 0 added rows, not re-numbered survivors"

    # The surviving file must KEEP its original row-id lineage (manifest first_row_id == 3).
    surviving = _data_file_row_ids(tbl)
    assert len(surviving) == 1
    surviving_manifest_frid, surviving_datafile_frid, surviving_rows = surviving[0]
    assert surviving_rows == 3
    assert surviving_manifest_frid == 3, "surviving rows must keep their original first_row_id (3), not be renumbered"

    # Data correctness.
    actual_ids = sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist())
    assert actual_ids == [10, 11, 12]


def test_v3_whole_file_delete_with_two_survivors_renumbers_none(v3_catalog: Catalog) -> None:
    """Three files; delete the middle file wholesale; the two survivors keep their row ids."""
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    tbl.append(_batch([0, 1]))  # row ids 0-1
    tbl = v3_catalog.load_table("ns.t")
    tbl.append(_batch([10, 11]))  # row ids 2-3  (deleted)
    tbl = v3_catalog.load_table("ns.t")
    tbl.append(_batch([20, 21]))  # row ids 4-5
    tbl = v3_catalog.load_table("ns.t")
    assert tbl.metadata.next_row_id == 6

    # delete the middle file (ids 10,11) wholesale using a bound-provable range predicate
    # (min=10,max=11 fully inside [10,20)), so this is a metadata-only whole-file delete.
    tbl.delete(delete_filter="id >= 10 and id < 20")
    tbl = v3_catalog.load_table("ns.t")

    assert tbl.metadata.next_row_id == 6, "no new row ids on a whole-file delete"
    frids = sorted(triple[0] for triple in _data_file_row_ids(tbl))
    # survivors must keep first_row_ids 0 and 4 (NOT renumbered to 0 and 2).
    assert frids == [0, 4], "survivors were re-numbered after deleting the middle file"
    actual_ids = sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist())
    assert actual_ids == [0, 1, 20, 21]


def test_v3_whole_file_delete_in_shared_manifest_preserves_survivor_row_ids(v3_catalog: Catalog) -> None:
    """When two data files share ONE manifest and one is deleted wholesale, the survivor must
    keep its row-id lineage.

    This is the strongest delete-lineage guard: the surviving file is REWRITTEN into a new
    manifest (the shared source manifest is dropped), so without preserving lineage the
    manifest-list writer renumbers the survivor (the historical bug: next_row_id jumped 6->9,
    survivor block frid 0->6). The fix materializes the survivor's absolute _row_id into
    DataFile field 142 and inherits the source manifest's first_row_id.
    """
    import itertools
    import uuid

    from pyiceberg.io.pyarrow import _dataframe_to_data_files

    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})

    # Two data files in a SINGLE manifest (one fast-append commit): row ids 0-2 and 3-5.
    with tbl.transaction() as txn:
        with txn.update_snapshot().fast_append() as fast_append:
            for ids in ([0, 1, 2], [100, 101, 102]):
                for data_file in _dataframe_to_data_files(
                    io=tbl.io,
                    df=_batch(ids),
                    table_metadata=txn.table_metadata,
                    write_uuid=uuid.uuid4(),
                    counter=itertools.count(),
                ):
                    fast_append.append_data_file(data_file)
    tbl = v3_catalog.load_table("ns.t")
    assert tbl.metadata.next_row_id == 6
    # one shared manifest with first_row_id == 0
    assert [m[0] for m in _data_file_row_ids(tbl)] == [0, 0]

    # Whole-file delete the first file (ids 0,1,2): bound-provable (max=2 < 5).
    tbl.delete(delete_filter="id < 5")
    tbl = v3_catalog.load_table("ns.t")

    # next_row_id must NOT advance and no new rows are added.
    assert tbl.metadata.next_row_id == 6, "survivor was re-numbered (next_row_id advanced)"
    delete_snap = tbl.metadata.current_snapshot()
    assert delete_snap is not None
    assert delete_snap.added_rows == 0

    surviving = _data_file_row_ids(tbl)
    assert len(surviving) == 1
    manifest_frid, datafile_frid, rows = surviving[0]
    assert rows == 3
    # The rewritten manifest inherits the source manifest's first_row_id (0)...
    assert manifest_frid == 0, "rewritten manifest must inherit the source manifest first_row_id"
    # ...and the survivor's absolute _row_id (3) is materialized into field 142.
    assert datafile_frid == 3, "survivor's _row_id (field 142) must be materialized to its original value (3)"

    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [100, 101, 102]


def test_v3_whole_file_delete_materializes_deleted_entry_first_row_id(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})

    _append_data_files_in_one_manifest(tbl, [[0, 1, 2], [100, 101, 102]])
    tbl = v3_catalog.load_table("ns.t")
    assert _data_file_row_ids(tbl) == [(0, None, 3), (0, None, 3)]

    tbl.delete(delete_filter="id >= 100")
    tbl = v3_catalog.load_table("ns.t")

    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    deleted_entries = [
        entry
        for manifest in snap.manifests(tbl.io)
        if manifest.content == ManifestContent.DATA
        for entry in manifest.fetch_manifest_entry(tbl.io, discard_deleted=False)
        if entry.status == ManifestEntryStatus.DELETED
    ]

    assert len(deleted_entries) == 1
    deleted_file = deleted_entries[0].data_file
    assert deleted_file.record_count == 3
    assert deleted_file.first_row_id == 3


def test_v3_partial_rewrite_delete_fails_loudly(v3_catalog: Catalog) -> None:
    """A copy-on-write delete that needs to REWRITE a data file must fail loudly on v3.

    Preserving _row_id lineage across a physical rewrite needs materialized per-row _row_id
    columns, which PyIceberg does not have. Rather than silently re-numbering survivors, the
    v3 path raises NotImplementedError.
    """
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    # Single file containing 0..5; deleting a subset forces a partial rewrite.
    tbl.append(_batch([0, 1, 2, 3, 4, 5]))
    tbl = v3_catalog.load_table("ns.t")

    with pytest.raises(NotImplementedError, match="copy-on-write delete"):
        tbl.delete(delete_filter="id in (2, 3)")

    # The table state is unchanged (no corruption, no renumbering).
    tbl = v3_catalog.load_table("ns.t")
    assert tbl.metadata.next_row_id == 6
    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [0, 1, 2, 3, 4, 5]


def test_v3_partial_rewrite_delete_caught_in_transaction_stages_nothing(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    tbl.append(_batch([0, 1, 2, 3, 4, 5]))
    tbl = v3_catalog.load_table("ns.t")

    raised = False
    with tbl.transaction() as txn:
        try:
            txn.delete("id in (2, 3)")
        except NotImplementedError as exc:
            assert "copy-on-write delete" in str(exc)
            raised = True

    assert raised
    tbl = v3_catalog.load_table("ns.t")
    assert tbl.metadata.next_row_id == 6
    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [0, 1, 2, 3, 4, 5]


def test_v2_partial_rewrite_delete_still_works(tmp_path: Path) -> None:
    """The loud v3 failure must NOT regress v2 copy-on-write deletes."""
    cat = InMemoryCatalog("t1v2", warehouse=f"file://{tmp_path}")
    cat.create_namespace("ns")
    tbl = cat.create_table("ns.t", schema=SCHEMA, properties={"format-version": "2"})
    tbl.append(_batch([0, 1, 2, 3, 4, 5]))
    tbl = cat.load_table("ns.t")
    tbl.delete(delete_filter="id in (2, 3)")
    tbl = cat.load_table("ns.t")
    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [0, 1, 4, 5]


def test_v3_upgrade_from_v2_seeds_next_row_id(tmp_path: Path) -> None:
    """v2 -> v3 upgrade must be reachable via the public API and seed next_row_id = 0."""
    cat = InMemoryCatalog("t1up", warehouse=f"file://{tmp_path}")
    cat.create_namespace("ns")
    tbl = cat.create_table("ns.t", schema=SCHEMA, properties={"format-version": "2"})
    assert tbl.metadata.format_version == 2
    # v2 metadata does not carry next-row-id at all.
    assert getattr(tbl.metadata, "next_row_id", None) is None

    with tbl.transaction() as txn:
        txn.upgrade_table_version(3)
    tbl = cat.load_table("ns.t")

    assert tbl.metadata.format_version == 3
    assert tbl.metadata.next_row_id == 0, "upgrade to v3 must seed next_row_id at 0"

    # And the upgraded table must support a v3 append with correct row lineage.
    tbl.append(_batch([1, 2, 3]))
    tbl = cat.load_table("ns.t")
    snap = tbl.metadata.current_snapshot()
    assert snap is not None
    assert snap.first_row_id == 0
    assert snap.added_rows == 3
    assert tbl.metadata.next_row_id == 3


def test_v3_upgrade_with_data_does_not_renumber_carried_v2_files(tmp_path: Path) -> None:
    """Upgrading a NON-empty v2 table to v3 must assign each carried-forward file a row id
    range exactly once and keep it stable across later commits.

    Regression: the module-level manifest cache is keyed only by manifest_path and previously
    returned the pre-upgrade v2 ManifestFile (first_row_id=None). The v3 manifest-list writer
    then re-assigned a fresh first_row_id to those carried files on EVERY subsequent commit,
    double-counting next_row_id (9 -> 15 instead of 10) and renumbering already-assigned rows.
    """
    cat = InMemoryCatalog("t1upd", warehouse=f"file://{tmp_path}")
    cat.create_namespace("ns")
    tbl = cat.create_table("ns.t", schema=SCHEMA, properties={"format-version": "2"})
    # 5 v2 rows across two data manifests, written BEFORE the upgrade.
    tbl.append(_batch([0, 1, 2]))
    tbl = cat.load_table("ns.t")
    tbl.append(_batch([10, 11]))
    tbl = cat.load_table("ns.t")

    with tbl.transaction() as txn:
        txn.upgrade_table_version(3)
    tbl = cat.load_table("ns.t")
    assert tbl.metadata.next_row_id == 0

    # First v3 append: 4 new rows. The 5 carried v2 rows get assigned ids for the first time.
    tbl.append(_batch([100, 101, 102, 103]))
    tbl = cat.load_table("ns.t")
    snap1 = tbl.metadata.current_snapshot()
    assert snap1 is not None
    after_first = tbl.metadata.next_row_id
    assert after_first == 9, "5 carried rows + 4 new rows must yield next_row_id == 9"
    frid_by_path_1 = {
        m.manifest_path: m.first_row_id for m in snap1.manifests(tbl.io) if m.content == ManifestContent.DATA
    }
    assert all(frid is not None for frid in frid_by_path_1.values()), "carried files must be assigned a first_row_id"

    # Second v3 append: only 1 NEW row. next_row_id must advance by exactly 1, NOT re-count carried rows.
    tbl.append(_batch([200]))
    tbl = cat.load_table("ns.t")
    snap2 = tbl.metadata.current_snapshot()
    assert snap2 is not None
    assert tbl.metadata.next_row_id == 10, "second append must advance next_row_id by exactly 1 new row"
    assert snap2.added_rows == 1

    # Immutability: every manifest present in both snapshots must keep the SAME first_row_id.
    frid_by_path_2 = {
        m.manifest_path: m.first_row_id for m in snap2.manifests(tbl.io) if m.content == ManifestContent.DATA
    }
    for path in set(frid_by_path_1) & set(frid_by_path_2):
        assert frid_by_path_1[path] == frid_by_path_2[path], "carried-forward file was renumbered across snapshots"

    assert sorted(row["id"] for row in tbl.scan().to_arrow().to_pylist()) == [0, 1, 2, 10, 11, 100, 101, 102, 103, 200]


def test_v3_add_snapshot_update_advances_next_row_id_from_snapshot_first_row_id(v3_catalog: Catalog) -> None:
    v3_catalog.create_namespace("ns")
    tbl = v3_catalog.create_table("ns.t", schema=SCHEMA, properties={"format-version": "3"})
    assert tbl.metadata.next_row_id == 0

    snapshot = Snapshot(
        snapshot_id=25,
        parent_snapshot_id=None,
        sequence_number=1,
        timestamp_ms=1602638593590,
        manifest_list="s3:/a/b/c.avro",
        summary=Summary(Operation.APPEND),
        schema_id=tbl.metadata.current_schema_id,
        first_row_id=tbl.metadata.next_row_id,
        added_rows=4,
    )

    new_metadata = update_table_metadata(tbl.metadata, (AddSnapshotUpdate(snapshot=snapshot),))
    assert new_metadata.next_row_id == 4
