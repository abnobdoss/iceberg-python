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
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from pyiceberg.table import CommitTableResponse, Table
from pyiceberg.table.refs import SnapshotRef, SnapshotRefType
from pyiceberg.table.snapshots import SnapshotLogEntry
from pyiceberg.table.update import RemoveSnapshotsUpdate, TableRequirement, TableUpdate, update_table_metadata
from pyiceberg.table.update.snapshot import ExpireSnapshots


def test_cannot_expire_protected_head_snapshot(table_v2: Table) -> None:
    """Test that a HEAD (branch) snapshot cannot be expired."""
    HEAD_SNAPSHOT = 3051729675574597004
    KEEP_SNAPSHOT = 3055729675574597004

    # Mock the catalog's commit_table method
    table_v2.catalog = MagicMock()
    # Simulate refs protecting HEAD_SNAPSHOT as a branch
    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": MagicMock(snapshot_id=HEAD_SNAPSHOT, snapshot_ref_type="branch"),
                "tag1": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="tag"),
            }
        }
    )
    # Assert fixture data
    assert any(ref.snapshot_id == HEAD_SNAPSHOT for ref in table_v2.metadata.refs.values())

    # Attempt to expire the HEAD snapshot and expect a ValueError
    with pytest.raises(ValueError, match=f"Snapshot with ID {HEAD_SNAPSHOT} is protected and cannot be expired."):
        table_v2.maintenance.expire_snapshots().by_id(HEAD_SNAPSHOT).commit()

    table_v2.catalog.commit_table.assert_not_called()


def test_cannot_expire_tagged_snapshot(table_v2: Table) -> None:
    """Test that a tagged snapshot cannot be expired."""
    TAGGED_SNAPSHOT = 3051729675574597004
    KEEP_SNAPSHOT = 3055729675574597004

    table_v2.catalog = MagicMock()
    # Simulate refs protecting TAGGED_SNAPSHOT as a tag
    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "tag1": MagicMock(snapshot_id=TAGGED_SNAPSHOT, snapshot_ref_type="tag"),
                "main": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="branch"),
            }
        }
    )
    assert any(ref.snapshot_id == TAGGED_SNAPSHOT for ref in table_v2.metadata.refs.values())

    with pytest.raises(ValueError, match=f"Snapshot with ID {TAGGED_SNAPSHOT} is protected and cannot be expired."):
        table_v2.maintenance.expire_snapshots().by_id(TAGGED_SNAPSHOT).commit()

    table_v2.catalog.commit_table.assert_not_called()


def test_cannot_expire_current_snapshot_without_ref(table_v2: Table) -> None:
    current_snapshot_id = table_v2.metadata.current_snapshot_id
    assert current_snapshot_id is not None
    non_current_snapshot_id = next(
        snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots if snapshot.snapshot_id != current_snapshot_id
    )

    table_v2.catalog = MagicMock()
    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": SnapshotRef(snapshot_id=non_current_snapshot_id, snapshot_ref_type=SnapshotRefType.BRANCH),
            }
        }
    )
    assert all(ref.snapshot_id != current_snapshot_id for ref in table_v2.metadata.refs.values())

    with pytest.raises(ValueError, match=f"Snapshot with ID {current_snapshot_id} is protected and cannot be expired."):
        table_v2.maintenance.expire_snapshots().by_id(current_snapshot_id).commit()

    table_v2.catalog.commit_table.assert_not_called()


def test_expire_unprotected_snapshot(table_v2: Table) -> None:
    """Test that an unprotected snapshot can be expired."""
    EXPIRE_SNAPSHOT = 3051729675574597004
    KEEP_SNAPSHOT = 3055729675574597004

    mock_response = CommitTableResponse(
        metadata=table_v2.metadata.model_copy(update={"snapshots": [KEEP_SNAPSHOT]}),
        metadata_location="mock://metadata/location",
        uuid=uuid4(),
    )
    table_v2.catalog = MagicMock()
    table_v2.catalog.commit_table.return_value = mock_response

    # Remove any refs that protect the snapshot to be expired
    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="branch"),
                "tag1": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="tag"),
            }
        }
    )

    # Assert fixture data
    assert all(ref.snapshot_id != EXPIRE_SNAPSHOT for ref in table_v2.metadata.refs.values())

    # Expire the snapshot
    table_v2.maintenance.expire_snapshots().by_id(EXPIRE_SNAPSHOT).commit()

    table_v2.catalog.commit_table.assert_called_once()
    remaining_snapshots = table_v2.metadata.snapshots
    assert EXPIRE_SNAPSHOT not in remaining_snapshots
    assert len(table_v2.metadata.snapshots) == 1


def test_expire_nonexistent_snapshot_raises(table_v2: Table) -> None:
    """Test that trying to expire a non-existent snapshot raises an error."""
    NONEXISTENT_SNAPSHOT = 9999999999999999999

    table_v2.catalog = MagicMock()
    table_v2.metadata = table_v2.metadata.model_copy(update={"refs": {}})

    with pytest.raises(ValueError, match=f"Snapshot with ID {NONEXISTENT_SNAPSHOT} does not exist."):
        table_v2.maintenance.expire_snapshots().by_id(NONEXISTENT_SNAPSHOT).commit()

    table_v2.catalog.commit_table.assert_not_called()


def test_expire_snapshots_by_timestamp_skips_protected(table_v2: Table) -> None:
    # Setup: two snapshots; both are old, but one is head/tag protected
    HEAD_SNAPSHOT = 3051729675574597004
    TAGGED_SNAPSHOT = 3055729675574597004

    # Add snapshots to metadata for timestamp/protected test
    from types import SimpleNamespace

    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": MagicMock(snapshot_id=HEAD_SNAPSHOT, snapshot_ref_type="branch"),
                "mytag": MagicMock(snapshot_id=TAGGED_SNAPSHOT, snapshot_ref_type="tag"),
            },
            "snapshots": [
                SimpleNamespace(snapshot_id=HEAD_SNAPSHOT, timestamp_ms=1, parent_snapshot_id=None),
                SimpleNamespace(snapshot_id=TAGGED_SNAPSHOT, timestamp_ms=1, parent_snapshot_id=None),
            ],
        }
    )
    table_v2.catalog = MagicMock()

    # Attempt to expire all snapshots before a future timestamp (so both are candidates)
    future_datetime = datetime.now() + timedelta(days=1)

    # Mock the catalog's commit_table to return the current metadata (simulate no change)
    mock_response = CommitTableResponse(
        metadata=table_v2.metadata,  # protected snapshots remain
        metadata_location="mock://metadata/location",
        uuid=uuid4(),
    )
    table_v2.catalog.commit_table.return_value = mock_response

    table_v2.maintenance.expire_snapshots().older_than(future_datetime).commit()
    # Update metadata to reflect the commit (as in other tests)
    table_v2.metadata = mock_response.metadata

    # Both protected snapshots should remain
    remaining_ids = {s.snapshot_id for s in table_v2.metadata.snapshots}
    assert HEAD_SNAPSHOT in remaining_ids
    assert TAGGED_SNAPSHOT in remaining_ids

    # No snapshots should have been expired (commit_table called, but with empty snapshot_ids)
    args, kwargs = table_v2.catalog.commit_table.call_args
    updates = args[2] if len(args) > 2 else ()
    # Find RemoveSnapshotsUpdate in updates
    remove_update = next((u for u in updates if getattr(u, "action", None) == "remove-snapshots"), None)
    assert remove_update is not None
    assert remove_update.snapshot_ids == []


def test_expire_snapshots_by_ids(table_v2: Table) -> None:
    """Test that multiple unprotected snapshots can be expired by IDs."""
    EXPIRE_SNAPSHOT_1 = 3051729675574597004
    EXPIRE_SNAPSHOT_2 = 3051729675574597005
    KEEP_SNAPSHOT = 3055729675574597004

    mock_response = CommitTableResponse(
        metadata=table_v2.metadata.model_copy(update={"snapshots": [KEEP_SNAPSHOT]}),
        metadata_location="mock://metadata/location",
        uuid=uuid4(),
    )
    table_v2.catalog = MagicMock()
    table_v2.catalog.commit_table.return_value = mock_response

    # Remove any refs that protect the snapshots to be expired
    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="branch"),
                "tag1": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="tag"),
            }
        }
    )

    # Add snapshots to metadata for multi-id test
    from types import SimpleNamespace

    table_v2.metadata = table_v2.metadata.model_copy(
        update={
            "refs": {
                "main": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="branch"),
                "tag1": MagicMock(snapshot_id=KEEP_SNAPSHOT, snapshot_ref_type="tag"),
            },
            "snapshots": [
                SimpleNamespace(snapshot_id=EXPIRE_SNAPSHOT_1, timestamp_ms=1, parent_snapshot_id=None),
                SimpleNamespace(snapshot_id=EXPIRE_SNAPSHOT_2, timestamp_ms=1, parent_snapshot_id=None),
                SimpleNamespace(snapshot_id=KEEP_SNAPSHOT, timestamp_ms=2, parent_snapshot_id=None),
            ],
        }
    )

    # Assert fixture data
    assert all(ref.snapshot_id not in (EXPIRE_SNAPSHOT_1, EXPIRE_SNAPSHOT_2) for ref in table_v2.metadata.refs.values())

    # Expire the snapshots
    table_v2.maintenance.expire_snapshots().by_ids([EXPIRE_SNAPSHOT_1, EXPIRE_SNAPSHOT_2]).commit()

    table_v2.catalog.commit_table.assert_called_once()
    remaining_snapshots = table_v2.metadata.snapshots
    assert EXPIRE_SNAPSHOT_1 not in remaining_snapshots
    assert EXPIRE_SNAPSHOT_2 not in remaining_snapshots
    assert len(table_v2.metadata.snapshots) == 1


def test_thread_safety_fix() -> None:
    """Test that ExpireSnapshots instances have isolated state."""
    # Create two ExpireSnapshots instances
    expire1 = ExpireSnapshots(Mock())
    expire2 = ExpireSnapshots(Mock())

    # Verify they have separate snapshot sets (this was the bug!)
    # Before fix: both would have the same id (shared class attribute)
    # After fix: they should have different ids (separate instance attributes)
    assert id(expire1._snapshot_ids_to_expire) != id(expire2._snapshot_ids_to_expire), (
        "ExpireSnapshots instances are sharing the same snapshot set - thread safety bug still exists"
    )

    # Test that modifications to one don't affect the other
    expire1._snapshot_ids_to_expire.add(1001)
    expire2._snapshot_ids_to_expire.add(2001)

    # Verify no cross-contamination of snapshot IDs
    assert 2001 not in expire1._snapshot_ids_to_expire, "Snapshot IDs are leaking between instances"
    assert 1001 not in expire2._snapshot_ids_to_expire, "Snapshot IDs are leaking between instances"


def test_concurrent_operations() -> None:
    """Test concurrent operations with separate ExpireSnapshots instances."""
    results: dict[str, set[int]] = {"expire1_snapshots": set(), "expire2_snapshots": set()}

    def worker1() -> None:
        expire1 = ExpireSnapshots(Mock())
        expire1._snapshot_ids_to_expire.update([1001, 1002, 1003])
        results["expire1_snapshots"] = expire1._snapshot_ids_to_expire.copy()

    def worker2() -> None:
        expire2 = ExpireSnapshots(Mock())
        expire2._snapshot_ids_to_expire.update([2001, 2002, 2003])
        results["expire2_snapshots"] = expire2._snapshot_ids_to_expire.copy()

    # Run both workers concurrently
    thread1 = threading.Thread(target=worker1)
    thread2 = threading.Thread(target=worker2)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Check for cross-contamination
    expected_1 = {1001, 1002, 1003}
    expected_2 = {2001, 2002, 2003}

    assert results["expire1_snapshots"] == expected_1, "Worker 1 snapshots contaminated"
    assert results["expire2_snapshots"] == expected_2, "Worker 2 snapshots contaminated"


def test_update_remove_snapshots_with_statistics(table_v2_with_statistics: Table) -> None:
    """
    Test removing snapshots from a table that has statistics.

    This test exercises the code path where RemoveStatisticsUpdate is instantiated
    within the RemoveSnapshotsUpdate handler. Before the fix for #2558, this would
    fail with: TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given
    """
    # The table has 2 snapshots with IDs: 3051729675574597004 and 3055729675574597004
    # Both snapshots have statistics associated with them
    REMOVE_SNAPSHOT = 3051729675574597004
    KEEP_SNAPSHOT = 3055729675574597004

    # Verify fixture assumptions
    assert len(table_v2_with_statistics.metadata.snapshots) == 2
    assert len(table_v2_with_statistics.metadata.statistics) == 2
    assert any(stat.snapshot_id == REMOVE_SNAPSHOT for stat in table_v2_with_statistics.metadata.statistics), (
        "Snapshot to remove should have statistics"
    )

    # This should trigger RemoveStatisticsUpdate instantiation for the removed snapshot
    update = RemoveSnapshotsUpdate(snapshot_ids=[REMOVE_SNAPSHOT])
    new_metadata = update_table_metadata(table_v2_with_statistics.metadata, (update,))

    # Verify the snapshot was removed
    assert len(new_metadata.snapshots) == 1
    assert new_metadata.snapshots[0].snapshot_id == KEEP_SNAPSHOT

    # Verify the statistics for the removed snapshot were also removed
    assert len(new_metadata.statistics) == 1
    assert new_metadata.statistics[0].snapshot_id == KEEP_SNAPSHOT
    assert not any(stat.snapshot_id == REMOVE_SNAPSHOT for stat in new_metadata.statistics), (
        "Statistics for removed snapshot should be gone"
    )


def _prepare_table_with_snapshots(
    table: Table,
    snapshot_ids_and_timestamps: list[tuple[int, int]],
    refs: dict[str, SnapshotRef] | None = None,
    current_snapshot_id: int | None = None,
) -> None:
    base_snapshot = table.metadata.snapshots[0]
    snapshots = []
    snapshot_log = []
    parent_snapshot_id = None

    for sequence_number, (snapshot_id, timestamp_ms) in enumerate(snapshot_ids_and_timestamps):
        snapshots.append(
            base_snapshot.model_copy(
                update={
                    "snapshot_id": snapshot_id,
                    "parent_snapshot_id": parent_snapshot_id,
                    "sequence_number": sequence_number,
                    "timestamp_ms": timestamp_ms,
                    "manifest_list": f"s3://bucket/test/{snapshot_id}.avro",
                }
            )
        )
        snapshot_log.append(SnapshotLogEntry(snapshot_id=snapshot_id, timestamp_ms=timestamp_ms))
        parent_snapshot_id = snapshot_id

    table.metadata = table.metadata.model_copy(
        update={
            "current_snapshot_id": current_snapshot_id,
            "refs": refs or {},
            "snapshots": snapshots,
            "snapshot_log": snapshot_log,
        }
    )


def _configure_commit_to_apply_updates(table: Table) -> None:
    def commit_table(
        _table: Table, _requirements: tuple[TableRequirement, ...], updates: tuple[TableUpdate, ...]
    ) -> CommitTableResponse:
        return CommitTableResponse(
            metadata=update_table_metadata(table.metadata, updates),
            metadata_location="mock://metadata/location",
            uuid=uuid4(),
        )

    table.catalog = MagicMock()
    table.catalog.commit_table.side_effect = commit_table


def test_retain_last_two_expires_surplus_unprotected_snapshots(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
            (104, 4000),
            (105, 5000),
        ],
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().retain_last(2).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {104, 105}


def test_retain_last_one_keeps_only_newest_unprotected_snapshot(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
        ],
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().retain_last(1).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {103}


def test_retain_last_keeps_current_snapshot_without_counting_it(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
            (104, 4000),
        ],
        refs={
            "main": SnapshotRef(snapshot_id=104, snapshot_ref_type=SnapshotRefType.BRANCH),
        },
        current_snapshot_id=101,
    )
    _configure_commit_to_apply_updates(table_v2)
    assert all(ref.snapshot_id != 101 for ref in table_v2.metadata.refs.values())

    table_v2.maintenance.expire_snapshots().retain_last(1).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {101, 103, 104}
    assert table_v2.metadata.current_snapshot_id == 101
    current_snapshot = table_v2.current_snapshot()
    assert current_snapshot is not None
    assert current_snapshot.snapshot_id == 101


def test_older_than_with_retain_last_keeps_newest_unprotected_floor(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
            (104, 4000),
            (105, 5000),
        ],
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().older_than(datetime(1970, 1, 1, 0, 0, 10)).retain_last(2).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {104, 105}


def test_older_than_with_retain_last_intersection(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
            (104, 4000),
            (105, 5000),
        ],
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().older_than(datetime(1970, 1, 1, 0, 0, 2, 500000)).retain_last(2).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {103, 104, 105}


def test_retain_last_tiebreak_uses_sequence_number(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (102, 1000),
            (101, 1000),
        ],
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().retain_last(1).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {101}


def test_retain_last_keeps_protected_snapshots_without_counting_them(table_v2: Table) -> None:
    _prepare_table_with_snapshots(
        table_v2,
        [
            (101, 1000),
            (102, 2000),
            (103, 3000),
            (104, 4000),
        ],
        refs={
            "old-tag": SnapshotRef(snapshot_id=101, snapshot_ref_type=SnapshotRefType.TAG),
            "main": SnapshotRef(snapshot_id=104, snapshot_ref_type=SnapshotRefType.BRANCH),
        },
        current_snapshot_id=104,
    )
    _configure_commit_to_apply_updates(table_v2)

    table_v2.maintenance.expire_snapshots().retain_last(1).commit()

    remaining_ids = {snapshot.snapshot_id for snapshot in table_v2.metadata.snapshots}
    assert remaining_ids == {101, 103, 104}


def test_retain_last_requires_at_least_one_snapshot(table_v2: Table) -> None:
    table_v2.catalog = MagicMock()

    with pytest.raises(ValueError, match="Number of snapshots to retain must be at least 1"):
        table_v2.maintenance.expire_snapshots().retain_last(0)

    table_v2.catalog.commit_table.assert_not_called()
