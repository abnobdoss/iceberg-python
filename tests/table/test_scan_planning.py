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

from typing import Any

import pyarrow as pa
import pytest

from pyiceberg.table import Table
from pyiceberg.table.scan_planning import (
    LocalScanPlanner,
    RestScanPlanner,
    RustScanPlanner,
    ScanPlanner,
    resolve_scan_planner,
)


class _FakeCatalog:
    def __init__(self, server_side: bool) -> None:
        self._server_side = server_side

    def supports_server_side_planning(self) -> bool:
        return self._server_side


class _FakeScan:
    """Minimal stand-in for DataScan that records how the planner consulted it."""

    def __init__(self, catalog: _FakeCatalog | None) -> None:
        self.catalog = catalog
        self.local_called = False
        self.server_side_called = False

    def _should_use_server_side_planning(self) -> bool:
        return self.catalog is not None and self.catalog.supports_server_side_planning()

    def _plan_files_local(self) -> list[Any]:
        self.local_called = True
        return ["local-task"]

    def _plan_files_server_side(self) -> list[Any]:
        self.server_side_called = True
        return ["server-task"]


def test_local_scan_planner_satisfies_protocol() -> None:
    assert isinstance(LocalScanPlanner(), ScanPlanner)
    assert isinstance(RestScanPlanner(), ScanPlanner)


def test_local_scan_planner_delegates_to_the_scan() -> None:
    scan = _FakeScan(catalog=None)

    tasks = list(LocalScanPlanner().plan_files(scan))  # type: ignore[arg-type]

    assert tasks == ["local-task"]
    assert scan.local_called and not scan.server_side_called


def test_rest_scan_planner_delegates_to_the_scan() -> None:
    scan = _FakeScan(catalog=_FakeCatalog(server_side=True))

    tasks = list(RestScanPlanner().plan_files(scan))  # type: ignore[arg-type]

    assert tasks == ["server-task"]
    assert scan.server_side_called and not scan.local_called


def test_resolve_scan_planner_returns_rest_when_server_side_supported() -> None:
    scan = _FakeScan(catalog=_FakeCatalog(server_side=True))

    assert isinstance(resolve_scan_planner(scan), RestScanPlanner)  # type: ignore[arg-type]


def test_resolve_scan_planner_returns_local_otherwise() -> None:
    assert isinstance(resolve_scan_planner(_FakeScan(catalog=None)), LocalScanPlanner)  # type: ignore[arg-type]
    assert isinstance(resolve_scan_planner(_FakeScan(catalog=_FakeCatalog(server_side=False))), LocalScanPlanner)  # type: ignore[arg-type]


def test_rust_scan_planner_is_an_honest_stub() -> None:
    # Native plan output cannot rebuild a faithful FileScanTask (no residual / partition / deletes),
    # so the planner refuses rather than silently dropping the residual. The fused read path is the
    # supported native route.
    with pytest.raises(NotImplementedError, match="faithful FileScanTask"):
        list(RustScanPlanner().plan_files(_FakeScan(catalog=None)))  # type: ignore[arg-type]


def test_data_scan_plan_files_uses_injected_planner(table_v2: Table) -> None:
    """An injected planner overrides resolution on a real DataScan and is handed the scan itself."""
    seen: list[Any] = []

    class _RecordingPlanner:
        def plan_files(self, scan: Any) -> list[Any]:
            seen.append(scan)
            return ["injected-task"]

    scan = table_v2.scan().update(scan_planner=_RecordingPlanner())

    tasks = list(scan.plan_files())

    assert tasks == ["injected-task"]
    assert seen == [scan]  # the planner receives the DataScan itself, not a copied context


def test_data_scan_plan_files_resolves_local_by_default(table_v2: Table) -> None:
    # No injected planner and no server-side catalog: resolution must pick the local planner.
    scan = table_v2.scan()

    assert scan.scan_planner is None
    assert isinstance(resolve_scan_planner(scan), LocalScanPlanner)


def test_data_scan_to_arrow_uses_native_batch_reader_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    table_v2: Table,
) -> None:
    schema = pa.schema([pa.field("id", pa.int32())])
    batch = pa.record_batch({"id": pa.array([1, 2], type=pa.int32())})
    scan = table_v2.scan().select("id")
    seen: list[Any] = []

    def _batch_reader(self: Any) -> pa.RecordBatchReader:
        seen.append(self)
        return pa.RecordBatchReader.from_batches(schema, [batch])

    monkeypatch.setenv("PYICEBERG_RUST_ARROW_SCAN", "1")
    monkeypatch.setattr(type(scan), "to_arrow_batch_reader", _batch_reader)

    table = scan.to_arrow()

    assert seen == [scan]
    assert table.column("id").to_pylist() == [1, 2]
