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
"""Pluggable scan-planning strategies.

A ``ScanPlanner`` turns a ``DataScan`` into the file scan tasks that back it. The Python (local)
and REST (server-side) strategies reproduce the existing planner selection exactly; the resolver
picks between them just as ``DataScan.plan_files`` used to. A planner can also be injected on a
``DataScan`` to override resolution, which is how the Rust strategy is opted into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyiceberg.table import DataScan, FileScanTask


@runtime_checkable
class ScanPlanner(Protocol):
    """Plan the file scan tasks for a ``DataScan``."""

    def plan_files(self, scan: DataScan) -> Iterable[FileScanTask]: ...


class LocalScanPlanner:
    """Plan files locally by reading manifests (PyIceberg's default planner)."""

    def plan_files(self, scan: DataScan) -> Iterable[FileScanTask]:
        return scan._plan_files_local()


class RestScanPlanner:
    """Plan files using REST server-side scan planning."""

    def plan_files(self, scan: DataScan) -> Iterable[FileScanTask]:
        return scan._plan_files_server_side()


class RustScanPlanner:
    """Placeholder for a pyiceberg-core planner that returns ``FileScanTask`` objects.

    Opt-in only: ``resolve_scan_planner`` never selects this, so it has to be injected on a
    ``DataScan``. It is intentionally unimplemented: pyiceberg-core's native plan output exposes
    only booleans/counts for predicate, partition data, and deletes, not the residual expression,
    partition ``Record``, or delete ``DataFile`` objects needed to rebuild a faithful PyIceberg
    ``FileScanTask`` — so reconstructing one here would silently drop the residual and break
    filter-on-dropped-column scans. Use the fused ``PYICEBERG_RUST_SCAN_PLANNING`` read path, which
    plans and reads natively and never round-trips through ``FileScanTask``.
    """

    def plan_files(self, scan: DataScan) -> Iterable[FileScanTask]:
        raise NotImplementedError(
            "Native pyiceberg-core planning to FileScanTask is not supported; "
            "the native plan output does not expose the residual, partition data, or deletes "
            "needed to rebuild a faithful FileScanTask. Use the fused PYICEBERG_RUST_SCAN_PLANNING "
            "read path instead."
        )


def resolve_scan_planner(scan: DataScan) -> ScanPlanner:
    """Pick the planner for ``scan``, reproducing the historical local/server-side selection.

    ``RustScanPlanner`` is never auto-selected; it is opt-in via ``DataScan.scan_planner``.
    """
    if scan._should_use_server_side_planning():
        return RestScanPlanner()
    return LocalScanPlanner()
