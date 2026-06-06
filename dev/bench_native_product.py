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
"""Benchmark PyIceberg scan backends and Arrow stream handoffs.

The parent process launches one child per case so wall time, CPU time, and peak
RSS are isolated from previous cases. Native routes report ``unsupported`` when
the installed ``pyiceberg-core`` wheel does not expose the scan bindings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa

from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.schema import Schema
from pyiceberg.table import PYICEBERG_RUST_SCAN_MODE
from pyiceberg.types import DoubleType, IntegerType, LongType, NestedField

ENGINES = ("pyarrow", "rust-read", "rust-plan-and-read", "capsule-pyarrow", "capsule-polars", "duckdb")
SHAPES = ("full", "projection", "filter")


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _table_data(start: int, rows: int) -> pa.Table:
    ids = pa.array(range(start, start + rows), type=pa.int64())
    part = pa.array(((value % 8) for value in range(start, start + rows)), type=pa.int32())
    payload = pa.array((float(value % 1024) for value in range(start, start + rows)), type=pa.float64())
    return pa.table({"id": ids, "part": part, "payload": payload})


def _create_table(warehouse: Path, rows: int, files: int) -> Any:
    if warehouse.exists():
        shutil.rmtree(warehouse)
    warehouse.mkdir(parents=True, exist_ok=True)

    catalog = InMemoryCatalog("bench", warehouse=warehouse.as_uri())
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "part", IntegerType(), required=False),
        NestedField(3, "payload", DoubleType(), required=False),
    )
    table = catalog.create_table("default.scan_bench", schema=schema)

    rows_per_file = max(1, rows // files)
    offset = 0
    for file_idx in range(files):
        chunk_rows = rows - offset if file_idx == files - 1 else rows_per_file
        table.append(_table_data(offset, chunk_rows))
        offset += chunk_rows
    return table


def _scan_for_shape(table: Any, shape: str) -> Any:
    if shape == "full":
        return table.scan()
    if shape == "projection":
        return table.scan(selected_fields=("id", "payload"))
    if shape == "filter":
        return table.scan(row_filter="part == 1", selected_fields=("id", "payload"))
    raise ValueError(f"Unknown shape: {shape}")


def _table_digest(table: pa.Table) -> dict[str, Any]:
    if "id" not in table.column_names:
        return {"rows": table.num_rows, "sum_id": None}
    return {"rows": table.num_rows, "sum_id": table["id"].combine_chunks().sum().as_py()}


def _run_scan(scan: Any, engine: str) -> dict[str, Any]:
    os.environ.pop(PYICEBERG_RUST_SCAN_MODE, None)
    if engine == "rust-read":
        importlib.import_module("pyiceberg_core.scan")
        os.environ[PYICEBERG_RUST_SCAN_MODE] = "rust-read"
    elif engine == "rust-plan-and-read":
        importlib.import_module("pyiceberg_core.scan")
        os.environ[PYICEBERG_RUST_SCAN_MODE] = "rust-plan-and-read"

    if engine in {"rust-read", "rust-plan-and-read"}:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Falling back to PyArrow scan.*", category=RuntimeWarning)
            return _table_digest(scan.to_arrow_batch_reader().read_all())
    if engine == "pyarrow":
        return _table_digest(scan.to_arrow_batch_reader().read_all())
    if engine == "capsule-pyarrow":
        return _table_digest(pa.table(scan))
    if engine == "capsule-polars":
        import polars as pl

        frame = pl.DataFrame(scan)
        return {"rows": frame.height, "sum_id": frame["id"].sum() if "id" in frame.columns else None}
    if engine == "duckdb":
        con = scan.to_duckdb("iceberg_scan")
        rows, sum_id = con.sql("select count(*) as rows, sum(id) as sum_id from iceberg_scan").fetchone()
        return {"rows": rows, "sum_id": sum_id}
    raise ValueError(f"Unknown engine: {engine}")


def _measure(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    rss_before = _rss_kib()
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    digest = fn()
    wall_ms = (time.perf_counter() - wall_before) * 1000
    cpu_ms = (time.process_time() - cpu_before) * 1000
    rss_after = _rss_kib()
    return {
        **digest,
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "peak_rss_kib": rss_after,
        "delta_peak_rss_kib": max(0, rss_after - rss_before),
    }


def _child(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {
        "engine": args.engine,
        "shape": args.shape,
        "rows_input": args.rows,
        "files": args.files,
        "status": "ok",
    }
    try:
        table = _create_table(Path(args.warehouse), args.rows, args.files)
        scan = _scan_for_shape(table, args.shape)
        result.update(_measure(lambda: _run_scan(scan, args.engine)))
    except Exception as exc:
        error_message = str(exc)
        result["status"] = (
            "unsupported"
            if "pyiceberg-core" in error_message
            or "pyiceberg_core" in error_message
            or "Falling back to PyArrow scan" in error_message
            else "error"
        )
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parent(args: argparse.Namespace) -> int:
    engines = args.engines.split(",") if args.engines else list(ENGINES)
    shapes = args.shapes.split(",") if args.shapes else list(SHAPES)
    for engine in engines:
        if engine not in ENGINES:
            raise ValueError(f"Unknown engine: {engine}")
    for shape in shapes:
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape: {shape}")

    records: list[dict[str, Any]] = []
    for shape in shapes:
        for engine in engines:
            for repeat in range(args.repeats):
                with tempfile.TemporaryDirectory(prefix="pyiceberg-bench-") as tmp:
                    cmd = [
                        sys.executable,
                        __file__,
                        "--child",
                        "--engine",
                        engine,
                        "--shape",
                        shape,
                        "--rows",
                        str(args.rows),
                        "--files",
                        str(args.files),
                        "--warehouse",
                        tmp,
                    ]
                    proc = subprocess.run(cmd, text=True, check=False, capture_output=True)
                    if proc.stderr:
                        print(proc.stderr, file=sys.stderr, end="")
                    record = json.loads(proc.stdout.strip().splitlines()[-1])
                    record["repeat"] = repeat
                    records.append(record)
                    print(json.dumps(record, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engines", help=f"Comma-separated subset of: {', '.join(ENGINES)}")
    parser.add_argument("--shapes", help=f"Comma-separated subset of: {', '.join(SHAPES)}")
    parser.add_argument("--output")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--shape", choices=SHAPES)
    parser.add_argument("--warehouse", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.child:
        return _child(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
