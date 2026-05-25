#!/usr/bin/env python3
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
"""Benchmark harness to compare PyArrow vs Rust-backed Arrow scans.

Usage:
  python3 dev/bench_arrow_scan.py --table default.test_limit [options]

Options:
  --table TEXT            Iceberg table identifier (default: default.test_limit)
  --uri TEXT              Catalog REST URI (default: http://localhost:8181)
  --selected-fields TEXT  Comma-separated list of fields to project
  --row-filter TEXT       Row filter expression (e.g. "idx > 5")
  --limit INTEGER         Max number of rows to scan
  --runs INTEGER          Number of runs to average (default: 5)
  --native-only           Only run the Rust-backed scan
  --pyarrow-only          Only run the PyArrow scan

This script outputs elapsed time, total rows, batch count, and peak memory usage.
"""

from __future__ import annotations

import argparse
import os
import time
import tracemalloc
from typing import Any

from pyiceberg.catalog import load_catalog
from pyiceberg.table import Table


def run_benchmark(
    table: Table,
    use_native: bool,
    selected_fields: tuple[str, ...] | None,
    row_filter: str | None,
    limit: int | None,
    runs: int,
) -> dict[str, Any]:
    # Set the environment variable
    os.environ["PYICEBERG_RUST_ARROW_SCAN"] = "1" if use_native else "0"

    scan_kwargs: dict[str, Any] = {}
    if selected_fields is not None:
        scan_kwargs["selected_fields"] = selected_fields
    if row_filter is not None:
        scan_kwargs["row_filter"] = row_filter
    if limit is not None:
        scan_kwargs["limit"] = limit

    scan = table.scan(**scan_kwargs)

    elapsed_times = []
    total_rows = 0
    total_batches = 0
    peak_memories = []

    # Warm up run
    try:
        reader = scan.to_arrow_batch_reader()
        for _ in reader:
            pass
    except Exception as exc:
        return {"error": str(exc)}

    for _ in range(runs):
        tracemalloc.start()
        start_time = time.perf_counter()

        reader = scan.to_arrow_batch_reader()
        rows = 0
        batches = 0
        for batch in reader:
            rows += batch.num_rows
            batches += 1

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_times.append(end_time - start_time)
        peak_memories.append(peak)
        total_rows = rows
        total_batches = batches

    avg_time = sum(elapsed_times) / runs
    avg_peak_mem = sum(peak_memories) / runs

    return {
        "avg_time_ms": avg_time * 1000,
        "rows": total_rows,
        "batches": total_batches,
        "peak_memory_kb": avg_peak_mem / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark harness comparing PyArrow and Native Rust-backed Arrow scans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--table", default="default.test_limit", help="Iceberg table identifier")
    parser.add_argument("--uri", default="http://localhost:8181", help="Catalog REST URI")
    parser.add_argument("--selected-fields", help="Comma-separated fields to project")
    parser.add_argument("--row-filter", help="Row filter expression")
    parser.add_argument("--limit", type=int, help="Limit number of rows")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--native-only", action="store_true", help="Only run native scans")
    parser.add_argument("--pyarrow-only", action="store_true", help="Only run pyarrow scans")

    args = parser.parse_args()

    selected_fields = tuple(args.selected_fields.split(",")) if args.selected_fields else None

    # Load catalog
    catalog = load_catalog("default", uri=args.uri)
    try:
        table = catalog.load_table(args.table)
    except Exception as e:
        print(f"Failed to load table '{args.table}': {e}")
        return

    print("=" * 60)
    print(f"Benchmarking table: {args.table}")
    print(f"Selected fields: {selected_fields}")
    print(f"Row filter:      {args.row_filter}")
    print(f"Limit:           {args.limit}")
    print(f"Runs:            {args.runs}")
    print("=" * 60)

    results = {}

    if not args.native_only:
        print("Running PyArrow benchmark...")
        results["PyArrow"] = run_benchmark(
            table,
            use_native=False,
            selected_fields=selected_fields,
            row_filter=args.row_filter,
            limit=args.limit,
            runs=args.runs,
        )

    if not args.pyarrow_only:
        print("Running Native Rust-backed benchmark...")
        results["Native (Rust)"] = run_benchmark(
            table,
            use_native=True,
            selected_fields=selected_fields,
            row_filter=args.row_filter,
            limit=args.limit,
            runs=args.runs,
        )

    print("\nBenchmark Results:")
    print(f"{'Engine':<20} | {'Avg Latency (ms)':<18} | {'Rows Read':<12} | {'Batches':<10} | {'Peak Memory (KB)':<18}")
    print("-" * 88)

    for engine, res in results.items():
        if "error" in res:
            print(f"{engine:<20} | Error: {res['error']}")
        else:
            print(
                f"{engine:<20} | "
                f"{res['avg_time_ms']:>17.2f} | "
                f"{res['rows']:>12} | "
                f"{res['batches']:>10} | "
                f"{res['peak_memory_kb']:>17.2f}"
            )


if __name__ == "__main__":
    main()
