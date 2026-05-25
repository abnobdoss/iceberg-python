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
"""Benchmark PyArrow and Rust-backed Arrow scans on realistic Iceberg tables.

Typical stress run:

  uv run python dev/bench_arrow_scan.py --refresh --runs 5 --warmups 1 \
    --s3-endpoint http://localhost:19000

The default stress profile creates 5M rows / 2k files for normal and
partitioned scans, plus a 1M row / 500 file merge-on-read positional delete
table. Each measured scan runs in a fresh child process so RSS includes native
Arrow/Rust allocations made by the Python client process.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROWS = 5_000_000
DEFAULT_FILES = 2_000
DEFAULT_DELETE_ROWS = 1_000_000
DEFAULT_DELETE_FILES = 500
DEFAULT_NAMESPACE = "default"
DEFAULT_TABLE_PREFIX = "bench_native_scan"


@dataclass(frozen=True)
class CatalogConfig:
    uri: str
    s3_endpoint: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_region: str
    s3_force_virtual_addressing: str


@dataclass(frozen=True)
class Scenario:
    name: str
    table: str
    selected_fields: tuple[str, ...] | None = None
    row_filter: str | None = None
    limit: int | None = None


@dataclass
class RunResult:
    engine: str
    scenario: str
    elapsed_ms: float
    rows: int
    batches: int
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float
    maxrss_mb: float
    tracemalloc_peak_mb: float


FALLBACK_WARNING_MARKERS = (
    "Falling back to PyArrow scan because pyiceberg-core cannot handle this scan",
    "Falling back to native task-based scan because Rust-planned scan failed",
)


def _catalog_props(config: CatalogConfig) -> dict[str, str]:
    return {
        "type": "rest",
        "uri": config.uri,
        "s3.endpoint": config.s3_endpoint,
        "s3.access-key-id": config.s3_access_key_id,
        "s3.secret-access-key": config.s3_secret_access_key,
        "s3.region": config.s3_region,
        "s3.force-virtual-addressing": config.s3_force_virtual_addressing,
    }


def _set_catalog_env(config: CatalogConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYICEBERG_CATALOG__DEFAULT__TYPE": "rest",
            "PYICEBERG_CATALOG__DEFAULT__URI": config.uri,
            "PYICEBERG_CATALOG__DEFAULT__S3__ENDPOINT": config.s3_endpoint,
            "PYICEBERG_CATALOG__DEFAULT__S3__ACCESS_KEY_ID": config.s3_access_key_id,
            "PYICEBERG_CATALOG__DEFAULT__S3__SECRET_ACCESS_KEY": config.s3_secret_access_key,
            "PYICEBERG_CATALOG__DEFAULT__S3__REGION": config.s3_region,
            "PYICEBERG_CATALOG__DEFAULT__S3__FORCE_VIRTUAL_ADDRESSING": config.s3_force_virtual_addressing,
        }
    )
    return env


def _table_name(namespace: str, table_prefix: str, suffix: str) -> str:
    return f"{namespace}.{table_prefix}_{suffix}"


def scenarios(namespace: str, table_prefix: str, rows: int, delete_rows: int) -> list[Scenario]:
    many = _table_name(namespace, table_prefix, "many_files")
    partitioned = _table_name(namespace, table_prefix, "partitioned")
    deletes = _table_name(namespace, table_prefix, "pos_deletes")
    many_manifests = _table_name(namespace, table_prefix, "many_manifests")
    halfway = rows // 2
    delete_halfway = delete_rows // 2
    return [
        Scenario("many_files_full", many),
        Scenario("many_files_project_id", many, selected_fields=("id",)),
        Scenario("many_files_filter_id", many, row_filter=f"id >= {halfway}", selected_fields=("id", "value")),
        Scenario("many_files_limit_1000", many, limit=1_000),
        Scenario("partition_pruned_part_7", partitioned, row_filter="part = 7", selected_fields=("id", "part", "value")),
        Scenario("partitioned_project", partitioned, selected_fields=("id", "part")),
        Scenario("pos_deletes_full", deletes),
        Scenario("pos_deletes_filter", deletes, row_filter=f"id >= {delete_halfway}", selected_fields=("id", "value")),
        Scenario("many_manifests_full", many_manifests),
        Scenario("many_manifests_filter", many_manifests, row_filter="id = 50", selected_fields=("id", "value")),
    ]


def provision(args: argparse.Namespace, config: CatalogConfig) -> None:
    from pyspark.sql import SparkSession

    from pyiceberg.catalog import load_catalog

    spark = SparkSession.builder.remote(args.spark_uri).getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", str(max(args.files, args.delete_files)))
    catalog = load_catalog("default", **_catalog_props(config))
    try:
        catalog.create_namespace(args.namespace)
    except Exception:
        pass

    many = _table_name(args.namespace, args.table_prefix, "many_files")
    partitioned = _table_name(args.namespace, args.table_prefix, "partitioned")
    deletes = _table_name(args.namespace, args.table_prefix, "pos_deletes")
    many_manifests = _table_name(args.namespace, args.table_prefix, "many_manifests")

    if args.refresh:
        for identifier in (many, partitioned, deletes, many_manifests):
            spark.sql(f"DROP TABLE IF EXISTS rest.{identifier}")

    if (
        _table_exists(catalog, many)
        and _table_exists(catalog, partitioned)
        and _table_exists(catalog, deletes)
        and _table_exists(catalog, many_manifests)
    ):
        print("Benchmark tables already exist; use --refresh to recreate them.")
        return

    print(f"Creating {many}: rows={args.rows:,}, files={args.files:,}")
    base = _benchmark_dataframe(spark, args.rows).repartition(args.files)
    base.writeTo(f"rest.{many}").using("iceberg").tableProperty("format-version", "2").createOrReplace()

    print(f"Creating {partitioned}: rows={args.rows:,}, files={args.files:,}, partitioned by part")
    partitioned_df = _benchmark_dataframe(spark, args.rows).repartition(args.files, "part")
    partitioned_df.writeTo(f"rest.{partitioned}").using("iceberg").tableProperty("format-version", "2").partitionedBy(
        "part"
    ).createOrReplace()

    print(f"Creating {deletes}: rows={args.delete_rows:,}, files={args.delete_files:,}, positional deletes ~=5%")
    delete_df = _benchmark_dataframe(spark, args.delete_rows).repartition(args.delete_files)
    delete_df.writeTo(f"rest.{deletes}").using("iceberg").tableProperty("format-version", "2").tableProperty(
        "write.delete.mode", "merge-on-read"
    ).tableProperty("write.update.mode", "merge-on-read").tableProperty("write.merge.mode", "merge-on-read").createOrReplace()
    spark.sql(f"DELETE FROM rest.{deletes} WHERE id % 20 = 0")

    print(f"Creating {many_manifests}: planning-heavy table with many manifests")
    small_df = _benchmark_dataframe(spark, 100).repartition(1)
    small_df.writeTo(f"rest.{many_manifests}").using("iceberg").tableProperty("format-version", "2").createOrReplace()
    for _ in range(20):
        small_df.writeTo(f"rest.{many_manifests}").append()


def _benchmark_dataframe(spark: Any, rows: int) -> Any:
    from pyspark.sql import functions as F

    return (
        spark.range(0, rows, 1, numPartitions=max(1, min(rows, 20_000)))
        .withColumn("part", (F.col("id") % F.lit(100)).cast("int"))
        .withColumn("value", (F.col("id") * F.lit(3)).cast("long"))
        .withColumn("payload", F.concat(F.lit("payload-"), F.col("id").cast("string")))
    )


def _table_exists(catalog: Any, identifier: str) -> bool:
    try:
        catalog.load_table(identifier)
        return True
    except Exception:
        return False


def validate_scenarios(config: CatalogConfig, scenario_list: list[Scenario], engines: list[str]) -> None:
    failures: list[str] = []
    for scenario in scenario_list:
        summaries = {}
        for engine in engines:
            summaries[engine] = _run_scan(config, scenario, engine, validate_only=True)

        ref_engine = engines[0]
        ref_summary = summaries[ref_engine]
        comparable_keys = ("rows", "batches", "columns", "checksum")
        for engine in engines[1:]:
            mismatches = [key for key in comparable_keys if ref_summary.get(key) != summaries[engine].get(key)]
            if mismatches:
                failures.append(
                    f"{scenario.name}: mismatched {', '.join(mismatches)} "
                    f"between {ref_engine}={ref_summary} and {engine}={summaries[engine]}"
                )
    if failures:
        raise RuntimeError("Validation failed:\n" + "\n".join(failures))


def _run_scan(config: CatalogConfig, scenario: Scenario, engine: str, validate_only: bool = False) -> dict[str, Any]:
    from pyiceberg.catalog import load_catalog

    os.environ["PYICEBERG_RUST_ARROW_SCAN"] = "1" if engine == "native-task" else "0"
    os.environ["PYICEBERG_RUST_PLANNED_ARROW_SCAN"] = "1" if engine == "native-planned" else "0"
    catalog = load_catalog("default", **_catalog_props(config))
    table = catalog.load_table(scenario.table)
    scan_kwargs: dict[str, Any] = {}
    if scenario.selected_fields is not None:
        scan_kwargs["selected_fields"] = scenario.selected_fields
    if scenario.row_filter is not None:
        scan_kwargs["row_filter"] = scenario.row_filter
    if scenario.limit is not None:
        scan_kwargs["limit"] = scenario.limit

    tracemalloc.start()
    rss_before_mb = _current_rss_mb()
    start = time.perf_counter()
    rows = 0
    batches = 0
    checksum = 0
    columns: list[str] | None = None
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        for batch in table.scan(**scan_kwargs).to_arrow_batch_reader():
            if columns is None:
                columns = batch.schema.names
            rows += batch.num_rows
            batches += 1
            checksum += _batch_checksum(batch)

    fallback_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if any(marker in str(warning.message) for marker in FALLBACK_WARNING_MARKERS)
    ]
    if fallback_warnings:
        raise RuntimeError(f"{engine} {scenario.name} used a fallback scan path: {fallback_warnings}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    rss_after_mb = _current_rss_mb()
    tracemalloc.stop()

    result = {
        "engine": engine,
        "scenario": scenario.name,
        "elapsed_ms": elapsed_ms,
        "rows": rows,
        "batches": batches,
        "columns": columns or [],
        "checksum": checksum,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "maxrss_mb": _maxrss_mb(),
    }
    if validate_only:
        return result
    return result


def _batch_checksum(batch: Any) -> int:
    import pyarrow.compute as pc

    checksum = 0
    table = batch.to_struct_array().flatten()
    for name in ("id", "part", "value"):
        if name in batch.schema.names:
            column = table[batch.schema.get_field_index(name)]
            if column.type.__class__.__name__ == "RunEndEncodedType":
                column = pc.run_end_decode(column)
            scalar = pc.sum(column)
            value = scalar.as_py()
            if value is not None:
                checksum += int(value)
    return checksum


def _maxrss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run_child(payload: dict[str, Any]) -> None:
    config = CatalogConfig(**payload["config"])
    scenario = Scenario(
        name=payload["scenario"]["name"],
        table=payload["scenario"]["table"],
        selected_fields=tuple(payload["scenario"]["selected_fields"]) if payload["scenario"]["selected_fields"] else None,
        row_filter=payload["scenario"]["row_filter"],
        limit=payload["scenario"]["limit"],
    )
    result = _run_scan(config, scenario, payload["engine"])
    print(json.dumps(result, sort_keys=True))


def measure_once(config: CatalogConfig, scenario: Scenario, engine: str) -> RunResult:
    import psutil

    payload = {
        "config": asdict(config),
        "scenario": {
            "name": scenario.name,
            "table": scenario.table,
            "selected_fields": scenario.selected_fields,
            "row_filter": scenario.row_filter,
            "limit": scenario.limit,
        },
        "engine": engine,
    }
    command = [sys.executable, str(Path(__file__).resolve()), "__run_once", json.dumps(payload)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_set_catalog_env(config),
    )
    ps_process = psutil.Process(process.pid)
    rss_before = _rss_mb(ps_process)
    peak_rss = rss_before
    while process.poll() is None:
        peak_rss = max(peak_rss, _rss_mb(ps_process))
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    peak_rss = max(peak_rss, _rss_mb(ps_process, default=peak_rss))
    if process.returncode != 0:
        raise RuntimeError(
            f"{engine} {scenario.name} failed with exit {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    line = stdout.strip().splitlines()[-1]
    payload_result = json.loads(line)
    return RunResult(
        engine=engine,
        scenario=scenario.name,
        elapsed_ms=float(payload_result["elapsed_ms"]),
        rows=int(payload_result["rows"]),
        batches=int(payload_result["batches"]),
        rss_before_mb=float(payload_result["rss_before_mb"]),
        rss_after_mb=float(payload_result["rss_after_mb"]),
        peak_rss_mb=peak_rss,
        maxrss_mb=float(payload_result["maxrss_mb"]),
        tracemalloc_peak_mb=float(payload_result["tracemalloc_peak_mb"]),
    )


def _rss_mb(process: Any, default: float = 0.0) -> float:
    try:
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return default


def run_benchmarks(
    args: argparse.Namespace, config: CatalogConfig, scenario_list: list[Scenario], engines: list[str]
) -> list[RunResult]:
    results: list[RunResult] = []

    for scenario in scenario_list:
        for engine in engines:
            for _ in range(args.warmups):
                measure_once(config, scenario, engine)
            for _ in range(args.runs):
                result = measure_once(config, scenario, engine)
                print(
                    f"{scenario.name:<30} {engine:<8} "
                    f"{result.elapsed_ms:>9.1f} ms rows={result.rows} "
                    f"rss_peak={result.peak_rss_mb:.1f} MB"
                )
                results.append(result)
    return results


def summarize(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.scenario, result.engine), []).append(result)

    rows: list[dict[str, Any]] = []
    for (scenario, engine), group in sorted(grouped.items()):
        latencies = [r.elapsed_ms for r in group]
        peak_rss = [r.peak_rss_mb for r in group]
        rss_deltas = [r.rss_after_mb - r.rss_before_mb for r in group]
        maxrss = [r.maxrss_mb for r in group]
        tracemalloc_peaks = [r.tracemalloc_peak_mb for r in group]
        median_ms = statistics.median(latencies)
        rows_read = group[-1].rows
        rows.append(
            {
                "scenario": scenario,
                "engine": engine,
                "runs": len(group),
                "rows": rows_read,
                "batches": group[-1].batches,
                "mean_ms": statistics.mean(latencies),
                "median_ms": median_ms,
                "p95_ms": _percentile(latencies, 95),
                "rows_per_sec": rows_read / (median_ms / 1000) if median_ms else 0,
                "rss_delta_mb": max(rss_deltas),
                "peak_rss_mb": max(peak_rss),
                "maxrss_mb": max(maxrss),
                "tracemalloc_peak_mb": max(tracemalloc_peaks),
            }
        )
    return rows


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scenario | Engine | Runs | Rows | Batches | Median ms | Mean ms | P95 ms | Rows/s | "
        "RSS delta MB | Peak RSS MB | Max RSS MB | Python peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {scenario} | {engine} | {runs} | {rows} | {batches} | {median_ms:.1f} | {mean_ms:.1f} | "
            "{p95_ms:.1f} | {rows_per_sec:.0f} | {rss_delta_mb:.1f} | {peak_rss_mb:.1f} | {maxrss_mb:.1f} | "
            "{tracemalloc_peak_mb:.1f} |".format(**row)
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress benchmark PyArrow vs Rust-backed Arrow scans.")
    parser.add_argument("--uri", default="http://localhost:8181", help="REST catalog URI")
    parser.add_argument("--spark-uri", default="sc://localhost:15002", help="Spark Connect URI used for provisioning")
    parser.add_argument("--s3-endpoint", default="http://localhost:9000", help="S3-compatible endpoint for the client")
    parser.add_argument("--s3-access-key-id", default="admin")
    parser.add_argument("--s3-secret-access-key", default="password")
    parser.add_argument("--s3-region", default="us-east-1")
    parser.add_argument("--s3-force-virtual-addressing", default="false")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--table-prefix", default=DEFAULT_TABLE_PREFIX)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--files", type=int, default=DEFAULT_FILES)
    parser.add_argument("--delete-rows", type=int, default=DEFAULT_DELETE_ROWS)
    parser.add_argument("--delete-files", type=int, default=DEFAULT_DELETE_FILES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--refresh", action="store_true", help="Drop and recreate benchmark tables")
    parser.add_argument("--skip-provision", action="store_true", help="Assume benchmark tables already exist")
    parser.add_argument("--skip-validation", action="store_true", help="Skip parity validation before timed runs")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--pyarrow-only", action="store_true")
    parser.add_argument("--engines", default="pyarrow,native-task,native-planned", help="Comma-separated engines to run")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "__run_once":
        run_child(json.loads(sys.argv[2]))
        return

    args = parse_args()
    config = CatalogConfig(
        uri=args.uri,
        s3_endpoint=args.s3_endpoint,
        s3_access_key_id=args.s3_access_key_id,
        s3_secret_access_key=args.s3_secret_access_key,
        s3_region=args.s3_region,
        s3_force_virtual_addressing=args.s3_force_virtual_addressing,
    )
    scenario_list = scenarios(args.namespace, args.table_prefix, args.rows, args.delete_rows)

    if args.native_only and args.pyarrow_only:
        raise ValueError("--native-only and --pyarrow-only are mutually exclusive")

    if args.native_only:
        engines = ["native-task", "native-planned"]
    elif args.pyarrow_only:
        engines = ["pyarrow"]
    else:
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    if not args.skip_provision:
        provision(args, config)

    if not args.skip_validation:
        print("Validating native and PyArrow scan parity...")
        validate_scenarios(config, scenario_list, engines)

    print("Running benchmark. Memory caveat: RSS is for the Python benchmark/client process, not Spark/REST/MinIO containers.")
    results = run_benchmarks(args, config, scenario_list, engines)
    summary_rows = summarize(results)
    table = markdown_table(summary_rows)
    print("\n" + table)

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"summary": summary_rows, "runs": [asdict(result) for result in results]}, indent=2, sort_keys=True) + "\n"
        )
    if args.markdown_out:
        args.markdown_out.write_text(table + "\n")


if __name__ == "__main__":
    main()
