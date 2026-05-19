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
import functools
import operator

import pyarrow as pa
from pyarrow import Table as pyarrow_table
from pyarrow import compute as pc

from pyiceberg.expressions import (
    AlwaysFalse,
    BooleanExpression,
    EqualTo,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThanOrEqual,
    Or,
)


def create_file_match_filter(df: pyarrow_table, join_cols: list[str]) -> BooleanExpression:
    """Build a conservative predicate for upsert file pruning.

    The returned predicate may match extra files, but must not exclude files that
    could contain a matching row. Exact row matching still happens downstream.
    """
    if len(df) == 0:
        return AlwaysFalse()

    per_col: list[BooleanExpression] = []
    for col in join_cols:
        col_arr = df.column(col)
        bounds = pc.min_max(col_arr).as_py()
        col_min, col_max = bounds["min"], bounds["max"]

        if col_min is None:
            per_col.append(IsNull(col))
            continue

        pred: BooleanExpression = GreaterThanOrEqual(col, col_min) & LessThanOrEqual(col, col_max)
        if pc.any(pc.is_null(col_arr)).as_py():
            pred = pred | IsNull(col)
        per_col.append(pred)

    return functools.reduce(operator.and_, per_col)


def create_match_filter(df: pyarrow_table, join_cols: list[str]) -> BooleanExpression:
    """Exact predicate over the source join keys; null in a key compares with IsNull."""
    unique_keys = df.select(join_cols).group_by(join_cols).aggregate([])
    if len(unique_keys) == 0:
        return AlwaysFalse()

    if len(join_cols) == 1:
        col = join_cols[0]
        vals = unique_keys[0].to_pylist()
        non_null = [v for v in vals if v is not None]
        if not non_null:
            return IsNull(col)
        in_pred: BooleanExpression = In(col, non_null)
        return in_pred if len(non_null) == len(vals) else in_pred | IsNull(col)

    row_preds = [
        functools.reduce(
            operator.and_,
            [EqualTo(c, row[c]) if row[c] is not None else IsNull(c) for c in join_cols],
        )
        for row in unique_keys.to_pylist()
    ]
    return row_preds[0] if len(row_preds) == 1 else Or(*row_preds)


def _default_scalar(arrow_type: pa.DataType) -> pa.Scalar:
    """Return a fixed non-null scalar of the given type for use as a null sentinel."""
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pa.scalar("", type=arrow_type)
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return pa.scalar(b"", type=arrow_type)
    if pa.types.is_fixed_size_binary(arrow_type):
        return pa.scalar(b"\x00" * arrow_type.byte_width, type=arrow_type)
    if pa.types.is_boolean(arrow_type):
        return pa.scalar(False, type=arrow_type)
    return pa.scalar(0, type=arrow_type)


def _augment_for_null_safe_join(table: pa.Table, join_cols: set[str]) -> pa.Table:
    """Augment join columns so pyarrow inner join treats null↔null as a match.

    Replaces nulls in each join col with a fixed same-type sentinel and appends an
    `__isnull_<col>` indicator. Joining on (col, indicator) then matches null rows to
    null rows. The sentinel is type-derived (not data-derived) so both sides agree.
    """
    for col in join_cols:
        if f"__isnull_{col}" in join_cols:
            raise ValueError(f"join column '__isnull_{col}' collides with the reserved null-indicator name")
    out = table
    for col in join_cols:
        arr = table.column(col)
        out = out.set_column(out.column_names.index(col), col, pc.fill_null(arr, _default_scalar(arr.type)))
        out = out.append_column(f"__isnull_{col}", pc.is_null(arr))
    return out


def has_duplicate_rows(df: pyarrow_table, join_cols: list[str]) -> bool:
    """Check for duplicate rows in a PyArrow table based on the join columns."""
    return len(df.select(join_cols).group_by(join_cols).aggregate([([], "count_all")]).filter(pc.field("count_all") > 1)) > 0


def get_rows_to_update(source_table: pa.Table, target_table: pa.Table, join_cols: list[str]) -> pa.Table:
    """
    Return a table with rows that need to be updated in the target table based on the join columns.

    The table is joined on the identifier columns, and then checked if there are any updated rows.
    Those are selected and everything is renamed correctly.
    """
    all_columns = set(source_table.column_names)
    join_cols_set = set(join_cols)

    non_key_cols = list(all_columns - join_cols_set)

    if has_duplicate_rows(target_table, join_cols):
        raise ValueError("Target table has duplicate rows, aborting upsert")

    if len(target_table) == 0:
        # When the target table is empty, there is nothing to update :)
        return source_table.schema.empty_table()

    # We need to compare non_key_cols in Python as PyArrow
    # 1. Cannot do a join when non-join columns have complex types
    # 2. Cannot compare columns with complex types
    # See: https://github.com/apache/arrow/issues/35785
    SOURCE_INDEX_COLUMN_NAME = "__source_index"
    TARGET_INDEX_COLUMN_NAME = "__target_index"

    if SOURCE_INDEX_COLUMN_NAME in join_cols or TARGET_INDEX_COLUMN_NAME in join_cols:
        raise ValueError(
            f"{SOURCE_INDEX_COLUMN_NAME} and {TARGET_INDEX_COLUMN_NAME} are reserved for joining "
            f"DataFrames, and cannot be used as column names"
        ) from None

    # Step 1: Prepare source index with join keys and a marker index
    # Cast only the join keys to target table types, so we can do the join
    # See: https://github.com/apache/arrow/issues/37542
    target_key_schema = pa.schema([target_table.schema.field(col) for col in join_cols])
    source_index = _augment_for_null_safe_join(
        source_table.select(join_cols_set).cast(target_key_schema), join_cols_set
    ).append_column(SOURCE_INDEX_COLUMN_NAME, pa.array(range(len(source_table))))

    # Step 2: Prepare target index with join keys and a marker
    target_index = _augment_for_null_safe_join(target_table.select(join_cols_set), join_cols_set).append_column(
        TARGET_INDEX_COLUMN_NAME, pa.array(range(len(target_table)))
    )

    # Step 3: Inner join on (key value, is-null indicator) per col — matches null↔null.
    join_keys = list(join_cols_set) + [f"__isnull_{c}" for c in join_cols_set]
    matching_indices = source_index.join(target_index, keys=join_keys, join_type="inner")

    # Step 4: Compare all rows using Python
    to_update_indices = []
    for source_idx, target_idx in zip(
        matching_indices[SOURCE_INDEX_COLUMN_NAME].to_pylist(),
        matching_indices[TARGET_INDEX_COLUMN_NAME].to_pylist(),
        strict=True,
    ):
        source_row = source_table.slice(source_idx, 1)
        target_row = target_table.slice(target_idx, 1)

        for key in non_key_cols:
            source_val = source_row.column(key)[0].as_py()
            target_val = target_row.column(key)[0].as_py()
            if source_val != target_val:
                to_update_indices.append(source_idx)
                break

    # Step 5: Take rows from source table using the indices and cast to target schema
    if to_update_indices:
        return source_table.take(to_update_indices)
    else:
        return source_table.schema.empty_table()
