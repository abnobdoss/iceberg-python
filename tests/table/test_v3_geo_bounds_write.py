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
"""Tests for the integrated v3 geospatial bounds write helper + bbox pruning.

``write_file`` now runs ``_geospatial_bounds_from_arrow`` over geometry/geography
WKB columns and merges the resulting ``lower_bounds``/``upper_bounds`` into the
data file. These tests exercise that integrated helper against real Arrow tables,
confirm the bounds serialize per spec, and confirm ``bbox_might_match`` prunes
and keeps files correctly against the persisted bounds (never a false negative).

Documented gap (see integration.md): a full ``Table.append`` of a geo column is
blocked on geoarrow-pyarrow <-> Iceberg schema/parquet interop (the WKB extension
type is neither inferred by ``visit_pyarrow`` nor written cleanly by the Parquet
writer). The bounds-computation and pruning code integrated here is fully real.
"""

import struct

import pyarrow as pa

from pyiceberg.io.pyarrow import _geospatial_bounds_from_arrow
from pyiceberg.schema import Schema
from pyiceberg.types import GeographyType, GeometryType, LongType, NestedField
from pyiceberg.utils.geospatial import deserialize_geospatial_bound
from pyiceberg.utils.geospatial_pruning import bbox_might_match


def _wkb_point(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def test_geospatial_bounds_from_arrow_geometry() -> None:
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "geom", GeometryType(), required=False),
    )
    arrow = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "geom": pa.array(
                [_wkb_point(0.0, 0.0), _wkb_point(10.0, 20.0), _wkb_point(-5.0, 3.0)],
                pa.large_binary(),
            ),
        }
    )

    lower, upper = _geospatial_bounds_from_arrow(schema, arrow)

    assert set(lower) == {2}
    assert set(upper) == {2}

    lower_bound = deserialize_geospatial_bound(lower[2])
    upper_bound = deserialize_geospatial_bound(upper[2])
    assert (lower_bound.x, lower_bound.y) == (-5.0, 0.0)
    assert (upper_bound.x, upper_bound.y) == (10.0, 20.0)

    # Pruning against the exact serialized bounds.
    assert bbox_might_match("st-intersects", _wkb_point(100.0, 100.0), lower[2], upper[2], is_geography=False) is False
    assert bbox_might_match("st-intersects", _wkb_point(5.0, 5.0), lower[2], upper[2], is_geography=False) is True
    # The file's own corner must never be a false negative.
    assert bbox_might_match("st-intersects", _wkb_point(-5.0, 0.0), lower[2], upper[2], is_geography=False) is True
    assert bbox_might_match("st-intersects", _wkb_point(10.0, 20.0), lower[2], upper[2], is_geography=False) is True


def test_geospatial_bounds_skips_nulls_and_nongeo() -> None:
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "geom", GeometryType(), required=False),
    )
    arrow = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "geom": pa.array([_wkb_point(1.0, 2.0), None, _wkb_point(3.0, 4.0)], pa.large_binary()),
        }
    )

    lower, upper = _geospatial_bounds_from_arrow(schema, arrow)
    # Non-geo column (id) gets no geo bounds.
    assert set(lower) == {2}
    lower_bound = deserialize_geospatial_bound(lower[2])
    upper_bound = deserialize_geospatial_bound(upper[2])
    assert (lower_bound.x, lower_bound.y) == (1.0, 2.0)
    assert (upper_bound.x, upper_bound.y) == (3.0, 4.0)


def test_geospatial_bounds_empty_when_all_null() -> None:
    schema = Schema(NestedField(2, "geom", GeometryType(), required=False))
    arrow = pa.table({"geom": pa.array([None, None], pa.large_binary())})

    lower, upper = _geospatial_bounds_from_arrow(schema, arrow)
    assert lower == {}
    assert upper == {}


def test_geospatial_bounds_geography_is_flagged() -> None:
    schema = Schema(NestedField(5, "geo", GeographyType(), required=False))
    arrow = pa.table({"geo": pa.array([_wkb_point(-170.0, 10.0), _wkb_point(170.0, -10.0)], pa.large_binary())})

    lower, upper = _geospatial_bounds_from_arrow(schema, arrow)
    assert set(lower) == {5}
    # geography longitude bounds use the antimeridian-minimal interval; a query
    # crossing the seam must still be kept against this wrapped bbox.
    assert bbox_might_match("st-intersects", _wkb_point(180.0, 0.0), lower[5], upper[5], is_geography=True) is True
