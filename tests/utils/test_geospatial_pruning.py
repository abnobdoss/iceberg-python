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
import struct
from random import Random

import pytest

from pyiceberg.utils.geospatial import (
    GeospatialBound,
    GeospatialStatsAggregator,
    extract_envelope_from_wkb,
    serialize_geospatial_bound,
)
from pyiceberg.utils.geospatial_pruning import bbox_might_match

_SUPPORTED_PREDICATES = ("st-contains", "st-intersects", "st-overlaps", "st-within")


def _point_wkb(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def _big_endian_point_wkb(x: float, y: float) -> bytes:
    return struct.pack(">BIdd", 0, 1, x, y)


def _point_xyzm_wkb(x: float, y: float, z: float, m: float) -> bytes:
    return struct.pack("<BIdddd", 1, 3001, x, y, z, m)


def _linestring_wkb(points: list[tuple[float, float]]) -> bytes:
    ordinates = [ordinate for point in points for ordinate in point]
    return struct.pack("<BII" + ("dd" * len(points)), 1, 2, len(points), *ordinates)


def _polygon_bbox_wkb(x_min: float, y_min: float, x_max: float, y_max: float) -> bytes:
    points = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min),
    ]
    ordinates = [ordinate for point in points for ordinate in point]
    return struct.pack("<BIII" + ("dd" * len(points)), 1, 3, 1, len(points), *ordinates)


def _bound(x: float, y: float, z: float | None = None, m: float | None = None) -> bytes:
    return serialize_geospatial_bound(GeospatialBound(x=x, y=y, z=z, m=m))


def _bounds(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[bytes, bytes]:
    return _bound(x_min, y_min), _bound(x_max, y_max)


def _stats_bounds(points: list[tuple[float, float]], is_geography: bool) -> tuple[bytes, bytes]:
    aggregator = GeospatialStatsAggregator(is_geography=is_geography)
    for x, y in points:
        aggregator.add(_point_wkb(x, y))

    lower_bound = aggregator.serialized_min()
    upper_bound = aggregator.serialized_max()
    assert lower_bound is not None
    assert upper_bound is not None
    return lower_bound, upper_bound


@pytest.mark.parametrize("predicate", ["st-intersects", "st-overlaps", "st-contains"])
def test_bbox_pruning_disjoint_geometry_returns_false(predicate: str) -> None:
    lower_bound, upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    query_wkb = _polygon_bbox_wkb(20.0, 20.0, 30.0, 30.0)

    assert not bbox_might_match(predicate, query_wkb, lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_overlapping_geometry_returns_true() -> None:
    lower_bound, upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    query_wkb = _polygon_bbox_wkb(5.0, 5.0, 15.0, 15.0)

    assert bbox_might_match("st-intersects", query_wkb, lower_bound, upper_bound, is_geography=False)


@pytest.mark.parametrize(("lower_bound", "upper_bound"), [(None, _bound(10.0, 10.0)), (_bound(0.0, 0.0), None)])
def test_bbox_pruning_missing_bounds_returns_true(lower_bound: bytes | None, upper_bound: bytes | None) -> None:
    assert bbox_might_match("st-intersects", _point_wkb(100.0, 100.0), lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_within_disjoint_returns_false() -> None:
    lower_bound, upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    query_wkb = _polygon_bbox_wkb(20.0, 20.0, 30.0, 30.0)

    assert not bbox_might_match("st-within", query_wkb, lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_within_overlapping_returns_true() -> None:
    lower_bound, upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    query_wkb = _polygon_bbox_wkb(5.0, 5.0, 15.0, 15.0)

    assert bbox_might_match("st-within", query_wkb, lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_antimeridian_wrapped_file_matches_query_inside_interval() -> None:
    lower_bound, upper_bound = _bounds(170.0, -10.0, -170.0, 10.0)

    assert bbox_might_match("st-intersects", _point_wkb(178.0, 0.0), lower_bound, upper_bound, is_geography=True)


def test_bbox_pruning_antimeridian_wrapped_file_prunes_query_outside_interval() -> None:
    lower_bound, upper_bound = _bounds(170.0, -10.0, -170.0, 10.0)

    assert not bbox_might_match("st-intersects", _point_wkb(0.0, 0.0), lower_bound, upper_bound, is_geography=True)


@pytest.mark.parametrize(
    ("file_x_min", "file_x_max", "query_x"),
    [
        (170.0, 180.0, 180.0),
        (170.0, 180.0, -180.0),
        (180.0, 180.0, 180.0),
        (180.0, 180.0, -180.0),
        (-180.0, -180.0, 180.0),
        (-180.0, -180.0, -180.0),
    ],
)
def test_bbox_pruning_geography_antimeridian_seam_is_closed(
    file_x_min: float,
    file_x_max: float,
    query_x: float,
) -> None:
    lower_bound, upper_bound = _bounds(file_x_min, 0.0, file_x_max, 10.0)

    assert bbox_might_match("st-intersects", _point_wkb(query_x, 5.0), lower_bound, upper_bound, is_geography=True)


@pytest.mark.parametrize(
    ("file_x_min", "file_y_min", "file_x_max", "file_y_max", "query_x", "query_y"),
    [
        (170.0, 0.0, 180.0, 10.0, 0.0, 5.0),
        (170.0, 0.0, 180.0, 10.0, -90.0, 5.0),
        (180.0, 0.0, 180.0, 10.0, 179.0, 5.0),
        (-180.0, 0.0, -180.0, 10.0, -179.0, 5.0),
        (170.0, 0.0, -170.0, 10.0, 0.0, 5.0),
    ],
)
def test_bbox_pruning_geography_negative_mutation_guard(
    file_x_min: float,
    file_y_min: float,
    file_x_max: float,
    file_y_max: float,
    query_x: float,
    query_y: float,
) -> None:
    # Mutation guard: keep at least five False assertions so a return-True stub fails loudly.
    lower_bound, upper_bound = _bounds(file_x_min, file_y_min, file_x_max, file_y_max)

    assert not bbox_might_match("st-intersects", _point_wkb(query_x, query_y), lower_bound, upper_bound, is_geography=True)


def test_bbox_pruning_big_endian_point_wkb_extracts_and_prunes() -> None:
    query_wkb = _big_endian_point_wkb(5.0, 6.0)
    envelope = extract_envelope_from_wkb(query_wkb, is_geography=False)

    assert envelope is not None
    assert envelope.x_min == 5.0
    assert envelope.x_max == 5.0
    assert envelope.y_min == 6.0
    assert envelope.y_max == 6.0

    matching_lower_bound, matching_upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    disjoint_lower_bound, disjoint_upper_bound = _bounds(20.0, 0.0, 30.0, 10.0)

    assert bbox_might_match(
        "st-intersects",
        query_wkb,
        matching_lower_bound,
        matching_upper_bound,
        is_geography=False,
    )
    assert not bbox_might_match(
        "st-intersects",
        query_wkb,
        disjoint_lower_bound,
        disjoint_upper_bound,
        is_geography=False,
    )


def test_bbox_pruning_false_negative_guard_for_points_inside_wrapped_file_bbox() -> None:
    lower_bound, upper_bound = _bounds(170.0, -20.0, -160.0, 20.0)

    for longitude, latitude in [
        (170.0, -20.0),
        (174.5, 3.0),
        (179.0, 19.0),
        (180.0, 0.0),
        (-179.5, -7.0),
        (-170.0, 20.0),
        (-160.0, 0.5),
    ]:
        assert bbox_might_match(
            "st-intersects",
            _point_wkb(longitude, latitude),
            lower_bound,
            upper_bound,
            is_geography=True,
        )


def test_bbox_pruning_ignores_z_and_m_dimensions() -> None:
    lower_bound = _bound(0.0, 0.0, z=0.0, m=0.0)
    upper_bound = _bound(10.0, 10.0, z=1.0, m=1.0)
    query_wkb = _point_xyzm_wkb(5.0, 5.0, 999.0, 999.0)

    assert bbox_might_match("st-intersects", query_wkb, lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_empty_query_returns_true() -> None:
    lower_bound, upper_bound = _bounds(0.0, 0.0, 10.0, 10.0)
    query_wkb = _linestring_wkb([])

    assert bbox_might_match("st-intersects", query_wkb, lower_bound, upper_bound, is_geography=False)


def test_bbox_pruning_geography_preserves_file_edge_point_after_circle_round_trip() -> None:
    points = [(-9.32, 29.55), (-158.16, 36.27), (52.97, 88.76)]
    lower_bound, upper_bound = _stats_bounds(points, is_geography=True)

    assert bbox_might_match(
        "st-intersects",
        _point_wkb(-158.16, 36.27),
        lower_bound,
        upper_bound,
        is_geography=True,
    )


@pytest.mark.parametrize(
    ("is_geography", "lon_range", "lat_range"),
    [
        (True, (-180.0, 180.0), (-90.0, 90.0)),
        (False, (-1_000_000.0, 1_000_000.0), (-1_000_000.0, 1_000_000.0)),
    ],
)
def test_bbox_pruning_never_prunes_files_own_points(
    is_geography: bool,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    rng = Random(1729 if is_geography else 2718)
    false_negatives: list[tuple[list[tuple[float, float]], tuple[float, float], str]] = []

    for _ in range(2000):
        points = [
            (
                rng.uniform(*lon_range),
                rng.uniform(*lat_range),
            )
            for _ in range(rng.randint(1, 5))
        ]
        lower_bound, upper_bound = _stats_bounds(points, is_geography=is_geography)

        for point in points:
            query_wkb = _point_wkb(*point)
            for predicate in _SUPPORTED_PREDICATES:
                if not bbox_might_match(
                    predicate,
                    query_wkb,
                    lower_bound,
                    upper_bound,
                    is_geography=is_geography,
                ):
                    false_negatives.append((points, point, predicate))

    assert false_negatives == []


def test_bbox_pruning_geography_still_prunes_clearly_disjoint_query() -> None:
    lower_bound, upper_bound = _stats_bounds(
        [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)],
        is_geography=True,
    )

    assert not bbox_might_match("st-intersects", _point_wkb(170.0, 80.0), lower_bound, upper_bound, is_geography=True)
