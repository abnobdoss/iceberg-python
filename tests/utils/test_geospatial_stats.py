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

from pyiceberg.utils.geospatial import GeospatialStatsAggregator, deserialize_geospatial_bound


def _point_wkb(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def _linestring_xyzm_wkb(points: list[tuple[float, float, float, float]]) -> bytes:
    ordinates = [ordinate for point in points for ordinate in point]
    return struct.pack("<BII" + ("dddd" * len(points)), 1, 3002, len(points), *ordinates)


def test_geospatial_stats_aggregator_accumulates_planar_bounds() -> None:
    aggregator = GeospatialStatsAggregator(is_geography=False)

    aggregator.add(_point_wkb(10.0, -3.0))
    aggregator.add(_linestring_xyzm_wkb([(-2.0, 4.0, 7.0, 3.0), (6.0, 9.0, 2.0, 8.0)]))

    min_bound = aggregator.min_bound()
    max_bound = aggregator.max_bound()
    assert min_bound is not None
    assert max_bound is not None
    assert min_bound.x == -2.0
    assert min_bound.y == -3.0
    assert min_bound.z == 2.0
    assert min_bound.m == 3.0
    assert max_bound.x == 10.0
    assert max_bound.y == 9.0
    assert max_bound.z == 7.0
    assert max_bound.m == 8.0

    serialized_min = aggregator.serialized_min()
    serialized_max = aggregator.serialized_max()
    assert serialized_min is not None
    assert serialized_max is not None
    assert deserialize_geospatial_bound(serialized_min) == aggregator.min_bound()
    assert deserialize_geospatial_bound(serialized_max) == aggregator.max_bound()


def test_geospatial_stats_aggregator_geography_wraps_antimeridian() -> None:
    aggregator = GeospatialStatsAggregator(is_geography=True)

    aggregator.add(_point_wkb(170.0, 1.0))
    aggregator.add(_point_wkb(-170.0, 2.0))

    min_bound = aggregator.min_bound()
    max_bound = aggregator.max_bound()

    assert min_bound is not None
    assert max_bound is not None
    assert min_bound.x > max_bound.x
    assert min_bound.x == 170.0
    assert max_bound.x == -170.0
    assert min_bound.y == 1.0
    assert max_bound.y == 2.0


def test_geospatial_stats_aggregator_empty_input_has_no_bounds() -> None:
    aggregator = GeospatialStatsAggregator(is_geography=False)

    assert aggregator.min_bound() is None
    assert aggregator.max_bound() is None
    assert aggregator.serialized_min() is None
    assert aggregator.serialized_max() is None
