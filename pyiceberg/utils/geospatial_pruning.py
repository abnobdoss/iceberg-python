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
"""Geospatial bounding-box pruning helpers.

Returns False ONLY when no row in the file can possibly match; any uncertainty
returns True. False negatives (wrongly returning False) would cause silent data loss.
"""

from __future__ import annotations

import math

from pyiceberg.utils.geospatial import deserialize_geospatial_bound, extract_envelope_from_wkb

_SUPPORTED_PREDICATES = {"st-contains", "st-intersects", "st-overlaps", "st-within"}
_BOUNDARY_ABSOLUTE_EPSILON = 1e-7
_BOUNDARY_RELATIVE_EPSILON = 1e-12


def bbox_might_match(
    predicate: str,
    query_wkb: bytes,
    lower_bound: bytes | None,
    upper_bound: bytes | None,
    is_geography: bool,
) -> bool:
    if predicate not in _SUPPORTED_PREDICATES:
        raise ValueError(f"Unsupported geospatial predicate for bbox pruning: {predicate}")

    if lower_bound is None or upper_bound is None:
        return True

    query_envelope = extract_envelope_from_wkb(query_wkb, is_geography)
    if query_envelope is None:
        return True

    lower = deserialize_geospatial_bound(lower_bound)
    upper = deserialize_geospatial_bound(upper_bound)

    y_overlaps = _scalar_intervals_overlap(lower.y, upper.y, query_envelope.y_min, query_envelope.y_max)
    if not y_overlaps:
        return False

    if is_geography:
        x_overlaps = _longitude_intervals_overlap(lower.x, upper.x, query_envelope.x_min, query_envelope.x_max)
    else:
        x_overlaps = _scalar_intervals_overlap(lower.x, upper.x, query_envelope.x_min, query_envelope.x_max)

    return x_overlaps


def _scalar_intervals_overlap(left_min: float, left_max: float, right_min: float, right_max: float) -> bool:
    left_start = min(left_min, left_max)
    left_end = max(left_min, left_max)
    right_start = min(right_min, right_max)
    right_end = max(right_min, right_max)

    # BBox pruning must be conservative: returning False can drop matching rows.
    # Geography bounds round-trip longitude through circle coordinates, which can
    # drift stored edges inward by tiny double-precision amounts. Expand only the
    # boundary comparisons so edge-equal queries remain "might match".
    epsilon = _interval_boundary_epsilon(left_start, left_end, right_start, right_end)
    return left_start <= right_end + epsilon and right_start <= left_end + epsilon


def _interval_boundary_epsilon(*values: float) -> float:
    finite_values = (abs(value) for value in values if math.isfinite(value))
    scale = max(finite_values, default=1.0)
    return _BOUNDARY_ABSOLUTE_EPSILON + (_BOUNDARY_RELATIVE_EPSILON * scale)


def _longitude_intervals_overlap(left_min: float, left_max: float, right_min: float, right_max: float) -> bool:
    left_segments = _longitude_interval_to_segments(left_min, left_max)
    right_segments = _longitude_interval_to_segments(right_min, right_max)

    return any(
        _longitude_segments_overlap(left_segment, right_segment)
        for left_segment in left_segments
        for right_segment in right_segments
    )


def _longitude_segments_overlap(left_segment: tuple[float, float], right_segment: tuple[float, float]) -> bool:
    left_start, left_end = left_segment
    right_start, right_end = right_segment

    return any(
        _scalar_intervals_overlap(left_start, left_end, right_start + shift, right_end + shift) for shift in (-360.0, 0.0, 360.0)
    )


def _longitude_interval_to_segments(x_min: float, x_max: float) -> list[tuple[float, float]]:
    start = _longitude_to_circle(x_min)
    end = _longitude_to_circle(x_max)

    if _is_full_longitude_interval(x_min, x_max):
        return [(0.0, 360.0)]

    if x_min <= x_max:
        return [(start, end)]

    return [(start, 360.0), (0.0, end)]


def _is_full_longitude_interval(x_min: float, x_max: float) -> bool:
    return math.isclose(_normalize_longitude(x_min), -180.0) and math.isclose(_normalize_longitude(x_max), 180.0)


def _normalize_longitude(value: float) -> float:
    normalized = ((value + 180.0) % 360.0) - 180.0
    if math.isclose(normalized, -180.0) and value > 0:
        return 180.0
    return normalized


def _longitude_to_circle(value: float) -> float:
    normalized = _normalize_longitude(value)
    if math.isclose(normalized, 180.0):
        return 360.0
    return normalized + 180.0
