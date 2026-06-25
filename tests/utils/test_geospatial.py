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
import math
import struct

import pytest

from pyiceberg.utils.geospatial import (
    GeospatialBound,
    deserialize_geospatial_bound,
    extract_envelope_from_wkb,
    merge_envelopes,
    serialize_geospatial_bound,
)


def test_geospatial_bound_serde_xy() -> None:
    raw = serialize_geospatial_bound(GeospatialBound(x=10.0, y=20.0))
    assert len(raw) == 16
    bound = deserialize_geospatial_bound(raw)
    assert bound.x == 10.0
    assert bound.y == 20.0
    assert bound.z is None
    assert bound.m is None


def test_geospatial_bound_serde_xyz() -> None:
    raw = serialize_geospatial_bound(GeospatialBound(x=10.0, y=20.0, z=30.0))
    assert len(raw) == 24
    bound = deserialize_geospatial_bound(raw)
    assert bound.x == 10.0
    assert bound.y == 20.0
    assert bound.z == 30.0
    assert bound.m is None


def test_geospatial_bound_serde_xym() -> None:
    raw = serialize_geospatial_bound(GeospatialBound(x=10.0, y=20.0, m=40.0))
    assert len(raw) == 32
    x, y, z, m = struct.unpack("<dddd", raw)
    assert x == 10.0
    assert y == 20.0
    assert math.isnan(z)
    assert m == 40.0

    bound = deserialize_geospatial_bound(raw)
    assert bound.x == 10.0
    assert bound.y == 20.0
    assert bound.z is None
    assert bound.m == 40.0


def test_geospatial_bound_serde_xyzm() -> None:
    raw = serialize_geospatial_bound(GeospatialBound(x=10.0, y=20.0, z=30.0, m=40.0))
    assert len(raw) == 32
    bound = deserialize_geospatial_bound(raw)
    assert bound.x == 10.0
    assert bound.y == 20.0
    assert bound.z == 30.0
    assert bound.m == 40.0


def test_geospatial_bound_serde_rejects_ambiguous_nan_z_with_m() -> None:
    with pytest.raises(ValueError, match="NaN z"):
        serialize_geospatial_bound(GeospatialBound(x=1.0, y=2.0, z=math.nan, m=5.0))

    xym = GeospatialBound(x=1.0, y=2.0, z=None, m=5.0)
    assert deserialize_geospatial_bound(serialize_geospatial_bound(xym)) == xym

    xyzm = GeospatialBound(x=1.0, y=2.0, z=3.0, m=5.0)
    assert deserialize_geospatial_bound(serialize_geospatial_bound(xyzm)) == xyzm


def test_extract_envelope_geometry() -> None:
    # LINESTRING(170 0, -170 1)
    wkb = struct.pack("<BIIdddd", 1, 2, 2, 170.0, 0.0, -170.0, 1.0)
    envelope = extract_envelope_from_wkb(wkb, is_geography=False)
    assert envelope is not None
    assert envelope.x_min == -170.0
    assert envelope.x_max == 170.0
    assert envelope.y_min == 0.0
    assert envelope.y_max == 1.0


def test_extract_envelope_geography_wraps_antimeridian() -> None:
    # LINESTRING(170 0, -170 1)
    wkb = struct.pack("<BIIdddd", 1, 2, 2, 170.0, 0.0, -170.0, 1.0)
    envelope = extract_envelope_from_wkb(wkb, is_geography=True)
    assert envelope is not None
    assert envelope.x_min > envelope.x_max
    assert envelope.x_min == 170.0
    assert envelope.x_max == -170.0
    assert envelope.y_min == 0.0
    assert envelope.y_max == 1.0


def test_extract_envelope_xyzm_linestring() -> None:
    # LINESTRING ZM (0 1 2 3, 4 5 6 7)
    wkb = struct.pack("<BII" + "dddd" * 2, 1, 3002, 2, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    envelope = extract_envelope_from_wkb(wkb, is_geography=False)
    assert envelope is not None
    assert envelope.x_min == 0.0
    assert envelope.x_max == 4.0
    assert envelope.y_min == 1.0
    assert envelope.y_max == 5.0
    assert envelope.z_min == 2.0
    assert envelope.z_max == 6.0
    assert envelope.m_min == 3.0
    assert envelope.m_max == 7.0


def test_extract_envelope_geography_edge_does_not_wrap_when_vertices_span_more_than_half() -> None:
    # LINESTRING(-120 0, 0 0, 120 0): the connected edges occupy the -120..120 arc
    # through 0. A vertex-set minimal-arc heuristic would instead wrap and exclude a
    # real edge (e.g. the segment containing longitude -60), dropping matching rows.
    wkb = struct.pack("<BII" + "dd" * 3, 1, 2, 3, -120.0, 0.0, 0.0, 0.0, 120.0, 0.0)
    envelope = extract_envelope_from_wkb(wkb, is_geography=True)
    assert envelope is not None
    assert envelope.x_min == -120.0
    assert envelope.x_max == 120.0
    assert envelope.x_min < envelope.x_max  # does NOT wrap the antimeridian


def test_extract_envelope_geography_multipoint_has_no_connecting_edges() -> None:
    # MULTIPOINT(-120 0, 0 0, 120 0): no edges connect the points, so the minimal
    # enclosing arc is the short way (wrapping the antimeridian) since the largest
    # gap among the three points is the -120..120 span through 0.
    point = lambda x, y: struct.pack("<BIdd", 1, 1, x, y)  # noqa: E731
    body = point(-120.0, 0.0) + point(0.0, 0.0) + point(120.0, 0.0)
    wkb = struct.pack("<BII", 1, 4, 3) + body
    envelope = extract_envelope_from_wkb(wkb, is_geography=True)
    assert envelope is not None
    assert envelope.x_min > envelope.x_max  # wraps: 120 .. -120 across +-180


def test_merge_geography_envelopes() -> None:
    left = extract_envelope_from_wkb(struct.pack("<BIIdddd", 1, 2, 2, 170.0, 0.0, -170.0, 1.0), is_geography=True)
    right = extract_envelope_from_wkb(struct.pack("<BIIdddd", 1, 2, 2, -160.0, 2.0, -120.0, 3.0), is_geography=True)
    assert left is not None
    assert right is not None

    merged = merge_envelopes(left, right, is_geography=True)
    assert merged.x_min > merged.x_max
    assert merged.x_min == 170.0
    assert merged.x_max == -120.0
    assert merged.y_min == 0.0
    assert merged.y_max == 3.0
