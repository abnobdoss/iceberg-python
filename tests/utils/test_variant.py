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

from __future__ import annotations

import struct
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from pyiceberg.types import IcebergType, NestedField, VariantType
from pyiceberg.utils.variant import _PRIMITIVE_FLOAT, decode_variant, encode_variant


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        1,
        -1,
        127,
        128,
        -129,
        32767,
        70000,
        5_000_000_000,
        3.14,
        "",
        "hi",
        "x" * 70,
        {"a": 1, "b": "x", "c": True},
        {"outer": {"inner": 42}, "list": [1, 2, 3]},
        {},
        [1, "two", None, {"k": 3.0}],
        [],
        Decimal("0"),
        Decimal("1.50"),
        Decimal("-3.14159"),
        Decimal("123456789012345.678"),
        Decimal("-9." + "9" * 37),
        b"",
        b"\x00\x01\x02bytes",
        date(2024, 1, 15),
        date(1969, 12, 31),
        datetime(2024, 1, 15, 12, 30, 45, 123456),
        datetime(2024, 1, 15, 12, 30, 45, 123456, tzinfo=timezone.utc),
    ],
)
def test_variant_round_trip(value: Any) -> None:
    metadata_bytes, value_bytes = encode_variant(value)

    assert metadata_bytes[0] & 0x0F == 1
    decoded = decode_variant(metadata_bytes, value_bytes)
    assert decoded == value
    if isinstance(value, Decimal):
        # Scale must survive the round trip, not just numeric equality.
        assert decoded.as_tuple() == value.as_tuple()


def test_short_string_uses_short_string_basic_type() -> None:
    _, value_bytes = encode_variant("hi")

    assert value_bytes[0] & 0x03 == 1


def test_long_string_uses_primitive_string() -> None:
    _, value_bytes = encode_variant("x" * 70)

    assert value_bytes[0] & 0x03 == 0
    assert value_bytes[0] >> 2 == 16


def test_object_field_ids_are_emitted_in_lexicographic_field_name_order() -> None:
    metadata_bytes, value_bytes = encode_variant({"c": 3, "a": 1, "b": 2})
    dictionary = _decode_metadata_dictionary(metadata_bytes)

    metadata = value_bytes[0]
    assert metadata & 0x03 == 2
    is_large = bool(metadata & 0x40)
    assert not is_large
    field_id_size = ((metadata >> 4) & 0x03) + 1
    num_elements = value_bytes[1]
    field_ids_start = 2
    field_ids = [
        int.from_bytes(value_bytes[offset : offset + field_id_size], "little")
        for offset in range(field_ids_start, field_ids_start + num_elements * field_id_size, field_id_size)
    ]

    assert [dictionary[field_id] for field_id in field_ids] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "value, expected_value_bytes",
    [
        # null: basic_type=0 (primitive), value_header=0 -> 0x00
        (None, b"\x00"),
        # true: primitive type 1 -> (1 << 2) | 0 = 0x04
        (True, b"\x04"),
        # false: primitive type 2 -> (2 << 2) | 0 = 0x08
        (False, b"\x08"),
        # int8 1: header (3 << 2) | 0 = 0x0c, then 0x01
        (1, b"\x0c\x01"),
        # int8 -1: 0x0c then 0xff (two's complement)
        (-1, b"\x0c\xff"),
        # int16 200: header (4 << 2) = 0x10, then 0xc8 0x00
        (200, b"\x10\xc8\x00"),
        # short string "hi": header (2 << 2) | 1 = 0x09, then bytes
        ("hi", b"\x09hi"),
        # empty short string: header (0 << 2) | 1 = 0x01
        ("", b"\x01"),
        # double 1.0: header (7 << 2) = 0x1c, then IEEE-754 LE
        (1.0, b"\x1c\x00\x00\x00\x00\x00\x00\xf0\x3f"),
        # decimal4 1.50: header (8 << 2) = 0x20, scale=2, unscaled=150 (LE int32)
        (Decimal("1.50"), b"\x20\x02\x96\x00\x00\x00"),
        # date epoch+0 (1970-01-01): header (11 << 2) = 0x2c, then 0 (LE int32)
        (date(1970, 1, 1), b"\x2c\x00\x00\x00\x00"),
        # binary b"ab": header (15 << 2) = 0x3c, length 2 (LE uint32), then bytes
        (b"ab", b"\x3c\x02\x00\x00\x00ab"),
    ],
)
def test_variant_value_bytes_match_spec(value: Any, expected_value_bytes: bytes) -> None:
    _, value_bytes = encode_variant(value)

    assert value_bytes == expected_value_bytes
    assert decode_variant(b"\x11\x00\x00", value_bytes) == value


def test_scalar_metadata_header_is_empty_dictionary() -> None:
    # version=1, sorted_strings=1, offset_size=1 -> 0x11; dict size 0; single offset 0.
    metadata_bytes, _ = encode_variant(42)

    assert metadata_bytes == b"\x11\x00\x00"


def test_metadata_header_offset_size_uses_top_two_bits() -> None:
    # Force a dictionary large enough (>255 bytes) to require a 2-byte offset size.
    value = {f"field_name_{index:03d}": index for index in range(30)}

    metadata_bytes, _ = encode_variant(value)

    header = metadata_bytes[0]
    assert header & 0x0F == 1  # version
    assert (header >> 4) & 0x01 == 1  # sorted_strings
    assert (header >> 5) & 0x01 == 0  # reserved bit must stay 0
    assert ((header >> 6) & 0x03) + 1 == 2  # offset_size lives in bits 6-7
    assert decode_variant(metadata_bytes, encode_variant(value)[1]) == value


def test_large_dictionary_metadata_round_trips() -> None:
    # A dictionary whose byte length exceeds 0xFFFF forces offset_size 3, which sets
    # bit 7 of the metadata header. Decoding must not mistake that for the reserved bit.
    value = {f"field_{index:020d}": index for index in range(4000)}

    metadata_bytes, value_bytes = encode_variant(value)

    assert ((metadata_bytes[0] >> 6) & 0x03) + 1 >= 3
    assert decode_variant(metadata_bytes, value_bytes) == value


def test_decode_float_primitive() -> None:
    # Variant float (primitive id 14) is read for interop, even though encoding emits
    # double for native Python floats. header = (14 << 2) = 0x38, then IEEE-754 LE float.
    value_bytes = bytes([_PRIMITIVE_FLOAT << 2]) + struct.pack("<f", 1.5)

    assert decode_variant(b"\x11\x00\x00", value_bytes) == 1.5


def test_variant_schema_to_arrow_raises_rather_than_mishandling() -> None:
    pytest.importorskip("pyarrow")
    from pyiceberg.io.pyarrow import schema_to_pyarrow
    from pyiceberg.schema import Schema

    schema = Schema(NestedField(1, "payload", VariantType(), required=False))

    # Arrow-native / shredded Variant is intentionally unsupported; it must error, not
    # silently emit a wrong Arrow type.
    with pytest.raises(ValueError, match="Type not recognized: variant"):
        schema_to_pyarrow(schema)


def test_variant_type() -> None:
    field = NestedField(1, "payload", VariantType(), required=False)

    assert str(VariantType()) == "variant"
    assert VariantType().minimum_format_version() == 3
    assert VariantType().model_dump_json() == '"variant"'
    assert VariantType.model_validate_json('"variant"') == VariantType()
    assert IcebergType.model_validate("variant") == VariantType()
    assert NestedField.model_validate_json(field.model_dump_json()) == field


def _decode_metadata_dictionary(metadata_bytes: bytes) -> list[str]:
    header = metadata_bytes[0]
    offset_size = ((header >> 6) & 0x03) + 1
    offset = 1
    dictionary_size = int.from_bytes(metadata_bytes[offset : offset + offset_size], "little")
    offset += offset_size

    offsets = []
    for _ in range(dictionary_size + 1):
        offsets.append(int.from_bytes(metadata_bytes[offset : offset + offset_size], "little"))
        offset += offset_size

    dictionary_bytes = metadata_bytes[offset:]
    return [dictionary_bytes[start:end].decode("utf-8") for start, end in zip(offsets, offsets[1:], strict=False)]
