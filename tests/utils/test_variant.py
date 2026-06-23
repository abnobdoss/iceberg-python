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

from typing import Any

import pytest

from pyiceberg.types import IcebergType, NestedField, VariantType
from pyiceberg.utils.variant import decode_variant, encode_variant


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
        [1, "two", None, {"k": 3.0}],
    ],
)
def test_variant_round_trip(value: Any) -> None:
    metadata_bytes, value_bytes = encode_variant(value)

    assert metadata_bytes[0] & 0x0F == 1
    assert decode_variant(metadata_bytes, value_bytes) == value


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
    offset_size = ((header >> 5) & 0x03) + 1
    offset = 1
    dictionary_size = int.from_bytes(metadata_bytes[offset : offset + offset_size], "little")
    offset += offset_size

    offsets = []
    for _ in range(dictionary_size + 1):
        offsets.append(int.from_bytes(metadata_bytes[offset : offset + offset_size], "little"))
        offset += offset_size

    dictionary_bytes = metadata_bytes[offset:]
    return [dictionary_bytes[start:end].decode("utf-8") for start, end in zip(offsets, offsets[1:])]
