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

"""Pure-python encoder and decoder for the Parquet Variant binary format.

This implements only the non-shredded (metadata + value) Variant encoding. Shredded
Variant and Arrow-native interop are not supported here; reading/writing Variant columns
through Arrow is blocked on Arrow issues #45937, #50131, and #50132. ``VariantType`` is
intentionally not registered with the Arrow schema visitors, so converting a schema that
contains a Variant to Arrow raises rather than silently producing wrong data.
"""

from __future__ import annotations

import struct
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

_METADATA_VERSION = 1

_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DATETIME_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

_BASIC_TYPE_PRIMITIVE = 0
_BASIC_TYPE_SHORT_STRING = 1
_BASIC_TYPE_OBJECT = 2
_BASIC_TYPE_ARRAY = 3

_PRIMITIVE_NULL = 0
_PRIMITIVE_TRUE = 1
_PRIMITIVE_FALSE = 2
_PRIMITIVE_INT8 = 3
_PRIMITIVE_INT16 = 4
_PRIMITIVE_INT32 = 5
_PRIMITIVE_INT64 = 6
_PRIMITIVE_DOUBLE = 7
_PRIMITIVE_DECIMAL4 = 8
_PRIMITIVE_DECIMAL8 = 9
_PRIMITIVE_DECIMAL16 = 10
_PRIMITIVE_DATE = 11
_PRIMITIVE_TIMESTAMP_TZ = 12
_PRIMITIVE_TIMESTAMP_NTZ = 13
_PRIMITIVE_FLOAT = 14
_PRIMITIVE_BINARY = 15
_PRIMITIVE_STRING = 16

_MAX_DECIMAL_SCALE = 38
_MAX_INT128 = (1 << 127) - 1
_MIN_INT128 = -(1 << 127)

_MAX_UINT32 = (1 << 32) - 1
_MIN_INT8 = -(1 << 7)
_MAX_INT8 = (1 << 7) - 1
_MIN_INT16 = -(1 << 15)
_MAX_INT16 = (1 << 15) - 1
_MIN_INT32 = -(1 << 31)
_MAX_INT32 = (1 << 31) - 1
_MIN_INT64 = -(1 << 63)
_MAX_INT64 = (1 << 63) - 1


def encode_variant(py_value: Any) -> tuple[bytes, bytes]:
    """Encode a Python value as an Iceberg/Parquet Variant binary pair.

    Args:
        py_value: A variant value represented with native Python types.

    Returns:
        A pair of metadata bytes and value bytes.
    """
    field_names: set[str] = set()
    _collect_field_names(py_value, field_names)
    dictionary_strings = sorted(field_names)
    dictionary = {name: index for index, name in enumerate(dictionary_strings)}

    return _encode_metadata(dictionary_strings), _encode_value(py_value, dictionary)


def decode_variant(metadata_bytes: bytes, value_bytes: bytes) -> Any:
    """Decode an Iceberg/Parquet Variant binary pair into native Python values."""
    dictionary = _decode_metadata(metadata_bytes)
    value, offset = _decode_value(value_bytes, 0, dictionary)
    if offset != len(value_bytes):
        raise ValueError("Variant value contains trailing bytes")
    return value


def _collect_field_names(value: Any, field_names: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Variant object field names must be strings: {key!r}")
            field_names.add(key)
            _collect_field_names(child, field_names)
    elif isinstance(value, list):
        for child in value:
            _collect_field_names(child, field_names)
    elif value is None or isinstance(value, (bool, int, float, str, bytes, Decimal, date, datetime)):
        return
    else:
        raise ValueError(f"Unsupported variant value: {value!r}")


def _encode_metadata(dictionary_strings: list[str]) -> bytes:
    encoded_strings = [value.encode("utf-8") for value in dictionary_strings]
    dictionary_bytes = b"".join(encoded_strings)
    offset_size = _unsigned_width(max(len(dictionary_strings), len(dictionary_bytes)))
    header = _METADATA_VERSION | 0x10 | ((offset_size - 1) << 6)

    offsets = [0]
    offset = 0
    for value in encoded_strings:
        offset += len(value)
        offsets.append(offset)

    return b"".join(
        [
            bytes([header]),
            _write_unsigned(len(dictionary_strings), offset_size),
            *(_write_unsigned(offset, offset_size) for offset in offsets),
            dictionary_bytes,
        ]
    )


def _decode_metadata(metadata_bytes: bytes) -> list[str]:
    if not metadata_bytes:
        raise ValueError("Variant metadata is empty")

    header = metadata_bytes[0]
    if header & 0x20:
        raise ValueError("Variant metadata reserved bit is set")
    version = header & 0x0F
    if version != _METADATA_VERSION:
        raise ValueError(f"Unsupported variant metadata version: {version}")

    offset_size = ((header >> 6) & 0x03) + 1
    offset = 1
    dictionary_size, offset = _read_unsigned(metadata_bytes, offset, offset_size)

    offsets = []
    for _ in range(dictionary_size + 1):
        value, offset = _read_unsigned(metadata_bytes, offset, offset_size)
        offsets.append(value)

    if not offsets or offsets[0] != 0:
        raise ValueError("Variant metadata dictionary offsets must start with zero")
    if any(left > right for left, right in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("Variant metadata dictionary offsets must be ordered")

    dictionary_bytes = metadata_bytes[offset:]
    if offsets[-1] != len(dictionary_bytes):
        raise ValueError("Variant metadata dictionary length does not match offsets")

    return [dictionary_bytes[start:end].decode("utf-8") for start, end in zip(offsets, offsets[1:], strict=False)]


def _encode_value(value: Any, dictionary: dict[str, int]) -> bytes:
    if value is None:
        return _primitive_header(_PRIMITIVE_NULL)
    if value is True:
        return _primitive_header(_PRIMITIVE_TRUE)
    if value is False:
        return _primitive_header(_PRIMITIVE_FALSE)
    if isinstance(value, int) and not isinstance(value, bool):
        return _encode_int(value)
    if isinstance(value, float):
        return _primitive_header(_PRIMITIVE_DOUBLE) + struct.pack("<d", value)
    if isinstance(value, Decimal):
        return _encode_decimal(value)
    if isinstance(value, datetime):
        return _encode_timestamp(value)
    if isinstance(value, date):
        return _primitive_header(_PRIMITIVE_DATE) + (value - _EPOCH_DATE).days.to_bytes(4, "little", signed=True)
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, bytes):
        return _encode_binary(value)
    if isinstance(value, dict):
        return _encode_object(value, dictionary)
    if isinstance(value, list):
        return _encode_array(value, dictionary)
    raise ValueError(f"Unsupported variant value: {value!r}")


def _primitive_header(primitive_type: int) -> bytes:
    return bytes([(primitive_type << 2) | _BASIC_TYPE_PRIMITIVE])


def _encode_int(value: int) -> bytes:
    if _MIN_INT8 <= value <= _MAX_INT8:
        return _primitive_header(_PRIMITIVE_INT8) + value.to_bytes(1, "little", signed=True)
    if _MIN_INT16 <= value <= _MAX_INT16:
        return _primitive_header(_PRIMITIVE_INT16) + value.to_bytes(2, "little", signed=True)
    if _MIN_INT32 <= value <= _MAX_INT32:
        return _primitive_header(_PRIMITIVE_INT32) + value.to_bytes(4, "little", signed=True)
    if _MIN_INT64 <= value <= _MAX_INT64:
        return _primitive_header(_PRIMITIVE_INT64) + value.to_bytes(8, "little", signed=True)
    raise ValueError(f"Variant integer out of int64 range: {value}")


def _encode_string(value: str) -> bytes:
    value_bytes = value.encode("utf-8")
    if len(value_bytes) < 64:
        return bytes([(len(value_bytes) << 2) | _BASIC_TYPE_SHORT_STRING]) + value_bytes
    if len(value_bytes) > _MAX_UINT32:
        raise ValueError("Variant string is too long")
    return _primitive_header(_PRIMITIVE_STRING) + _write_unsigned(len(value_bytes), 4) + value_bytes


def _encode_binary(value: bytes) -> bytes:
    if len(value) > _MAX_UINT32:
        raise ValueError("Variant binary value is too long")
    return _primitive_header(_PRIMITIVE_BINARY) + _write_unsigned(len(value), 4) + value


def _encode_decimal(value: Decimal) -> bytes:
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"Variant decimal cannot encode non-finite value: {value}")
    scale = -exponent
    if scale < 0 or scale > _MAX_DECIMAL_SCALE:
        raise ValueError(f"Variant decimal scale must be in [0, 38]: {scale}")
    unscaled = int("".join(str(digit) for digit in digits) or "0")
    if sign:
        unscaled = -unscaled

    if _MIN_INT32 <= unscaled <= _MAX_INT32:
        primitive_type, width = _PRIMITIVE_DECIMAL4, 4
    elif _MIN_INT64 <= unscaled <= _MAX_INT64:
        primitive_type, width = _PRIMITIVE_DECIMAL8, 8
    elif _MIN_INT128 <= unscaled <= _MAX_INT128:
        primitive_type, width = _PRIMITIVE_DECIMAL16, 16
    else:
        raise ValueError(f"Variant decimal unscaled value out of int128 range: {unscaled}")

    return _primitive_header(primitive_type) + bytes([scale]) + unscaled.to_bytes(width, "little", signed=True)


def _encode_timestamp(value: datetime) -> bytes:
    if value.tzinfo is not None:
        micros = round((value - _EPOCH_DATETIME_UTC).total_seconds() * 1_000_000)
        primitive_type = _PRIMITIVE_TIMESTAMP_TZ
    else:
        micros = round((value - _EPOCH_DATETIME_UTC.replace(tzinfo=None)).total_seconds() * 1_000_000)
        primitive_type = _PRIMITIVE_TIMESTAMP_NTZ
    return _primitive_header(primitive_type) + micros.to_bytes(8, "little", signed=True)


def _encode_object(value: dict[Any, Any], dictionary: dict[str, int]) -> bytes:
    items = []
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Variant object field names must be strings: {key!r}")
        items.append((key, child))
    items.sort(key=lambda item: item[0])

    encoded_values = [_encode_value(child, dictionary) for _, child in items]
    value_region = b"".join(encoded_values)
    field_offsets = _offsets(encoded_values)
    field_ids = [dictionary[key] for key, _ in items]

    field_offset_size = _unsigned_width(len(value_region))
    field_id_size = _unsigned_width(max(field_ids, default=0))
    is_large = len(items) > 255
    header = _BASIC_TYPE_OBJECT | ((field_offset_size - 1) << 2) | ((field_id_size - 1) << 4) | (0x40 if is_large else 0)

    return b"".join(
        [
            bytes([header]),
            _write_unsigned(len(items), 4 if is_large else 1),
            *(_write_unsigned(field_id, field_id_size) for field_id in field_ids),
            *(_write_unsigned(offset, field_offset_size) for offset in field_offsets),
            value_region,
        ]
    )


def _encode_array(value: list[Any], dictionary: dict[str, int]) -> bytes:
    encoded_values = [_encode_value(child, dictionary) for child in value]
    value_region = b"".join(encoded_values)
    field_offsets = _offsets(encoded_values)

    field_offset_size = _unsigned_width(len(value_region))
    is_large = len(value) > 255
    header = _BASIC_TYPE_ARRAY | ((field_offset_size - 1) << 2) | (0x10 if is_large else 0)

    return b"".join(
        [
            bytes([header]),
            _write_unsigned(len(value), 4 if is_large else 1),
            *(_write_unsigned(offset, field_offset_size) for offset in field_offsets),
            value_region,
        ]
    )


def _offsets(encoded_values: list[bytes]) -> list[int]:
    offsets = [0]
    offset = 0
    for value in encoded_values:
        offset += len(value)
        offsets.append(offset)
    return offsets


def _decode_value(value_bytes: bytes, offset: int, dictionary: list[str]) -> tuple[Any, int]:
    if offset >= len(value_bytes):
        raise ValueError("Unexpected end of variant value")

    metadata = value_bytes[offset]
    offset += 1
    basic_type = metadata & 0x03
    value_header = metadata >> 2

    if basic_type == _BASIC_TYPE_PRIMITIVE:
        return _decode_primitive(value_header, value_bytes, offset)
    if basic_type == _BASIC_TYPE_SHORT_STRING:
        return _read_utf8(value_bytes, offset, value_header)
    if basic_type == _BASIC_TYPE_OBJECT:
        return _decode_object(metadata, value_bytes, offset, dictionary)
    if basic_type == _BASIC_TYPE_ARRAY:
        return _decode_array(metadata, value_bytes, offset, dictionary)

    raise ValueError(f"Unsupported variant basic type: {basic_type}")


def _decode_primitive(primitive_type: int, value_bytes: bytes, offset: int) -> tuple[Any, int]:
    if primitive_type == _PRIMITIVE_NULL:
        return None, offset
    if primitive_type == _PRIMITIVE_TRUE:
        return True, offset
    if primitive_type == _PRIMITIVE_FALSE:
        return False, offset
    if primitive_type == _PRIMITIVE_INT8:
        return _read_signed(value_bytes, offset, 1)
    if primitive_type == _PRIMITIVE_INT16:
        return _read_signed(value_bytes, offset, 2)
    if primitive_type == _PRIMITIVE_INT32:
        return _read_signed(value_bytes, offset, 4)
    if primitive_type == _PRIMITIVE_INT64:
        return _read_signed(value_bytes, offset, 8)
    if primitive_type == _PRIMITIVE_DOUBLE:
        _require_available(value_bytes, offset, 8)
        return struct.unpack_from("<d", value_bytes, offset)[0], offset + 8
    if primitive_type == _PRIMITIVE_FLOAT:
        _require_available(value_bytes, offset, 4)
        return struct.unpack_from("<f", value_bytes, offset)[0], offset + 4
    if primitive_type == _PRIMITIVE_DECIMAL4:
        return _read_decimal(value_bytes, offset, 4)
    if primitive_type == _PRIMITIVE_DECIMAL8:
        return _read_decimal(value_bytes, offset, 8)
    if primitive_type == _PRIMITIVE_DECIMAL16:
        return _read_decimal(value_bytes, offset, 16)
    if primitive_type == _PRIMITIVE_DATE:
        days, offset = _read_signed(value_bytes, offset, 4)
        return _EPOCH_DATE + timedelta(days=days), offset
    if primitive_type == _PRIMITIVE_TIMESTAMP_TZ:
        micros, offset = _read_signed(value_bytes, offset, 8)
        return _EPOCH_DATETIME_UTC + timedelta(microseconds=micros), offset
    if primitive_type == _PRIMITIVE_TIMESTAMP_NTZ:
        micros, offset = _read_signed(value_bytes, offset, 8)
        return _EPOCH_DATETIME_UTC.replace(tzinfo=None) + timedelta(microseconds=micros), offset
    if primitive_type == _PRIMITIVE_BINARY:
        length, offset = _read_unsigned(value_bytes, offset, 4)
        _require_available(value_bytes, offset, length)
        return value_bytes[offset : offset + length], offset + length
    if primitive_type == _PRIMITIVE_STRING:
        length, offset = _read_unsigned(value_bytes, offset, 4)
        return _read_utf8(value_bytes, offset, length)
    raise ValueError(f"Unsupported variant primitive type: {primitive_type}")


def _read_decimal(value_bytes: bytes, offset: int, width: int) -> tuple[Decimal, int]:
    _require_available(value_bytes, offset, 1)
    scale = value_bytes[offset]
    offset += 1
    if scale > _MAX_DECIMAL_SCALE:
        raise ValueError(f"Variant decimal scale must be in [0, 38]: {scale}")
    unscaled, offset = _read_signed(value_bytes, offset, width)
    sign = 1 if unscaled < 0 else 0
    digits = tuple(int(digit) for digit in str(abs(unscaled)))
    return Decimal((sign, digits, -scale)), offset


def _decode_object(metadata: int, value_bytes: bytes, offset: int, dictionary: list[str]) -> tuple[dict[str, Any], int]:
    if metadata & 0x80:
        raise ValueError("Variant object reserved bit is set")

    field_offset_size = ((metadata >> 2) & 0x03) + 1
    field_id_size = ((metadata >> 4) & 0x03) + 1
    is_large = bool(metadata & 0x40)

    num_elements, offset = _read_unsigned(value_bytes, offset, 4 if is_large else 1)

    field_ids = []
    for _ in range(num_elements):
        field_id, offset = _read_unsigned(value_bytes, offset, field_id_size)
        if field_id >= len(dictionary):
            raise ValueError(f"Variant object field id out of range: {field_id}")
        field_ids.append(field_id)

    field_offsets, offset = _read_offsets(value_bytes, offset, field_offset_size, num_elements)
    value_region_start = offset
    value_region_end = value_region_start + field_offsets[-1]
    _require_available(value_bytes, value_region_start, field_offsets[-1])

    result = {}
    for index, field_id in enumerate(field_ids):
        start = value_region_start + field_offsets[index]
        expected_end = value_region_start + field_offsets[index + 1]
        field_value, actual_end = _decode_value(value_bytes[:value_region_end], start, dictionary)
        if actual_end != expected_end:
            raise ValueError("Variant object field value length does not match offset")
        result[dictionary[field_id]] = field_value

    return result, value_region_end


def _decode_array(metadata: int, value_bytes: bytes, offset: int, dictionary: list[str]) -> tuple[list[Any], int]:
    if metadata & 0xE0:
        raise ValueError("Variant array reserved bits are set")

    field_offset_size = ((metadata >> 2) & 0x03) + 1
    is_large = bool(metadata & 0x10)

    num_elements, offset = _read_unsigned(value_bytes, offset, 4 if is_large else 1)
    field_offsets, offset = _read_offsets(value_bytes, offset, field_offset_size, num_elements)

    value_region_start = offset
    value_region_end = value_region_start + field_offsets[-1]
    _require_available(value_bytes, value_region_start, field_offsets[-1])

    result = []
    for index in range(num_elements):
        start = value_region_start + field_offsets[index]
        expected_end = value_region_start + field_offsets[index + 1]
        item, actual_end = _decode_value(value_bytes[:value_region_end], start, dictionary)
        if actual_end != expected_end:
            raise ValueError("Variant array element value length does not match offset")
        result.append(item)

    return result, value_region_end


def _read_offsets(value_bytes: bytes, offset: int, offset_size: int, num_elements: int) -> tuple[list[int], int]:
    offsets = []
    for _ in range(num_elements + 1):
        value, offset = _read_unsigned(value_bytes, offset, offset_size)
        offsets.append(value)

    if not offsets or offsets[0] != 0:
        raise ValueError("Variant offsets must start with zero")
    if any(left > right for left, right in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("Variant offsets must be ordered")
    return offsets, offset


def _read_utf8(value_bytes: bytes, offset: int, length: int) -> tuple[str, int]:
    _require_available(value_bytes, offset, length)
    return value_bytes[offset : offset + length].decode("utf-8"), offset + length


def _read_signed(value_bytes: bytes, offset: int, width: int) -> tuple[int, int]:
    _require_available(value_bytes, offset, width)
    return int.from_bytes(value_bytes[offset : offset + width], "little", signed=True), offset + width


def _read_unsigned(value_bytes: bytes, offset: int, width: int) -> tuple[int, int]:
    _require_available(value_bytes, offset, width)
    return int.from_bytes(value_bytes[offset : offset + width], "little", signed=False), offset + width


def _write_unsigned(value: int, width: int) -> bytes:
    if value < 0 or value > (1 << (width * 8)) - 1:
        raise ValueError(f"Value {value} does not fit in {width} bytes")
    return value.to_bytes(width, "little", signed=False)


def _require_available(value_bytes: bytes, offset: int, length: int) -> None:
    if offset < 0 or length < 0 or offset + length > len(value_bytes):
        raise ValueError("Unexpected end of variant value")


def _unsigned_width(value: int) -> int:
    if value < 0:
        raise ValueError(f"Negative values cannot be encoded as unsigned offsets: {value}")
    if value <= 0xFF:
        return 1
    if value <= 0xFFFF:
        return 2
    if value <= 0xFFFFFF:
        return 3
    if value <= _MAX_UINT32:
        return 4
    raise ValueError(f"Value is too large for a four-byte unsigned field: {value}")
