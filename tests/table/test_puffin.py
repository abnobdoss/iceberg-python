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
import json
import random
import zlib
from os import path
from pathlib import Path

import pytest
from pyroaring import BitMap

from pyiceberg import __version__
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.table.puffin import (
    DELETION_VECTOR_MAGIC,
    MAGIC_BYTES,
    PROPERTY_REFERENCED_DATA_FILE,
    PuffinFile,
    PuffinWriter,
    _deserialize_bitmap,
)


def _open_file(file: str) -> bytes:
    cur_dir = path.dirname(path.realpath(__file__))
    with open(f"{cur_dir}/bitmaps/{file}", "rb") as f:
        return f.read()


def test_map_empty() -> None:
    puffin = _open_file("64mapempty.bin")

    expected: list[BitMap] = []
    actual = _deserialize_bitmap(puffin)

    assert expected == actual


def test_map_bitvals() -> None:
    puffin = _open_file("64map32bitvals.bin")

    expected = [BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    actual = _deserialize_bitmap(puffin)

    assert expected == actual


def test_map_spread_vals() -> None:
    puffin = _open_file("64mapspreadvals.bin")

    expected = [
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        BitMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ]
    actual = _deserialize_bitmap(puffin)

    assert expected == actual


def test_map_high_vals() -> None:
    puffin = _open_file("64maphighvals.bin")

    with pytest.raises(ValueError, match="Key 4022190063 is too large, max 2147483647 to maintain compatibility with Java impl"):
        _ = _deserialize_bitmap(puffin)


def _new_writer(tmp_path: Path, created_by: str | None = None) -> tuple[PuffinWriter, Path]:
    puffin_path = tmp_path / "test.puffin"
    return PuffinWriter(PyArrowFileIO().new_output(str(puffin_path)), created_by=created_by), puffin_path


def test_puffin_round_trip(tmp_path: Path) -> None:
    # Define some deletion positions for a file
    deletions = [5, (1 << 32) + 1, 5]  # Test with a high-bit position and duplicate

    file_path = "path/to/data.parquet"

    # Write the Puffin file
    writer, puffin_path = _new_writer(tmp_path, created_by="my-test-app")
    writer.set_blob(positions=deletions, referenced_data_file=file_path)
    size = writer.finish()

    # Read the Puffin file back
    puffin_bytes = puffin_path.read_bytes()
    assert size == len(puffin_bytes)
    reader = PuffinFile(puffin_bytes)

    # Assert footer metadata
    assert reader.footer.properties["created-by"] == "my-test-app"
    assert len(reader.footer.blobs) == 1

    blob_meta = reader.footer.blobs[0]
    assert blob_meta.properties[PROPERTY_REFERENCED_DATA_FILE] == file_path
    assert blob_meta.properties["cardinality"] == str(len(set(deletions)))

    # Assert the content of deletion vectors
    read_vectors = reader.to_vector()

    assert file_path in read_vectors
    assert read_vectors[file_path].to_pylist() == sorted(set(deletions))


def test_puffin_round_trip_with_sparse_bitmap_keys(tmp_path: Path) -> None:
    # High bits 0 and 2 are present while 1 is absent; the writer must emit sorted keys
    # and the reader pads the missing key with an empty bitmap.
    positions = [3, (2 << 32) + 4]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()

    vectors = PuffinFile(puffin_path.read_bytes()).to_vector()
    assert vectors["file.parquet"].to_pylist() == positions


def test_dv_roundtrip_empty_high_key(tmp_path: Path) -> None:
    positions = [(3 << 32) + 0, (3 << 32) + 9]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()

    vectors = PuffinFile(puffin_path.read_bytes()).to_vector()
    assert vectors["file.parquet"].to_pylist() == positions


def test_dv_roundtrip_large_sparse(tmp_path: Path) -> None:
    random_generator = random.Random(7321)
    positions: list[int] = [
        (1 << 32) - 1,
        1 << 32,
        (2 << 32) - 1,
        2 << 32,
        (7 << 32) + 123,
    ]

    for key in (0, 1, 3, 7):
        base = key << 32
        positions.extend(base + low for low in range(8_000, 14_000))
        positions.extend(base + random_generator.randrange(0, 0xFFFFFFFF) for _ in range(6_500))

    expected = sorted(set(positions))
    assert len(expected) > 50_000

    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()

    vectors = PuffinFile(puffin_path.read_bytes()).to_vector()
    assert vectors["file.parquet"].to_pylist() == expected


def test_dv_roundtrip_boundary_positions(tmp_path: Path) -> None:
    positions = [0, 1, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, (2 << 32) + 5]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()

    vectors = PuffinFile(puffin_path.read_bytes()).to_vector()
    assert vectors["file.parquet"].to_pylist() == positions


def test_dv_blob_crc_independently_verified(tmp_path: Path) -> None:
    positions = [0, 7, (1 << 32) - 1, 1 << 32, (4 << 32) + 11]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()
    puffin_bytes = puffin_path.read_bytes()

    footer_payload_size = int.from_bytes(puffin_bytes[-12:-8], "little")
    footer_bytes = puffin_bytes[-(footer_payload_size + 12) : -12]
    footer = json.loads(footer_bytes)
    blob = footer["blobs"][0]
    blob_bytes = puffin_bytes[blob["offset"] : blob["offset"] + blob["length"]]

    length_prefix = int.from_bytes(blob_bytes[0:4], "big")
    magic_and_vector = blob_bytes[4 : 4 + length_prefix]
    stored_crc = int.from_bytes(blob_bytes[4 + length_prefix : 8 + length_prefix], "big")

    assert magic_and_vector[: len(DELETION_VECTOR_MAGIC)] == DELETION_VECTOR_MAGIC
    assert length_prefix == len(DELETION_VECTOR_MAGIC) + len(magic_and_vector[len(DELETION_VECTOR_MAGIC) :])
    assert blob["length"] == 4 + length_prefix + 4
    assert stored_crc == zlib.crc32(magic_and_vector)


def test_dv_vector_body_is_portable_croaring(tmp_path: Path) -> None:
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, (1 << 32) + 100, (1 << 32) + 101]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()
    puffin_bytes = puffin_path.read_bytes()

    footer_payload_size = int.from_bytes(puffin_bytes[-12:-8], "little")
    footer_bytes = puffin_bytes[-(footer_payload_size + 12) : -12]
    footer = json.loads(footer_bytes)
    blob = footer["blobs"][0]
    blob_bytes = puffin_bytes[blob["offset"] : blob["offset"] + blob["length"]]

    length_prefix = int.from_bytes(blob_bytes[0:4], "big")
    assert blob_bytes[4:8] == DELETION_VECTOR_MAGIC
    vector_body = blob_bytes[8 : 4 + length_prefix]

    cursor = 0
    count = int.from_bytes(vector_body[cursor : cursor + 8], "little")
    cursor += 8
    assert count == 2

    bitmaps_by_key: dict[int, BitMap] = {}
    keys = []
    for index in range(count):
        key = int.from_bytes(vector_body[cursor : cursor + 4], "little")
        keys.append(key)
        cursor += 4

        bitmap_bytes = vector_body[cursor:]
        if index == 0:
            assert bitmap_bytes[:2] in (b"\x3a\x30", b"\x3b\x30")

        bitmap = BitMap.deserialize(bitmap_bytes)
        bitmaps_by_key[key] = bitmap
        cursor += len(bitmap.serialize())

    assert keys == [0, 1]
    assert keys == sorted(keys)
    assert set(bitmaps_by_key[0]) == set(range(10))
    assert set(bitmaps_by_key[1]) == {100, 101}
    assert cursor == len(vector_body)


def test_dv_vector_body_byte_exact_portable_format(tmp_path: Path) -> None:
    # Byte-exactness against the RoaringBitmap "portable" format the Iceberg/Puffin spec
    # mandates, independent of pyroaring's own reader. The expected bytes are constructed
    # by hand from the RoaringFormatSpec so this would fail if the writer ever emitted the
    # non-portable (native CRoaring frame-of-reference) layout instead.
    import struct

    # A single key (0) whose sub-positions form an array container (no runs): {1, 5, 100, 7000}.
    positions = [1, 5, 100, 7000]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()
    puffin_bytes = puffin_path.read_bytes()

    footer_payload_size = int.from_bytes(puffin_bytes[-12:-8], "little")
    footer = json.loads(puffin_bytes[-(footer_payload_size + 12) : -12])
    blob = footer["blobs"][0]
    blob_bytes = puffin_bytes[blob["offset"] : blob["offset"] + blob["length"]]

    length_prefix = int.from_bytes(blob_bytes[0:4], "big")
    assert blob_bytes[4:8] == DELETION_VECTOR_MAGIC
    vector_body = blob_bytes[8 : 4 + length_prefix]

    # Portable 64-bit layout: 8-byte LE bitmap count, then 4-byte LE key, then the 32-bit
    # portable RoaringBitmap. The 32-bit bitmap uses the SERIAL_COOKIE_NO_RUNCONTAINER
    # (0x303a) array-container layout: cookie, container count, descriptive header
    # (key, cardinality-1), the offset header, then explicit 16-bit values.
    expected = struct.pack("<Q", 1)  # one 32-bit bitmap
    expected += struct.pack("<I", 0)  # high key 0
    expected += struct.pack("<I", 0x0000303A)  # no-run cookie
    expected += struct.pack("<I", 1)  # one container
    expected += struct.pack("<HH", 0, len(positions) - 1)  # container key 0, cardinality-1
    expected += struct.pack("<I", 4 + 4 + 4 + 4)  # offset of container data after the header
    expected += struct.pack(f"<{len(positions)}H", *positions)  # array container values

    assert vector_body == expected


def test_dv_duplicate_positions_deduped(tmp_path: Path) -> None:
    positions = [(1 << 32) + 3, 0, (1 << 32) + 3, 7, 0, (1 << 32), 7, 3]
    expected = sorted(set(positions))
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()

    reader = PuffinFile(puffin_path.read_bytes())
    blob = reader.footer.blobs[0]
    assert blob.properties["cardinality"] == str(len(expected))
    assert reader.to_vector()["file.parquet"].to_pylist() == expected


def test_write_and_read_puffin_file(tmp_path: Path) -> None:
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=[1, 2, 3], referenced_data_file="file1.parquet")
    writer.set_blob(positions=[4, 5, 6], referenced_data_file="file2.parquet")
    writer.finish()

    reader = PuffinFile(puffin_path.read_bytes())

    assert len(reader.footer.blobs) == 1
    blob = reader.footer.blobs[0]

    assert blob.properties["referenced-data-file"] == "file2.parquet"
    assert blob.properties["cardinality"] == "3"
    assert blob.type == "deletion-vector-v1"
    # Reserved field id of the row position column (Java MetadataColumns.ROW_POSITION, INT_MAX - 2);
    # required for Java/Spark interoperability.
    assert blob.fields == [2147483645]
    assert blob.snapshot_id == -1
    assert blob.sequence_number == -1
    assert blob.compression_codec is None

    vectors = reader.to_vector()
    assert len(vectors) == 1
    assert "file1.parquet" not in vectors
    assert vectors["file2.parquet"].to_pylist() == [4, 5, 6]


def test_deletion_vector_blob_framing_is_spec_compliant(tmp_path: Path) -> None:
    # PuffinFile reads only the serialized vector, skipping the blob's length prefix,
    # deletion-vector magic and CRC-32. Assert that framing directly at the byte level so
    # the bytes an external reader (Java/Spark) relies on stay spec-compliant.
    positions = [0, 1, 5, (1 << 32) + 7]
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=positions, referenced_data_file="file.parquet")
    writer.finish()
    puffin_bytes = puffin_path.read_bytes()

    # The Puffin file begins with the magic.
    assert puffin_bytes[:4] == MAGIC_BYTES

    blob = PuffinFile(puffin_bytes).footer.blobs[0]
    blob_bytes = puffin_bytes[blob.offset : blob.offset + blob.length]

    # Layout: length (4B big-endian) | DV magic (4B) | vector | CRC-32 (4B big-endian),
    # where the length and CRC-32 both cover the magic bytes plus the vector.
    length_prefix = int.from_bytes(blob_bytes[0:4], "big")
    dv_magic = blob_bytes[4:8]
    vector = blob_bytes[8 : 4 + length_prefix]
    crc = int.from_bytes(blob_bytes[4 + length_prefix : 8 + length_prefix], "big")

    assert dv_magic == DELETION_VECTOR_MAGIC
    assert length_prefix == len(dv_magic) + len(vector)
    assert blob.length == 4 + length_prefix + 4
    assert crc == zlib.crc32(dv_magic + vector)


def test_puffin_file_with_no_blobs(tmp_path: Path) -> None:
    writer, puffin_path = _new_writer(tmp_path)
    writer.finish()

    reader = PuffinFile(puffin_path.read_bytes())
    assert len(reader.footer.blobs) == 0
    assert len(reader.to_vector()) == 0


def test_puffin_writer_default_created_by(tmp_path: Path) -> None:
    writer, puffin_path = _new_writer(tmp_path)
    writer.finish()

    reader = PuffinFile(puffin_path.read_bytes())
    assert reader.footer.properties["created-by"] == f"PyIceberg version {__version__}"


def test_set_blob_rejects_negative_positions(tmp_path: Path) -> None:
    writer, _ = _new_writer(tmp_path)
    with pytest.raises(ValueError, match="Invalid position: -1"):
        writer.set_blob(positions=[1, -1], referenced_data_file="file.parquet")


def test_set_blob_rejects_empty_positions(tmp_path: Path) -> None:
    writer, _ = _new_writer(tmp_path)
    with pytest.raises(ValueError, match="Deletion vector must contain at least one position"):
        writer.set_blob(positions=[], referenced_data_file="file.parquet")


def test_set_blob_rejects_position_exceeding_java_key_range(tmp_path: Path) -> None:
    writer, _ = _new_writer(tmp_path)
    with pytest.raises(ValueError, match="Key 2147483648 is too large, max 2147483647"):
        writer.set_blob(positions=[(2**31) << 32], referenced_data_file="file.parquet")


def test_set_blob_failure_preserves_previous_blob(tmp_path: Path) -> None:
    writer, puffin_path = _new_writer(tmp_path)
    writer.set_blob(positions=[1, 2, 3], referenced_data_file="good.parquet")

    with pytest.raises(ValueError, match="Invalid position: -1"):
        writer.set_blob(positions=[5, -1], referenced_data_file="bad.parquet")

    writer.finish()

    reader = PuffinFile(puffin_path.read_bytes())
    assert len(reader.footer.blobs) == 1
    assert reader.footer.blobs[0].properties[PROPERTY_REFERENCED_DATA_FILE] == "good.parquet"
    vectors = reader.to_vector()
    assert "bad.parquet" not in vectors
    assert vectors["good.parquet"].to_pylist() == [1, 2, 3]
