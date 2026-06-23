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
import io
import math
import zlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pyroaring import BitMap, FrozenBitMap

from pyiceberg import __version__
from pyiceberg.io import OutputFile
from pyiceberg.typedef import IcebergBaseModel

if TYPE_CHECKING:
    import pyarrow as pa

# Short for: Puffin Fratercula arctica, version 1
MAGIC_BYTES = b"PFA1"
DELETION_VECTOR_MAGIC = b"\xd1\xd3\x39\x64"
EMPTY_BITMAP = FrozenBitMap()
MAX_JAVA_SIGNED = int(math.pow(2, 31)) - 1
PROPERTY_REFERENCED_DATA_FILE = "referenced-data-file"
# Reserved field id of the row position (_pos) metadata column, referenced by
# deletion-vector-v1 blob metadata (Java: MetadataColumns.ROW_POSITION)
ROW_POSITION_FIELD_ID = 2147483645


def _deserialize_bitmap(pl: bytes) -> list[BitMap]:
    number_of_bitmaps = int.from_bytes(pl[0:8], byteorder="little")
    pl = pl[8:]

    bitmaps = []
    last_key = -1
    for _ in range(number_of_bitmaps):
        key = int.from_bytes(pl[0:4], byteorder="little")
        if key < 0:
            raise ValueError(f"Invalid unsigned key: {key}")
        if key <= last_key:
            raise ValueError("Keys must be sorted in ascending order")
        if key > MAX_JAVA_SIGNED:
            raise ValueError(f"Key {key} is too large, max {MAX_JAVA_SIGNED} to maintain compatibility with Java impl")
        pl = pl[4:]

        while last_key < key - 1:
            bitmaps.append(EMPTY_BITMAP)
            last_key += 1

        bm = BitMap().deserialize(pl)
        # TODO: Optimize this
        pl = pl[len(bm.serialize()) :]
        bitmaps.append(bm)

        last_key = key

    return bitmaps


def _serialize_bitmaps(bitmaps: dict[int, BitMap]) -> bytes:
    """
    Serialize a dictionary of bitmaps into a byte array.

    The format is:
    - 8 bytes: number of bitmaps (little-endian)
    - For each bitmap:
        - 4 bytes: key (little-endian)
        - n bytes: serialized bitmap
    """
    with io.BytesIO() as out:
        sorted_keys = sorted(bitmaps.keys())

        # number of bitmaps
        out.write(len(sorted_keys).to_bytes(8, "little"))

        for key in sorted_keys:
            if key < 0:
                raise ValueError(f"Invalid unsigned key: {key}")
            if key > MAX_JAVA_SIGNED:
                raise ValueError(f"Key {key} is too large, max {MAX_JAVA_SIGNED} to maintain compatibility with Java impl")

            # key
            out.write(key.to_bytes(4, "little"))
            # bitmap
            out.write(bitmaps[key].serialize())
        return out.getvalue()


class PuffinBlobMetadata(IcebergBaseModel):
    type: Literal["deletion-vector-v1"] = Field()
    fields: list[int] = Field()
    snapshot_id: int = Field(alias="snapshot-id")
    sequence_number: int = Field(alias="sequence-number")
    offset: int = Field()
    length: int = Field()
    compression_codec: str | None = Field(alias="compression-codec", default=None)
    properties: dict[str, str] = Field(default_factory=dict)


class Footer(IcebergBaseModel):
    blobs: list[PuffinBlobMetadata] = Field()
    properties: dict[str, str] = Field(default_factory=dict)


def _bitmaps_to_chunked_array(bitmaps: list[BitMap]) -> "pa.ChunkedArray":
    import pyarrow as pa

    return pa.chunked_array(
        ([(key_pos << 32) + pos for pos in bitmap] for key_pos, bitmap in enumerate(bitmaps)),
        type=pa.int64(),
    )


class PuffinFile:
    footer: Footer
    _deletion_vectors: dict[str, list[BitMap]]

    def __init__(self, puffin: bytes) -> None:
        for magic_bytes in [puffin[:4], puffin[-4:]]:
            if magic_bytes != MAGIC_BYTES:
                raise ValueError(f"Incorrect magic bytes, expected {MAGIC_BYTES!r}, got {magic_bytes!r}")

        # One flag is set, the rest should be zero
        # byte 0 (first)
        # - bit 0 (lowest bit): whether FooterPayload is compressed
        # - all other bits are reserved for future use and should be set to 0 on write
        flags = puffin[-8:-4]
        if flags[0] != 0:
            raise ValueError("The Puffin-file has a compressed footer, which is not yet supported")

        # 4 byte integer is always signed, in a two's complement representation, stored little-endian.
        footer_payload_size_int = int.from_bytes(puffin[-12:-8], byteorder="little")

        self.footer = Footer.model_validate_json(puffin[-(footer_payload_size_int + 12) : -12])
        puffin = puffin[8:]

        self._deletion_vectors = {
            blob.properties[PROPERTY_REFERENCED_DATA_FILE]: _deserialize_bitmap(puffin[blob.offset : blob.offset + blob.length])
            for blob in self.footer.blobs
        }

    def to_vector(self) -> dict[str, "pa.ChunkedArray"]:
        return {path: _bitmaps_to_chunked_array(bitmaps) for path, bitmaps in self._deletion_vectors.items()}


class PuffinWriter:
    """Writes a Puffin file containing a single deletion-vector-v1 blob to an output file."""

    _output_file: OutputFile
    _blobs: list[PuffinBlobMetadata]
    _blob_payloads: list[bytes]
    _created_by: str

    def __init__(self, output_file: OutputFile, created_by: str | None = None) -> None:
        self._output_file = output_file
        self._blobs = []
        self._blob_payloads = []
        self._created_by = created_by if created_by is not None else f"PyIceberg version {__version__}"

    def set_blob(
        self,
        positions: Iterable[int],
        referenced_data_file: str,
    ) -> None:
        """Set the deletion vector blob for a data file, replacing any previously set blob.

        Args:
            positions: Zero-based positions of the deleted rows in the referenced data file.
            referenced_data_file: Location of the data file the deletion vector applies to.
        """
        # We only support one blob at the moment
        self._blobs = []
        self._blob_payloads = []

        bitmaps: dict[int, BitMap] = {}
        for pos in positions:
            if pos < 0:
                raise ValueError(f"Invalid position: {pos}, positions must be non-negative")
            key = pos >> 32
            low_bits = pos & 0xFFFFFFFF
            if key not in bitmaps:
                bitmaps[key] = BitMap()
            bitmaps[key].add(low_bits)

        if not bitmaps:
            raise ValueError("Deletion vector must contain at least one position")

        cardinality = sum(len(bm) for bm in bitmaps.values())
        vector_payload = _serialize_bitmaps(bitmaps)

        # deletion-vector-v1 blob layout: combined length of magic and vector (4 bytes, big-endian),
        # the DV magic bytes, the serialized vector, and a CRC-32 checksum of magic + vector (4 bytes, big-endian)
        blob_content = DELETION_VECTOR_MAGIC + vector_payload
        self._blob_payloads.append(
            len(blob_content).to_bytes(4, "big") + blob_content + zlib.crc32(blob_content).to_bytes(4, "big")
        )

        self._blobs.append(
            PuffinBlobMetadata(
                type="deletion-vector-v1",
                fields=[ROW_POSITION_FIELD_ID],
                # -1 means the snapshot id and sequence number are inherited at commit time
                snapshot_id=-1,
                sequence_number=-1,
                # offset and length are placeholders; finish() fills them in when assembling the file
                offset=0,
                length=0,
                properties={PROPERTY_REFERENCED_DATA_FILE: referenced_data_file, "cardinality": str(cardinality)},
                compression_codec=None,
            )
        )

    def finish(self) -> int:
        """Write the Puffin file to the output file and return its size in bytes."""
        with io.BytesIO() as out:
            out.write(MAGIC_BYTES)

            blobs_metadata: list[PuffinBlobMetadata] = []
            for blob_metadata, blob_payload in zip(self._blobs, self._blob_payloads, strict=True):
                blobs_metadata.append(blob_metadata.model_copy(update={"offset": out.tell(), "length": len(blob_payload)}))
                out.write(blob_payload)

            footer = Footer(blobs=blobs_metadata, properties={"created-by": self._created_by})
            footer_payload_bytes = footer.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")

            out.write(MAGIC_BYTES)
            out.write(footer_payload_bytes)
            out.write(len(footer_payload_bytes).to_bytes(4, "little"))
            out.write((0).to_bytes(4, "little"))  # flags
            out.write(MAGIC_BYTES)

            puffin_bytes = out.getvalue()

        with self._output_file.create(overwrite=True) as output_stream:
            output_stream.write(puffin_bytes)

        return len(puffin_bytes)
