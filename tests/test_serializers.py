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
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockFixture

from pyiceberg.serializers import (
    Compressor,
    FromInputFile,
    GzipCompressor,
    NoopCompressor,
    ToOutputFile,
    metadata_file_extension,
)
from pyiceberg.table import StaticTable, TableProperties
from pyiceberg.table.locations import load_location_provider
from pyiceberg.table.metadata import TableMetadataV1, TableMetadataV2
from pyiceberg.table.update import AssertRefSnapshotId, TableRequirement
from pyiceberg.typedef import IcebergBaseModel


def test_legacy_current_snapshot_id(
    mocker: MockFixture, tmp_path_factory: pytest.TempPathFactory, example_table_metadata_no_snapshot_v1: dict[str, Any]
) -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    metadata_location = str(tmp_path_factory.mktemp("metadata") / f"{uuid.uuid4()}.metadata.json")
    metadata = TableMetadataV1(**example_table_metadata_no_snapshot_v1)
    ToOutputFile.table_metadata(metadata, PyArrowFileIO().new_output(location=metadata_location), overwrite=True)
    static_table = StaticTable.from_metadata(metadata_location)
    assert static_table.metadata.current_snapshot_id is None

    mocker.patch.dict(os.environ, values={"PYICEBERG_LEGACY_CURRENT_SNAPSHOT_ID": "True"})

    ToOutputFile.table_metadata(metadata, PyArrowFileIO().new_output(location=metadata_location), overwrite=True)
    with PyArrowFileIO().new_input(location=metadata_location).open() as input_stream:
        metadata_json_bytes = input_stream.read()
    assert json.loads(metadata_json_bytes)["current-snapshot-id"] == -1
    backwards_compatible_static_table = StaticTable.from_metadata(metadata_location)
    assert backwards_compatible_static_table.metadata.current_snapshot_id is None
    assert backwards_compatible_static_table.metadata == static_table.metadata


def test_null_serializer_field() -> None:
    class ExampleRequest(IcebergBaseModel):
        requirements: tuple[TableRequirement, ...]

    request = ExampleRequest(requirements=(AssertRefSnapshotId(ref="main", snapshot_id=None),))
    dumped_json = request.model_dump_json()
    expected_json = """{"type":"assert-ref-snapshot-id","ref":"main","snapshot-id":null}"""
    assert expected_json in dumped_json


@pytest.mark.parametrize(
    ("location", "compressor_type"),
    [
        ("s3://bucket/table/metadata/00000-table.gz.metadata.json", GzipCompressor),
        ("s3://bucket/table/metadata/00000-table.metadata.json", NoopCompressor),
        ("s3://bucket/table/metadata/00000-table.lz4.metadata.json", NoopCompressor),
    ],
)
def test_get_compressor_detects_metadata_compression(location: str, compressor_type: type[Compressor]) -> None:
    assert isinstance(Compressor.get_compressor(location), compressor_type)


@pytest.mark.parametrize(
    ("codec_name", "compressor_type"),
    [
        (None, NoopCompressor),
        ("none", NoopCompressor),
        ("gzip", GzipCompressor),
        ("GZIP", GzipCompressor),
    ],
)
def test_from_codec_name(codec_name: str | None, compressor_type: type[Compressor]) -> None:
    assert isinstance(Compressor.from_codec_name(codec_name), compressor_type)


@pytest.mark.parametrize("codec_name", ["zstd", "lz4", "snappy", "bogus"])
def test_from_codec_name_raises_for_unknown_codec(codec_name: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported metadata compression codec: {codec_name}"):
        Compressor.from_codec_name(codec_name)


@pytest.mark.parametrize("codec_name", ["none", "gzip"])
def test_table_metadata_round_trip_with_compression(
    tmp_path: Path, example_table_metadata_v2: dict[str, Any], codec_name: str
) -> None:
    from pyiceberg.io.pyarrow import PyArrowFileIO

    metadata = TableMetadataV2(**example_table_metadata_v2)
    file_io = PyArrowFileIO()
    metadata_location = str(tmp_path / f"{uuid.uuid4()}{metadata_file_extension(codec_name)}")

    ToOutputFile.table_metadata(metadata, file_io.new_output(location=metadata_location), overwrite=True)

    raw_bytes = Path(metadata_location).read_bytes()
    if codec_name == "gzip":
        # Java writes gzip-compressed metadata when this codec is set; assert we match by
        # checking the gzip magic bytes rather than relying on the (lossless) round trip.
        assert raw_bytes[:2] == b"\x1f\x8b"
    else:
        assert raw_bytes.lstrip().startswith(b"{")

    parsed_metadata = FromInputFile.table_metadata(file_io.new_input(location=metadata_location))
    assert parsed_metadata == metadata
    assert parsed_metadata.table_uuid == metadata.table_uuid
    assert parsed_metadata.schema() == metadata.schema()


@pytest.mark.parametrize(
    ("table_properties", "expected_suffix"),
    [
        ({}, ".metadata.json"),
        ({TableProperties.WRITE_METADATA_COMPRESSION: "none"}, ".metadata.json"),
        ({TableProperties.WRITE_METADATA_COMPRESSION: "gzip"}, ".gz.metadata.json"),
    ],
)
def test_new_table_metadata_file_location_uses_metadata_compression(
    table_properties: dict[str, str], expected_suffix: str
) -> None:
    provider = load_location_provider(table_location="table_location", table_properties=table_properties)

    metadata_location = provider.new_table_metadata_file_location(new_version=3)

    assert metadata_location.startswith("table_location/metadata/00003-")
    assert metadata_location.endswith(expected_suffix)


def test_new_table_metadata_file_location_raises_for_unknown_compression() -> None:
    provider = load_location_provider(
        table_location="table_location", table_properties={TableProperties.WRITE_METADATA_COMPRESSION: "bogus"}
    )

    with pytest.raises(ValueError, match="Unsupported metadata compression codec: bogus"):
        provider.new_table_metadata_file_location(new_version=3)
