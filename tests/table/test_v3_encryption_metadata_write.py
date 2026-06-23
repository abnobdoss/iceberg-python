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
"""End-to-end test that v3 ``encryption-keys`` metadata round-trips through a real metadata file.

The v3 write path is now enabled (the write gate was lifted in the foundation),
so ``TableMetadataV3`` serializes through the production ``ToOutputFile``/
``FromInputFile`` code path. This proves the ``encryption-keys`` field and the
snapshot ``key-id`` field survive an actual on-disk metadata file written and
read back with the catalog's own serializers.
"""

import json

import pyarrow as pa
import pytest

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.serializers import FromInputFile, ToOutputFile
from pyiceberg.table.encryption import EncryptedKey


@pytest.fixture()
def catalog(tmp_path):  # type: ignore[no-untyped-def]
    catalog = SqlCatalog(
        "test",
        uri=f"sqlite:///{tmp_path}/cat.db",
        warehouse=f"file://{tmp_path}",
    )
    catalog.create_namespace("ns")
    return catalog


def test_encryption_keys_round_trip_through_real_v3_metadata_file(catalog, tmp_path) -> None:  # type: ignore[no-untyped-def]
    schema = pa.schema([("id", pa.int64())])
    table = catalog.create_table("ns.enc", schema=schema, properties={"format-version": "3"})
    table.append(pa.table({"id": [1, 2, 3]}))

    metadata = table.metadata
    assert metadata.format_version == 3

    enriched = metadata.model_copy(
        update={
            "encryption_keys": [
                EncryptedKey(
                    key_id="key-1",
                    encrypted_key_metadata="ZW5jcnlwdGVkLW1ldGE=",
                    encrypted_by_id="kms-root",
                    properties={"scheme": "AES-GCM"},
                ),
                EncryptedKey(key_id="key-2", encrypted_key_metadata="c2Vjb25k"),
            ]
        }
    )

    io = PyArrowFileIO()
    metadata_path = f"file://{tmp_path}/v3-with-encryption.metadata.json"
    ToOutputFile.table_metadata(enriched, io.new_output(metadata_path), overwrite=True)

    # The raw bytes on disk really contain the v3 encryption-keys structure with spec aliases.
    raw = io.new_input(metadata_path).open().read().decode("utf-8")
    payload = json.loads(raw)
    assert "encryption-keys" in payload
    assert payload["encryption-keys"][0]["key-id"] == "key-1"
    assert payload["encryption-keys"][0]["encrypted-key-metadata"] == "ZW5jcnlwdGVkLW1ldGE="
    assert payload["encryption-keys"][0]["encrypted-by-id"] == "kms-root"
    assert payload["encryption-keys"][0]["properties"] == {"scheme": "AES-GCM"}
    assert payload["encryption-keys"][1]["key-id"] == "key-2"
    # encrypted-by-id / properties are optional and omitted when unset
    assert "encrypted-by-id" not in payload["encryption-keys"][1]

    # Read the file back through the production deserializer.
    parsed = FromInputFile.table_metadata(io.new_input(metadata_path))
    assert parsed.format_version == 3
    assert parsed.encryption_keys is not None
    assert len(parsed.encryption_keys) == 2
    assert parsed.encryption_keys[0] == EncryptedKey(
        key_id="key-1",
        encrypted_key_metadata="ZW5jcnlwdGVkLW1ldGE=",
        encrypted_by_id="kms-root",
        properties={"scheme": "AES-GCM"},
    )
    assert parsed.encryption_keys[1].key_id == "key-2"
    assert parsed.encryption_keys[1].encrypted_by_id is None


def test_snapshot_key_id_round_trip_through_real_v3_metadata_file(catalog, tmp_path) -> None:  # type: ignore[no-untyped-def]
    schema = pa.schema([("id", pa.int64())])
    table = catalog.create_table("ns.enc_snap", schema=schema, properties={"format-version": "3"})
    table.append(pa.table({"id": [1, 2, 3]}))

    metadata = table.metadata
    snapshot = metadata.snapshots[0]
    keyed_snapshot = snapshot.model_copy(update={"key_id": "snap-key-7"})
    enriched = metadata.model_copy(update={"snapshots": [keyed_snapshot]})

    io = PyArrowFileIO()
    metadata_path = f"file://{tmp_path}/v3-snap-keyid.metadata.json"
    ToOutputFile.table_metadata(enriched, io.new_output(metadata_path), overwrite=True)

    raw = json.loads(io.new_input(metadata_path).open().read().decode("utf-8"))
    assert raw["snapshots"][0]["key-id"] == "snap-key-7"

    parsed = FromInputFile.table_metadata(io.new_input(metadata_path))
    assert parsed.snapshots[0].key_id == "snap-key-7"
