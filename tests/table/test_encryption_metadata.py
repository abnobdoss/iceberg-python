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

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from pyiceberg.table.encryption import EncryptedKey
from pyiceberg.table.metadata import TableMetadataV3
from pyiceberg.table.snapshots import Snapshot


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"key-id": "key-a"},
        {"encrypted-key-metadata": "ZW5jcnlwdGVkLW1ldGFkYXRh"},
    ],
)
def test_encrypted_key_requires_key_id_and_metadata(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EncryptedKey.model_validate(payload)


def test_encrypted_key_deserialization_with_all_fields() -> None:
    encrypted_key = EncryptedKey.model_validate(
        {
            "key-id": "key-a",
            "encrypted-key-metadata": "ZW5jcnlwdGVkLW1ldGFkYXRh",
            "encrypted-by-id": "root-key",
            "properties": {"kms": "test", "purpose": "table"},
        }
    )

    assert encrypted_key.key_id == "key-a"
    assert encrypted_key.encrypted_key_metadata == "ZW5jcnlwdGVkLW1ldGFkYXRh"
    assert encrypted_key.encrypted_by_id == "root-key"
    assert encrypted_key.properties == {"kms": "test", "purpose": "table"}


def test_encrypted_key_deserialization_with_required_fields_only() -> None:
    encrypted_key = EncryptedKey.model_validate(
        {
            "key-id": "key-a",
            "encrypted-key-metadata": "ZW5jcnlwdGVkLW1ldGFkYXRh",
        }
    )

    assert encrypted_key.key_id == "key-a"
    assert encrypted_key.encrypted_key_metadata == "ZW5jcnlwdGVkLW1ldGFkYXRh"
    assert encrypted_key.encrypted_by_id is None
    assert encrypted_key.properties is None


def test_encrypted_key_serialization_round_trip_uses_aliases() -> None:
    encrypted_key = EncryptedKey.model_validate(
        {
            "key-id": "key-a",
            "encrypted-key-metadata": "ZW5jcnlwdGVkLW1ldGFkYXRh",
            "encrypted-by-id": "root-key",
            "properties": {"kms": "test"},
        }
    )

    serialized = encrypted_key.model_dump_json(by_alias=True)

    assert EncryptedKey.model_validate_json(serialized) == encrypted_key
    assert '"key-id"' in serialized
    assert '"encrypted-key-metadata"' in serialized
    assert "key_id" not in serialized
    assert "encrypted_key_metadata" not in serialized


def test_snapshot_key_id_deserialization_and_serialization() -> None:
    snapshot = Snapshot.model_validate(
        {
            "snapshot-id": 25,
            "timestamp-ms": 1602638573590,
            "manifest-list": "s3:/a/b/c.avro",
            "key-id": "manifest-list-key",
        }
    )
    snapshot_without_key = Snapshot.model_validate(
        {
            "snapshot-id": 26,
            "timestamp-ms": 1602638573591,
            "manifest-list": "s3:/a/b/d.avro",
        }
    )

    assert snapshot.key_id == "manifest-list-key"
    assert snapshot_without_key.key_id is None
    assert '"key-id":"manifest-list-key"' in snapshot.model_dump_json(by_alias=True)


def test_table_metadata_v3_encryption_keys_deserialization(example_table_metadata_v3: dict[str, Any]) -> None:
    metadata_dict = deepcopy(example_table_metadata_v3)
    metadata_dict["encryption-keys"] = [
        {
            "key-id": "key-a",
            "encrypted-key-metadata": "ZW5jcnlwdGVkLW1ldGFkYXRh",
        }
    ]

    metadata = TableMetadataV3(**metadata_dict)

    assert metadata.encryption_keys is not None
    assert metadata.encryption_keys[0].key_id == "key-a"
    assert metadata.encryption_keys[0].encrypted_key_metadata == "ZW5jcnlwdGVkLW1ldGFkYXRh"


def test_table_metadata_v3_without_encryption_keys_deserialization(example_table_metadata_v3: dict[str, Any]) -> None:
    metadata = TableMetadataV3(**deepcopy(example_table_metadata_v3))

    assert metadata.encryption_keys is None
