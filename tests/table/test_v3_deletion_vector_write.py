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
"""End-to-end tests for committing a v3 deletion vector and reading it back.

These exercise the full integration: write a v3 table, write a Puffin deletion
vector for one of its data files, emit a DELETES manifest entry carrying the
``referenced_data_file``/``content_offset``/``content_size_in_bytes`` v3 fields,
commit it through the snapshot producer, then scan the table and confirm the
existing reader applies the deletion vector.
"""

import pyarrow as pa
import pytest

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.manifest import DataFileContent, FileFormat
from pyiceberg.table.puffin import write_deletion_vector


@pytest.fixture()
def catalog(tmp_path):  # type: ignore[no-untyped-def]
    catalog = SqlCatalog(
        "test",
        uri=f"sqlite:///{tmp_path}/cat.db",
        warehouse=f"file://{tmp_path}",
    )
    catalog.create_namespace("ns")
    return catalog


def _commit_dv(catalog, table_identifier, positions):  # type: ignore[no-untyped-def]
    table = catalog.load_table(table_identifier)
    tasks = list(table.scan().plan_files())
    assert len(tasks) == 1
    data_file = tasks[0].file

    dv_path = data_file.file_path.replace(".parquet", "-dv.puffin")
    dv_data_file = write_deletion_vector(
        output_file=table.io.new_output(dv_path),
        referenced_data_file=data_file.file_path,
        positions=positions,
        partition=data_file.partition,
        spec_id=data_file.spec_id,
    )
    with table.transaction() as txn:
        with txn.update_snapshot().append_deletion_vectors() as dv:
            dv.append_data_file(dv_data_file)
    return data_file, dv_data_file


def test_commit_and_read_deletion_vector(catalog) -> None:  # type: ignore[no-untyped-def]
    schema = pa.schema([("id", pa.int64())])
    table = catalog.create_table("ns.dv", schema=schema, properties={"format-version": "3"})
    table.append(pa.table({"id": list(range(10))}))

    data_file, dv_data_file = _commit_dv(catalog, "ns.dv", [2, 4, 6])

    # The DV manifest entry models a position-delete file in Puffin format.
    assert dv_data_file.content == DataFileContent.POSITION_DELETES
    assert dv_data_file.file_format == FileFormat.PUFFIN
    assert dv_data_file.referenced_data_file == data_file.file_path
    # content_offset brackets the framed deletion-vector blob (length+magic+vector+crc)
    # measured from the start of the Puffin file; size covers the whole framed blob.
    assert dv_data_file.content_offset == 4
    assert dv_data_file.content_size_in_bytes > 0

    # The existing reader applies the deletion vector.
    table = catalog.load_table("ns.dv")
    ids = sorted(table.scan().to_arrow().column("id").to_pylist())
    assert ids == [0, 1, 3, 5, 7, 8, 9]


def test_deletion_vector_manifest_entry_persisted_on_disk(catalog) -> None:  # type: ignore[no-untyped-def]
    schema = pa.schema([("id", pa.int64())])
    table = catalog.create_table("ns.dv2", schema=schema, properties={"format-version": "3"})
    table.append(pa.table({"id": list(range(8))}))

    data_file, _ = _commit_dv(catalog, "ns.dv2", [0, 7])

    # Read the committed manifest entry straight off disk and assert the v3 DV fields are there.
    table = catalog.load_table("ns.dv2")
    snapshot = table.current_snapshot()
    assert snapshot is not None

    delete_entries = []
    delete_manifests = []
    for manifest in snapshot.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io):
            if entry.data_file.content == DataFileContent.POSITION_DELETES:
                delete_entries.append(entry.data_file)
                delete_manifests.append(manifest)

    assert len(delete_entries) == 1
    dv = delete_entries[0]
    assert dv.file_format == FileFormat.PUFFIN
    assert dv.referenced_data_file == data_file.file_path
    assert dv.content_offset is not None
    assert dv.content_size_in_bytes is not None

    # The manifest itself is a DELETES manifest (content == 1), not a DATA manifest.
    from pyiceberg.manifest import ManifestContent

    assert delete_manifests[0].content == ManifestContent.DELETES

    ids = sorted(table.scan().to_arrow().column("id").to_pylist())
    assert ids == [1, 2, 3, 4, 5, 6]


def test_deletion_vectors_require_v3(catalog) -> None:  # type: ignore[no-untyped-def]
    schema = pa.schema([("id", pa.int64())])
    table = catalog.create_table("ns.v2", schema=schema, properties={"format-version": "2"})
    table.append(pa.table({"id": [1, 2, 3]}))

    with pytest.raises(ValueError, match="format version 3"):
        with table.transaction() as txn:
            txn.update_snapshot().append_deletion_vectors()
