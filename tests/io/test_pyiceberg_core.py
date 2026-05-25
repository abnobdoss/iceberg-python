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

import sys
from types import ModuleType
from typing import Any

import pytest

from pyiceberg.expressions import And, EqualTo, IsNull, StartsWith
from pyiceberg.io import FileIO, InputFile, OutputFile
from pyiceberg.io.pyiceberg_core import (
    DEFAULT_NATIVE_ARROW_BATCH_SIZE,
    arrow_batch_reader_from_pyiceberg_core,
    arrow_batch_reader_from_pyiceberg_core_planned,
    can_read_projected_schema_with_pyiceberg_core,
    delete_file_to_pyiceberg_core,
    expression_to_pyiceberg_core,
    file_io_to_pyiceberg_core,
    file_scan_task_to_pyiceberg_core,
    schema_to_pyiceberg_core,
)
from pyiceberg.manifest import DataFile, DataFileContent
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import FileScanTask
from pyiceberg.table.name_mapping import MappedField, NameMapping
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import FloatType, IntegerType, NestedField, StringType


class CoreObject:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class CoreSchema(CoreObject):
    @classmethod
    def from_json(cls, schema_json: str) -> CoreSchema:
        return cls(schema_json=schema_json)


class CoreFileIO(CoreObject):
    @classmethod
    def from_props(cls, properties: dict[str, str]) -> CoreFileIO:
        return cls(properties=properties)


class CorePredicate(CoreObject):
    @staticmethod
    def always_true() -> CorePredicate:
        return CorePredicate(kind="always_true")

    @staticmethod
    def always_false() -> CorePredicate:
        return CorePredicate(kind="always_false")

    def and_(self, other: CorePredicate) -> CorePredicate:
        return CorePredicate(op="and", left=self, right=other)

    def or_(self, other: CorePredicate) -> CorePredicate:
        return CorePredicate(op="or", left=self, right=other)

    def negate(self) -> CorePredicate:
        return CorePredicate(op="not", child=self)


class CoreReference(CoreObject):
    def _predicate(self, op: str, *args: Any) -> CorePredicate:
        return CorePredicate(op=op, name=self.args[0], args=args)

    def eq(self, value: Any) -> CorePredicate:
        return self._predicate("eq", value)

    def starts_with(self, value: Any) -> CorePredicate:
        return self._predicate("starts_with", value)

    def is_null(self) -> CorePredicate:
        return self._predicate("is_null")


class CoreReader(CoreObject):
    last_init: CoreReader | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        CoreReader.last_init = self

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return CoreObject(*args, **kwargs)


class CoreTable(CoreObject):
    last_init: CoreTable | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        CoreTable.last_init = self

    @classmethod
    def from_metadata_json(cls, file_io: Any, identifier: list[str], metadata_json: str, *args: Any, **kwargs: Any) -> CoreTable:
        return cls(*args, file_io=file_io, identifier=identifier, metadata_json=metadata_json, **kwargs)

    def read_arrow(self, *args: Any, **kwargs: Any) -> Any:
        return CoreObject(*args, **kwargs)


class FakeFileIO(FileIO):
    def new_input(self, location: str) -> InputFile:
        raise NotImplementedError

    def new_output(self, location: str) -> OutputFile:
        raise NotImplementedError

    def delete(self, location: str | InputFile | OutputFile) -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def fake_pyiceberg_core(monkeypatch: pytest.MonkeyPatch) -> None:
    root = ModuleType("pyiceberg_core")

    schema: Any = ModuleType("pyiceberg_core.schema")
    schema.Schema = CoreSchema

    file_io: Any = ModuleType("pyiceberg_core.file_io")
    file_io.FileIO = CoreFileIO

    expression: Any = ModuleType("pyiceberg_core.expression")
    expression.Predicate = CorePredicate
    expression.Reference = CoreReference

    scan: Any = ModuleType("pyiceberg_core.scan")
    scan.DeleteFile = CoreObject
    scan.FileScanTask = CoreObject
    scan.ArrowReader = CoreReader
    scan.Table = CoreTable

    monkeypatch.setitem(sys.modules, "pyiceberg_core", root)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.schema", schema)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.file_io", file_io)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.expression", expression)
    monkeypatch.setitem(sys.modules, "pyiceberg_core.scan", scan)


def test_schema_to_pyiceberg_core_feature_gates_old_core_wheels(monkeypatch: pytest.MonkeyPatch, simple_schema: Schema) -> None:
    monkeypatch.setitem(sys.modules, "pyiceberg_core", ModuleType("pyiceberg_core"))
    monkeypatch.delitem(sys.modules, "pyiceberg_core.schema", raising=False)

    with pytest.raises(NotImplementedError, match="does not expose native scan bindings"):
        schema_to_pyiceberg_core(simple_schema)


@pytest.fixture
def simple_schema() -> Schema:
    return Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType()),
        schema_id=3,
    )


def test_schema_to_pyiceberg_core_uses_lazy_core_schema_json(simple_schema: Schema) -> None:
    converted = schema_to_pyiceberg_core(simple_schema)

    assert converted.kwargs["schema_json"] == simple_schema.model_dump_json(by_alias=True, exclude_none=True)


def test_file_io_to_pyiceberg_core_uses_file_io_properties() -> None:
    converted = file_io_to_pyiceberg_core(FakeFileIO(properties={"s3.region": "us-east-1"}))

    assert converted.kwargs == {"properties": {"s3.region": "us-east-1"}}


def test_file_io_to_pyiceberg_core_filters_non_string_properties() -> None:
    converted = file_io_to_pyiceberg_core(FakeFileIO(properties={"s3.region": "us-east-1", "auth.manager": object()}))

    assert converted.kwargs == {"properties": {"s3.region": "us-east-1"}}


def test_file_io_to_pyiceberg_core_maps_path_style_s3_property() -> None:
    converted = file_io_to_pyiceberg_core(FakeFileIO(properties={"s3.force-virtual-addressing": "false"}))

    assert converted.kwargs == {
        "properties": {
            "s3.force-virtual-addressing": "false",
            "s3.path-style-access": "true",
        }
    }


def test_expression_to_pyiceberg_core_converts_expression_tree(simple_schema: Schema) -> None:
    converted = expression_to_pyiceberg_core(And(EqualTo("id", 34), StartsWith("data", "abc")), simple_schema)

    assert converted.kwargs["op"] == "and"
    assert converted.kwargs["left"].kwargs == {"op": "eq", "name": "id", "args": (34,)}
    assert converted.kwargs["right"].kwargs == {"op": "starts_with", "name": "data", "args": ("abc",)}


def test_expression_to_pyiceberg_core_converts_unary_expression(simple_schema: Schema) -> None:
    converted = expression_to_pyiceberg_core(IsNull("data"), simple_schema)

    assert converted.kwargs == {"op": "is_null", "name": "data", "args": ()}


def test_expression_to_pyiceberg_core_requires_schema_for_unbound_expression() -> None:
    with pytest.raises(NotImplementedError, match="without a Schema"):
        expression_to_pyiceberg_core(EqualTo("id", 34))


def test_expression_to_pyiceberg_core_raises_clear_error_for_unsupported_expression() -> None:
    from pyiceberg.expressions import IsNaN

    nan_schema = Schema(NestedField(1, "value", FloatType()))
    with pytest.raises(NotImplementedError, match="unsupported unary predicate"):
        expression_to_pyiceberg_core(IsNaN("value"), nan_schema)


def test_delete_file_to_pyiceberg_core_converts_delete_file_payload() -> None:
    delete_file = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path="s3://warehouse/table/delete.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=1,
        file_size_in_bytes=123,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    delete_file.spec_id = 7

    converted = delete_file_to_pyiceberg_core(delete_file)

    assert converted.args == ("s3://warehouse/table/delete.parquet", 123, "position-deletes")
    assert converted.kwargs == {"partition_spec_id": 7, "equality_ids": None}


def test_delete_file_to_pyiceberg_core_converts_equality_deletes() -> None:
    delete_file = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES,
        file_path="s3://warehouse/table/eq-delete.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=1,
        file_size_in_bytes=123,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
        equality_ids=[1],
    )
    delete_file.spec_id = 7

    converted = delete_file_to_pyiceberg_core(delete_file)
    assert converted.args == ("s3://warehouse/table/eq-delete.parquet", 123, "equality-deletes")
    assert converted.kwargs == {"partition_spec_id": 7, "equality_ids": [1]}


def test_delete_file_to_pyiceberg_core_raises_value_error_for_equality_delete_without_ids() -> None:
    delete_file = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES,
        file_path="s3://warehouse/table/eq-delete.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=1,
        file_size_in_bytes=123,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
        equality_ids=None,
    )
    delete_file.spec_id = 7

    with pytest.raises(ValueError, match="equality_ids is required for equality delete file"):
        delete_file_to_pyiceberg_core(delete_file)


def test_file_scan_task_to_pyiceberg_core_converts_task_payload(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={1: 20},
        value_counts={1: 10},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"\x01\x00\x00\x00"},
        upper_bounds={1: b"\x0a\x00\x00\x00"},
    )
    data_file.spec_id = 7
    partition_spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"))
    name_mapping = NameMapping([MappedField(field_id=1, names=["id"])])

    task = FileScanTask(data_file, residual=EqualTo("id", 3))
    converted = file_scan_task_to_pyiceberg_core(
        task,
        simple_schema,
        partition_spec=partition_spec,
        name_mapping=name_mapping,
        case_sensitive=False,
    )

    assert converted.kwargs["data_file_path"] == "s3://warehouse/table/data.parquet"
    assert converted.kwargs["data_file_format"] == "parquet"
    assert converted.kwargs["file_size_in_bytes"] == 1234
    assert converted.kwargs["length"] == 1234
    assert converted.kwargs["record_count"] == 10
    assert converted.kwargs["partition_data"] == ["bucket-1"]
    assert converted.kwargs["partition_spec"] == partition_spec.model_dump_json(by_alias=True, exclude_none=True)
    assert converted.kwargs["name_mapping"] == name_mapping.model_dump_json(by_alias=True, exclude_none=True)
    assert converted.kwargs["case_sensitive"] is False
    assert set(converted.kwargs["project_field_ids"]) == {1, 2}
    assert converted.kwargs["predicate"].kwargs == {"op": "eq", "name": "id", "args": (3,)}


def test_file_scan_task_to_pyiceberg_core_adds_filter_only_field_to_read_projection(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={1: 20},
        value_counts={1: 10},
        null_value_counts={1: 0},
        nan_value_counts={},
        lower_bounds={1: b"\x01\x00\x00\x00"},
        upper_bounds={1: b"\x0a\x00\x00\x00"},
    )
    data_file.spec_id = 0
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=3)

    task = FileScanTask(data_file, residual=EqualTo("data", "abc"))
    converted = file_scan_task_to_pyiceberg_core(task, simple_schema, projected_schema=projected_schema)

    assert converted.kwargs["project_field_ids"] == [1, 2]


def test_can_read_projected_schema_with_pyiceberg_core_requires_filter_fields(simple_schema: Schema) -> None:
    projected_schema = Schema(NestedField(1, "id", IntegerType(), required=True), schema_id=3)

    assert not can_read_projected_schema_with_pyiceberg_core(simple_schema, projected_schema, EqualTo("data", "abc"), True)
    assert can_read_projected_schema_with_pyiceberg_core(simple_schema, projected_schema, EqualTo("id", 1), True)


def test_file_scan_task_to_pyiceberg_core_requires_partition_spec_for_partitioned_task(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    data_file.spec_id = 7

    with pytest.raises(ValueError, match="partition_spec is required"):
        file_scan_task_to_pyiceberg_core(FileScanTask(data_file), simple_schema)


def test_arrow_batch_reader_from_pyiceberg_core_with_partition_and_name_mapping(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record("bucket-1"),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    data_file.spec_id = 1
    partition_spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="id"))
    name_mapping = NameMapping([MappedField(field_id=1, names=["id"])])

    task = FileScanTask(data_file)

    reader = arrow_batch_reader_from_pyiceberg_core(
        FakeFileIO({}),
        [task],
        simple_schema,
        simple_schema,
        {1: partition_spec},
        name_mapping,
    )

    assert isinstance(reader, CoreObject)
    assert CoreReader.last_init is not None
    assert CoreReader.last_init.kwargs["batch_size"] == DEFAULT_NATIVE_ARROW_BATCH_SIZE
    assert len(reader.args) == 2
    assert isinstance(reader.args[0], CoreSchema)
    assert isinstance(reader.args[1], list)
    assert len(reader.args[1]) == 1

    converted_task = reader.args[1][0]
    assert converted_task.kwargs["partition_spec"] == partition_spec.model_dump_json(by_alias=True, exclude_none=True)
    assert converted_task.kwargs["name_mapping"] == name_mapping.model_dump_json(by_alias=True, exclude_none=True)


def test_arrow_batch_reader_from_pyiceberg_core_with_limit(simple_schema: Schema) -> None:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://warehouse/table/data.parquet",
        file_format="PARQUET",
        partition=Record(),
        record_count=10,
        file_size_in_bytes=1234,
        column_sizes={},
        value_counts={},
        null_value_counts={},
        nan_value_counts={},
        lower_bounds={},
        upper_bounds={},
    )
    data_file.spec_id = 0
    task = FileScanTask(data_file)

    reader = arrow_batch_reader_from_pyiceberg_core(
        FakeFileIO({}),
        [task],
        simple_schema,
        simple_schema,
        {},
        None,
        limit=5,
    )

    assert isinstance(reader, CoreObject)
    assert reader.kwargs.get("max_rows") == 5
    assert CoreReader.last_init is not None
    assert CoreReader.last_init.kwargs["batch_size"] == DEFAULT_NATIVE_ARROW_BATCH_SIZE
    assert CoreReader.last_init.kwargs["data_file_concurrency_limit"] == 1


def test_arrow_batch_reader_from_pyiceberg_core_planned(simple_schema: Schema) -> None:
    from pyiceberg.expressions import EqualTo
    from pyiceberg.table.metadata import TableMetadataV2

    metadata = TableMetadataV2(
        location="s3://warehouse/table",
        last_sequence_number=1,
        last_updated_ms=1600000000000,
        last_column_id=2,
        schemas=[simple_schema],
        current_schema_id=simple_schema.schema_id,
        partition_specs=[PartitionSpec()],
        default_spec_id=0,
        last_partition_id=1000,
        default_sort_order_id=0,
        sort_orders=[],
        properties={},
        snapshots=[],
        snapshot_log=[],
        metadata_log=[],
    )

    reader = arrow_batch_reader_from_pyiceberg_core_planned(
        FakeFileIO({}),
        metadata,
        simple_schema,
        EqualTo("id", 123),
        selected_fields=("id",),
        table_identifier=("ns", "tbl"),
        snapshot_id=5,
        case_sensitive=True,
        limit=10,
    )

    assert isinstance(reader, CoreObject)
    assert CoreTable.last_init is not None
    assert CoreTable.last_init.kwargs["identifier"] == ["ns", "tbl"]
    assert CoreTable.last_init.kwargs["metadata_json"] == metadata.model_dump_json(by_alias=True, exclude_none=True)
    assert isinstance(CoreTable.last_init.kwargs["file_io"], CoreFileIO)

    assert isinstance(reader.args[0], CoreSchema)
    assert reader.kwargs["selected_fields"] == ["id"]
    assert reader.kwargs["snapshot_id"] == 5
    assert reader.kwargs["case_sensitive"] is True
    assert reader.kwargs["max_rows"] == 10

    pred = reader.kwargs["predicate"]
    assert isinstance(pred, CorePredicate)
    assert pred.kwargs["op"] == "eq"
    assert pred.kwargs["name"] == "id"
    assert pred.kwargs["args"] == (123,)


def test_arrow_batch_reader_from_pyiceberg_core_planned_star_fields(simple_schema: Schema) -> None:
    from pyiceberg.expressions import AlwaysTrue
    from pyiceberg.table.metadata import TableMetadataV2

    metadata = TableMetadataV2(
        location="s3://warehouse/table",
        last_sequence_number=1,
        last_updated_ms=1600000000000,
        last_column_id=2,
        schemas=[simple_schema],
        current_schema_id=simple_schema.schema_id,
        partition_specs=[PartitionSpec()],
        default_spec_id=0,
        last_partition_id=1000,
        default_sort_order_id=0,
        sort_orders=[],
        properties={},
        snapshots=[],
        snapshot_log=[],
        metadata_log=[],
    )

    reader = arrow_batch_reader_from_pyiceberg_core_planned(
        FakeFileIO({}),
        metadata,
        simple_schema,
        AlwaysTrue(),
        selected_fields=("*",),
        table_identifier=None,
        snapshot_id=None,
        case_sensitive=True,
        limit=None,
    )
    assert reader.kwargs["selected_fields"] is None
