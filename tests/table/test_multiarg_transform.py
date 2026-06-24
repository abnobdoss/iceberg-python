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
import re
from collections.abc import Callable
from typing import Any

import pytest

from pyiceberg.partitioning import (
    MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED,
    PartitionField,
    PartitionFieldValue,
    PartitionKey,
    PartitionSpec,
)
from pyiceberg.schema import Schema
from pyiceberg.table.sorting import NullOrder, SortDirection, SortField
from pyiceberg.transforms import IdentityTransform, Transform, TruncateTransform
from pyiceberg.types import IcebergType, IntegerType, NestedField


def test_partition_field_multi_source_ids_roundtrip() -> None:
    payload = '{"source-ids":[1,2],"field-id":1000,"transform":"identity","name":"x"}'

    field = PartitionField.model_validate_json(payload)

    assert field.source_ids == (1, 2)
    assert field.source_id is None
    serialized = json.loads(field.model_dump_json())
    assert serialized["source-ids"] == [1, 2]
    assert "source-id" not in serialized


def test_partition_field_multi_source_id_boundaries_raise_clear_error() -> None:
    schema = Schema(NestedField(1, "x", IntegerType()), NestedField(2, "y", IntegerType()))
    field = PartitionField.model_validate_json('{"source-ids":[1,2],"field-id":1000,"transform":"identity","name":"x"}')
    spec = PartitionSpec(field)

    with pytest.raises(ValueError, match=re.escape(MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED)):
        _ = field.single_source_id

    with pytest.raises(ValueError, match=re.escape(MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED)):
        spec.partition_type(schema)

    partition_key = PartitionKey(
        field_values=[PartitionFieldValue(field=field, value=1)],
        partition_spec=spec,
        schema=schema,
    )
    with pytest.raises(ValueError, match=re.escape(MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED)):
        _ = partition_key.partition


def test_partition_spec_multi_source_compatibility_checks_all_sources() -> None:
    schema = Schema(NestedField(1, "x", IntegerType()), NestedField(2, "y", IntegerType()))
    field = PartitionField.model_validate_json('{"source-ids":[1,2],"field-id":1000,"transform":"identity","name":"x"}')
    spec = PartitionSpec(field)

    spec.check_compatible(schema)


def test_single_source_serializer_honors_by_alias() -> None:
    field = PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="x")
    sort_field = SortField(
        source_id=2,
        transform=IdentityTransform(),
        direction=SortDirection.ASC,
        null_order=NullOrder.NULLS_FIRST,
    )

    # by_alias=False must use the python field name only (no duplicate aliased key)
    assert field.model_dump(by_alias=False) == {
        "source_id": 1,
        "field_id": 1000,
        "transform": "identity",
        "name": "x",
    }
    assert sort_field.model_dump(by_alias=False)["source_id"] == 2
    assert "source-id" not in sort_field.model_dump(by_alias=False)

    # by_alias=True keeps the spec-aliased key
    assert field.model_dump(by_alias=True)["source-id"] == 1
    assert "source_id" not in field.model_dump(by_alias=True)


def test_sort_field_multi_source_ids_roundtrip() -> None:
    payload = '{"source-ids":[1,2],"transform":"identity","direction":"asc","null-order":"nulls-first"}'

    field = SortField.model_validate_json(payload)

    assert field.source_ids == (1, 2)
    assert field.source_id is None
    serialized = json.loads(field.model_dump_json())
    assert serialized["source-ids"] == [1, 2]
    assert "source-id" not in serialized

    with pytest.raises(ValueError, match=re.escape(MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED)):
        _ = field.single_source_id


def test_partition_field_single_source_ids_serializes_as_source_id() -> None:
    payload = '{"source-ids":[19],"field-id":1000,"transform":"truncate[10]","name":"x"}'

    field = PartitionField.model_validate_json(payload)

    assert field.source_ids == (19,)
    assert field.source_id == 19
    expected = PartitionField(source_id=19, field_id=1000, transform=TruncateTransform(width=10), name="x")
    assert field == expected
    assert hash(field) == hash(expected)
    serialized = json.loads(field.model_dump_json())
    assert serialized["source-id"] == 19
    assert "source-ids" not in serialized


def test_sort_field_single_source_ids_serializes_as_source_id() -> None:
    payload = '{"source-ids":[19],"transform":"identity","direction":"asc","null-order":"nulls-first"}'

    field = SortField.model_validate_json(payload)

    assert field.source_ids == (19,)
    assert field.source_id == 19
    expected = SortField(
        source_id=19,
        transform=IdentityTransform(),
        direction=SortDirection.ASC,
        null_order=NullOrder.NULLS_FIRST,
    )
    assert field == expected
    assert hash(field) == hash(expected)
    serialized = json.loads(field.model_dump_json())
    assert serialized["source-id"] == 19
    assert "source-ids" not in serialized


def test_empty_source_ids_rejected() -> None:
    with pytest.raises(Exception, match="Empty source-ids is not allowed"):
        PartitionField.model_validate_json('{"source-ids":[],"field-id":1000,"transform":"identity","name":"x"}')

    with pytest.raises(Exception, match="Empty source-ids is not allowed"):
        SortField.model_validate_json('{"source-ids":[],"transform":"identity","direction":"asc","null-order":"nulls-first"}')


class SyntheticAddTransform(Transform[list[int], int]):
    root: str = "synthetic-add"

    def __init__(self) -> None:
        super().__init__("synthetic-add")

    def transform(self, source: IcebergType) -> Callable[[list[int] | None], int | None]:
        raise NotImplementedError

    def transform_multi(self, source_types: list[IcebergType]) -> Callable[[list[Any]], int | None]:
        assert source_types == [IntegerType(), IntegerType()]

        def add(values: list[Any]) -> int | None:
            if any(value is None for value in values):
                return None
            return values[0] + values[1]

        return add

    def can_transform(self, source: IcebergType) -> bool:
        return False

    def result_type(self, source: IcebergType) -> IcebergType:
        return IntegerType()

    def project(self, name: str, pred: Any) -> Any:
        return None

    def strict_project(self, name: str, pred: Any) -> Any:
        return None

    def pyarrow_transform(self, source: IcebergType) -> Callable[[Any], Any]:
        raise NotImplementedError


def test_synthetic_multi_arg_transform_dispatch() -> None:
    transform = SyntheticAddTransform().transform_multi([IntegerType(), IntegerType()])

    assert transform([1, 2]) == 3
    assert transform([10, 5]) == 15
    assert transform([1, None]) is None

    with pytest.raises(NotImplementedError, match="This transform does not support multiple arguments"):
        IdentityTransform().transform_multi([IntegerType(), IntegerType()])
