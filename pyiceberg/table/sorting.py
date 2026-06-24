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
# pylint: disable=keyword-arg-before-vararg
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    SerializationInfo,
    WithJsonSchema,
    model_serializer,
    model_validator,
)

from pyiceberg.exceptions import ValidationError
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform, Transform, parse_transform
from pyiceberg.typedef import IcebergBaseModel
from pyiceberg.types import IcebergType

MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED = (
    "Multi-argument transform evaluation is not yet supported (no concrete multi-arg transform is defined in the Iceberg spec)"
)


class SortDirection(Enum):
    ASC = "asc"
    DESC = "desc"

    def __str__(self) -> str:
        """Return the string representation of the SortDirection class."""
        return self.name

    def __repr__(self) -> str:
        """Return the string representation of the SortDirection class."""
        return f"SortDirection.{self.name}"


class NullOrder(Enum):
    NULLS_FIRST = "nulls-first"
    NULLS_LAST = "nulls-last"

    def __str__(self) -> str:
        """Return the string representation of the NullOrder class."""
        return self.name.replace("_", " ")

    def __repr__(self) -> str:
        """Return the string representation of the NullOrder class."""
        return f"NullOrder.{self.name}"


class SortField(IcebergBaseModel):
    """Sort order field.

    Args:
      source_id (int): Source column id from the table’s schema.
      transform (str): Transform that is used to produce values to be sorted on from the source column.
                       This is the same transform as described in partition transforms.
      direction (SortDirection): Sort direction, that can only be either asc or desc.
      null_order (NullOrder): Null order that describes the order of null values when sorted.
                              Can only be either nulls-first or nulls-last.
    """

    def __init__(
        self,
        source_id: int | None = None,
        transform: Transform[Any, Any] | Callable[[IcebergType], Transform[Any, Any]] | None = None,
        direction: SortDirection | None = None,
        null_order: NullOrder | None = None,
        **data: Any,
    ):
        if source_id is not None:
            data["source-id"] = source_id
        if transform is not None:
            data["transform"] = transform
        if direction is not None:
            data["direction"] = direction
        if null_order is not None:
            data["null-order"] = null_order
        super().__init__(**data)

    @model_validator(mode="before")
    def set_null_order(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["direction"] = values["direction"] if values.get("direction") else SortDirection.ASC
        if not values.get("null-order"):
            values["null-order"] = NullOrder.NULLS_FIRST if values["direction"] == SortDirection.ASC else NullOrder.NULLS_LAST
        return values

    @model_validator(mode="before")
    @classmethod
    def map_source_ids_onto_source_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            source_ids_key = "source-ids" if "source-ids" in data else "source_ids" if "source_ids" in data else None
            if source_ids_key:
                source_ids = data[source_ids_key]
                if isinstance(source_ids, (list, tuple)):
                    if len(source_ids) == 0:
                        raise ValueError("Empty source-ids is not allowed")
                    if len(source_ids) == 1:
                        data["source-id"] = source_ids[0]
                    else:
                        data.pop("source-id", None)
                        data.pop("source_id", None)
        return data

    source_id: int | None = Field(alias="source-id", default=None)
    source_ids: tuple[int, ...] | None = Field(alias="source-ids", default=None, repr=False)
    transform: Annotated[  # type: ignore
        Transform,
        BeforeValidator(parse_transform),
        PlainSerializer(lambda c: str(c), return_type=str),  # pylint: disable=W0108
        WithJsonSchema({"type": "string"}, mode="serialization"),
    ] = Field(default=IdentityTransform())
    direction: SortDirection = Field()
    null_order: NullOrder = Field(alias="null-order")

    @model_validator(mode="after")
    def validate_source_ids_present(self) -> "SortField":
        if self.source_ids is not None and len(self.source_ids) == 0:
            raise ValueError("Empty source-ids is not allowed")
        if self.source_id is None and self.source_ids is None:
            raise ValueError("source-id or source-ids must be present")
        return self

    @property
    def source_ids_normalized(self) -> tuple[int, ...]:
        if self.source_ids is not None and len(self.source_ids) > 1:
            return self.source_ids
        if self.source_id is not None:
            return (self.source_id,)
        return self.source_ids or ()

    @property
    def single_source_id(self) -> int:
        ids = self.source_ids_normalized
        if len(ids) != 1:
            raise ValueError(MULTI_ARGUMENT_TRANSFORM_UNSUPPORTED)
        return ids[0]

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:
        result = handler(self)
        single_key, multi_key = ("source-id", "source-ids") if info.by_alias else ("source_id", "source_ids")
        source_ids = self.source_ids_normalized
        if len(source_ids) > 1:
            result.pop(single_key, None)
            result[multi_key] = list(source_ids)
        else:
            result.pop(multi_key, None)
            if source_ids:
                result[single_key] = source_ids[0]
        return result

    def __str__(self) -> str:
        """Return the string representation of the SortField class."""
        source_ids = ", ".join(str(source_id) for source_id in self.source_ids_normalized)
        if isinstance(self.transform, IdentityTransform):
            # In the case of an identity transform, we can omit the transform
            return f"{source_ids} {self.direction} {self.null_order}"
        else:
            return f"{self.transform}({source_ids}) {self.direction} {self.null_order}"

    def __eq__(self, other: Any) -> bool:
        """Return the equality of two instances of the SortField class."""
        if not isinstance(other, SortField):
            return False
        return (
            self.source_ids_normalized == other.source_ids_normalized
            and self.transform == other.transform
            and self.direction == other.direction
            and self.null_order == other.null_order
        )

    def __hash__(self) -> int:
        """Return a hash consistent with SortField equality."""
        return hash((self.source_ids_normalized, self.transform, self.direction, self.null_order))


INITIAL_SORT_ORDER_ID = 1


class SortOrder(IcebergBaseModel):
    """Describes how the data is sorted within the table.

    Users can sort their data within partitions by columns to gain performance.

    The order of the sort fields within the list defines the order in which the sort is applied to the data.

    Args:
      fields (List[SortField]): The fields how the table is sorted.

    Keyword Args:
      order_id (int): An unique id of the sort-order of a table.
    """

    order_id: int = Field(alias="order-id", default=INITIAL_SORT_ORDER_ID)
    fields: list[SortField] = Field(default_factory=list)

    def __init__(self, *fields: SortField, **data: Any):
        if fields:
            data["fields"] = fields
        super().__init__(**data)

    @property
    def is_unsorted(self) -> bool:
        return len(self.fields) == 0

    def __str__(self) -> str:
        """Return the string representation of the SortOrder class."""
        result_str = "["
        if self.fields:
            result_str += "\n  " + "\n  ".join([str(field) for field in self.fields]) + "\n"
        result_str += "]"
        return result_str

    def __repr__(self) -> str:
        """Return the string representation of the SortOrder class."""
        fields = f"{', '.join(repr(column) for column in self.fields)}, " if self.fields else ""
        return f"SortOrder({fields}order_id={self.order_id})"

    def check_compatible(self, schema: Schema) -> None:
        for field in self.fields:
            for source_id in field.source_ids_normalized:
                source_field = schema._lazy_id_to_field.get(source_id)
                if source_field is None:
                    raise ValidationError(f"Cannot find source column for sort field: {field}")
                if not source_field.field_type.is_primitive:
                    raise ValidationError(f"Cannot sort by non-primitive source field: {source_field}")
                if not field.transform.can_transform(source_field.field_type):
                    raise ValidationError(
                        f"Invalid source field {source_field.name} with type {source_field.field_type} "
                        + f"for transform: {field.transform}"
                    )


UNSORTED_SORT_ORDER_ID = 0
UNSORTED_SORT_ORDER = SortOrder(order_id=UNSORTED_SORT_ORDER_ID)


def assign_fresh_sort_order_ids(
    sort_order: SortOrder, old_schema: Schema, fresh_schema: Schema, sort_order_id: int = INITIAL_SORT_ORDER_ID
) -> SortOrder:
    if sort_order.is_unsorted:
        return UNSORTED_SORT_ORDER

    fresh_fields = []
    for field in sort_order.fields:
        source_id = field.single_source_id
        original_field = old_schema.find_column_name(source_id)
        if original_field is None:
            raise ValueError(f"Could not find in old schema: {field}")
        fresh_field = fresh_schema.find_field(original_field)
        if fresh_field is None:
            raise ValueError(f"Could not find field in fresh schema: {original_field}")
        fresh_fields.append(
            SortField(
                source_id=fresh_field.field_id,
                transform=field.transform,
                direction=field.direction,
                null_order=field.null_order,
            )
        )

    return SortOrder(*fresh_fields, order_id=sort_order_id)
