from nvidia.omniverse.storage.fileobject.v1alpha import fileobject_pb2 as _fileobject_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VersionsOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VERSIONS_ORDER_UNSPECIFIED: _ClassVar[VersionsOrder]
    VERSIONS_ORDER_NEWEST_FIRST: _ClassVar[VersionsOrder]
    VERSIONS_ORDER_OLDEST_FIRST: _ClassVar[VersionsOrder]
    VERSIONS_ORDER_BY_KEY: _ClassVar[VersionsOrder]
VERSIONS_ORDER_UNSPECIFIED: VersionsOrder
VERSIONS_ORDER_NEWEST_FIRST: VersionsOrder
VERSIONS_ORDER_OLDEST_FIRST: VersionsOrder
VERSIONS_ORDER_BY_KEY: VersionsOrder

class EnumerateVersionsRequest(_message.Message):
    __slots__ = ("resource_address",)
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    def __init__(self, resource_address: _Optional[str] = ...) -> None: ...

class EnumerateVersionsResponse(_message.Message):
    __slots__ = ("items", "versions_order")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    VERSIONS_ORDER_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[VersionInfo]
    versions_order: VersionsOrder
    def __init__(self, items: _Optional[_Iterable[_Union[VersionInfo, _Mapping]]] = ..., versions_order: _Optional[_Union[VersionsOrder, str]] = ...) -> None: ...

class VersionInfo(_message.Message):
    __slots__ = ("resource_info", "sorting_key", "resource_address")
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    SORTING_KEY_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    resource_info: _fileobject_pb2.ResourceInfo
    sorting_key: str
    resource_address: str
    def __init__(self, resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ..., sorting_key: _Optional[str] = ..., resource_address: _Optional[str] = ...) -> None: ...
