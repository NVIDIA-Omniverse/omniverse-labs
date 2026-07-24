from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Chunk(_message.Message):
    __slots__ = ("chunk",)
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    chunk: bytes
    def __init__(self, chunk: _Optional[bytes] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ("data_object_size", "last_modified_timestamp")
    DATA_OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    LAST_MODIFIED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    data_object_size: int
    last_modified_timestamp: _timestamp_pb2.Timestamp
    def __init__(self, data_object_size: _Optional[int] = ..., last_modified_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Header(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Redirect(_message.Message):
    __slots__ = ("redirect_target_url", "method", "additional_headers")
    REDIRECT_TARGET_URL_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_HEADERS_FIELD_NUMBER: _ClassVar[int]
    redirect_target_url: str
    method: str
    additional_headers: _containers.RepeatedCompositeFieldContainer[Header]
    def __init__(self, redirect_target_url: _Optional[str] = ..., method: _Optional[str] = ..., additional_headers: _Optional[_Iterable[_Union[Header, _Mapping]]] = ...) -> None: ...

class AddressInfo(_message.Message):
    __slots__ = ("resource_address", "metadata")
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    metadata: Metadata
    def __init__(self, resource_address: _Optional[str] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class ResourceInfo(_message.Message):
    __slots__ = ("resource_identity", "metadata")
    RESOURCE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    resource_identity: ResourceIdentity
    metadata: Metadata
    def __init__(self, resource_identity: _Optional[_Union[ResourceIdentity, _Mapping]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class ResourceIdentity(_message.Message):
    __slots__ = ("encoded_identity",)
    ENCODED_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    encoded_identity: str
    def __init__(self, encoded_identity: _Optional[str] = ...) -> None: ...
