from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetMetadataRequest(_message.Message):
    __slots__ = ("uri", "user_metadata_keys")
    URI_FIELD_NUMBER: _ClassVar[int]
    USER_METADATA_KEYS_FIELD_NUMBER: _ClassVar[int]
    uri: str
    user_metadata_keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, uri: _Optional[str] = ..., user_metadata_keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetMetadataResponse(_message.Message):
    __slots__ = ("user_metadata",)
    class UserMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: UserMetadataValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[UserMetadataValue, _Mapping]] = ...) -> None: ...
    USER_METADATA_FIELD_NUMBER: _ClassVar[int]
    user_metadata: _containers.MessageMap[str, UserMetadataValue]
    def __init__(self, user_metadata: _Optional[_Mapping[str, UserMetadataValue]] = ...) -> None: ...

class UpdateMetadataRequest(_message.Message):
    __slots__ = ("uri", "user_metadata_key", "user_metadata", "expected_etag")
    URI_FIELD_NUMBER: _ClassVar[int]
    USER_METADATA_KEY_FIELD_NUMBER: _ClassVar[int]
    USER_METADATA_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ETAG_FIELD_NUMBER: _ClassVar[int]
    uri: str
    user_metadata_key: str
    user_metadata: _struct_pb2.Value
    expected_etag: str
    def __init__(self, uri: _Optional[str] = ..., user_metadata_key: _Optional[str] = ..., user_metadata: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., expected_etag: _Optional[str] = ...) -> None: ...

class UpdateMetadataResponse(_message.Message):
    __slots__ = ("etag",)
    ETAG_FIELD_NUMBER: _ClassVar[int]
    etag: str
    def __init__(self, etag: _Optional[str] = ...) -> None: ...

class DeleteMetadataRequest(_message.Message):
    __slots__ = ("uri", "user_metadata_key", "expected_etag")
    URI_FIELD_NUMBER: _ClassVar[int]
    USER_METADATA_KEY_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ETAG_FIELD_NUMBER: _ClassVar[int]
    uri: str
    user_metadata_key: str
    expected_etag: str
    def __init__(self, uri: _Optional[str] = ..., user_metadata_key: _Optional[str] = ..., expected_etag: _Optional[str] = ...) -> None: ...

class DeleteMetadataResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class UserMetadataValue(_message.Message):
    __slots__ = ("value", "etag")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    ETAG_FIELD_NUMBER: _ClassVar[int]
    value: _struct_pb2.Value
    etag: str
    def __init__(self, value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., etag: _Optional[str] = ...) -> None: ...
