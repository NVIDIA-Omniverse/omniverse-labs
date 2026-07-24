from nvidia.omniverse.storage.fileobject.v1alpha import fileobject_pb2 as _fileobject_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FolderMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FOLDER_MODE_UNSPECIFIED: _ClassVar[FolderMode]
    FOLDER_MODE_NATIVE: _ClassVar[FolderMode]
    FOLDER_MODE_NO_EMPTY: _ClassVar[FolderMode]
    FOLDER_MODE_HYBRID: _ClassVar[FolderMode]
FOLDER_MODE_UNSPECIFIED: FolderMode
FOLDER_MODE_NATIVE: FolderMode
FOLDER_MODE_NO_EMPTY: FolderMode
FOLDER_MODE_HYBRID: FolderMode

class ListRequest(_message.Message):
    __slots__ = ("folder",)
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    folder: FolderAddress
    def __init__(self, folder: _Optional[_Union[FolderAddress, _Mapping]] = ...) -> None: ...

class ListResponse(_message.Message):
    __slots__ = ("subfolder_addresses", "sub_resource_addresses")
    SUBFOLDER_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    SUB_RESOURCE_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    subfolder_addresses: _containers.RepeatedCompositeFieldContainer[FolderAddress]
    sub_resource_addresses: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, subfolder_addresses: _Optional[_Iterable[_Union[FolderAddress, _Mapping]]] = ..., sub_resource_addresses: _Optional[_Iterable[str]] = ...) -> None: ...

class ListStatRequest(_message.Message):
    __slots__ = ("folder",)
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    folder: FolderAddress
    def __init__(self, folder: _Optional[_Union[FolderAddress, _Mapping]] = ...) -> None: ...

class ListStatResponse(_message.Message):
    __slots__ = ("subfolder_addresses", "entries")
    SUBFOLDER_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    subfolder_addresses: _containers.RepeatedCompositeFieldContainer[FolderAddress]
    entries: _containers.RepeatedCompositeFieldContainer[ListItem]
    def __init__(self, subfolder_addresses: _Optional[_Iterable[_Union[FolderAddress, _Mapping]]] = ..., entries: _Optional[_Iterable[_Union[ListItem, _Mapping]]] = ...) -> None: ...

class DeleteFolderRequest(_message.Message):
    __slots__ = ("folder",)
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    folder: FolderAddress
    def __init__(self, folder: _Optional[_Union[FolderAddress, _Mapping]] = ...) -> None: ...

class DeleteFolderResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CreateFolderRequest(_message.Message):
    __slots__ = ("folder",)
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    folder: FolderAddress
    def __init__(self, folder: _Optional[_Union[FolderAddress, _Mapping]] = ...) -> None: ...

class CreateFolderResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListItem(_message.Message):
    __slots__ = ("resource_address", "resource_info")
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    resource_info: _fileobject_pb2.ResourceInfo
    def __init__(self, resource_address: _Optional[str] = ..., resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ...) -> None: ...

class GetFolderModeRequest(_message.Message):
    __slots__ = ("folder",)
    FOLDER_FIELD_NUMBER: _ClassVar[int]
    folder: FolderAddress
    def __init__(self, folder: _Optional[_Union[FolderAddress, _Mapping]] = ...) -> None: ...

class GetFolderModeResponse(_message.Message):
    __slots__ = ("folder_mode",)
    FOLDER_MODE_FIELD_NUMBER: _ClassVar[int]
    folder_mode: FolderMode
    def __init__(self, folder_mode: _Optional[_Union[FolderMode, str]] = ...) -> None: ...

class FolderAddress(_message.Message):
    __slots__ = ("uri",)
    URI_FIELD_NUMBER: _ClassVar[int]
    uri: str
    def __init__(self, uri: _Optional[str] = ...) -> None: ...
