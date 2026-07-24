from nvidia.omniverse.storage.fileobject.v1alpha import fileobject_pb2 as _fileobject_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DownloadPreference(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DOWNLOAD_PREFERENCE_UNSPECIFIED: _ClassVar[DownloadPreference]
    DOWNLOAD_PREFERENCE_BODY: _ClassVar[DownloadPreference]
    DOWNLOAD_PREFERENCE_REDIRECT: _ClassVar[DownloadPreference]

class UploadPreference(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UPLOAD_PREFERENCE_UNSPECIFIED: _ClassVar[UploadPreference]
    UPLOAD_PREFERENCE_BODY: _ClassVar[UploadPreference]
    UPLOAD_PREFERENCE_REDIRECT: _ClassVar[UploadPreference]
    UPLOAD_PREFERENCE_MULTIPART: _ClassVar[UploadPreference]

class UploadMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UPLOAD_METHOD_UNSPECIFIED: _ClassVar[UploadMethod]
    UPLOAD_METHOD_POST: _ClassVar[UploadMethod]
    UPLOAD_METHOD_PUT: _ClassVar[UploadMethod]
DOWNLOAD_PREFERENCE_UNSPECIFIED: DownloadPreference
DOWNLOAD_PREFERENCE_BODY: DownloadPreference
DOWNLOAD_PREFERENCE_REDIRECT: DownloadPreference
UPLOAD_PREFERENCE_UNSPECIFIED: UploadPreference
UPLOAD_PREFERENCE_BODY: UploadPreference
UPLOAD_PREFERENCE_REDIRECT: UploadPreference
UPLOAD_PREFERENCE_MULTIPART: UploadPreference
UPLOAD_METHOD_UNSPECIFIED: UploadMethod
UPLOAD_METHOD_POST: UploadMethod
UPLOAD_METHOD_PUT: UploadMethod

class EnumerateRequest(_message.Message):
    __slots__ = ("resource_address",)
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    def __init__(self, resource_address: _Optional[str] = ...) -> None: ...

class EnumerateResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[_fileobject_pb2.AddressInfo]
    def __init__(self, items: _Optional[_Iterable[_Union[_fileobject_pb2.AddressInfo, _Mapping]]] = ...) -> None: ...

class StatRequest(_message.Message):
    __slots__ = ("resource_address",)
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    def __init__(self, resource_address: _Optional[str] = ...) -> None: ...

class StatResponse(_message.Message):
    __slots__ = ("resource_info",)
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    resource_info: _fileobject_pb2.ResourceInfo
    def __init__(self, resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ...) -> None: ...

class ReadRequest(_message.Message):
    __slots__ = ("resource_identity", "download_preference")
    RESOURCE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_PREFERENCE_FIELD_NUMBER: _ClassVar[int]
    resource_identity: _fileobject_pb2.ResourceIdentity
    download_preference: DownloadPreference
    def __init__(self, resource_identity: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ..., download_preference: _Optional[_Union[DownloadPreference, str]] = ...) -> None: ...

class ReadResponse(_message.Message):
    __slots__ = ("metadata", "chunk", "redirect")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    REDIRECT_FIELD_NUMBER: _ClassVar[int]
    metadata: _fileobject_pb2.Metadata
    chunk: _fileobject_pb2.Chunk
    redirect: _fileobject_pb2.Redirect
    def __init__(self, metadata: _Optional[_Union[_fileobject_pb2.Metadata, _Mapping]] = ..., chunk: _Optional[_Union[_fileobject_pb2.Chunk, _Mapping]] = ..., redirect: _Optional[_Union[_fileobject_pb2.Redirect, _Mapping]] = ...) -> None: ...

class ReadFromAddressRequest(_message.Message):
    __slots__ = ("resource_address", "download_preference")
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DOWNLOAD_PREFERENCE_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    download_preference: DownloadPreference
    def __init__(self, resource_address: _Optional[str] = ..., download_preference: _Optional[_Union[DownloadPreference, str]] = ...) -> None: ...

class ReadFromAddressResponse(_message.Message):
    __slots__ = ("resource_info", "chunk", "redirect")
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    REDIRECT_FIELD_NUMBER: _ClassVar[int]
    resource_info: _fileobject_pb2.ResourceInfo
    chunk: _fileobject_pb2.Chunk
    redirect: _fileobject_pb2.Redirect
    def __init__(self, resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ..., chunk: _Optional[_Union[_fileobject_pb2.Chunk, _Mapping]] = ..., redirect: _Optional[_Union[_fileobject_pb2.Redirect, _Mapping]] = ...) -> None: ...

class FetchWriteTypeInfoRequest(_message.Message):
    __slots__ = ("destination_resource_address",)
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    destination_resource_address: str
    def __init__(self, destination_resource_address: _Optional[str] = ...) -> None: ...

class WriteTypeForSizeInterval(_message.Message):
    __slots__ = ("minimum_data_object_size", "maximum_data_object_size", "preferred_upload_method")
    MINIMUM_DATA_OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_DATA_OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    PREFERRED_UPLOAD_METHOD_FIELD_NUMBER: _ClassVar[int]
    minimum_data_object_size: int
    maximum_data_object_size: int
    preferred_upload_method: UploadPreference
    def __init__(self, minimum_data_object_size: _Optional[int] = ..., maximum_data_object_size: _Optional[int] = ..., preferred_upload_method: _Optional[_Union[UploadPreference, str]] = ...) -> None: ...

class FetchWriteTypeInfoResponse(_message.Message):
    __slots__ = ("write_type_intervals",)
    WRITE_TYPE_INTERVALS_FIELD_NUMBER: _ClassVar[int]
    write_type_intervals: _containers.RepeatedCompositeFieldContainer[WriteTypeForSizeInterval]
    def __init__(self, write_type_intervals: _Optional[_Iterable[_Union[WriteTypeForSizeInterval, _Mapping]]] = ...) -> None: ...

class WriteRequest(_message.Message):
    __slots__ = ("params", "chunk")
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    params: WriteParameters
    chunk: _fileobject_pb2.Chunk
    def __init__(self, params: _Optional[_Union[WriteParameters, _Mapping]] = ..., chunk: _Optional[_Union[_fileobject_pb2.Chunk, _Mapping]] = ...) -> None: ...

class WriteParameters(_message.Message):
    __slots__ = ("destination_resource_address", "previous_version", "data_object_size", "upload_preference")
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    DATA_OBJECT_SIZE_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_PREFERENCE_FIELD_NUMBER: _ClassVar[int]
    destination_resource_address: str
    previous_version: _fileobject_pb2.ResourceIdentity
    data_object_size: int
    upload_preference: UploadPreference
    def __init__(self, destination_resource_address: _Optional[str] = ..., previous_version: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ..., data_object_size: _Optional[int] = ..., upload_preference: _Optional[_Union[UploadPreference, str]] = ...) -> None: ...

class WriteRedirectProperties(_message.Message):
    __slots__ = ("redirect_target_url", "method", "additional_headers", "completion_header_names")
    REDIRECT_TARGET_URL_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_HEADERS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_HEADER_NAMES_FIELD_NUMBER: _ClassVar[int]
    redirect_target_url: str
    method: UploadMethod
    additional_headers: _containers.RepeatedCompositeFieldContainer[_fileobject_pb2.Header]
    completion_header_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, redirect_target_url: _Optional[str] = ..., method: _Optional[_Union[UploadMethod, str]] = ..., additional_headers: _Optional[_Iterable[_Union[_fileobject_pb2.Header, _Mapping]]] = ..., completion_header_names: _Optional[_Iterable[str]] = ...) -> None: ...

class WriteChunksAccepted(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WriteResponse(_message.Message):
    __slots__ = ("write_chunks_accepted", "resource_info", "write_redirect", "multipart_upload")
    WRITE_CHUNKS_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    WRITE_REDIRECT_FIELD_NUMBER: _ClassVar[int]
    MULTIPART_UPLOAD_FIELD_NUMBER: _ClassVar[int]
    write_chunks_accepted: WriteChunksAccepted
    resource_info: _fileobject_pb2.ResourceInfo
    write_redirect: WriteRedirectProperties
    multipart_upload: CreateMultipartUploadResponse
    def __init__(self, write_chunks_accepted: _Optional[_Union[WriteChunksAccepted, _Mapping]] = ..., resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ..., write_redirect: _Optional[_Union[WriteRedirectProperties, _Mapping]] = ..., multipart_upload: _Optional[_Union[CreateMultipartUploadResponse, _Mapping]] = ...) -> None: ...

class CompleteRedirectUploadRequest(_message.Message):
    __slots__ = ("destination_resource_address", "additional_headers")
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_HEADERS_FIELD_NUMBER: _ClassVar[int]
    destination_resource_address: str
    additional_headers: _containers.RepeatedCompositeFieldContainer[_fileobject_pb2.Header]
    def __init__(self, destination_resource_address: _Optional[str] = ..., additional_headers: _Optional[_Iterable[_Union[_fileobject_pb2.Header, _Mapping]]] = ...) -> None: ...

class CompleteRedirectUploadResponse(_message.Message):
    __slots__ = ("resource_info",)
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    resource_info: _fileobject_pb2.ResourceInfo
    def __init__(self, resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ...) -> None: ...

class CreateMultipartUploadResponse(_message.Message):
    __slots__ = ("upload_id", "first_part_write_redirect", "maximum_parts_number", "minimum_size_per_part", "maximum_size_per_part")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    FIRST_PART_WRITE_REDIRECT_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_PARTS_NUMBER_FIELD_NUMBER: _ClassVar[int]
    MINIMUM_SIZE_PER_PART_FIELD_NUMBER: _ClassVar[int]
    MAXIMUM_SIZE_PER_PART_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    first_part_write_redirect: WriteRedirectProperties
    maximum_parts_number: int
    minimum_size_per_part: int
    maximum_size_per_part: int
    def __init__(self, upload_id: _Optional[str] = ..., first_part_write_redirect: _Optional[_Union[WriteRedirectProperties, _Mapping]] = ..., maximum_parts_number: _Optional[int] = ..., minimum_size_per_part: _Optional[int] = ..., maximum_size_per_part: _Optional[int] = ...) -> None: ...

class UploadPartRequest(_message.Message):
    __slots__ = ("upload_id", "destination_resource_address", "part_number", "part_count")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    PART_COUNT_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    destination_resource_address: str
    part_number: int
    part_count: int
    def __init__(self, upload_id: _Optional[str] = ..., destination_resource_address: _Optional[str] = ..., part_number: _Optional[int] = ..., part_count: _Optional[int] = ...) -> None: ...

class UploadPartResponse(_message.Message):
    __slots__ = ("part_write_redirects",)
    PART_WRITE_REDIRECTS_FIELD_NUMBER: _ClassVar[int]
    part_write_redirects: _containers.RepeatedCompositeFieldContainer[WriteRedirectProperties]
    def __init__(self, part_write_redirects: _Optional[_Iterable[_Union[WriteRedirectProperties, _Mapping]]] = ...) -> None: ...

class CompletedUploadPart(_message.Message):
    __slots__ = ("part_number", "headers")
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    HEADERS_FIELD_NUMBER: _ClassVar[int]
    part_number: int
    headers: _containers.RepeatedCompositeFieldContainer[_fileobject_pb2.Header]
    def __init__(self, part_number: _Optional[int] = ..., headers: _Optional[_Iterable[_Union[_fileobject_pb2.Header, _Mapping]]] = ...) -> None: ...

class CompleteMultipartUploadRequest(_message.Message):
    __slots__ = ("upload_id", "destination_resource_address", "parts")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PARTS_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    destination_resource_address: str
    parts: _containers.RepeatedCompositeFieldContainer[CompletedUploadPart]
    def __init__(self, upload_id: _Optional[str] = ..., destination_resource_address: _Optional[str] = ..., parts: _Optional[_Iterable[_Union[CompletedUploadPart, _Mapping]]] = ...) -> None: ...

class CompleteMultipartUploadResponse(_message.Message):
    __slots__ = ("resource_info",)
    RESOURCE_INFO_FIELD_NUMBER: _ClassVar[int]
    resource_info: _fileobject_pb2.ResourceInfo
    def __init__(self, resource_info: _Optional[_Union[_fileobject_pb2.ResourceInfo, _Mapping]] = ...) -> None: ...

class AbortMultipartUploadRequest(_message.Message):
    __slots__ = ("upload_id", "destination_resource_address")
    UPLOAD_ID_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    upload_id: str
    destination_resource_address: str
    def __init__(self, upload_id: _Optional[str] = ..., destination_resource_address: _Optional[str] = ...) -> None: ...

class AbortMultipartUploadResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("resource_address", "previous_version")
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    previous_version: _fileobject_pb2.ResourceIdentity
    def __init__(self, resource_address: _Optional[str] = ..., previous_version: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CopyRequest(_message.Message):
    __slots__ = ("source_resource_identity", "destination_resource_address", "previous_version")
    SOURCE_RESOURCE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    source_resource_identity: _fileobject_pb2.ResourceIdentity
    destination_resource_address: str
    previous_version: _fileobject_pb2.ResourceIdentity
    def __init__(self, source_resource_identity: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ..., destination_resource_address: _Optional[str] = ..., previous_version: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ...) -> None: ...

class CopyResponse(_message.Message):
    __slots__ = ("resource_identity",)
    RESOURCE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    resource_identity: _fileobject_pb2.ResourceIdentity
    def __init__(self, resource_identity: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ...) -> None: ...

class MoveRequest(_message.Message):
    __slots__ = ("source_resource_address", "source_previous_version", "destination_resource_address", "destination_previous_version")
    SOURCE_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_PREVIOUS_VERSION_FIELD_NUMBER: _ClassVar[int]
    source_resource_address: str
    source_previous_version: _fileobject_pb2.ResourceIdentity
    destination_resource_address: str
    destination_previous_version: _fileobject_pb2.ResourceIdentity
    def __init__(self, source_resource_address: _Optional[str] = ..., source_previous_version: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ..., destination_resource_address: _Optional[str] = ..., destination_previous_version: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ...) -> None: ...

class MoveResponse(_message.Message):
    __slots__ = ("resource_identity",)
    RESOURCE_IDENTITY_FIELD_NUMBER: _ClassVar[int]
    resource_identity: _fileobject_pb2.ResourceIdentity
    def __init__(self, resource_identity: _Optional[_Union[_fileobject_pb2.ResourceIdentity, _Mapping]] = ...) -> None: ...

class GetOptimisticLockingSupportRequest(_message.Message):
    __slots__ = ("resource_address",)
    RESOURCE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    resource_address: str
    def __init__(self, resource_address: _Optional[str] = ...) -> None: ...

class GetOptimisticLockingSupportResponse(_message.Message):
    __slots__ = ("supports_write", "supports_delete", "supports_copy", "supports_move")
    SUPPORTS_WRITE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_DELETE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_COPY_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_MOVE_FIELD_NUMBER: _ClassVar[int]
    supports_write: bool
    supports_delete: bool
    supports_copy: bool
    supports_move: bool
    def __init__(self, supports_write: bool = ..., supports_delete: bool = ..., supports_copy: bool = ..., supports_move: bool = ...) -> None: ...
