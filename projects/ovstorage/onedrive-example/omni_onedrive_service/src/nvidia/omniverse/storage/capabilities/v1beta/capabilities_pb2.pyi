from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ListServicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListServicesResponse(_message.Message):
    __slots__ = ("services",)
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    services: _containers.RepeatedCompositeFieldContainer[ServiceEntry]
    def __init__(self, services: _Optional[_Iterable[_Union[ServiceEntry, _Mapping]]] = ...) -> None: ...

class ServiceEntry(_message.Message):
    __slots__ = ("service_name", "service_versions")
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    SERVICE_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    service_versions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, service_name: _Optional[str] = ..., service_versions: _Optional[_Iterable[str]] = ...) -> None: ...

class ListTopLevelAddressesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListTopLevelAddressesResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[TopLevelAddressEntry]
    def __init__(self, items: _Optional[_Iterable[_Union[TopLevelAddressEntry, _Mapping]]] = ...) -> None: ...

class TopLevelAddressEntry(_message.Message):
    __slots__ = ("top_level_address",)
    TOP_LEVEL_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    top_level_address: str
    def __init__(self, top_level_address: _Optional[str] = ...) -> None: ...
