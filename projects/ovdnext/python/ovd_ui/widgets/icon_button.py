# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small vector-authored controls using public OVUI image providers.

Icons stay independent of the text font's glyph coverage. The standalone
image loader cannot decode SVG; committed 4x PNGs come from the adjacent SVG
sources. Static icons share a RasterImageProvider; retained two-state icons
update a ByteImageProvider from cached PNG pixels without an asynchronous load.
"""
from functools import cache
from pathlib import Path

import omni.ui as ui

_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "icons"


@cache
def icon_url(name):
    path = _DIRECTORY / f"{name}.png"
    if not path.is_file():
        raise ValueError(f"Unknown workspace icon: {name}")
    return str(path)


@cache
def _provider(path):
    return ui.RasterImageProvider(path)


@cache
def _icon_pixels(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        pixels=np.array(source.convert("RGBA"),dtype="uint8")
    pixels.setflags(write=False)
    return pixels,[pixels.shape[1],pixels.shape[0]]


class IconButton:
    """Native button with a non-interactive themed image overlay."""
    def __init__(self, icon, *, text="", size=16, alternate_icon=None, **kwargs):
        # Tooltips may be assigned after construction (live status, row tags).
        # The public hover guard excludes widgets covered by another window.
        kwargs.setdefault("explicit_hover", True)
        self._image_url=icon_url(icon)
        self._alternate_url=icon_url(alternate_icon) if alternate_icon else None
        self._alternate=False
        # A TreeView caches child layout even when a nested Frame rebuilds.
        # Synchronous byte updates also avoid RasterImageProvider's URL-load
        # race when a tagged row is rebuilt or clicked again before loading.
        self._dynamic_provider=ui.ByteImageProvider() if alternate_icon else None
        if self._dynamic_provider is not None:
            self._dynamic_provider.set_data_array(*_icon_pixels(self._image_url))
        self._size=size
        self._name=kwargs.get("name", "")
        self.image=None
        self.label=None
        with ui.ZStack(width=kwargs.pop("width",ui.Fraction(1)),
                       height=kwargs.pop("height",ui.Fraction(1))):
            # The standalone Button's empty label collapses its hit rectangle
            # to padding only. A space retains the normal glyph-row baseline.
            self.button=ui.Button(" ",**kwargs)
            with ui.VStack(opaque_for_mouse_events=False):
                ui.Spacer()
                with ui.HStack(height=max(size,16),spacing=6 if text else 0,
                               opaque_for_mouse_events=False):
                    ui.Spacer()
                    self.image_frame=ui.Frame(width=size,height=max(size,16),build_fn=self._build_image,
                                              opaque_for_mouse_events=False)
                    if text:
                        self.label=ui.Label(text,name=self._name,width=len(text)*6,
                            style_type_name_override="Button.Label",opaque_for_mouse_events=False)
                    ui.Spacer()
                ui.Spacer()

    def _build_image(self):
        with ui.VStack(opaque_for_mouse_events=False):
            ui.Spacer()
            self.image=ui.ImageWithProvider(self._dynamic_provider or _provider(self._image_url),
                width=self._size,height=self._size,name=self._name,
                style_type_name_override="Button.Image",enabled=self.button.enabled,
                opaque_for_mouse_events=False)
            ui.Spacer()

    @property
    def alternate(self):return self._alternate

    @alternate.setter
    def alternate(self,value):
        if value and self._alternate_url is None:raise ValueError("No alternate icon configured")
        if bool(value)==self._alternate:return
        self._alternate=bool(value)
        if self._dynamic_provider is not None:
            self._dynamic_provider.set_data_array(*_icon_pixels(
                self._alternate_url if self._alternate else self._image_url))

    @property
    def image_url(self): return self._image_url

    @image_url.setter
    def image_url(self, path):
        if path!=self._image_url:
            self._image_url=path
            if self._dynamic_provider is None:self.image_frame.rebuild()
            elif not self._alternate:self._dynamic_provider.set_data_array(*_icon_pixels(path))

    @property
    def enabled(self): return self.button.enabled

    @enabled.setter
    def enabled(self, value):
        self.button.enabled=value
        if self.image is not None:self.image.enabled=value
        if self.label is not None:self.label.enabled=value

    @property
    def tooltip(self): return self.button.tooltip

    @tooltip.setter
    def tooltip(self, value): self.button.tooltip=value


icon_button=IconButton
