"""
gpu_display.py — Real-time OpenGL display window for ovrtx rendered frames.

Primary path (NVIDIA GPU with CUDA):
    ovrtx CUDA buffer → cudaGraphicsGLRegisterImage → GPU-to-GPU copy → GL present
    Zero CPU readback.

Fallback path (no CUDA interop available):
    ovrtx buffer → map to CPU (numpy) → glTexSubImage2D upload → GL present
    Still uses a native OS window instead of rerun, just with a PCIe roundtrip.

Dependencies:
    pip install glfw PyOpenGL numpy

Optional (for GPU-direct path):
    pip install cuda-python   (or cupy with CUDA toolkit)
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Optional

import glfw
import numpy as np
from OpenGL import GL

# Try to import CUDA interop support
try:
    from cuda import cuda as _cu
    from cuda import cudart as _cudart
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


@dataclass(frozen=True)
class NavigationInput:
    """Snapshot of first-person navigation input for one display frame."""

    forward: float = 0.0
    strafe: float = 0.0
    lift: float = 0.0
    look_dx: float = 0.0
    look_dy: float = 0.0
    fast: bool = False
    slow: bool = False
    look_active: bool = False


def _check_cuda(result):
    """Check CUDA driver/runtime API return codes."""
    if isinstance(result, tuple):
        err = result[0]
        if hasattr(err, 'value') and err.value != 0:
            raise RuntimeError(f"CUDA error: {err}")
        return result[1] if len(result) > 1 else None
    return result


class GPUDisplay:
    """OpenGL window that displays ovrtx-rendered frames with optional CUDA interop.

    Usage:
        display = GPUDisplay(width=1280, height=720, title="ov-fmi")
        # In render loop:
        if display.should_close():
            break
        display.present_frame(render_var_mapping)  # from ovrtx step
        display.poll_events()
        # Cleanup:
        display.destroy()
    """

    def __init__(self, width: int = 1280, height: int = 720, title: str = "ov-fmi"):
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")

        # Request OpenGL 3.3 core (widely supported)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)  # vsync

        self._width = width
        self._height = height
        self._tex_width = 0
        self._tex_height = 0

        # Create fullscreen quad VAO + shader
        self._shader = self._create_shader()
        self._vao = self._create_fullscreen_quad()
        self._texture = GL.glGenTextures(1)

        # Configure texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # CUDA interop state
        self._cuda_resource = None
        self._cuda_interop_active = False

        # Mouse tracking for first-person camera navigation.
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        self._have_mouse_pos = False
        self._mouse_captured = False

        # Window resize callback
        glfw.set_framebuffer_size_callback(self._window, self._framebuffer_resize)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos)

    def _framebuffer_resize(self, window, width, height):
        GL.glViewport(0, 0, width, height)
        self._width = width
        self._height = height

    def _cursor_pos(self, window, x, y):
        if not self._have_mouse_pos:
            self._mouse_x = float(x)
            self._mouse_y = float(y)
            self._have_mouse_pos = True
            return
        self._mouse_dx += float(x) - self._mouse_x
        self._mouse_dy += float(y) - self._mouse_y
        self._mouse_x = float(x)
        self._mouse_y = float(y)

    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    def poll_events(self):
        glfw.poll_events()

    def is_focused(self) -> bool:
        return bool(glfw.get_window_attrib(self._window, glfw.FOCUSED))

    def _is_key_down(self, key: int) -> bool:
        return glfw.get_key(self._window, key) == glfw.PRESS

    def _is_mouse_button_down(self, button: int) -> bool:
        return glfw.get_mouse_button(self._window, button) == glfw.PRESS

    def _consume_mouse_delta(self) -> tuple[float, float]:
        dx = self._mouse_dx
        dy = self._mouse_dy
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        return dx, dy

    def set_mouse_capture(self, captured: bool):
        """Capture/release the cursor for mouse-look."""
        if self._window is None or self._mouse_captured == captured:
            return

        self._mouse_captured = captured
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        self._have_mouse_pos = False
        mode = glfw.CURSOR_DISABLED if captured else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self._window, glfw.CURSOR, mode)

    def get_navigation_input(self) -> NavigationInput:
        """Return WASD/mouse-look input for this frame.

        Hold the right mouse button to capture the cursor and look around.
        Releasing the button, pressing Escape, or unfocusing the window releases
        the cursor. Mouse deltas are consumed by this call.
        """
        if self._window is None:
            return NavigationInput()

        if not self.is_focused():
            self.set_mouse_capture(False)
            self._consume_mouse_delta()
            return NavigationInput()

        if self._is_key_down(glfw.KEY_ESCAPE):
            self.set_mouse_capture(False)
            self._consume_mouse_delta()
            return NavigationInput()

        look_active = self._is_mouse_button_down(glfw.MOUSE_BUTTON_RIGHT)
        self.set_mouse_capture(look_active)
        dx, dy = self._consume_mouse_delta()
        if not look_active:
            dx = 0.0
            dy = 0.0

        forward = float(self._is_key_down(glfw.KEY_W)) - float(self._is_key_down(glfw.KEY_S))
        strafe = float(self._is_key_down(glfw.KEY_D)) - float(self._is_key_down(glfw.KEY_A))
        lift = float(self._is_key_down(glfw.KEY_E)) - float(self._is_key_down(glfw.KEY_Q))
        fast = (
            self._is_key_down(glfw.KEY_LEFT_SHIFT)
            or self._is_key_down(glfw.KEY_RIGHT_SHIFT)
        )
        slow = (
            self._is_key_down(glfw.KEY_LEFT_CONTROL)
            or self._is_key_down(glfw.KEY_RIGHT_CONTROL)
        )

        return NavigationInput(
            forward=forward,
            strafe=strafe,
            lift=lift,
            look_dx=dx,
            look_dy=dy,
            fast=fast,
            slow=slow,
            look_active=look_active,
        )

    def present_frame_cpu(self, pixels: np.ndarray):
        """Upload a CPU numpy array [H, W, C] (uint8 RGBA or RGB) to the GL texture and draw.

        This is the fallback path when CUDA interop is not available.
        """
        if pixels is None or pixels.size == 0:
            return

        h, w = pixels.shape[0], pixels.shape[1]
        channels = pixels.shape[2] if pixels.ndim == 3 else 1

        if channels == 4:
            fmt = GL.GL_RGBA
        elif channels == 3:
            fmt = GL.GL_RGB
        else:
            return  # unsupported format

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)

        # Reallocate texture if dimensions changed
        if w != self._tex_width or h != self._tex_height:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                w, h, 0, fmt, GL.GL_UNSIGNED_BYTE, pixels
            )
            self._tex_width = w
            self._tex_height = h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0,
                w, h, fmt, GL.GL_UNSIGNED_BYTE, pixels
            )

        self._draw_quad()

    def present_frame_cuda(self, cuda_ptr: int, width: int, height: int, channels: int = 4):
        """Copy a CUDA device pointer directly to the GL texture (zero CPU readback).

        cuda_ptr: device pointer to the rendered frame (uint8, RGBA, row-major)
        width, height: frame dimensions
        channels: 3 or 4
        """
        if not _HAS_CUDA:
            raise RuntimeError("cuda-python not available for GPU-direct display")

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)

        # Reallocate texture if size changed
        if width != self._tex_width or height != self._tex_height:
            fmt = GL.GL_RGBA if channels == 4 else GL.GL_RGB
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                width, height, 0, fmt, GL.GL_UNSIGNED_BYTE, None
            )
            self._tex_width = width
            self._tex_height = height

            # Re-register texture with CUDA
            if self._cuda_resource is not None:
                _check_cuda(_cudart.cudaGraphicsUnregisterResource(self._cuda_resource))
                self._cuda_resource = None

            self._cuda_resource = _check_cuda(
                _cudart.cudaGraphicsGLRegisterImage(
                    int(self._texture),
                    GL.GL_TEXTURE_2D,
                    _cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
                )
            )
            self._cuda_interop_active = True

        # Map the GL texture into CUDA, copy rendered frame, unmap
        _check_cuda(_cudart.cudaGraphicsMapResources(1, self._cuda_resource, None))
        try:
            array = _check_cuda(
                _cudart.cudaGraphicsSubResourceGetMappedArray(self._cuda_resource, 0, 0)
            )
            # cudaMemcpy2DToArray: dst_array, dst_x, dst_y, src_ptr, src_pitch, width_bytes, height, kind
            pitch = width * channels
            _check_cuda(_cudart.cudaMemcpy2DToArray(
                array, 0, 0,
                cuda_ptr, pitch,
                pitch, height,
                _cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            ))
        finally:
            _check_cuda(_cudart.cudaGraphicsUnmapResources(1, self._cuda_resource, None))

        self._draw_quad()

    def _draw_quad(self):
        """Draw the fullscreen textured quad and swap buffers."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self._shader)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        glfw.swap_buffers(self._window)

    def destroy(self):
        """Release all GL/CUDA resources and close the window."""
        if self._cuda_resource is not None and _HAS_CUDA:
            try:
                _check_cuda(_cudart.cudaGraphicsUnregisterResource(self._cuda_resource))
            except Exception:
                pass
            self._cuda_resource = None

        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
            self._texture = 0
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._shader:
            GL.glDeleteProgram(self._shader)
            self._shader = 0

        if self._window:
            self.set_mouse_capture(False)
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

    @staticmethod
    def _create_shader() -> int:
        """Create a minimal vertex+fragment shader for fullscreen textured quad."""
        vert_src = """
        #version 330 core
        const vec2 pos[4] = vec2[](
            vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1)
        );
        const vec2 uv[4] = vec2[](
            vec2(0, 1), vec2(1, 1), vec2(0, 0), vec2(1, 0)
        );
        out vec2 frag_uv;
        void main() {
            gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
            frag_uv = uv[gl_VertexID];
        }
        """
        frag_src = """
        #version 330 core
        in vec2 frag_uv;
        out vec4 color;
        uniform sampler2D tex;
        void main() {
            color = texture(tex, frag_uv);
        }
        """
        vert = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert, vert_src)
        GL.glCompileShader(vert)
        if not GL.glGetShaderiv(vert, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"Vertex shader error: {GL.glGetShaderInfoLog(vert).decode()}")

        frag = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag, frag_src)
        GL.glCompileShader(frag)
        if not GL.glGetShaderiv(frag, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"Fragment shader error: {GL.glGetShaderInfoLog(frag).decode()}")

        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vert)
        GL.glAttachShader(prog, frag)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(f"Shader link error: {GL.glGetProgramInfoLog(prog).decode()}")

        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)
        return prog

    @staticmethod
    def _create_fullscreen_quad() -> int:
        """Create a VAO for the fullscreen quad (vertex-less — uses gl_VertexID)."""
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        GL.glBindVertexArray(0)
        return vao


def is_cuda_interop_available() -> bool:
    """Check if CUDA↔OpenGL interop is available."""
    return _HAS_CUDA
