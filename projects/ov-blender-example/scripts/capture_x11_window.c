/* SPDX-License-Identifier: Apache-2.0
 * Capture an existing X11 drawable as RGB PPM, without altering its pixels.
 * Build: cc capture_x11_window.c -O2 -lX11 -o <output>/capture_x11_window
 * Pass a decimal/hex window id, or 0 for the root window. */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned char channel(unsigned long pixel, unsigned long mask) {
    if (!mask) return 0;
    while (!(mask & 1)) { mask >>= 1; pixel >>= 1; }
    return (unsigned char)(((pixel & mask) * 255) / mask);
}

int main(int argc, char **argv) {
    Display *display = XOpenDisplay(NULL);
    if (!display) { fprintf(stderr, "Cannot open display\n"); return 1; }
    Window window = argc > 1 ? strtoul(argv[1], NULL, 0) : 0;
    if (!window) window = DefaultRootWindow(display);
    XWindowAttributes attrs;
    if (!XGetWindowAttributes(display, window, &attrs)) return 2;
    XImage *image = XGetImage(display, window, 0, 0, attrs.width, attrs.height, AllPlanes, ZPixmap);
    if (!image) { fprintf(stderr, "Cannot capture drawable\n"); return 3; }
    fprintf(stdout, "P6\n%d %d\n255\n", attrs.width, attrs.height);
    unsigned char *row = malloc((size_t)attrs.width * 3);
    if (!row) return 4;
    for (int y = 0; y < attrs.height; ++y) {
        for (int x = 0; x < attrs.width; ++x) {
            unsigned long pixel = XGetPixel(image, x, y);
            row[x * 3] = channel(pixel, image->red_mask);
            row[x * 3 + 1] = channel(pixel, image->green_mask);
            row[x * 3 + 2] = channel(pixel, image->blue_mask);
        }
        if (fwrite(row, 3, attrs.width, stdout) != (size_t)attrs.width) return 5;
    }
    free(row);
    XDestroyImage(image);
    XCloseDisplay(display);
    return 0;
}
