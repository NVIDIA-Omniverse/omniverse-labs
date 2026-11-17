# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned-curve interaction for the original PvdPlot/Qwt gesture contract.

Left-drag pans x/y; right-drag stores a rectangle on the zoom stack; Ctrl+Right
and Home return to its base; Shift+O/I traverse the stack. QwtMagnifier wheel
up expands the axes by 1/0.9 around their center (its unused subclass inverted
this, but PvdPlot instantiates the base). +/- use 0.9 and its reciprocal.
Pan/magnification do not push rectangle history, matching separate Qwt tools.
Hover samples the nearest recorded frame; only double-click navigates playback.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Curve:
    name: str
    values: tuple[float | None, ...]
    unit: str = ""


@dataclass(frozen=True)
class PlotRange:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self):return self.right-self.left
    @property
    def height(self):return self.top-self.bottom


class PlotModel:
    """UI-neutral zoom stack and exact measurement cursor; no capture parsing."""
    def __init__(self, frames, curves):
        self.frames = tuple(frames)
        self.curves = tuple(curves)
        if not self.frames or any(b <= a for a,b in zip(self.frames,self.frames[1:])):
            raise ValueError("Graph frames must be nonempty and strictly increasing")
        if not self.curves or any(len(c.values) != len(self.frames) for c in self.curves):
            raise ValueError("Each curve must have one entry for every recorded frame")
        finite = [v for c in self.curves for v in c.values if v is not None and math.isfinite(v)]
        self.has_values = bool(finite)
        low, high = (min(finite), max(finite)) if finite else (0.,1.)
        padding = max((high-low)*.05,abs(high)*.01,1e-6)
        left, right = float(self.frames[0]), float(self.frames[-1])
        if left == right:left -= .5;right += .5
        self.base = PlotRange(left,right,low-padding,high+padding)
        self.view = self.base
        self.history = [self.base]
        self.history_index = 0
        self.cursor_frame = None

    def _valid(self, bounds):
        values = bounds.left,bounds.right,bounds.bottom,bounds.top
        return (all(math.isfinite(v) and abs(v)<1e35 for v in values)
                and bounds.width >= max(self.base.width*1e-9,1e-6)
                and bounds.height >= max(self.base.height*1e-9,1e-12))

    def set_view(self, bounds, *, record=False):
        if not self._valid(bounds):return False
        if record and bounds != self.history[self.history_index]:
            self.history = self.history[:self.history_index+1]+[bounds]
            # Bounded UI history; keep the original base at index zero.
            if len(self.history)>256:self.history.pop(1)
            self.history_index = len(self.history)-1
        changed = bounds != self.view
        self.view = bounds
        return changed

    def home(self):
        self.history_index = 0
        return self.set_view(self.base)

    def undo(self):
        self.history_index = max(0,self.history_index-1)
        return self.set_view(self.history[self.history_index])

    def redo(self):
        self.history_index = min(len(self.history)-1,self.history_index+1)
        return self.set_view(self.history[self.history_index])

    def magnify(self, factor):
        if not math.isfinite(factor) or factor <= 0:return False
        r = self.view
        x,y = (r.left+r.right)*.5,(r.bottom+r.top)*.5
        return self.set_view(PlotRange(x-r.width*factor*.5,x+r.width*factor*.5,
                                      y-r.height*factor*.5,y+r.height*factor*.5))

    def wheel(self, delta):
        return self.magnify(.9**(-max(-30.,min(30.,delta))))

    def pan(self, dx, dy, *, origin=None):
        r = origin or self.view
        return self.set_view(PlotRange(r.left-dx*r.width,r.right-dx*r.width,
                                      r.bottom+dy*r.height,r.top+dy*r.height))

    def rectangle(self, x1,y1,x2,y2, *, origin=None):
        r = origin or self.view
        left,right = sorted((r.left+x1*r.width,r.left+x2*r.width))
        bottom,top = sorted((r.top-y1*r.height,r.top-y2*r.height))
        return self.set_view(PlotRange(left,right,bottom,top),record=True)

    def frame_range(self, first,last):
        if not math.isfinite(first) or not math.isfinite(last) or first>last:
            raise ValueError("Choose a frame range whose start is before its end")
        first,last = max(self.frames[0],first),min(self.frames[-1],last)
        if first>last:raise ValueError("The range contains no recorded frames")
        if first==last:first-=.5;last+=.5
        return self.set_view(PlotRange(first,last,self.view.bottom,self.view.top),record=True)

    def point(self, x_fraction,y_fraction):
        return self.view.left+x_fraction*self.view.width, self.view.top-y_fraction*self.view.height

    def hover(self, x_fraction):
        position = self.view.left+x_fraction*self.view.width
        if position<self.frames[0] or position>self.frames[-1]:
            self.cursor_frame = None
            return None
        index = bisect_left(self.frames,position)
        if index and (index == len(self.frames) or position-self.frames[index-1]<self.frames[index]-position):index-=1
        self.cursor_frame = self.frames[index]
        return tuple((c.name,c.values[index],c.unit) for c in self.curves)

    def cursor_values(self):
        if self.cursor_frame is None:return ()
        i=bisect_left(self.frames,self.cursor_frame)
        return tuple((c.name,c.values[i],c.unit) for c in self.curves)

    def display_paths(self, curve, *, limit=4096):
        """Clipped polylines in normalized chart coordinates (y increases up).

        Keep original frame/value positions and exact edge intersections. When
        more than ``limit`` samples are visible, each bucket keeps its first,
        last, minimum and maximum sample in temporal order; a bucket containing
        missing values remains a gap. At most 4*limit vertices enter clipping.
        This avoids native Plot's off-axis clamping and interpolated spike loss.
        Single recorded samples return singleton paths for point markers.
        """
        if limit<2:raise ValueError("Graph display limit must be at least two")
        r=self.view
        first=max(0,bisect_left(self.frames,r.left)-1)
        last=min(len(self.frames),bisect_right(self.frames,r.right)+1)
        count=last-first
        indices=[]
        if count<=limit:indices=list(range(first,last))
        else:
            for bucket in range(limit):
                begin=first+count*bucket//limit
                end=first+count*(bucket+1)//limit
                values=curve.values[begin:end]
                if any(v is None or not math.isfinite(v) for v in values):
                    indices.append(None)
                    continue
                indices.extend(sorted({begin,end-1,begin+values.index(min(values)),begin+values.index(max(values))}))
        paths=[]
        current=[]
        previous=None
        for index in indices+[None]:
            value=curve.values[index] if index is not None else None
            if value is None or not math.isfinite(value):
                if current:paths.append(tuple(current));current=[]
                previous=None
                continue
            point=((self.frames[index]-r.left)/r.width,(value-r.bottom)/r.height)
            if previous is None:
                if 0<=point[0]<=1 and 0<=point[1]<=1:current=[point]
            else:
                clipped=_clip_segment(previous,point)
                if clipped:
                    a,b=clipped
                    if current and current[-1]!=a:paths.append(tuple(current));current=[]
                    if not current:current.append(a)
                    if current[-1]!=b:current.append(b)
                elif current:paths.append(tuple(current));current=[]
            previous=point
        return tuple(paths)


def _clip_segment(a,b):
    """Liang–Barsky clipping to the unit square, preserving true intersections."""
    dx,dy=b[0]-a[0],b[1]-a[1]
    low,high=0.,1.
    for p,q in ((-dx,a[0]),(dx,1-a[0]),(-dy,a[1]),(dy,1-a[1])):
        if p==0:
            if q<0:return None
        elif p<0:low=max(low,q/p)
        else:high=min(high,q/p)
        if low>high:return None
    return ((a[0]+low*dx,a[1]+low*dy),(a[0]+high*dx,a[1]+high*dy))
