# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched GPU drawing of owned native debug primitives."""
import moderngl
import numpy as np

from .render_settings import DisplayMode

# Categories published by the native OverlaySnapshot, with Qt actor-ID gates.
SELECTION_OVERLAYS = ((1, "Bounds"), (2, "Actor axes"),
                      (4, "Linear velocity"), (8, "Angular velocity"), (16, "Joint frames"),
                      (32, "Contacts"), (256, "Center of mass"))
# Actor axes, velocities, joint frames, and scene-query vectors. Legacy contact
# normals also use Arrow primitives but must retain contact-pass depth testing.
ON_TOP_ARROW_CATEGORIES = (2, 4, 8, 16, 64)


def overlay_visibility(snapshot, scene_id, hidden=(), *, selected=(), modes=None):
    """Shared visible/picking predicate over one immutable native publication."""
    mask=np.ones(len(snapshot.kinds),dtype=bool)
    if scene_id is not None:
        mask &= (snapshot.scene_ids==scene_id)|(snapshot.scene_ids<=0)
    if hidden:
        mask &= ~np.isin(snapshot.object_ids,list(hidden))
        # Preserve the original contact/shape relationship visibility rules.
        related_hidden=np.isin(snapshot.related_ids,list(hidden))|(snapshot.related_ids<=0)
        mask &= ~(related_hidden.all(axis=1)&(snapshot.related_ids>0).any(axis=1))
        # A joint frame belongs to both actors; hiding either hides its draw
        # and pick rows. Contacts retain their established pair visibility.
        mask &= ~((snapshot.categories==16)&np.isin(snapshot.related_ids,list(hidden)).any(axis=1))
    selected_rows=None
    for category,_ in SELECTION_OVERLAYS:
        mode=(modes or {}).get(category,DisplayMode.ALL)
        category_rows=snapshot.categories==category
        if mode==DisplayMode.NONE:
            mask &= ~category_rows
        elif mode in (DisplayMode.SELECTED,DisplayMode.UNSELECTED):
            if selected_rows is None:
                selected_rows=np.isin(snapshot.object_ids,list(selected))
            rows=selected_rows
            if category in (16,32):  # Joint frames also match the joint itself.
                related=((snapshot.related_ids>0)&np.isin(snapshot.related_ids,list(selected))).any(axis=1)
                rows=rows|related if category==16 else related
            keep=rows if mode==DisplayMode.SELECTED else ~rows
            if category==32:
                keep |= (snapshot.related_ids==0).all(axis=1)  # No recorded pair: leave visible.
            mask &= ~category_rows | keep
    return mask

VERTEX="""#version 330
uniform mat4 view_projection;
in vec3 position;
in vec4 tint;
in uint instance_id;
out vec4 vertex_color;
flat out uint object_id;
void main() {
    gl_Position=view_projection*vec4(position,1.0);
    gl_Position.y*=-1.0;
    gl_Position.z-=gl_Position.w*1e-6;
    vertex_color=tint;
    object_id=instance_id;
}
"""
FRAGMENT="""#version 330
uniform bool picking;
in vec4 vertex_color;
flat in uint object_id;
out vec4 color;
void main() {
    if(picking) {
        if(object_id==0u || object_id==0xffffffffu) discard;
        uint id=object_id+1u;
        color=vec4(float(id&255u),float((id>>8u)&255u),float((id>>16u)&255u),float((id>>24u)&255u))/255.0;
    } else color=vertex_color;
}
"""
VERTEX_DTYPE=np.dtype([("position","f4",3),("tint","f4",4),("id","u4")])


class OverlayRenderer:
    def __init__(self,context):
        self.context=context
        self.program=context.program(vertex_shader=VERTEX,fragment_shader=FRAGMENT)
        self.buffers={}
        self.snapshot=None
        self.draw_count=0
        self.primitive_count=0
        self.palette=None
        self.modes={}

    def set_mode(self,category,mode):
        if category not in dict(SELECTION_OVERLAYS) or isinstance(mode,bool):
            raise ValueError("Unsupported overlay category or display mode")
        self.modes[category]=DisplayMode(mode)

    def set_snapshot(self,snapshot):
        self.snapshot=snapshot

    def _upload(self,cache_key,positions,colors,ids):
        if not len(positions):
            return None
        records=np.zeros(len(positions),dtype=VERTEX_DTYPE)
        records["position"],records["tint"],records["id"]=positions,colors,ids
        data=records.tobytes()
        if cache_key not in self.buffers:
            buffer=self.context.buffer(reserve=len(data),dynamic=True)
            vao=self.context.vertex_array(self.program,[(buffer,"3f 4f 1u","position","tint","instance_id")])
            self.buffers[cache_key]=(buffer,vao)
        buffer,vao=self.buffers[cache_key]
        if len(data)!=buffer.size: buffer.orphan(len(data))
        buffer.write(data)
        return vao,len(records)

    def draw(self,view_projection,eye,distance,scene_id,hidden=(),*,selected=(),picking=False,framebuffer=None):
        """Draw with OvUI's deliberately always-on-top visual debug arrows.

        Pass the currently bound framebuffer (including an MSAA target) to
        depth-ignore debug arrows in a nested scope. Contact arrows always keep
        caller depth. Without a target, all primitives retain caller-controlled
        depth behavior. Picking always retains it: an occluded arrow does not
        steal its occluding body's hit even when its visual overlay is on top.

        This is not exact Qt parity: Qt routes these arrows to its depth-off
        pass only in simplified mode (>1000 renderable actors); otherwise it
        submits depth-tested 3D arrow geometry.
        """
        snapshot=self.snapshot
        self.draw_count=self.primitive_count=0
        if snapshot is None or not snapshot.ready or not len(snapshot.kinds):
            return
        mask=overlay_visibility(snapshot,scene_id,hidden,selected=selected,modes=self.modes)
        self.primitive_count=int(mask.sum())
        self.program["view_projection"].write(view_projection.T.astype("f4").tobytes())
        self.program["picking"].value=picking
        point=mask & (snapshot.kinds==1)
        triangle=mask & (snapshot.kinds==3)
        line=mask & (snapshot.kinds==2)
        arrow=mask & (snapshot.kinds==4)
        colors=self.palette.overlay_colors(snapshot) if self.palette is not None else snapshot.colors
        line_positions=[snapshot.positions[line,:2].reshape(-1,3)]
        line_colors=[colors[line,:2].reshape(-1,4)]
        line_ids=[np.repeat(snapshot.object_ids[line],2)]
        arrow_batch=None
        if arrow.any():
            positions=snapshot.positions[arrow,:2]
            arrow_colors=colors[arrow,:2]
            ids=snapshot.object_ids[arrow]
            delta=positions[:,1]-positions[:,0]
            length=np.linalg.norm(delta,axis=1)
            direction=delta/np.maximum(length[:,None],1e-20)
            toward_eye=eye-positions[:,1]
            side=np.cross(direction,toward_eye)
            side_length=np.linalg.norm(side,axis=1)
            fallback=np.cross(direction,[0.,0.,1.])
            side[side_length<1e-8]=fallback[side_length<1e-8]
            side/=np.maximum(np.linalg.norm(side,axis=1)[:,None],1e-20)
            head=np.minimum(length*.2,max(distance*.012,1e-5))[:,None]
            left=positions[:,1]-direction*head+side*head*.45
            right=positions[:,1]-direction*head-side*head*.45
            wings=np.stack((positions[:,1],left,positions[:,1],right),axis=1)
            on_top=np.isin(snapshot.categories[arrow],ON_TOP_ARROW_CATEGORIES)
            if framebuffer is None or picking:
                on_top[:]=False
            for depth_independent,rows in ((False,~on_top),(True,on_top)):
                if not rows.any():
                    continue
                arrow_positions=(positions[rows].reshape(-1,3),wings[rows].reshape(-1,3))
                arrow_tints=(arrow_colors[rows].reshape(-1,4),np.repeat(arrow_colors[rows,1],4,axis=0))
                arrow_ids=(np.repeat(ids[rows],2),np.repeat(ids[rows],4))
                if depth_independent:
                    arrow_batch=(moderngl.LINES,np.concatenate(arrow_positions),np.concatenate(arrow_tints),np.concatenate(arrow_ids))
                else:
                    line_positions.extend(arrow_positions)
                    line_colors.extend(arrow_tints)
                    line_ids.extend(arrow_ids)
        batches=((moderngl.POINTS,snapshot.positions[point,0],colors[point,0],snapshot.object_ids[point]),
                 (moderngl.LINES,np.concatenate(line_positions),np.concatenate(line_colors),np.concatenate(line_ids)),
                 (moderngl.TRIANGLES,snapshot.positions[triangle].reshape(-1,3),colors[triangle].reshape(-1,4),np.repeat(snapshot.object_ids[triangle],3)))
        if arrow_batch is not None:
            # OvUI deliberately draws depth-tested contacts after on-top arrows
            # so visible contacts can cover gizmos at the same screen position.
            batches=(arrow_batch,)+batches
        self.context.point_size=7. if picking else 4.
        self.context.line_width=3. if picking else 1.5
        for batch in batches:
            mode,positions,colors,ids=batch
            if selected and not picking:
                colors=colors.copy()
                selected_rows=np.isin(ids,list(selected))
                if self.palette is None:
                    colors[selected_rows]=(.8,.94,.49,1.)
                else:
                    colors[selected_rows,:3]+=self.palette.gizmo("selectionTint")[:3]
            # Both passes draw LINES, but their independent sizes must not
            # repeatedly orphan the same buffer on every unchanged frame.
            cache_key=(mode,"on_top") if batch is arrow_batch else mode
            uploaded=self._upload(cache_key,positions,colors,ids)
            if uploaded:
                vao,count=uploaded
                if batch is arrow_batch:
                    # A scope restores the caller state and leaves opaque depth
                    # untouched for contacts, query geometry, and later passes.
                    with self.context.scope(framebuffer=framebuffer,enable_only=moderngl.BLEND):
                        vao.render(mode,vertices=count)
                else:
                    vao.render(mode,vertices=count)
                self.draw_count+=1
        self.context.line_width=1.

    def close(self):
        for buffer,vao in self.buffers.values():
            vao.release();buffer.release()
        self.program.release()
