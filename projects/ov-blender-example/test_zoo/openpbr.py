# SPDX-License-Identifier: Apache-2.0
"""Direct USD probes, distinct from the add-on's Principled conversion path.

Names/types follow ASWF OpenPBR1.1.1 and MaterialX ND_open_pbr_surface_surfaceshader.
https://academysoftwarefoundation.github.io/OpenPBR/
"""
VALUES={
 'base_diffuse_roughness':.8,'specular_weight':.2,'specular_color':(.1,.3,1),
 'specular_roughness_anisotropy':.8,'specular_ior':2.4,
 'transmission_depth':.5,'transmission_scatter':(.1,.3,.7),'transmission_scatter_anisotropy':.7,
 'transmission_dispersion_scale':1.,'transmission_dispersion_abbe_number':12.,
 'subsurface_weight':.8,'subsurface_radius':.3,'subsurface_radius_scale':(1,.3,.1),
 'subsurface_scatter_anisotropy':.7,'coat_weight':1.,'coat_color':(.1,.4,1),
 'coat_roughness_anisotropy':.8,'coat_ior':2.,'coat_darkening':.2,
 'fuzz_weight':1.,'fuzz_color':(.2,.7,1),'fuzz_roughness':.8,
 'emission_luminance':500.,'thin_film_weight':1.,'thin_film_thickness':.65,'thin_film_ior':1.8,
 'geometry_opacity':.35,'geometry_thin_walled':True,'geometry_coat_normal':(.35,0,.94)}


def parameters(variant,edited=False):
    result={'base_color':(.45,.1,.025),'specular_roughness':.23}
    if variant.startswith('transmission') or variant in {'frosted_glass','geometry_thin_walled'}:
        result.update(transmission_weight=1.,transmission_color=(.1,.6,.8),specular_roughness=.08)
    if variant.startswith('subsurface'):result.update(subsurface_weight=.8,subsurface_color=(.9,.2,.06))
    if variant.startswith('coat') or variant in {'geometry_coat_normal','pearlescent_coat'}:result['coat_weight']=1.
    if variant.startswith('fuzz'):result['fuzz_weight']=1.
    if variant.startswith('thin_film') or variant=='pearlescent_coat':result['thin_film_weight']=1.
    if variant=='coated_metal':result.update(base_metalness=1.,coat_weight=1.,coat_roughness=.08)
    elif variant=='frosted_glass':result.update(specular_roughness=.5,transmission_depth=.25)
    elif variant=='pearlescent_coat':result.update(coat_weight=1.,thin_film_thickness=.45,coat_ior=1.8)
    else:result[variant]=VALUES[variant]
    if edited:
        if variant in VALUES:
            value=VALUES[variant]
            result[variant]=(not value if isinstance(value,bool) else tuple(reversed(value)) if isinstance(value,tuple)
                             else value*.4)
        else:result['base_color']=(.025,.5,.15)
    return result


def author_overlay(source,target,mesh_paths,variant,edited=False):
    from pxr import Usd,UsdShade,Sdf,Gf
    stage=Usd.Stage.CreateNew(str(target));stage.GetRootLayer().subLayerPaths=[str(source)]
    mat=UsdShade.Material.Define(stage,'/ZooDirectOpenPBR')
    shader=UsdShade.Shader.Define(stage,'/ZooDirectOpenPBR/Surface')
    shader.CreateIdAttr('ND_open_pbr_surface_surfaceshader')
    values=parameters(variant,edited)
    for name,value in values.items():
        kind=(Sdf.ValueTypeNames.Bool if isinstance(value,bool) else
              Sdf.ValueTypeNames.Vector3f if isinstance(value,tuple) and name.startswith('geometry_') else
              Sdf.ValueTypeNames.Color3f if isinstance(value,tuple) else Sdf.ValueTypeNames.Float)
        shader.CreateInput(name,kind).Set(Gf.Vec3f(*value) if isinstance(value,tuple) else value)
    mat.CreateSurfaceOutput('mtlx').ConnectToSource(shader.CreateOutput('out',Sdf.ValueTypeNames.Token))
    for path in mesh_paths:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat)
    stage.GetRootLayer().Save()
    return values
