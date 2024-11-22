import open3d as o3d
import numpy as np
import plotly.graph_objects as go

from dash import html
from dash import dcc
from dash import Dash

import os, tqdm, glob
from PIL import Image
import torch
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras, 
    OrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    look_at_rotation
)
import matplotlib.pyplot as plt


# https://github.com/isl-org/Open3D/blob/74df0a3882565c35ce827c96a052a02fabc1cad7/python/open3d/visualization/draw_plotly.py#L191
# draw_plotly utils
def get_point_object(geometry, point_sample_factor=1):
    points = np.asarray(geometry.points)
    colors = None
    if geometry.has_colors():
        colors = np.asarray(geometry.colors)
    elif geometry.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
    else:
        geometry.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(geometry.colors)
    if (point_sample_factor > 0 and point_sample_factor < 1):
        indices = np.random.choice(len(points),
                                   (int)(len(points) * point_sample_factor),
                                   replace=False)
        points = points[indices]
        colors = colors[indices]
    scatter_3d = go.Scatter3d(x=points[:, 0],
                              y=points[:, 1],
                              z=points[:, 2],
                              mode='markers',
                              marker=dict(size=1, color=colors))
    return scatter_3d


def get_mesh_object(geometry,):
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)
    vertexcolor = np.asarray(geometry.vertex_colors)

    if np.sum((vertexcolor[:,0]==vertexcolor[:,1])*1) == vertexcolor.shape[0]:
        pl_mygrey = [0, 'rgb(0, 0, 0)'], [1., 'rgb(255,255,255)']
        mesh_3d = go.Mesh3d(x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=triangles[:, 0],
                        j=triangles[:, 1],
                        k=triangles[:, 2],
                        # vertexcolor=vertexcolor,
                        flatshading=True,
                        showscale=True,
                        colorscale=pl_mygrey,
                        intensity=vertexcolor[:, 0],
                        lighting=dict(ambient=0.18,
                                      diffuse=1,
                                      fresnel=0.1,
                                      specular=1,
                                      roughness=0.05,
                                      facenormalsepsilon=1e-15,
                                      vertexnormalsepsilon=1e-15),
                        lightposition=dict(x=100, y=200, z=0))
    
    else:
        mesh_3d = go.Mesh3d(x=vertices[:, 0],
                            y=vertices[:, 1],
                            z=vertices[:, 2],
                            i=triangles[:, 0],
                            j=triangles[:, 1],
                            k=triangles[:, 2],
                            vertexcolor=vertexcolor,
                            flatshading=True,
                            # showscale=True,
                            # colorscale=pl_mygrey,
                            # intensity=vertices[:, 0],
                            lighting=dict(ambient=0.18,
                                        diffuse=1,
                                        fresnel=0.1,
                                        specular=1,
                                        roughness=0.05,
                                        facenormalsepsilon=1e-15,
                                        vertexnormalsepsilon=1e-15),
                            lightposition=dict(x=100, y=200, z=0))
    return mesh_3d


def get_wireframe_object(geometry):
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)
    x = []
    y = []
    z = []
    tri_points = np.asarray(vertices)[triangles]
    for point in tri_points:
        x.extend([point[k % 3][0] for k in range(4)] + [None])
        y.extend([point[k % 3][1] for k in range(4)] + [None])
        z.extend([point[k % 3][2] for k in range(4)] + [None])
    wireframe = go.Scatter3d(x=x,
                             y=y,
                             z=z,
                             mode='lines',
                             line=dict(color='rgb(70,70,70)', width=1))
    return wireframe


def get_lineset_object(geometry):
    x = []
    y = []
    z = []
    line_points = np.asarray(geometry.points)[np.asarray(geometry.lines)]
    for point in line_points:
        x.extend([point[k % 2][0] for k in range(2)] + [None])
        y.extend([point[k % 2][1] for k in range(2)] + [None])
        z.extend([point[k % 2][2] for k in range(2)] + [None])
    line_3d = go.Scatter3d(x=x, y=y, z=z, mode='lines')
    return line_3d


def get_graph_objects(geometry_list,
                      mesh_show_wireframe=False,
                      point_sample_factor=1):

    graph_objects = []
    for geometry in geometry_list:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            graph_objects.append(get_point_object(geometry,
                                                  point_sample_factor))

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            graph_objects.append(get_mesh_object(geometry))
            if (mesh_show_wireframe):
                graph_objects.append(get_wireframe_object(geometry))

        if geometry_type == o3d.geometry.Geometry.Type.LineSet:
            graph_objects.append(get_lineset_object(geometry))

    return graph_objects


def get_max_bound(geometry_list):
    max_bound = [0, 0, 0]

    for geometry in geometry_list:
        bound = np.subtract(geometry.get_max_bound(), geometry.get_min_bound())
        max_bound = np.fmax(bound, max_bound)
    return max_bound


def get_geometry_center(geometry_list):
    center = [0, 0, 0]
    for geometry in geometry_list:
        center += geometry.get_center()
    np.divide(center, len(geometry_list))
    return center


def get_plotly_fig(geometry_list,
                   width=600,
                   height=400,
                   mesh_show_wireframe=False,
                   point_sample_factor=1,
                   front=None,
                   lookat=None,
                   up=None,
                   zoom=1.0):
    graph_objects = get_graph_objects(geometry_list, mesh_show_wireframe,
                                      point_sample_factor)
    geometry_center = get_geometry_center(geometry_list)
    max_bound = get_max_bound(geometry_list)
    # adjust camera to plotly-style
    if up is not None:
        plotly_up = dict(x=up[0], y=up[1], z=up[2])
    else:
        plotly_up = dict(x=0, y=0, z=1)

    if lookat is not None:
        lookat = [
            (i - j) / k for i, j, k in zip(lookat, geometry_center, max_bound)
        ]
        plotly_center = dict(x=lookat[0], y=lookat[1], z=lookat[2])
    else:
        plotly_center = dict(x=0, y=0, z=0)

    if front is not None:
        normalize_factor = np.sqrt(np.abs(np.sum(front)))
        front = [i / normalize_factor for i in front]
        plotly_eye = dict(x=zoom * 5 * front[0] + plotly_center['x'],
                          y=zoom * 5 * front[1] + plotly_center['y'],
                          z=zoom * 5 * front[2] + plotly_center['z'])
    else:
        plotly_eye = None

    camera = dict(up=plotly_up, center=plotly_center, eye=plotly_eye)
    fig = go.Figure(data=graph_objects,
                    layout=dict(
                        showlegend=False,
                        width=width,
                        height=height,
                        margin=dict(
                            l=0,
                            r=0,
                            b=0,
                            t=0,
                        ),
                        scene_camera=camera,
                    ))
    return fig


def draw_plotly(geometry_list,
                window_name='Open3D',
                width=600,
                height=400,
                mesh_show_wireframe=False,
                point_sample_factor=1,
                front=None,
                lookat=None,
                up=None,
                zoom=1.0):

    fig = get_plotly_fig(geometry_list, width, height, mesh_show_wireframe,
                         point_sample_factor, front, lookat, up, zoom)
    fig.show()


def draw_plotly_server(geometry_list,
                       window_name='Open3D',
                       width=1080,
                       height=960,
                       mesh_show_wireframe=False,
                       point_sample_factor=1,
                       front=None,
                       lookat=None,
                       up=None,
                       zoom=1.0,
                       port=8050):

    fig = get_plotly_fig(geometry_list, width, height, mesh_show_wireframe,
                         point_sample_factor, front, lookat, up, zoom)
    app = Dash(window_name)
    app.layout = html.Div([
        html.H3(window_name),
        html.Div(
            [
                dcc.Graph(id="graph-camera", figure=fig),
            ],
            style={
                "width": "100%",
                "display": "inline-block",
                "padding": "0 0"
            },
        ),
    ])
    app.run_server(debug=False, port=port)




# open3d utils
def make_pcd(point_cloud, color=None, per_vertex_color=None, estimate_normals=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color is not None:
        # pcd.colors = o3d.utility.Vector3dVector(color)
        pcd.paint_uniform_color(color)
    if per_vertex_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(per_vertex_color)
    
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=100))
    return pcd

def make_line_set(points, edges, line_color = None, per_line_color=None):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    if per_line_color is not None:
        line_set.colors = o3d.utility.Vector3dVector(per_line_color)
    
    return line_set

def make_mesh(vertices, faces, vertex_colors=None):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    if vertex_colors is not None: 
        # print(vertices.shape, vertex_colors.shape, vertex_colors.min(), vertex_colors.max())
        # print(vertex_colors[:10])
        # vertex_colors = plt.get_cmap('plasma')(vertex_colors)
        # print(vertex_colors[:10])
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[:,:3])
        # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return o3d_mesh

def make_mesh_v2(vertices, faces, vertex_colors=None):
    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces)
    # o3d_mesh.compute_vertex_normals()
    if vertex_colors is not None: 
        # print(vertex_colors.shape)
        o3d_mesh.vertex.colors = o3d.core.Tensor(vertex_colors)
    
    return o3d_mesh



def visualize_3d(meshes, vertex_colors=None, clean_via_connected_components=False):
    verts = meshes.verts_padded()
    faces = meshes.faces_padded()
    if vertex_colors is not None: vertex_colors = np.copy(vertex_colors.cpu())

    if clean_via_connected_components:
        o3d_mesh = make_mesh(verts[0].cpu().numpy(), faces[0].cpu().numpy(), verts[0].cpu().numpy())
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                o3d_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)

        import copy
        print("Show mesh with small clusters removed")
        mesh_0 = copy.deepcopy(o3d_mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 10000
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        triangles_remove = np.asarray(faces[0].cpu().numpy())[triangles_to_remove]
        
        vertex_colors[triangles_remove.reshape(-1)] = 0.
        
        ## normalizing the visible vertex colors.
        
        ## top 95%
        min_value = 0.
        max_value = 0.37385
        threshold_value = 0.112
        
        threshold_value = (threshold_value - min_value) / (max_value - min_value)
        vertex_colors[:,0] = (vertex_colors[:,0] - min_value) / (max_value - min_value)
        vertex_colors[:,1] = (vertex_colors[:,1] - min_value) / (max_value - min_value)
        vertex_colors[:,2] = (vertex_colors[:,2] - min_value) / (max_value - min_value)
        vertex_colors[vertex_colors<threshold_value]=0.
        mesh_0 = make_mesh(np.asarray(mesh_0.vertices), np.asarray(mesh_0.triangles), vertex_colors=vertex_colors)
        # draw_plotly([mesh_0])
        o3d_mesh = mesh_0
    
    else:
        o3d_mesh = make_mesh(verts[0].cpu().numpy(), faces[0].cpu().numpy(), vertex_colors=vertex_colors)
        # draw_plotly([o3d_mesh])

    return vertex_colors, o3d_mesh



# copied from: https://pytorch3d.org/tutorials/render_textured_meshes
def load_mesh(mesh_path, device):
    mesh = load_objs_as_meshes([mesh_path], device=torch.device(device))
    # plt.figure(figsize=(7,7))
    try:
        texture_image=mesh.textures.maps_padded()
    except:
        mesh.textures = TexturesVertex(torch.ones_like(mesh.verts_padded()))
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # plt.axis("off")
    return mesh

def load_mesh(mesh_path, device, visualize_texture_map=False):
    mesh = load_objs_as_meshes([mesh_path], device=torch.device(device))
    updated_vertices = mesh.verts_padded() # / 2.

    ## to convert from threestudio to pytorch 3d -- https://github.com/threestudio-project/threestudio/issues/95#issuecomment-1573474026
    mesh = mesh.update_padded(
        new_verts_padded=
            torch.cat((updated_vertices[:,:,1:2], updated_vertices[:,:,2:3], updated_vertices[:,:,0:1]), dim=-1)
    )
    try:
        texture_image=mesh.textures.maps_padded()
        if visualize_texture_map:
            plt.figure(figsize=(7,7))
            plt.imshow(texture_image.squeeze().cpu().numpy())
            plt.axis("off")

    except:
        mesh.textures = TexturesVertex(torch.ones_like(mesh.verts_padded()))

    return mesh

def create_renderer(camera_position, proj_matrix, device, elevation, azimuth, camera_distances):
    # R, T = look_at_view_transform(eye=camera_position, up=((-1, 0, 0),), at=((0,0,0),))
    R, T = look_at_view_transform(
        dist = camera_distances,
        elev=elevation,
        azim=azimuth,
        degrees= True,
        eye = None,
        at=((0, 0, 0),),  # (1, 3)
        up=((0, 1, 0),),  # (1, 3)
        device =device,
    )
    proj_matrix[:,:,0]*=-1
    cameras = PerspectiveCameras(device=device, R=R, T=T, K=proj_matrix, image_size=[[512, 512],])

    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        # lights=lights
        lights=None
    )

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader        
    )

    return renderer, rasterizer, cameras


# reference link: https://github.com/facebookresearch/pytorch3d/issues/158
# reference link: https://github.com/facebookresearch/pytorch3d/issues/126
def render_mesh(renderer, rasterizer, cameras, meshes):
    images = renderer(meshes)
    fragments = rasterizer(meshes)
    meshes_vertices = meshes.verts_padded() 

    # # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face  

    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = meshes.faces_packed() 
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = meshes.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0]).cuda()   # (V,)
    
    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()   # (num_visible_faces )

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices 
    vertex_visibility_map[unique_visible_verts_idx] = 1.0

    verts_screen = cameras.transform_points_screen(meshes_vertices)
    # verts_screen = verts_screen * vertex_visibility_map[None, :, None]
    
    return images, verts_screen, vertex_visibility_map


def save_rendered_colored_mesh(mesh, vertex_data, all_batch_data, save_dir, device):
    
    new_texture = TexturesVertex(verts_features=torch.from_numpy(vertex_data[None]).to(device))
    mesh.textures = new_texture

    print('rendering variance maps')
    for idx in tqdm.tqdm(range(len(all_batch_data))):
        if idx%2!=0: continue

        batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
        
        camera_position = batch_data['camera_positions'].cpu().numpy()
        proj_matrix = batch_data['proj_mtx'].cpu().numpy()
        renderer, rasterizer, cameras = create_renderer(camera_position, proj_matrix, device=device, elevation=batch_data['elevation'], azimuth=batch_data['azimuth'], camera_distances=batch_data['camera_distances'])
        rendered_images, rendered_verts_screen, rendered_verts_visibility = render_mesh(renderer, rasterizer, cameras, mesh)
        
        img_to_save = np.asarray(rendered_images[0, ..., :3].cpu().numpy()*255, dtype=np.uint8)
        if not os.path.exists(os.path.join(save_dir, 'dino_variance_maps')):
            os.makedirs(os.path.join(save_dir, 'dino_variance_maps'), exist_ok=True)
        np.save(os.path.join(save_dir, 'dino_variance_maps', str(idx).zfill(4) + '.npy'), rendered_images.cpu().numpy())
        Image.fromarray(img_to_save).save(os.path.join(save_dir, 'dino_variance_maps', str(idx).zfill(4) + '.png'))


def compute_dino_3d_consistency(mesh, dino_features_verts_3d, visibility_verts_3d, is_visualize_flag=False):
    print('computing 3d metrics')
    dino_features_verts_3d = torch.stack(dino_features_verts_3d)
    visibility_verts_3d = torch.stack(visibility_verts_3d)
    
    dino_features_verts_3d = dino_features_verts_3d * visibility_verts_3d[...,None].to(dino_features_verts_3d.get_device())
    dino_verts_mean = torch.sum(dino_features_verts_3d, dim=0) / (torch.sum(visibility_verts_3d[...,None], dim=0) + 1e-8)
    
    mean_centered = dino_features_verts_3d - (dino_verts_mean[None, ...] * visibility_verts_3d[...,None].to(dino_features_verts_3d.get_device()))
    squared_mean_centered = mean_centered ** 2
    non_zero_sum_squared_diff = torch.sum(squared_mean_centered, dim=0)
    dino_verts_variance = non_zero_sum_squared_diff / (torch.sum(visibility_verts_3d[...,None], dim=0) + 1e-8)
    dino_verts_std = torch.sqrt(dino_verts_variance)
    
    dino_verts_variance = dino_verts_variance.mean(dim=-1)
    dino_verts_std = dino_verts_std.mean(dim=-1)
    
    if is_visualize_flag:
        # Some options for visualization. Visualize whatever you want :)
        visualize_3d(mesh, vertex_colors=dino_verts_variance[...,None].repeat(1,3))
        visualize_3d(mesh, vertex_colors=dino_verts_variance[:,:3])
        visualize_3d(mesh, vertex_colors=dino_verts_mean[:, 1:4])
        visualize_3d(mesh, vertex_colors=dino_verts_variance[...,None].repeat(1,3), clean_via_connected_components=True)
        normalized_cleaned_dino_verts_std, cleaned_mesh = visualize_3d(mesh, vertex_colors=dino_verts_std[...,None].repeat(1,3), clean_via_connected_components=True)
        
    # return cleaned_mesh, normalized_cleaned_dino_verts_std, dino_verts_mean, dino_verts_std, dino_verts_variance
    return dino_verts_mean, dino_verts_std, dino_verts_variance