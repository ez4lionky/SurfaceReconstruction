import open3d as o3d
import numpy as np


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


pcd_info = np.loadtxt('KingsCollege_xyzrgb.txt')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_info[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pcd_info[:, 3:6]/255)
o3d.visualization.draw_geometries([pcd])

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
display_inlier_outlier(pcd, ind)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
bbox = pcd.get_axis_aligned_bounding_box()

# o3d.visualization.draw_geometries([pcd])
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=11)

mesh = mesh.crop(bbox)
# dec_mesh = mesh.simplify_quadric_decimation(100000)
dec_mesh = mesh

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

o3d.visualization.draw_geometries([dec_mesh])
o3d.io.write_triangle_mesh("./KingsCollege_mesh.ply", dec_mesh)
