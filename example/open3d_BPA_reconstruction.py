import open3d as o3d
import numpy as np

pcd_info = np.loadtxt('sample_w_normals.xyz', skiprows=1)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_info[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(pcd_info[:, 3:6]/255)
pcd.normals = o3d.utility.Vector3dVector(pcd_info[:, 6:9])

# o3d.visualization.draw_geometries([pcd])
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

bpa_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

o3d.visualization.draw_geometries([dec_mesh])
