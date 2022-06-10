import numpy as np
from obj_io import ObjMesh, loadObj, saveObj
import argparse


def calcFaceCenters(verts, indices):
    fetched = verts[indices]
    centers = np.mean(fetched, axis=1)
    return centers


def nearest_neighbors(query,  # [Q, dim]
                      data,  # [D, dim]
                      num_nn=1,
                      dist_func=lambda x, y: np.linalg.norm(x-y, axis=-1)):
    n_data, _ = data.shape
    qd = np.repeat(query[:, None], n_data, axis=1)  # [Q, D, dim]
    dist_all = dist_func(qd, data)  # [Q, D]
    if num_nn == 1:
        nn = np.argmin(dist_all, axis=-1)[..., None]  # [Q, 1]
    else:
        nn = np.argsort(dist_all, axis=-1)  # [Q, D]
        if num_nn > 1:
            nn = nn[:, :num_nn]  # [Q, N]
        # if num_nn <= 0 , return all
    return nn, np.take_along_axis(dist_all, nn, axis=-1)


def calcVertexWiseCorrespondence(a_verts, b_verts):
    nn = nearest_neighbors(a_verts, b_verts, 1)
    return nn


def calcQuadTriCorrespondence(quad_verts, quad_indices,
                              tri_verts, tri_indices):
    tv2qv_table, _ = calcVertexWiseCorrespondence(quad_verts, tri_verts)
    tri_indices_q = tv2qv_table[tri_indices].squeeze(-1)
    num_tri = len(tri_indices_q)
    num_quad = len(quad_indices)
    q2t = {k: [] for k in range(num_quad)}
    t2q = {k: [] for k in range(num_tri)}
    # TODO: batch
    for j, q_i in enumerate(quad_indices):
        tris = np.where(np.isin(tri_indices_q, q_i).sum(axis=-1) == 3)[0]
        for tri_i in tris.tolist():
            q2t[j].append(tri_i)
            t2q[tri_i].append(j)
    return q2t, t2q


def visualizeQuadTriCorrespondence(quad_path, tri_path, quad, tri, q2t):
    rand_colors = np.random.randint(0, 255, size=(quad.indices.shape[0], 3))
    vis_quad_verts, vis_quad_indices, vis_quad_colors = [], [], []
    vis_tri_verts, vis_tri_indices, vis_tri_colors = [], [], []
    for i, q_index in enumerate(quad.indices):
        for q_vi in q_index:
            vis_quad_verts.append(quad.verts[q_vi])
            vis_quad_colors.append(rand_colors[i])
        quad_count = len(vis_quad_indices)
        vis_quad_indices.append([quad_count * 4, quad_count * 4 + 1,
                                quad_count * 4 + 2, quad_count * 4 + 3])
        for t_index in tri.indices[q2t[i]]:
            for t_vi in t_index:
                vis_tri_verts.append(tri.verts[t_vi])
                vis_tri_colors.append(rand_colors[i])
            tri_count = len(vis_tri_indices)
            vis_tri_indices.append([tri_count * 3, tri_count * 3 + 1,
                                    tri_count * 3 + 2])
    vis_quad = ObjMesh(verts=vis_quad_verts,
                       indices=vis_quad_indices,
                       vert_colors=vis_quad_colors)
    vis_tri = ObjMesh(verts=vis_tri_verts, indices=vis_tri_indices,
                      vert_colors=vis_tri_colors)
    saveObj(quad_path, vis_quad)
    saveObj(tri_path, vis_tri)


def writeDict(path, d):
    if path == '':
        return
    with open(path, 'w') as fp:
        for k in sorted(d.keys()):
            fp.write(' '.join([str(x) for x in d[k]]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
             description='Quad and triangle mesh correspondence')
    parser.add_argument('quad', help='Quad .obj')
    parser.add_argument('tri', help='Tri .obj')
    parser.add_argument('--quad_vis', default='./quad_vis.obj')
    parser.add_argument('--tri_vis', default='./tri_vis.obj')
    parser.add_argument('--q2t', default='./q2t.txt')
    parser.add_argument('--t2q', default='./t2q.txt')

    args = parser.parse_args()

    quad_path = args.quad
    tri_path = args.tri

    quad = loadObj(quad_path)
    tri = loadObj(tri_path)

    q2t_path = args.q2t
    t2q_path = args.t2q
    q2t, t2q = calcQuadTriCorrespondence(quad.verts, quad.indices,
                                         tri.verts, tri.indices)
    writeDict(q2t_path, q2t)
    writeDict(t2q_path, t2q)

    quad_vis_path = args.quad_vis
    tri_vis_path = args.tri_vis
    if quad_vis_path != '' or tri_vis_path != '':
        visualizeQuadTriCorrespondence(quad_vis_path,
                                       tri_vis_path, quad, tri, q2t)
