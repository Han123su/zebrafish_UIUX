import numpy as np
import networkx as nx
import cv2
from math import dist
from skimage import morphology
from skan import csr, Skeleton
from plantcv import plantcv as pcv
from scipy.interpolate import CubicSpline

def extract_skeleton(img):
    thres_min_size = 20 # 移除骨架分支的最小size
    # obtain binary skeleton

    skeleton = morphology.skeletonize(img)
    skeleton = skeleton.astype(np.uint8) * 255
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton, size=thres_min_size) # remove branch
    skeleton_img = pruned_skeleton > 0

    skeleton_img = morphology.skeletonize(skeleton_img).astype(bool)
    #######################################################
#    graph_class = csr.Skeleton(skeleton_img) # coordiantes
#    stats = csr.branch_statistics(graph_class.graph)
#    
#    # https://skeleton-analysis.org/stable/api/skan.csr.html
#    for ii in range(np.size(stats, axis=0)):
#        if stats[ii, 2] <= thres_min_size and stats[ii, 3] == 1:
#            # remove the branch
#            for jj in range(np.size(graph_class.path_coordinates(ii), axis=0)):
#                skeleton_img[int(graph_class.path_coordinates(ii)[jj, 0]), int(
#                    graph_class.path_coordinates(ii)[jj, 1])] = False
    
    # during the short branch removing process it can happen that some branches are not longer connected as the complete three branches intersection is removed
    # therefor the remaining skeleton is dilatated and then skeletonized again
    #######################################################

    sk_dilation = morphology.binary_dilation(skeleton_img)
    sk_final = morphology.skeletonize(sk_dilation)

    sk = Skeleton(sk_final)
    path_coor = [sk.path_coordinates(ii) for ii in range(sk.n_paths)]
    skeleton_idx = np.array(
        [path_coor[ix][j] for ix in range(len(path_coor)) for j in range(len(path_coor[ix]))])
    # cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('img', img.astype(np.uint8)*255)
    # cv2.waitKey(1)
    return sk_final.astype(np.uint8) * 255, img, sk, path_coor, skeleton_idx


def travel_all_path(G, endpoints):
    max_T, max_path, max_len, max_source = None, None, -1, -1
    for node in endpoints:
        T = dfs_tree_with_weight(G, source=node)
        # print(node, list(T.edges))
        path_c = nx.dag_longest_path(T, weight="w")
        current_val = 0
        for u, v in zip(path_c[:-1], path_c[1:]):
            current_val += T.get_edge_data(u, v)["w"]

        if current_val > max_len:
            max_T = T
            max_source = node
            max_len = current_val
            max_path = path_c
    return max_path


def build_tree_nodes(sk):
    pts, ll, endpoints, new_sk_idx = [], [], [], []
    for i in range(sk.n_paths):
        pts.append(tuple(sk.path_coordinates(i)[0]))
        pts.append(tuple(sk.path_coordinates(i)[-1]))
    for i in range(0, len(pts), 2):
        ll.append((pts[i], pts[i + 1], {"w": dist(pts[i], pts[i + 1])}))

    for i in range(len(pts)):
        cnt = 0
        for j in range(len(pts)):
            if pts[i] != pts[j]:
                cnt += 1
            if cnt == len(pts) - 1:
                endpoints.append(pts[i])
                break

    return nx.Graph(ll), endpoints


def dfs_tree_with_weight(G, source=None, depth_limit=None):
    T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G)
    else:
        T.add_node(source)
    T.add_edges_from([
        (u, v, G.get_edge_data(u, v))
        for u, v in nx.dfs_edges(G, source, depth_limit)
    ])
    return T


def dfs_longest_path(sk, path_coor, max_path):
    new_sk_idx = []
    for i in range(sk.n_paths):
        for j in range(len(max_path) - 1):
            if (path_coor[i][0] == max_path[j]).all() and (
                    path_coor[i][-1] == max_path[j + 1]).all():
                if new_sk_idx == []:
                    new_sk_idx.extend(path_coor[i])
                else:
                    new_sk_idx.extend(path_coor[i][1:])
    new_sk_idx = np.array(new_sk_idx)
    return new_sk_idx

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
def curve_fitting(curve):
    y = curve[:, 0]
    x = curve[:, 1]
    xx = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    xx = np.cumsum(xx)
    xx = (np.c_[np.array([0]), [xx]])[0]
    cs = CubicSpline(xx, curve)
    new_x = np.linspace(0, xx[-1], 20)

    # plt.figure(1)
    # plt.imshow(img)
    # plt.plot(x, y, 'b--')
    # # plt.plot(x, new_y, 'r--')
    # plt.plot(cs(new_x)[:, 1], cs(new_x)[:, 0], 'g--')
    # plt.show()
    # plt.close()
    return cs(new_x)
