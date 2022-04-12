import torch

# cohen_sutherland line-box intersection algorithm:
#    https://www.geeksforgeeks.org/line-clipping-set-1-cohen-sutherland-algorithm/
#    https://www.cs.drexel.edu/~david/Classes/CS430/Lectures/L-14_Color.6.pdf

# note that the box has to be axis-aligned

# x: left-->right, y: lower-->upper, z: behind-->front
# region code
INSIDE, LEFT, RIGHT, LOWER, UPPER, BEHIND, FRONT = 0, 1, 2, 4, 8, 16, 32


def _get_region_code(p, box_min_xyz, box_max_xyz):
    '''
    p: [..., 3]
    box_minxyz: tuple
    box_maxxyz: tuple
    '''
    code = INSIDE * torch.ones_like(p[..., 0]).int()

    eps = 1e-4
    # update mask
    mask = (p[..., 0] < box_min_xyz[0] - eps).int()
    code = (code | LEFT) * mask + code * (1 - mask)
    mask = (p[..., 0] > box_max_xyz[0] + eps).int()
    code = (code | RIGHT) * mask + code * (1 - mask)

    mask = (p[..., 1] < box_min_xyz[1] - eps).int()
    code = (code | LOWER) * mask + code * (1 - mask)
    mask = (p[..., 1] > box_max_xyz[1] + eps).int()
    code = (code | UPPER) * mask + code * (1 - mask)

    mask = (p[..., 2] < box_min_xyz[2] - eps).int()
    code = (code | BEHIND) * mask + code * (1 - mask)
    mask = (p[..., 2] > box_max_xyz[2] + eps).int()
    code = (code | FRONT) * mask + code * (1 - mask)

    return code


def check_inside_box(p, box_min_xyz=[-1., -1., -1.], box_max_xyz=[1., 1., 1.]):
    code = _get_region_code(p, box_min_xyz, box_max_xyz)
    return code == 0


def intersect_box(start_p, end_p, box_min_xyz, box_max_xyz):
    '''
    start_p, end_p: [..., 3]
    box_min_xyz: tuple([xmin, ymin, zmin])
    box_max_xyz: tuple([xmax, ymax, zmax])
    '''
    start_p_orig = start_p
    end_p_orig = end_p

    process_mask = torch.zeros_like(start_p[..., 0]).int()
    no_intersect_mask = process_mask.clone()

    start_p = start_p.clone()
    end_p = end_p.clone()
    while (process_mask.sum() < process_mask.numel()):
        start_p_code = _get_region_code(start_p, box_min_xyz, box_max_xyz)
        end_p_code = _get_region_code(end_p, box_min_xyz, box_max_xyz)
        # case 1: both points are inside box
        direct_accept_mask = ((start_p_code == 0) & (end_p_code == 0)).int()
        process_mask |= direct_accept_mask
        # case 2: both points are outside box and have the same region code
        direct_reject_mask = (start_p_code & end_p_code != 0).int()
        process_mask |= direct_reject_mask

        no_intersect_mask |= direct_reject_mask

        # remaining two cases:
        #   1) both points are outside box but have different region code, and no intersection with box
        #   2) some segment of line lies within the box

        # at least one endpoint is outside the box, pick it
        start_p_outside_mask = (start_p_code != 0).int()
        code_out = start_p_code * start_p_outside_mask + \
            end_p_code * (1 - start_p_outside_mask)
        # compute intersection
        intersect_p = torch.zeros_like(start_p)
        # x axis
        mask_tmp = (code_out & LEFT) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_min_xyz[0] - start_p[..., 0]
                          ) / (end_p[..., 0] - start_p[..., 0])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]
        mask_tmp = (code_out & RIGHT) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_max_xyz[0] - start_p[..., 0]
                          ) / (end_p[..., 0] - start_p[..., 0])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]
        # y axis
        mask_tmp = (code_out & LOWER) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_min_xyz[1] - start_p[..., 1]
                          ) / (end_p[..., 1] - start_p[..., 1])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]
        mask_tmp = (code_out & UPPER) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_max_xyz[1] - start_p[..., 1]
                          ) / (end_p[..., 1] - start_p[..., 1])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]
        # z axis
        mask_tmp = (code_out & BEHIND) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_min_xyz[2] - start_p[..., 2]
                          ) / (end_p[..., 2] - start_p[..., 2])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]
        mask_tmp = (code_out & FRONT) != 0
        if mask_tmp.sum() > 0:
            lambda_val = (box_max_xyz[2] - start_p[..., 2]
                          ) / (end_p[..., 2] - start_p[..., 2])
            p_tmp = start_p + lambda_val.unsqueeze(-1) * (end_p - start_p)
            mask_tmp &= torch.isfinite(p_tmp).all(dim=-1)
            intersect_p[mask_tmp] = p_tmp[mask_tmp]

        # update start_p and end_p
        mask = (1 - process_mask) & start_p_outside_mask
        mask = mask.float().unsqueeze(-1)
        start_p = intersect_p * mask + start_p * (1 - mask)

        mask = (1 - process_mask) & (1-start_p_outside_mask)
        mask = mask.float().unsqueeze(-1)
        end_p = intersect_p * mask + end_p * (1 - mask)

    # sanity
    eps = 1e-4
    intersect_mask = (
        1-no_intersect_mask) & (torch.norm(start_p-end_p, dim=-1) > eps).int()

    # naninf_mask = ~torch.isfinite(start_p)
    # start_p[naninf_mask] = start_p_orig[naninf_mask]
    # naninf_mask = ~torch.isfinite(end_p)
    # end_p[naninf_mask] = end_p_orig[naninf_mask]

    return start_p, end_p, intersect_mask.float()


def axis_align_oriented_box(bbx_pts):
    '''
    bbx_pts is [8, 3]
    '''
    # pick an arbitrary corner point as origin
    origin_pt = bbx_pts[0]
    dists = np.linalg.norm(bbx_pts - origin_pt.reshape((1, 3)), axis=1)
    idx = np.argsort(dists)[1:]
    # determine x axis
    x_pt = bbx_pts[idx[0]]
    # pick y axis according to right-handed rule
    y_pt = bbx_pts[idx[1]]
    z_pt = bbx_pts[idx[2]]
    if np.dot(np.cross(x_pt-origin_pt, y_pt-origin_pt), z_pt-origin_pt) < 0:
        y_pt = bbx_pts[idx[2]]
        z_pt = bbx_pts[idx[1]]
    x_dir = (x_pt-origin_pt) / np.linalg.norm(x_pt-origin_pt)
    y_dir = (y_pt-origin_pt) / np.linalg.norm(y_pt-origin_pt)
    z_dir = (z_pt-origin_pt) / np.linalg.norm(z_pt-origin_pt)
    rot = np.stack((x_dir, y_dir, z_dir), axis=0)
    return rot


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # lines: [N, 2, 3]
    x, y, z = np.meshgrid(
        np.array([-1, 1]), np.array([-1, 1]), np.array([-1, 1]))
    xyz = np.stack([x.reshape(-1), y.reshape(-1),
                   z.reshape(-1)], axis=1)  # [8, 3]
    lines = []
    colors = []
    for i in range(xyz.shape[0]):
        for j in range(i+1, xyz.shape[0]):
            pt1 = xyz[i]
            pt2 = xyz[j]
            # only flip 1 symbol
            if np.sum(np.abs(pt1 - pt2) > 0) == 1:
                lines.append(np.stack((pt1, pt2), axis=0))
                colors.append((0., 0., 1., 1.))
    lines = np.stack(lines, axis=0)
    ax.add_collection(Line3DCollection(lines, colors=colors))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # random rays
    radius = 2.
    start_p = np.random.randn(20, 10, 3)
    start_p = start_p / \
        np.linalg.norm(start_p, axis=-1, keepdims=True) * radius
    end_p = np.random.randn(20, 10, 3)
    end_p = end_p / np.linalg.norm(end_p, axis=-1, keepdims=True) * radius

    start_p, end_p, mask = intersect_box(torch.from_numpy(start_p), torch.from_numpy(end_p),
                                         box_min_xyz=[-1., -1., -1.], box_max_xyz=[1., 1., 1.])
    for i in range(start_p.shape[0]):
        for j in range(start_p.shape[1]):
            if mask[i, j] > 0:
                # print(mask)
                start_p_tmp = start_p[i, j].numpy()
                end_p_tmp = end_p[i, j].numpy()
                segs = np.stack((start_p_tmp, end_p_tmp), axis=0)[
                    np.newaxis, :, :]
                colors = [(1., 0., 0., 1.)]
                ax.add_collection(Line3DCollection(segs, colors=colors))
            else:
                start_p_tmp = start_p[i, j].numpy()
                end_p_tmp = end_p[i, j].numpy()
                segs = np.stack((start_p_tmp, end_p_tmp), axis=0)[
                    np.newaxis, :, :]
                colors = [(1., 1., 0., 1.)]
                ax.add_collection(Line3DCollection(segs, colors=colors))
            # # print(mask)
            # start_p_tmp = start_p[i, j].numpy()
            # end_p_tmp = end_p[i, j].numpy()
            # segs = np.stack((start_p_tmp, end_p_tmp), axis=0)[np.newaxis, :, :]
            # colors = [(1., 0., 0., 1.)]
            # ax.add_collection(Line3DCollection(segs, colors=colors))

    # # random rays
    # for i in range(20):
    #     radius = 2.
    #     # start_p = np.array([0., 0., 0.]).reshape((1, 3))
    #     # end_p = np.array([0.5, 0.5, 1.5]).reshape((1, 3))
    #     start_p = np.random.randn(1, 3)
    #     start_p = start_p / np.linalg.norm(start_p.flatten()) * radius
    #     end_p = np.random.randn(1, 3)
    #     end_p = end_p / np.linalg.norm(end_p.flatten()) * radius
    #
    #     start_p, end_p, mask = intersect_box(torch.from_numpy(start_p), torch.from_numpy(end_p),
    #                                          box_min_xyz=[-1., -1., -1.], box_max_xyz=[1., 1., 1.])
    #     if mask.sum() > 0:
    #         # print(mask)
    #         start_p = start_p.numpy()
    #         end_p = end_p.numpy()
    #         segs = np.concatenate((start_p, end_p), axis=0)[np.newaxis, :, :]
    #         colors = [(1., 0., 0., 1.)]
    #         ax.add_collection(Line3DCollection(segs, colors=colors))

    plt.savefig('./tmp.png')
    plt.show()
