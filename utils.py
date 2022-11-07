import json
import torch
import numpy as np
import socket
import math
import matplotlib.pyplot as plt
import io
import os
import PIL.Image
from torchvision.transforms import ToTensor


class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def write_log(writer, time, obj_pc, R, t, loss, loss1, loss2, sigma, T_gt, euler_gt, trans_gt, real=False):
    writer.add_scalar('loss', loss, time)
    writer.add_scalar('matching energy', loss1, time)  # loss one is not normalized (surface field changes with sigma)
    writer.add_scalar('keypoint energy', loss2, time)
    writer.add_scalar('soothing sigma', sigma, time)
    if not real:
        euler = get_Euler(R.clone().detach().cpu().numpy()) * 180 / np.pi
        d_euler = np.minimum(abs(euler - euler_gt), abs(euler - (euler / abs(euler)) * 360 - euler_gt))
        delta_tr = np.sum((t[0].clone().detach().cpu().numpy() / 1. - trans_gt) ** 2) ** (0.5)
        writer.add_scalar('delta translation', delta_tr, time)
        delta_rot = np.sum((d_euler) ** 2) ** (0.5)
        writer.add_scalar('delta rotation', delta_rot,
                          time)
        obj_pc_ = np.concatenate((obj_pc.T, np.ones((1, obj_pc.shape[0]))), axis=0)
        T_pred = np.block([[R.clone().detach().cpu().numpy(), t.clone().detach().cpu().numpy().T],
                           [np.zeros((1, 3)), np.ones((1, 1))]])
        gt_opc = (T_gt @ obj_pc_).T[:, :3]
        pred_opc = (T_pred @ obj_pc_).T[:, :3]
        distance = np.linalg.norm(gt_opc - pred_opc, axis=1)
        mean_distance = np.mean(distance)
        writer.add_scalar('3D-ADD', mean_distance, time)

    # print("Distance:", mean_distance)
    # print("delta_translation:", delta_tr)
    # print("delta_rotation:", delta_rot)
    # print("loss:", loss)
    # print("loss 1:", loss1)
    # print("loss 2:", loss2)
    # print("smoothing sigma", sigma)


def save_transform(R, t, root):
    path = os.path.join(root, 'output')
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(os.path.join(path, 'output_transform.npy'), 'wb') as f:
        output_transform = torch.cat([R, t.squeeze(0)[..., None]], dim=-1).detach().clone().cpu().numpy()
        np.save(f, output_transform)


def render_fused_image(nerf1, nerf2,  # z-fighting overlay of two nerfs
                       pose1, pose2,
                       near=0.3,
                       far=0.8,
                       focal_length=None,
                       principal_point=None,
                       resolution=(512, 512),
                       chunk=16,
                       batch_size=1,
                       scale=1,
                       device=torch.device('cuda')):
    if principal_point is None:
        principal_point = (resolution[0] / 2, resolution[1] / 2)

    if focal_length is None:
        focal_length = max(resolution[0], resolution[1])

    x = torch.arange(resolution[0]) + 0.5
    y = torch.arange(resolution[1]) + 0.5

    x, y = torch.meshgrid(x, y, indexing="xy")
    ip_points = torch.stack([
        (x - principal_point[0]) / focal_length,
        (y - principal_point[1]) / focal_length,
        torch.ones_like(x),
    ],
        axis=-1).reshape((-1, 3))
    ip_points = ip_points.to(pose1.device)

    ws_ip_points1 = apply_transform(pose1, ip_points) / scale
    ws_eye1 = apply_transform(pose1, torch.zeros((1, 3)).to(pose1.device)) / scale
    eyes_tiled1 = ws_eye1.repeat(resolution[0] * resolution[1], 1)  # [HW, 3]
    ray_dirs1 = ws_ip_points1 - eyes_tiled1
    ray_dirs1 /= torch.linalg.norm(ray_dirs1, axis=-1, keepdims=True)  # [HW, 3]
    rays1 = torch.cat([eyes_tiled1, ray_dirs1], axis=-1).to(device)

    ws_ip_points2 = apply_transform(pose2, ip_points)
    ws_eye2 = apply_transform(pose2, torch.zeros((1, 3)).to(pose2.device))
    eyes_tiled2 = ws_eye2.repeat(resolution[0] * resolution[1], 1)  # [HW, 3]
    ray_dirs2 = ws_ip_points2 - eyes_tiled2
    ray_dirs2 /= torch.linalg.norm(ray_dirs2, axis=-1, keepdims=True)  # [HW, 3]

    rgb = torch.zeros((rays1.shape[0], 3)).to(device)
    mask = torch.zeros((rays1.shape[0], 1)).to(device)

    batch_count = torch.tensor(rays1.shape[0] / chunk)
    batch_count = torch.ceil(batch_count).long()

    depths = nerf1.sample_depth(batch_size, near=near, far=far)[0].to(pose1.device)  # [HW,N,1]
    positions1 = eyes_tiled1[:, None] + ray_dirs1[:, None] * depths / scale  # [HW,N,3]
    positions2 = eyes_tiled2[:, None] + ray_dirs2[:, None] * depths  # [HW,N,3]

    rays1 = ray_dirs1[:, None].repeat((1, depths.shape[1], 1))  # [HW,N,3]
    rays2 = ray_dirs2[:, None].repeat((1, depths.shape[1], 1))  # [HW,N,3]
    for n in torch.arange(batch_count):
        positions_chunk1 = positions1[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)
        rays_chunk1 = rays1[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)
        positions_chunk2 = positions2[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)
        rays_chunk2 = rays2[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)

        rgb_n1, density_n1 = nerf1.nerf.call_eval(positions_chunk1.to(device),
                                                  rays_chunk1.to(device))  # check shape. goal [HW, N, 3|1]
        density_n1 = density_n1.reshape(chunk, depths.shape[1]) / scale
        rgb_n1 = rgb_n1.reshape(chunk, depths.shape[1], 3)

        rgb_n2, density_n2 = nerf2.nerf.call_eval(positions_chunk2.to(device),
                                                  rays_chunk2.to(device))  # check shape. goal [HW, N, 3|1]
        density_n2 = density_n2.reshape(chunk, depths.shape[1])
        rgb_n2 = rgb_n2.reshape(chunk, depths.shape[1], 3)

        rgb_n1[:, :, 0], rgb_n1[:, :, 1], rgb_n1[:, :, 2] = (0.8 * 176 / 255. + 0.2 * rgb_n1[:, :, 0]), (
                0.8 * 164 / 255 + 0.2 * rgb_n1[:, :, 1]), (0.8 * 101 / 255 + 0.2 * rgb_n1[:, :, 2])
        rgb_n2[:, :, 0], rgb_n2[:, :, 1], rgb_n2[:, :, 2] = (0.8 * 90 / 255 + 0.2 * rgb_n2[:, :, 0]), (
                0.8 * 115 / 255 + 0.2 * rgb_n2[:, :, 1]), (0.8 * 151 / 255 + 0.2 * rgb_n2[:, :, 2])

        density_fused = torch.maximum(density_n1, density_n2)

        sample_depth = depths[n * chunk:(n + 1) * chunk, :, 0].to(device)

        intervals = sample_depth[..., 1:] - sample_depth[..., :-1]
        before_intervals = torch.cat([intervals[..., :1], intervals], axis=-1)
        after_intervals = torch.cat([intervals, intervals[..., :1]], axis=-1)
        delta = (before_intervals + after_intervals) / 2

        quantity = delta * density_fused

        cumsumex = lambda t: torch.cumsum(torch.cat(
            [torch.zeros_like(t[..., :1]), t[..., :-1]], axis=-1),
            axis=-1)
        transmittance = torch.exp(cumsumex(-quantity))[..., None]
        cross_section = 1.0 - torch.exp(-quantity)[..., None]

        weights = transmittance * cross_section

        ###

        quantity1 = delta * density_n1

        cumsumex = lambda t: torch.cumsum(torch.cat(
            [torch.zeros_like(t[..., :1]), t[..., :-1]], axis=-1),
            axis=-1)
        transmittance1 = torch.exp(cumsumex(-quantity1))[..., None]
        cross_section1 = 1.0 - torch.exp(-quantity1)[..., None]

        weights1 = transmittance1 * cross_section1
        ###

        quantity2 = delta * density_n2

        cumsumex = lambda t: torch.cumsum(torch.cat(
            [torch.zeros_like(t[..., :1]), t[..., :-1]], axis=-1),
            axis=-1)
        transmittance2 = torch.exp(cumsumex(-quantity2))[..., None]
        cross_section2 = 1.0 - torch.exp(-quantity2)[..., None]

        weights2 = transmittance2 * cross_section2
        ###
        w1 = (weights1 > weights2).float()
        w2 = (weights1 <= weights2).float()

        rgb_fused = w1 * rgb_n1 + w2 * rgb_n2

        radiance = torch.sum(weights * rgb_fused, axis=-2)
        alpha = torch.sum(weights, axis=-2)

        rgb[n * chunk:(n + 1) * chunk] = radiance.detach()
        mask[n * chunk:(n + 1) * chunk] = alpha.detach()

    rgb = rgb.reshape(resolution[0], resolution[1], 3)
    mask = mask.reshape(resolution[0], resolution[1], 1)

    rgb = rgb + (1.0 - mask)

    return rgb


def show(writer, R, t, nerf1, nerf2, view, focal, samples, As, time, scale=1, image_size=(512, 512), near=0.3, far=0.8, chunk=256,
         device=torch.device('cuda')):
    print("rendering images, might take a few minutes...")
    s_ext = view.clone().detach()
    s = torch.cat([invert(torch.cat([R[:3], t.squeeze(0)[..., None]], dim=-1)),
                   torch.tensor([[0., 0., 0., 1.]]).to(device)]) @ s_ext.squeeze(0)
    rgbs_fused = render_fused_image(nerf1, nerf2, s.unsqueeze(0).to(device).float(), s_ext.to(device),
                                    focal_length=focal, scale=scale, near=near, far=far)

    rgb1 = nerf1.render_image_fine((s).unsqueeze(0).to(device).float(), focal_length=focal, scale=1, near=near, far=far, chunk=chunk)
    rgb2 = nerf2.render_image_fine(s_ext, focal_length=focal, scale=1, near=near, far=far, chunk=chunk)
    rgb1 = rgb1.detach().clone().cpu().numpy()
    rgb2 = rgb2.detach().clone().cpu().numpy()
    rgbs_fused = rgbs_fused.detach().clone().cpu().numpy()

    image1 = PIL.Image.fromarray(np.uint8((rgb1) * 255))
    image1 = ToTensor()(image1)
    writer.add_image('predicted', image1, time)

    image2 = PIL.Image.fromarray(np.uint8((rgbs_fused) * 255))
    image2 = ToTensor()(image2)
    writer.add_image('z-fighting', image2, time)

    image3 = PIL.Image.fromarray(np.uint8((rgb2) * 255))
    image3 = ToTensor()(image3)
    writer.add_image('target', image3, time)

    ps = project_points(samples[0] * scale, focal, s, image_size=image_size)
    f = plt.figure()
    plt.axis('off')
    if time > 0:
        plt.scatter(ps[:, 0], ps[:, 1], marker='.', color="red", s=8)
    aas = project_points(As * scale, focal, s, image_size=image_size).detach().clone().cpu().numpy()  # scaling
    plt.scatter(aas[:, 0], aas[:, 1], marker='x', color="chartreuse", s=30)
    plt.imshow(rgbs_fused)
    buff = io.BytesIO()
    plt.savefig(buff, format='png')
    buff.seek(0)
    plt.show()
    f.clear()
    plt.close(f)
    image = PIL.Image.open(buff)
    image = ToTensor()(image)
    writer.add_image('Samples', image, time)


def parse_raw_camera(pose_raw):  # extrinsics
    R = torch.diag(torch.tensor([1, -1, -1]))
    t = torch.zeros(R.shape[:-1], device=R.device)
    pose_flip = torch.cat([torch.cat([R, t[..., None]], dim=-1), torch.tensor([[0., 0., 0., 1]])], dim=0)
    pose = (pose_raw @ pose_flip)[..., :3, :]
    return pose


def load_poses(file, res):
    raw_W, raw_H = res
    meta_fname = file
    with open(meta_fname) as file:
        meta = json.load(file)
    p_list = meta["frames"]
    focal = 0.5 * raw_W / np.tan(0.5 * meta["camera_angle_x"])
    pose_raw_all = [torch.tensor(f["transform_matrix"], dtype=torch.float32) for f in p_list]
    pose_all = torch.stack([parse_raw_camera(p) for p in pose_raw_all], dim=0)
    return focal, pose_all


def apply_transform(matrix, points):
    points = torch.cat([points, torch.ones_like(points[..., :1]).to(points.device)], axis=-1)
    points = points[..., None]
    result = matrix @ points
    return result[..., :3, 0]


def get_intrinsic(focal, image_size=(512, 512), device=torch.device('cuda')):
    intrinsic = torch.tensor([
        [focal, 0., image_size[0] / 2, 0],
        [0., focal, image_size[1] / 2, 0],
        [0., 0., 1, 0],
        [0., 0., 0, 1]]).to(device).float()
    return intrinsic


def to_hom(X):
    # get homogeneous coordinates
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def invert(pose):
    # invert a camera pose
    R, t = pose[..., :3], pose[..., 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = (-R_inv @ t)[..., 0]
    pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)
    return pose_inv


def world2cam(X, pose):
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def get_Euler(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def make_rot_matrix(thetac, alphac, gammac, device=torch.device('cuda')):
    theta = thetac.sigmoid() * torch.pi * 2
    alpha = alphac.sigmoid() * torch.pi * 2
    gamma = gammac.sigmoid() * torch.pi * 2
    m1 = torch.cat((torch.cat((theta.cos().unsqueeze(0),
                               -theta.sin().unsqueeze(0), torch.tensor([0.0]).to(device))).unsqueeze(0),
                    torch.cat((theta.sin().unsqueeze(0), theta.cos().unsqueeze(0),
                               torch.tensor([0.0]).to(device))).unsqueeze(0),
                    torch.tensor([[0.0, 0.0, 1.0]]).to(device)), 0).float()

    m2 = torch.cat((torch.cat((alpha.cos().unsqueeze(0), torch.tensor([0.0]).to(device),
                               alpha.sin().unsqueeze(0))).unsqueeze(0),
                    torch.tensor([[0.0, 1.0, 0.0]]).to(device),
                    torch.cat((-alpha.sin().unsqueeze(0), torch.tensor([0.0]).to(device),
                               alpha.cos().unsqueeze(0))).unsqueeze(0),
                    ), 0).float()

    m3 = torch.cat(
        (torch.tensor([[1.0, 0.0, 0.0]]).to(device), torch.cat((torch.tensor([0.0]).to(device),
                                                                gamma.cos().unsqueeze(0),
                                                                -gamma.sin().unsqueeze(0))).unsqueeze(0),
         torch.cat((torch.tensor([0.0]).to(device), gamma.sin().unsqueeze(0),
                    gamma.cos().unsqueeze(0))).unsqueeze(0),
         ), 0).float()

    R = (m1 @ m2 @ m3)
    return R


def generate_camera_point_spherical(r, theta=None, phi=None, phi_low=np.pi / 4, r_scale=1., z_offset=0.):
    phi_low = np.pi / 4 if phi_low is None else phi_low
    r_scale = 1. if r_scale is None else r_scale
    z_offset = 0.2 if z_offset is None else z_offset
    r = r * r_scale
    if theta is None and phi is None:
        theta, phi = np.random.rand() * 2 * np.pi, np.random.uniform(low=phi_low,
                                                                     high=np.pi / 2)

    z = np.sin(phi) * r
    xy = r * np.cos(phi)
    y = np.cos(theta) * xy
    x = np.sin(theta) * xy

    forward = np.array([-x, -y, z_offset - z])
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rot = np.stack([right, up, -forward], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = np.array([x, y, z])
    pose = parse_raw_camera(torch.tensor(pose).float())
    pose = torch.concat((pose, torch.tensor([[0, 0, 0.0, 1]]).to(pose.device)), axis=0)

    return pose


def load_sample_extrinsic(file_p, i, res=(512, 512)):
    poses, _ = load_poses(file_p, res)
    return poses[i]


def sample_cameras(file_p, num):  # farthest point sampling of the training camera views
    with open(file_p) as file:
        t_dict = json.load(file)
    poses_raw = np.array(np.asarray([i["transform_matrix"] for i in t_dict["frames"]], dtype=np.float64))
    poses = poses_raw[:, :3]  # [n,3,4]
    poses[:, :, 2] = -poses[:, :, 2].copy()
    poses[:, :, 1] = -poses[:, :, 1].copy()
    camera_positions = poses[:, :, 3]  # [n,3]
    index = np.random.randint(poses.shape[0])
    samples = [camera_positions[index]]
    sample_poses = [poses[index]]
    cp_list = list(camera_positions)
    sp_list = list(poses)
    cp_list.pop(index)
    for i in range(num - 1):
        n = len(samples)
        m = len(cp_list)
        if m <= 0:
            break
        cp_array = np.repeat(np.expand_dims(np.array(cp_list), 0), n, axis=0)  # [n,m,3]
        samples_array = np.repeat(np.expand_dims(np.array(samples), 1), m, axis=1)  # [n,m,3]
        diff = np.sum((cp_array - samples_array) ** 2, axis=-1)  # [n,m]
        argmax = np.argmax(np.min(diff, axis=0))
        samples += [cp_list[argmax]]
        sample_poses += [sp_list[argmax]]
        cp_list.pop(argmax)
        sp_list.pop(argmax)
    sample_poses = np.array(sample_poses)

    return torch.tensor(sample_poses)  # [num, 3]


def make_random_extrinsic(sample, r_scale=1, z_offset=0, phi_low=np.pi / 4):  # make a random view on the sphere
    sample = invert(sample)
    r = torch.norm(sample[:3, 3])
    return generate_camera_point_spherical(r, r_scale=r_scale, z_offset=z_offset, phi_low=phi_low)


def get_top_view(sample):
    sample = invert(sample)
    r = torch.norm(sample[:3, 3])
    return generate_camera_point_spherical(r, theta=0, phi=np.pi / 2)


def inverse_projection(extrinsic, focal, depth, x, image_size=(512, 512)):
    intrinsic = get_intrinsic(focal, image_size=image_size).clone().detach().cpu().numpy()
    intrinsici = np.linalg.inv(intrinsic)
    pose = extrinsic[:3].detach().clone().cpu().numpy()

    far = np.max(depth) * 0.9
    roA = np.array([pose[:, 3]]).T
    rdA = pose @ intrinsici @ np.concatenate((x, np.ones((x.shape[0], 2))), axis=-1).T - roA
    rdA /= np.linalg.norm(rdA, axis=0, keepdims=True)

    f = (roA + depth.T * rdA).T
    f = f[depth[:, 0, 0] < far]

    points = (f).tolist()

    return points


def triangulate(A1, A2, extrinsic1, extrinsic2, focal, image_size=(512, 512)):
    num = A1.shape[0]
    intrinsic = get_intrinsic(focal, image_size=image_size).detach().clone().cpu().numpy()
    intrinsici = np.linalg.inv(intrinsic)
    extrinsic1 = extrinsic1[0].numpy()
    extrinsic2 = extrinsic2[0].numpy()

    roA = np.array([extrinsic1[:, 3]]).T
    roB = np.array([extrinsic2[:, 3]]).T

    rdAs = []
    rdBs = []
    for i in range(num):
        rdA = extrinsic1 @ intrinsici @ np.array([[A1[i, 0], A1[i, 1], 1, 1]]).T - roA
        rdA = rdA / np.linalg.norm(rdA)
        rdA = rdA[:3]
        rdAs += [rdA]
        rdB = extrinsic2 @ intrinsici @ np.array([[A2[i, 0], A2[i, 1], 1, 1]]).T - roB
        rdB = rdB / np.linalg.norm(rdB)
        rdB = rdB[:3]
        rdBs += [rdB]

    roB = roB[:3]
    roA = roA[:3]

    As = []
    for i in range(num):
        tA = np.dot(np.cross((roB.T[0] - roA.T[0]), rdBs[i].T[0]), np.cross(rdAs[i].T[0], rdBs[i].T[0])) / np.dot(
            (np.cross(rdAs[i].T[0], rdBs[i].T[0])), (np.cross(rdAs[i].T[0], rdBs[i].T[0])))
        PA = roA + tA * rdAs[i]

        tB = np.dot(np.cross((roA.T[0] - roB.T[0]), rdAs[i].T[0]), np.cross(rdBs[i].T[0], rdAs[i].T[0])) / np.dot(
            (np.cross(rdBs[i].T[0], rdAs[i].T[0])), (np.cross(rdBs[i].T[0], rdAs[i].T[0])))
        PB = roB + tB * rdBs[i]

        As += [(PA + PB) / 2]
    return torch.tensor(np.array(As))


def visualize_points(points, vis, win_name, title="Sample Points", change_color=True, snum=0):
    num = points.shape[0]
    points = points.view(num, 3)
    b = 2 if change_color else 1
    Ys = torch.tensor([1] * (num - snum) + [b] * (snum))

    vis.scatter(
        X=points,
        Y=Ys,
        win=win_name,
        opts=dict(
            title=title,
            markersize=2,
            xtickmin=-0.7,
            xtickmax=0.7,
            xtickstep=0.1,
            ytickmin=-0.7,
            ytickmax=0.7,
            ytickstep=0.1,
            ztickmin=0,
            ztickmax=2,
            ztickstep=0.1)
    )


def load_keypoints(file_p):
    with open(file_p) as file:
        t_dict = json.load(file)

    if t_dict["keypoints"] == 0:
        return None
    As = []
    A_exts = []
    for i in t_dict["keypoints"]['1'].keys():
        As += [np.array(t_dict["keypoints"]['1'][i])]
        with open(os.path.normpath(i)) as f:
            ext_dict = json.load(f)
        A_exts += [torch.tensor(
            np.array(ext_dict["transform_matrix"], dtype=np.float64))]

    Bs = []
    B_exts = []
    for i in t_dict["keypoints"]['2'].keys():
        Bs += [np.array(t_dict["keypoints"]['2'][i])]
        with open(os.path.normpath(i)) as f:
            ext_dict = json.load(f)
        B_exts += [torch.tensor(
            np.array(ext_dict["transform_matrix"], dtype=np.float64))]
    return As, Bs, A_exts, B_exts


def project_points(points, focal, pose, image_size=(512, 512)):
    intrinsic = get_intrinsic(focal, image_size=image_size)
    extrinsic = invert(pose[:3])
    extrinsic = extrinsic.detach().clone().cpu()
    extrinsic = torch.cat((extrinsic, torch.tensor([[0, 0, 0, 1.0]])), 0)
    intrinsic = intrinsic.detach().clone().cpu()
    p = torch.cat((points.detach().clone().cpu(), torch.ones(points.shape[0], 1)), -1)
    x = p @ extrinsic.transpose(0, 1)

    x = x / x[:, 2].view(x.shape[0], 1)

    return (x @ intrinsic.transpose(0, 1))[:, :2]


def jitter_points(samples, d):  # [1,n,3]
    noise = (torch.rand(samples.shape) - 0.5) * 2
    norm = torch.sqrt(torch.sum(noise ** 2, axis=-1)).unsqueeze(-1)
    noise = noise / norm * d
    return samples + noise.to(samples.device)


def find_min_dist(new_set, samples):  # [1,n,3], [1,n,3]

    n = samples.shape[1]
    nn = new_set.unsqueeze(1).repeat((1, n, 1, 1))
    ss = samples.unsqueeze(-2).repeat((1, 1, n, 1))
    return torch.min(torch.sum((nn - ss) ** 2, -1), axis=1)[0]  # [1,n]


def aux_initial_samples(kps, wrapper1, sigmas, rho):  # add 10 auxiliary points close to initial keypoints
    smpls = []
    for i in range(10):
        new_samples = jitter_points((kps[:, :, 0]).unsqueeze(0).float(), rho * (3 ** (0.5)) / 5)

        opds = wrapper1.get_surface_value(new_samples, sigmas).squeeze(0)
        maxx = torch.max(opds)
        tthresh = maxx * np.exp(-2)  # two sigmas away

        for i in range(new_samples.shape[1]):
            if opds[i] >= tthresh:
                smpls += [new_samples[0, i].numpy()]

    return np.array(smpls)


def draw_new_samples(samples, rho, R, t, wrapper1, wrapper2, c, sigmas, obj_bigness, scale=1):
    new_samples = jitter_points(samples, rho)
    dists = find_min_dist(new_samples, samples)
    smpls = torch.tensor([]).to(samples.device)

    surface_1_prev = wrapper1.get_surface_value(samples, sigmas)
    surface_1_prev = surface_1_prev.transpose(0, 1).clone().detach().cpu().numpy()
    maxx = np.max(surface_1_prev)
    thresh = 15 * c
    thresh_2 = maxx * np.exp(-2)  # 2 sigmas away

    for i in range(samples.shape[1]):  # filter previous samples
        if surface_1_prev[i] >= thresh_2:
            smpls = torch.cat((smpls, samples[0, i].unsqueeze(0)))

    if smpls.shape[0] >= 3000:  # prevent sample size from getting too big, slows down registration
        return smpls.unsqueeze(0)

    W = torch.matmul(scale * new_samples, R.transpose(0, 1)) + t

    surface_1 = wrapper1.get_surface_value(new_samples, sigmas).transpose(0, 1).clone().detach().cpu().numpy()
    surface_2 = wrapper2.get_surface_value(W, scale * sigmas).transpose(0, 1).clone().detach().cpu().numpy()
    delta = np.abs(surface_2 - surface_1)

    for i in range(new_samples.shape[1]):
        if dists[0, i] >= (rho * obj_bigness / 10) ** 2 and delta[i] <= thresh and surface_1[i] >= thresh_2 and \
                surface_2[i] >= thresh_2:
            smpls = torch.cat((smpls, new_samples[0, i].unsqueeze(0)))

    return smpls.unsqueeze(0)
