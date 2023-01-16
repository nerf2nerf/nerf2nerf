import argparse
import json
import io
import PIL.Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
from torch.nn import MSELoss
from torch.nn import Linear
from torch.optim import Adam
import matplotlib.pyplot as plt
import utils as uu
import os


def get_focal(file, resolution=(512, 512)):
    raw_H, raw_W = resolution[0], resolution[1]
    meta_fname = file
    with open(meta_fname) as file:
        meta = json.load(file)
    focal = 0.5 * raw_W / np.tan(0.5 * meta["camera_angle_x"])
    return focal


class Teacher():
    def __init__(self, path, device=torch.device("cuda")):
        self.nerf = torch.jit.load(path).to(device)

    def get_prob(self,
                 extrinsics,
                 focal_length,
                 level,
                 sigma,
                 cameras,
                 num_rays=128,
                 resolution=(512, 512),
                 device=torch.device("cuda"),
                 full_images=False, s_points=None):

        g_samples = 10  # guassian smoothing number of samples
        delta = 0.05
        cut_off = 0.5
        H, W = resolution[0], resolution[1]
        poses = extrinsics[:, :3, :]

        batch_size = poses.shape[0]
        principal_point = (resolution[0] / 2, resolution[1] / 2)

        x = torch.arange(resolution[0]) + 0.5
        y = torch.arange(resolution[1]) + 0.5

        x, y = torch.meshgrid(x, y, indexing="xy")
        ip_points = torch.stack([
            (x - principal_point[0]) / focal_length,
            (y - principal_point[1]) / focal_length,
            torch.ones_like(x),
        ], axis=-1).reshape((-1, 3))

        ip_points = ip_points.repeat(batch_size, 1, 1).to(poses.device)  # [B, HW, 3]
        ws_ip_points = uu.apply_transform(poses, ip_points)
        centers = uu.apply_transform(poses, torch.zeros_like(ip_points).to(poses.device))
        ray_dirs = ws_ip_points - centers
        ray_dirs /= torch.linalg.norm(ray_dirs, axis=-1, keepdims=True)  # [B, HW, 3]

        if full_images:
            # center[z] + d*ray[z] = level -> d = (level - center[z])/ray[z]
            depth_samples = ((level - centers[:, :, 2]) / ray_dirs[:, :, 2])
            depth_samples = depth_samples.unsqueeze(-1).unsqueeze(-1)  # [B,HW,N=1,1]
            points_3D_samples_orig = (
                    centers[:, :, None] + ray_dirs[:, :, None] * depth_samples).clone().detach().cpu()  # [B,HW,N,3]
        else:
            points_3D_samples_orig = s_points[:, :, None].clone().detach().cpu()  # [B,M,N,3]

        # add Guassain noise to samples
        points_3D_samples = points_3D_samples_orig.unsqueeze(-2).repeat(1, 1, 1, g_samples, 1)  # [B,HW, N, N', 3]
        shape = points_3D_samples.shape[:-1]
        noise = torch.stack((torch.normal(0., sigma[0], size=shape), torch.normal(0., sigma[1], size=shape),
                             torch.normal(0., sigma[2], size=shape)), axis=-1)
        points_3D_samples += noise  # [B,HW, N,N', 3]

        # account for different view directions, reshape to match points shape
        camera_samples = cameras[None, :, None, None, :].expand(
            (shape[0], cameras.shape[0], shape[1], shape[2], 3)).float()  # [B,V,HW,N,3]

        points_3D_samples = points_3D_samples.to(device)
        camera_samples = camera_samples.to(device)

        prob_samples = None
        if full_images:
            ret_all = []
            for c in range(0, H * W, num_rays):
                ray_idx = torch.arange(c, min(c + num_rays, H * W), device=device)
                ppoints = points_3D_samples[0, ray_idx, 0].unsqueeze(0).repeat((cameras.shape[0], 1, 1, 1))
                ppoints_shape = ppoints.shape
                ppoints = ppoints.reshape(ppoints_shape[0] * ppoints_shape[1] * ppoints_shape[2], ppoints_shape[3])
                pcameras = camera_samples[0, :, ray_idx, 0].unsqueeze(-2).repeat(1, 1, g_samples, 1).reshape(
                    ppoints_shape[0] * ppoints_shape[1] * ppoints_shape[2], ppoints_shape[3])
                op_sample, transmitt_sample = self.get_transmittance_and_opacity(delta, pcameras, ppoints,
                                                                                 ppoints_shape)

                s_sample = (op_sample * transmitt_sample).max(axis=0)[0]
                s_sample = (s_sample >= cut_off).float()  # make_binary
                pprob_samples = s_sample.mean(axis=-2).unsqueeze(0)

                ret_all.append(pprob_samples.clone().detach())
            # group all slices of images
            ret_all = torch.cat(ret_all, dim=1)
            prob_samples = ret_all.view(-1, W, H, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
        else:
            for g in range(g_samples):
                all_s_sample = None
                for c in range(cameras.shape[0]):
                    ppoints = points_3D_samples[:, :, :, g, :]
                    ppoints_shape = ppoints.shape
                    ppoints = ppoints.reshape(ppoints_shape[0] * ppoints_shape[1] * ppoints_shape[2], ppoints_shape[3])
                    pcameras = camera_samples[:, c, :, :, :].reshape(
                        ppoints_shape[0] * ppoints_shape[1] * ppoints_shape[2], ppoints_shape[3])
                    op_sample, transmitt_sample = self.get_transmittance_and_opacity(delta, pcameras, ppoints,
                                                                                     ppoints_shape)
                    s_sample = transmitt_sample * op_sample
                    if all_s_sample is None:
                        all_s_sample = s_sample  # [B,M,N,1]
                    else:
                        all_s_sample = torch.maximum(s_sample, all_s_sample)
                all_s_sample = (all_s_sample >= cut_off).float()  # make binary
                if prob_samples is None:
                    prob_samples = all_s_sample
                else:
                    prob_samples += all_s_sample
            prob_samples /= g_samples

        return prob_samples, points_3D_samples_orig  # [B,M,N,1] | [B,1,H,W] , [B,M,N,3] | [B,HW,N,3]

    def get_transmittance_and_opacity(self, delta, pcameras, ppoints, ppoints_shape):
        prays = ppoints - pcameras
        prays /= torch.linalg.norm(prays, axis=-1, keepdims=True)
        density_sample = self.nerf.call_eval(ppoints, prays)[1].unsqueeze(-1).clone().detach().cpu()
        density_sample = density_sample.reshape(ppoints_shape[0], ppoints_shape[1], ppoints_shape[2],
                                                1)
        op_sample = 1 - torch.exp(-density_sample * delta)
        transmitt_sample = self.nerf.transmittance_eval(ppoints, pcameras).unsqueeze(
            -1).clone().detach().cpu()
        transmitt_sample = transmitt_sample.reshape(ppoints_shape[0], ppoints_shape[1], ppoints_shape[2], 1)
        return op_sample, transmitt_sample


class Distilled_MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(Distilled_MLP, self).__init__()

        self.layer1 = (Linear(n_inputs, 256))
        self.activation1 = ReLU()
        self.layer2 = (Linear(256, 256))
        self.activation2 = ReLU()
        self.layer3 = (Linear(256, 256))
        self.activation3 = ReLU()
        self.layer4 = (Linear(256, 256))
        self.activation4 = ReLU()
        self.layer5 = (Linear(256, 256))
        self.activation5 = ReLU()
        self.layer6 = (Linear(256, 256))
        self.activation6 = ReLU()
        self.layer7 = (Linear(256, 256))
        self.activation7 = ReLU()
        self.layer8 = (Linear(256, 1))
        self.activation8 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        X = self.layer1(X)
        X = self.activation1(X)

        X = self.layer2(X)
        X = self.activation2(X)

        X = self.layer3(X)
        X = self.activation3(X)

        X = self.layer4(X)
        X = self.activation4(X)

        X = self.layer5(X)
        X = self.activation5(X)

        X = self.layer6(X)
        X = self.activation6(X)

        X = self.layer7(X)
        X = self.activation7(X)

        X = self.layer8(X)
        X = self.activation8(X)

        return X

    def integrated_positional_encoding(self, inputs, L, sigma):
        shape = inputs.shape
        dim = shape[1]
        freq = 2 ** torch.arange(L, device=inputs.device, dtype=torch.float32) * np.pi  # [L]
        freq2 = 4 ** torch.arange(L, device=inputs.device, dtype=torch.float32) * (np.pi ** 2)  # [L]
        freq2 = freq2.reshape((1, L))
        diag_covs = []
        for i in range(dim):
            diag_covs += [torch.exp(-0.5 * (sigma[:, i].unsqueeze(-1) ** 2) @ freq2).unsqueeze(-2)]
        diag_cov = torch.cat(diag_covs, axis=-2)

        spectrum = inputs[..., None] * freq  # [N,3,L]

        sin, cos = spectrum.sin(), spectrum.cos()  # [N,3,L]

        input_enc = torch.stack([diag_cov * sin, diag_cov * cos], dim=-2)  # [N,2,3,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [N,6L]

        input_enc = torch.cat([inputs, input_enc], dim=-1)  # [N,3+6L]

        return input_enc


def render_prob(pose, focal_length, level, L, sigma, model, resolution=(512, 512)):  # sigma (1,3)
    poses = pose[:, :3, :]
    batch_size = poses.shape[0]
    principal_point = (resolution[0] / 2, resolution[1] / 2)

    x = torch.arange(resolution[0]) + 0.5
    y = torch.arange(resolution[1]) + 0.5

    x, y = torch.meshgrid(x, y, indexing="xy")
    ip_points = torch.stack([
        (x - principal_point[0]) / focal_length,
        (y - principal_point[1]) / focal_length,
        torch.ones_like(x),
    ], axis=-1).reshape((-1, 3))

    ip_points = ip_points.repeat(batch_size, 1, 1).to(poses.device)  # [B, HW, 3]
    ws_ip_points = uu.apply_transform(poses, ip_points)
    centers = uu.apply_transform(poses, torch.zeros_like(ip_points).to(poses.device))
    ray_dirs = ws_ip_points - centers
    ray_dirs /= torch.linalg.norm(ray_dirs, axis=-1, keepdims=True)  # [B, HW, 3]

    depth_samples = (level - centers[:, :, 2]) / ray_dirs[:, :, 2]
    depth_samples = depth_samples.unsqueeze(-1).unsqueeze(-1)  # [B=1,HW,N=1,1]

    points_3D_samples_orig = (centers[:, :, None] + ray_dirs[:, :, None] * depth_samples).clone().detach()  # [B,HW,N,3]
    points = points_3D_samples_orig.view(depth_samples.shape[0] * depth_samples.shape[1] * depth_samples.shape[2],
                                         3)  # [HW,3]

    sigma = sigma.repeat(points.shape[0], 1).clone().detach()
    points_enc = model.integrated_positional_encoding(points, L, sigma)
    probs_hat = model(points_enc)  # [HW, 1]
    probs_hat = probs_hat.view(resolution[0], resolution[1], 1)
    return probs_hat


def show_pred_image(sig, level, top_view, focal, L, model):
    path = os.path.join(root, "distill_output")
    if not os.path.isdir(path):
        os.makedirs(path)
    rgb = render_prob(top_view, focal, level, L, sig, model)
    f = plt.figure()
    rgb = rgb.detach().clone().cpu().numpy()
    rgb[0, 0] = 0
    rgb[0, 1] = 1.
    plt.imshow(rgb)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.savefig(os.path.join(path, "predicted.png"))
    plt.show()
    f.clear()
    plt.close(f)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    writer.add_image('predicted', image, i)


def show_gt_image(sig, teacher_net, level, cameras):
    f = plt.figure()
    rgb = teacher_net.get_prob(top_view, focal, level, sig[0], cameras, full_images=True)[
        0].clone().cpu().numpy()
    rgb[0, 0, 0] = 0
    rgb[0, 0, 1] = 1.
    rgb = np.moveaxis(rgb[0], 0, -1)
    plt.imshow(rgb)
    plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.savefig(os.path.join(root, "distill_output", "gt.png"))
    plt.show()
    f.clear()
    plt.close(f)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    writer.add_image('GT', image, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--scene_no', type=int)
    parser.add_argument('-ab', '--a_or_b')
    args = parser.parse_args()
    real = True if args.scene_no == 4 or args.scene_no == 5 else False

    root = os.path.join(os.getcwd(), "scenes", "scene_" + str(args.scene_no))
    focal = get_focal(os.path.join(root, "transforms.json"))
    cameras = uu.sample_cameras(os.path.join(root, "transforms.json"), 15)[:, :, 3]

    path = os.path.join(root, args.a_or_b, 'nerf_model_' + args.a_or_b + '.pt')
    device = torch.device("cuda")
    teacher_net = Teacher(path)
    L = 7
    sigma_levels = [[0.0, 0.0, 0.0], [0.004, 0.004, 0.004], [0.008, 0.008, 0.008]]
    probs, xs, sigmas = [], [], []

    top_view = uu.get_top_view(uu.load_sample_extrinsic(os.path.join(root, "transforms.json"), 0)).unsqueeze(0).to(
        device).float()
    iters = 20
    balancing_iters = 50

    if not real:
        if args.scene_no < 4:
            bounds_shift = torch.tensor([0.5, 0.5, 0])  # scene bounding box center and range
            bounds_range = torch.tensor([1.2, 1.2, 0.6])
        else:
            bounds_shift = torch.tensor([0.5, 0.5, -1.75])
            bounds_range = torch.tensor([1.2, 1.2, 0.2])
    elif args.scene_no == 4:
        bounds_shift = torch.tensor([0.5, 0.5, -2.75])
        bounds_range = torch.tensor([1.2, 1.2, 0.32])
    elif args.scene_no == 5:
        bounds_shift = torch.tensor([0.5, 0.5, -1.83])
        bounds_range = torch.tensor([1.2, 1.2, 0.3])

    for i in range(iters):
        print("Gathering dataset...iteration:", i + 1, "/", iters)
        s_points = ((torch.rand((1, 10 ** 5, 3)) - bounds_shift) * bounds_range).to(
            device)
        for sigma in sigma_levels:
            prob, x = teacher_net.get_prob(top_view, focal, None, torch.tensor(sigma), cameras, s_points=s_points)
            probs += [prob]
            xs += [x]
            sigmas += [sigma] * prob.shape[1]

    probs = torch.cat(probs, axis=0)
    xs = torch.cat(xs, axis=0)
    xs = xs.view(xs.shape[0] * xs.shape[1] * xs.shape[2], 3)
    probs = probs.view(probs.shape[0] * probs.shape[1] * probs.shape[2], 1)

    ones = xs[(probs > torch.quantile(probs, 0.97)).repeat(1, 3)].reshape(-1, 3)  # balance 0-1 classes

    for s in [0.001, 0.01]:  # add new data points near ones
        for i in range(balancing_iters):
            x_new = (ones + torch.normal(0, s, size=ones.size())).unsqueeze(0)
            for sigma in sigma_levels:
                prob, x = teacher_net.get_prob(top_view, focal, None, torch.tensor(sigma), cameras, s_points=x_new)
                prob = prob.view(prob.shape[0] * prob.shape[1] * prob.shape[2], 1)
                x = x.view(x.shape[0] * x.shape[1] * x.shape[2], 3)
                xs = torch.cat((xs, x), axis=0)
                probs = torch.cat((probs, prob), axis=0)
                sigmas += [sigma] * prob.shape[0]
    probs = probs.detach().clone().float().to(device)
    xs = xs.detach().clone().float().cpu()
    idx = torch.randperm(probs.shape[0])
    probs = probs[idx]
    xs = xs[idx]
    sigmas = torch.tensor(sigmas).float().cpu()
    sigmas = sigmas[idx]
    print("Data collection finished.")

    model = Distilled_MLP(6 * L + 3).to(device)
    model.scale = 1
    xs_enc = model.integrated_positional_encoding(xs, L, sigmas).to(device)
    sigmas = sigmas.to(device)

    writer = SummaryWriter(comment='views')
    epochs = 60
    batch_size = 2048
    iters = xs.shape[0] // batch_size

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    ep = 1e-10
    save_after = 10
    level = 0.37  # horizontal cut z-value to show, might need to move up or down for different scenes to catch a good cut

    for i in range(epochs):
        for j in range(iters):
            x_i = xs_enc[j * batch_size: (j + 1) * batch_size]
            probs_i = probs[j * batch_size: (j + 1) * batch_size].flatten()
            probs_hat = model(x_i)
            loss = torch.mean(probs_hat.flatten() - probs_i * torch.log(probs_hat.flatten() + ep))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("loss:", i, loss)
        if i % save_after == 0:
            torch.save(model.state_dict(), os.path.join(root, args.a_or_b, "distilled_" + args.a_or_b + "_7.ckpt"))

            # adjust sigma and level
            show_pred_image(torch.tensor([[0.0, 0.0, 0.0]]).float().to(device), level, top_view, focal, L, model)
            if i == 0:
                show_gt_image(torch.tensor([[0.0, 0.0, 0.0]]).float().to(device), teacher_net, level, cameras)
