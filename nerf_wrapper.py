import torch
from PIL import Image
import numpy as np
import json
import os
from distill import Distilled_MLP
import utils as uu


class NeRFWrapper():
    def __init__(self, nerf_path, distilled_path):
        self.nerf = torch.jit.load(nerf_path).cuda()
        self.L = 7
        self.distill = Distilled_MLP(6 * self.L + 3)
        self.distill.load_state_dict(torch.load(distilled_path))
        self.distill.eval()
        self.distill.to('cuda')

    def get_surface_value(self, xyzs, sigma, device=torch.device("cuda")):
        xyzs = xyzs[0].to(device)
        sigmas = sigma.repeat(xyzs.shape[0], 1)
        xs_enc = self.distill.integrated_positional_encoding(xyzs, self.L, sigmas)
        prob = self.distill(xs_enc).transpose(0, 1)
        return prob

    def query_nerf(self, positions, directions):
        rgb, density = self.nerf.call_eval(positions, directions)
        return rgb, density

    def query_transmittance(self, positions, cam_origin):
        transmittance = self.nerf.transmittance_eval(positions, cam_origin)
        return transmittance

    def composite(self, ray, rgb_samples, density_samples, depth_samples, setbg_opaque=False):
        ray_length = ray.norm(dim=-1, keepdim=True).squeeze(-1)  # [HW,N,1]
        depth_intv_samples = depth_samples[..., 1:, 0] - depth_samples[..., :-1, 0]  # [HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples, torch.empty_like(depth_intv_samples[..., :1]).fill_(1e10)],
                                       dim=2)  # [HW,N]
        dist_samples = depth_intv_samples * ray_length  # [HW,N]
        sigma_delta = density_samples.squeeze(-1) * dist_samples  # [HW,N]
        alpha = 1 - (-sigma_delta).exp_()  # [HW,N]
        T = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=2).cumsum(
            dim=2)).exp_()  # [HW,N]
        prob = (T * alpha)[..., None]  # [HW,N,1]
        rgb = (rgb_samples * prob).sum(dim=2)  # [HW,4]

        opacity = prob.sum(dim=2)  # [HW,1]
        depth = (depth_samples * prob).sum(dim=2) / (opacity + 1e-7)  # [HW,1]
        if setbg_opaque:
            rgb = rgb + (1.0 - opacity)
        return rgb, depth, opacity, prob  # [HW,K], [HW,N,1]

    def sample_depth(self,
                     batch_size,
                     num_rays=None,
                     near=0.3,
                     far=0.8,
                     resolution=(512, 512),
                     sample_intvs=500,
                     nerf_depth_param='metric',
                     ):
        depth_min, depth_max = near, far
        H, W = resolution[0], resolution[1]
        num_rays = num_rays or H * W

        rand_samples = torch.rand((batch_size, num_rays, sample_intvs, 1))
        rand_samples += torch.arange(sample_intvs)[None, None, :, None].float()  # [B,HW,N,1]
        depth_samples = rand_samples / sample_intvs * (depth_max - depth_min) + depth_min  # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1 / (depth_samples + 1e-8),
        )[nerf_depth_param]
        return depth_samples

    def render_image_fine(self,
                          pose,
                          near=0.3,
                          far=0.8,
                          focal_length=None,
                          principal_point=None,
                          resolution=(512, 512),
                          fine_samples=256,
                          coarse_samples=1024,
                          chunk=256,
                          save_p=None,
                          index=None,
                          scale=1,
                          device=torch.device('cuda')):
        inv_scale = 1 / scale
        total_samples = coarse_samples + fine_samples

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
        pose = pose.clone().detach().cpu()

        ws_ip_points = uu.apply_transform(pose, ip_points) * inv_scale
        ws_eye = uu.apply_transform(pose, torch.zeros((1, 3))) * inv_scale
        eyes_tiled = ws_eye.repeat(resolution[0] * resolution[1], 1)  # [HW, 3]
        ray_dirs = ws_ip_points - eyes_tiled
        ray_dirs /= torch.linalg.norm(ray_dirs, axis=-1, keepdims=True)  # [HW, 3]
        rays = torch.cat([eyes_tiled, ray_dirs], axis=-1).to(device)

        rgb = torch.zeros((rays.shape[0], 3)).to(device)
        mask = torch.zeros((rays.shape[0], 1)).to(device)

        batch_count = torch.tensor(rays.shape[0] / chunk)
        batch_count = torch.ceil(batch_count).long()

        for n in torch.arange(batch_count):
            rgba_n, depth_n = self.nerf.trace_eval(  # coarse and fine sampling + render incorporated in nerf jit model
                rays[n * chunk:(n + 1) * chunk].to(device),
                torch.tensor(total_samples).to(device),
                torch.tensor(coarse_samples).to(device),
                torch.tensor(near).to(device),
                torch.tensor(far).to(device))

            rgb[n * chunk:(n + 1) * chunk] = rgba_n[..., :3].detach()
            mask[n * chunk:(n + 1) * chunk] = rgba_n[..., 3].unsqueeze(-1).detach()

        rgb = rgb + (1 - mask)
        rgb = rgb.reshape(resolution[0], resolution[1], 3)
        if save_p:
            image = rgb.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))
            image.save(os.path.join(save_p, "rgb_" + str(index)))
            data = {"transform_matrix": pose.clone().detach().cpu().numpy().tolist()}
            with open(os.path.join(save_p, str(index) + ".json"), 'w') as outfile:
                json.dump(data, outfile)
        return rgb[..., :3]

    def render_fast(self,
                    pose,
                    near=0.3,
                    far=0.8,
                    focal_length=None,
                    principal_point=None,
                    resolution=(512, 512),
                    chunk=256,
                    batch_size=1,
                    device=torch.device('cuda')):  # use this function for faster rendering but less psnr

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
        ip_points = ip_points.to(pose.device)

        ws_ip_points = uu.apply_transform(pose, ip_points)
        ws_eye = uu.apply_transform(pose, torch.zeros((1, 3)).to(pose.device))
        eyes_tiled = ws_eye.repeat(resolution[0] * resolution[1], 1)  # [HW, 3]
        ray_dirs = ws_ip_points - eyes_tiled
        ray_dirs /= torch.linalg.norm(ray_dirs, axis=-1, keepdims=True)  # [HW, 3]
        rays = torch.cat([eyes_tiled, ray_dirs], axis=-1).to(device)

        rgb = torch.zeros((rays.shape[0], 3)).to(device)
        mask = torch.zeros((rays.shape[0], 1)).to(device)
        depth = torch.zeros((rays.shape[0], 1)).to(device)

        batch_count = torch.tensor(rays.shape[0] / chunk)
        batch_count = torch.ceil(batch_count).long()

        depths = self.sample_depth(batch_size, near=near, far=far)[0].to(pose.device)  # [HW,N,1]
        positions = eyes_tiled[:, None] + ray_dirs[:, None] * depths  # [HW,N,3]

        rays = ray_dirs[:, None].repeat((1, depths.shape[1], 1))  # [HW,N,3]

        for n in torch.arange(batch_count):
            positions_chunk = positions[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)
            rays_chunk = rays[n * chunk:(n + 1) * chunk].reshape(chunk * depths.shape[1], 3)

            rgb_n, density_n = self.nerf.call_eval(positions_chunk.to(device), rays_chunk.to(device))
            density_n = density_n.reshape(chunk, depths.shape[1])
            rgb_n = rgb_n.reshape(chunk, depths.shape[1], 3)

            sample_depth = depths[n * chunk:(n + 1) * chunk, :, 0].to(device)

            intervals = sample_depth[..., 1:] - sample_depth[..., :-1]
            before_intervals = torch.cat([intervals[..., :1], intervals], axis=-1)
            after_intervals = torch.cat([intervals, intervals[..., :1]], axis=-1)
            delta = (before_intervals + after_intervals) / 2

            quantity = delta * density_n

            cumsumex = lambda t: torch.cumsum(torch.cat(
                [torch.zeros_like(t[..., :1]), t[..., :-1]], axis=-1),
                axis=-1)
            transmittance = torch.exp(cumsumex(-quantity))[..., None]
            cross_section = 1.0 - torch.exp(-quantity)[..., None]

            weights = transmittance * cross_section

            radiance = torch.sum(weights * rgb_n, axis=-2)
            alpha = torch.sum(weights, axis=-2)
            depth_n = torch.sum(weights * sample_depth[..., None],
                                axis=-2)

            rgb[n * chunk:(n + 1) * chunk] = radiance.detach()
            mask[n * chunk:(n + 1) * chunk] = alpha.detach()
            depth[n * chunk:(n + 1) * chunk] = depth_n.detach()

        rgb = rgb.reshape(resolution[0], resolution[1], 3)
        mask = mask.reshape(resolution[0], resolution[1], 1)
        depth = depth.reshape(resolution[0], resolution[1], 1)

        rgb = rgb + (1.0 - mask)
        depth = depth + (1.0 - mask) * far  # far plane
        return rgb, depth


if __name__ == "__main__":  # to test the wrapper
    root = os.path.join(os.getcwd(), "scenes", "scene_" + str(1))
    wrapper = NeRFWrapper(os.path.join(root, 'b', 'nerf_model_b.pt'), os.path.join(root, 'b', 'distilled_b_1.ckpt'))

    focal, poses = uu.load_poses(os.path.join(root, "transforms.json"), (512, 512))
    pose = poses[0]

    print("Rendering frame, this may take a few minutes.")
    image = wrapper.render_image_fine(pose.float(),
                                      resolution=(512, 512))

    image = image.cpu().numpy() * 255
    image = Image.fromarray(image.astype(np.uint8))
    image.save("image.png")
