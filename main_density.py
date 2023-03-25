import torch
from nerf_wrapper import NeRFWrapper
import utils as uu
import numpy as np
import torch.nn as nn
import robust_loss_pytorch.general
import visdom
import json
import os
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='bust_no_vis')
    args = parser.parse_args()
    with open(os.path.join(os.getcwd(), 'options', args.yaml + ".yaml"), "r") as f:
        opt = uu.AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

    np.random.seed(seed=10)
    root = os.path.join(os.getcwd(), "scenes", "scene_" + str(opt.scene_no))
    writer = SummaryWriter(comment='views')
    kp_iters = 2000 if not opt.real else 4000
    vis = None
    if opt.use_vis:
        is_open = uu.check_socket_open('localhost', opt.vis_port)
        retry = None
        while not is_open:
            retry = input("visdom port ({}) not open, retry? (y/n) ".format(opt.vis_port))
            if retry not in ["y", "n"]: continue
            if retry == "y":
                is_open = uu.check_socket_open('localhost', opt.vis_port)
            else:
                break
        vis = visdom.Visdom(server='localhost', port=opt.vis_port, env='Reg')

    with torch.cuda.device(opt.device):
        wrapper1 = NeRFWrapper(os.path.join(root, 'a', 'nerf_model_a.pt'),
                               os.path.join(root, 'a', 'distilled_a_1.ckpt'))

        wrapper2 = NeRFWrapper(os.path.join(root, 'b', 'nerf_model_b.pt'),
                               os.path.join(root, 'b', 'distilled_b_1.ckpt'))

        focal, poses = uu.load_poses(os.path.join(root, 'transforms.json'), opt.image_size)
        sample_ext = poses[0]
        p1 = torch.concat((sample_ext, torch.tensor([[0, 0, 0.0, 1]]).to(sample_ext.device)), axis=0)
        s1 = uu.make_random_extrinsic(sample_ext, r_scale=opt.r_scale, z_offset=opt.z_offset,
                                      phi_low=opt.phi_low).unsqueeze(0).to(
            opt.device).float()

        if opt.extract_pc:
            points = []
            sample_exts = uu.sample_cameras(os.path.join(root, "transforms.json"), 8)
            counter = 0
            for i in sample_exts:
                rgbs, depth = wrapper2.render_fast((i).unsqueeze(0).to(opt.device).float(), focal_length=focal,
                                                   near=opt.near, far=opt.far, chunk=opt.chunk)

                samples = torch.tensor([[2 * i, 2 * j] for i in range(256) for j in
                                        range(256)])  # use every other pixel for less noise and faster extraction
                sample_depth = depth[samples[:, 1], samples[:, 0]].unsqueeze(-1)

                points += uu.inverse_projection(i, focal,
                                                sample_depth.clone().detach().cpu().numpy(),
                                                samples.clone().detach().cpu().numpy(), image_size=opt.image_size)
                print(counter)
                counter += 1
                uu.visualize_points(torch.tensor(np.array(points)), vis, 'keys', title='surfaces')
            with open(root + '/b/extracted_pc.json', 'w') as f:
                json.dump(points, f)

            print("Sample surface points generated.")
            exit(0)

        # get ground truth poses and gt object point cloud model if available (synth)
        if not opt.real:
            with open(os.path.join(root, "gt_transform.json")) as file:
                gt_poses = json.load(file)

            with open(os.path.join(root, "object_point_clouds", opt.object_name + ".json")) as file:
                obj_pc = np.array(json.load(file))

            A = np.array(gt_poses[opt.object_name]['scene_a'])
            B = gt_poses[opt.object_name]['scene_b']
            T_gt = B @ np.linalg.inv(A)
            euler_gt = uu.get_Euler(T_gt[:3, :3]) * 180 / np.pi
            trans_gt = T_gt[:3, 3]
        else:
            T_gt = None
            euler_gt = None
            trans_gt = None
            obj_pc = None

        try:
            k = uu.load_keypoints(os.path.join(root, opt.object_name + "_keypoints.json"))
        except:
            # make key-point annotation images
            for i in range(2):
                ext = uu.make_random_extrinsic(sample_ext, r_scale=opt.r_scale, z_offset=opt.z_offset,
                                               phi_low=opt.phi_low).unsqueeze(0).to(opt.device).float()
                wrapper1.render_image_fine(ext, focal_length=focal, save_p=os.path.join(root, "a"), index=str(i + 1),
                                           near=opt.near, far=opt.far, chunk=opt.chunk)

            for i in range(2):
                ext = uu.make_random_extrinsic(sample_ext, r_scale=opt.r_scale, z_offset=opt.z_offset,
                                               phi_low=opt.phi_low).unsqueeze(0).to(opt.device).float()
                wrapper2.render_image_fine(ext, focal_length=focal, save_p=os.path.join(root, "b"), index=str(i + 1),
                                           near=opt.near, far=opt.far, chunk=opt.chunk)

            print("Please fill keypoints file.")
            exit(0)

        As_2d, Bs_2d, A_exts, B_exts = k[0], k[1], k[2], k[3]
        As = uu.triangulate(As_2d[0], As_2d[1], A_exts[0], A_exts[1], focal, image_size=opt.image_size)
        Bs = uu.triangulate(Bs_2d[0], Bs_2d[1], B_exts[0], B_exts[1], focal, image_size=opt.image_size)

        obj_diameter = np.linalg.norm(
            np.max(As.detach().clone().cpu().numpy(), axis=0) - np.min(As.detach().clone().cpu().numpy(), axis=0))
        if opt.use_vis:
            uu.visualize_points(torch.cat((As * opt.scale, Bs), 0), vis, 'keys', snum=len(Bs),
                                title='Key Points')
        r = float(torch.norm(sample_ext[:3, 3] * opt.scale))
        rho = r * (3 ** (0.5) / 100)  # diameter of a grid cell of a 100x100 resolution of the space
        sigma_h = 0.008 if opt.sigma_h is None else opt.sigma_h
        sigma_l = 0.004 if opt.sigma_l is None else opt.sigma_l
        sigmas = torch.tensor([[sigma_h, sigma_h, sigma_h]]).float().to(opt.device)
        sigmas_f = torch.tensor([[sigma_l, sigma_l, sigma_l]]).float().to(opt.device)

        samples = np.concatenate(
            (As[:, :, 0].clone().detach().numpy(), uu.aux_initial_samples(As, wrapper1, sigmas, rho)))

        if opt.use_vis:
            uu.visualize_points(torch.cat([torch.tensor(samples), As[0].T], axis=0), vis,
                                'samples', snum=len(As)
                                )
        X = torch.tensor(samples).to(opt.device).float().unsqueeze(0)

        As = As.view(As.shape[0], 3).detach().clone().float().to(opt.device)
        Bs = Bs.view(Bs.shape[0], 3).detach().clone().float().to(opt.device)

        # fixed_surface_values = wrapper1.get_surface_value(X, sigmas)
        # fixed_surface_values = fixed_surface_values.detach().clone()
        fixed_density_values = wrapper1.query_density(X.squeeze(0)).unsqueeze(0)
        fixed_density_values = fixed_density_values.detach().clone()

        thetac = nn.Parameter(torch.tensor((0.00001)).to(opt.device), requires_grad=True, ).float()
        alphac = nn.Parameter(torch.tensor((0.00001)).to(opt.device), requires_grad=True, ).float()
        gammac = nn.Parameter(torch.tensor((0.00001)).to(opt.device), requires_grad=True, ).float()

        t = nn.Parameter(torch.tensor([[0.0001, 0.0001, 0.0001]]).to(opt.device), requires_grad=True)

        optimizer1 = torch.optim.Adam(
            [{'params': [thetac, alphac, gammac], 'lr': 1e-1}, {'params': [t], 'lr': 5e-2}])

        ##### keypoint solution #####
        for time in range(kp_iters):
            R = uu.make_rot_matrix(thetac, alphac, gammac)
            delta = torch.mean((torch.matmul(opt.scale * As, R.transpose(0, 1)) + t - Bs) ** 2)
            loss = delta
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
        uu.write_log(writer, -1, obj_pc, R, t, 0, 0, 0, 0, T_gt, euler_gt, trans_gt, real=opt.real)
        ########
        if opt.use_vis:
            uu.visualize_points(torch.cat((torch.matmul(opt.scale * As, R.transpose(0, 1)) + t, Bs), 0), vis,
                                'keys', snum=len(Bs), title='Key Points')

        w2 = 1
        sigmas_i = sigmas.clone().detach()
        print("optimization start!")
        adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=torch.float, device=opt.device, alpha_lo=1e-6, alpha_hi=1.0,
            scale_init=1)
        optimizer = torch.optim.Adam(
            [{'params': [thetac, alphac, gammac], 'lr': 2e-2 if not opt.lr else opt.lr},
             {'params': [t], 'lr': 1e-2 if not opt.lr else opt.lr},
             {'params': list(adaptive.parameters()), 'lr': 1e-2}])

        for time in range(opt.iters + 1):
            R = uu.make_rot_matrix(thetac, alphac, gammac)
            if time % opt.show_freq == 0 and time > 0:
                uu.save_transform(R, t, root)
                uu.show(writer, R, t, wrapper1, wrapper2, s1, focal, X, As, time, scale=opt.scale,
                        image_size=opt.image_size, near=opt.near, far=opt.far, chunk=opt.chunk)
            W = torch.matmul((opt.scale) * X, R.transpose(0, 1)) + t
            # moving_surface_values = wrapper2.get_surface_value(W, (opt.scale) * sigmas)
            # print('moving_surface_values shape: ', moving_surface_values.shape) # [1, N]
            # print('w shape: ', W.shape) # [1. N, 3]
            moving_density_values = wrapper1.query_density(W.squeeze(0)).unsqueeze(0)

            # delta = torch.abs(moving_surface_values - fixed_surface_values)
            delta = torch.abs(moving_density_values - fixed_density_values)
            delta = delta.transpose(0, 1).float()
            loss1 = torch.mean(adaptive.lossfun(delta)) # matching energy
            loss2 = torch.mean((torch.matmul(opt.scale * As, R.transpose(0, 1)) + t - Bs) ** 2) # keypoint energy
            loss = (1 - w2) * loss1 + w2 * loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if time % opt.freq == 0 and time > 0:
                uu.write_log(writer, time, obj_pc, R, t, loss, loss1, loss2, sigmas[0, 0], T_gt, euler_gt, trans_gt,
                             real=opt.real)
                c = adaptive.scale()[0].clone().detach().cpu().numpy()
                w2 = 0.5 * (1 + np.cos(time * np.pi / opt.iters))  # Trigonometric additive cooling
                sigmas = sigmas_f + 0.5 * (sigmas_i - sigmas_f) * (1 + np.cos(time * np.pi / opt.iters))
                X = uu.draw_new_samples(X, rho, R, t, wrapper1, wrapper2, c, sigmas, opt.obj_bigness)
                if opt.use_vis:
                    uu.visualize_points(torch.cat([X[0].clone().detach() * opt.scale, As * opt.scale], axis=0), vis,
                                        'samples', snum=len(As)
                                        )
                    uu.visualize_points(torch.cat(
                        ((torch.matmul(As * opt.scale, R.transpose(0, 1)) + t), Bs,
                         torch.tensor([[0, 0, 0.0]]).to(opt.device)), 0),
                        vis, 'keys', snum=len(list(Bs)) + 1, title='Key Points')

                W = torch.matmul((opt.scale) * X.detach().clone().float().to(opt.device), R.transpose(0, 1)) + t
                # fixed_surface_values = wrapper1.get_surface_value(X, sigmas)
                # fixed_surface_values = fixed_surface_values.detach().clone()
                # print('fixed_surface_values shape: ', fixed_surface_values.shape) # [1, N]
                fixed_density_values = wrapper1.query_density(X.squeeze(0)).unsqueeze(0)
                fixed_density_values = fixed_density_values.detach().clone()