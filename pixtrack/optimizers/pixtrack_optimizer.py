from typing import Optional, Tuple
import torch
from torch import Tensor
from pixloc.pixlib.models.learned_optimizer import LearnedOptimizer
from pixloc.pixlib.geometry.costs import DirectAbsoluteCost
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.geometry.optimization import optimizer_step
import logging
logger = logging.getLogger(__name__)

class DirectAbsoluteCostDepth(DirectAbsoluteCost):
    def residuals(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor, D_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            do_gradients: bool = False, N_ref: Tensor = None):

        p3D_q = T_w2q * p3D
        N_ref = T_w2q * N_ref
        p2D, visible = camera.world2image(p3D_q)

        FD_query = torch.cat((F_query, D_query), dim=0)

        FD_p2D_raw, valid, gradients_fd = self.interpolator(
            FD_query, p2D, return_gradients=do_gradients)

        F_p2D_raw = FD_p2D_raw[:, :-4]
        N_p2D_raw = FD_p2D_raw[:, -3:]
        D_p2D_raw = FD_p2D_raw[:, -4].unsqueeze(1)
        gradients = gradients_fd[:, :-4]
        gradients_norms = gradients_fd[:, -3:]
        gradients_depth = gradients_fd[:, -4].unsqueeze(1)

        valid = valid & visible

        if confidences is not None:
            C_ref, C_query = confidences
            C_query_p2D, _, _ = self.interpolator(
                C_query, p2D, return_gradients=False)
            weight = C_ref * C_query_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_ref
        res_depth = (D_p2D_raw - p3D_q[:, 2].unsqueeze(-1)) * 2.
        res_norms = (N_p2D_raw - N_ref) * 1.

        info = (p3D_q, F_p2D_raw, gradients)
        info_depth = (res_depth, D_p2D_raw, gradients_depth)
        info_norms = (res_norms, N_p2D_raw, gradients_norms)

        return res, valid, weight, F_p2D, info, info_depth, info_norms

    def jacobian_depth(
            self, T_w2q: Pose, camera: Camera,
            p3D_q: Tensor, F_p2D_raw: Tensor, J_f_p2D: Tensor):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        J_p2D_T = J_p2D_p3D @ J_p3D_T
        J = J_f_p2D @ J_p2D_T

        # Extra term for depth loss
        J = J - J_p3D_T[:, 2].unsqueeze(1)

        return J, J_p2D_T

    def jacobian_norms(
            self, T_w2q: Pose, camera: Camera,
            p3D_q: Tensor, F_p2D_raw: Tensor, J_f_p2D: Tensor):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            J_f_p2D = J_normalization(F_p2D_raw) @ J_f_p2D

        J_p2D_T = J_p2D_p3D @ J_p3D_T
        J = J_f_p2D @ J_p2D_T

        # Extra term for depth loss
        J = J - J_p3D_T

        return J, J_p2D_T


    def residual_jacobian(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor, D_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None, N_ref=None):

        res, valid, weight, F_p2D, info, info_depth, info_norms = self.residuals(
            T_w2q, camera, p3D, F_ref, F_query, D_query, confidences, True, N_ref)
        J, _ = self.jacobian(T_w2q, camera, *info)

        p3D_q = info[0]
        res_depth, D_p2D_raw, gradients_depth = info_depth
        res_norms, N_p2D_raw, gradients_norms = info_norms

        J_d, _ = self.jacobian_depth(T_w2q, camera, p3D_q, D_p2D_raw, gradients_depth)
        J_n, _ = self.jacobian_norms(T_w2q, camera, N_ref, N_p2D_raw, gradients_norms)

        J_fd = torch.cat((J, J_d), dim=1)
        res_fd = torch.cat((res, res_depth), dim=1)

        J_fdn = torch.cat((J, J_d, J_n), dim=1)
        res_fdn = torch.cat((res, res_depth, res_norms), dim=1)
        if self.use_depth:
            J = J_fd
            res = res_fd

        if False: #self.use_norms:
            J = J_fdn
            res = res_fdn

        if True:
            J = J_d
            res = res_depth

        if False:
            J = torch.cat((J_d, J_n), dim=1)
            res = torch.cat((res_depth, res_norms), dim=1)

        return res, valid, weight, F_p2D, J

class PixTrackOptimizer(LearnedOptimizer):
    def early_stop(self, **args):
        stop = False
        if not self.training and (args["i"] % 1) == 0:
            T_delta, grad = args["T_delta"], args["grad"]
            grad_norm = torch.norm(grad.detach(), dim=-1)
            small_grad = grad_norm < self.conf.grad_stop_criteria
            dR, dt = T_delta.magnitude()
            small_step = (dt < self.conf.dt_stop_criteria) & (
                dR < self.conf.dR_stop_criteria
            )
            if torch.all(small_step | small_grad):
                stop = True
        return stop

    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None, 
             D_query: Tensor = None, N_ref = None):

        if not isinstance(self.cost_fn, DirectAbsoluteCostDepth):
             self.cost_fn = DirectAbsoluteCostDepth(self.cost_fn.interpolator,
                               normalize=self.cost_fn.normalize)
             self.cost_fn.use_depth = self.use_depth

        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, D_query, W_ref_query, N_ref)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = self.dampingnet()

        for i in range(self.conf.num_iters):
            res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 10)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            if w_unc is not None:
                weights *= w_unc
            if self.conf.jacobi_scaling:
                J, J_scaling = self.J_scaling(J, J_scaling, valid)

            # solve the linear system
            g, H = self.build_system(J, res, weights)
            delta = optimizer_step(g, H, lambda_, mask=~failed)
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            # compute the pose update
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)
            if self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost):
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed
