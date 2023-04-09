from typing import Optional, Tuple
import torch
from torch import Tensor
from pixloc.pixlib.models.learned_optimizer import LearnedOptimizer
from pixloc.pixlib.geometry.costs import DirectAbsoluteCost
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.geometry.optimization import optimizer_step


class DirectAbsoluteCostDepth(DirectAbsoluteCost):
    def residuals(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor, D_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            do_gradients: bool = False):

        p3D_q = T_w2q * p3D
        p2D, visible = camera.world2image(p3D_q)

        FD_query = torch.cat((F_query, D_query), dim=0)

        FD_p2D_raw, valid, gradients_fd = self.interpolator(
            FD_query, p2D, return_gradients=do_gradients)

        F_p2D_raw = FD_p2D_raw[:, :-1]
        D_p2D_raw = FD_p2D_raw[:, -1].unsqueeze(1)
        gradients = gradients_fd[:, :-1]
        gradients_depth = gradients_fd[:, -1].unsqueeze(1)

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
        res_depth = D_p2D_raw - p3D_q[:, 2].unsqueeze(-1)

        info = (p3D_q, F_p2D_raw, gradients)
        info_depth = (res_depth, D_p2D_raw, gradients_depth)

        return res, valid, weight, F_p2D, info, info_depth

    def residual_jacobian(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor, D_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None):

        res, valid, weight, F_p2D, info, info_depth = self.residuals(
            T_w2q, camera, p3D, F_ref, F_query, D_query, confidences, True)
        J, _ = self.jacobian(T_w2q, camera, *info)
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
             D_query: Tensor = None):

        if not isinstance(self.cost_fn, DirectAbsoluteCostDepth):
             self.cost_fn = DirectAbsoluteCostDepth(self.cost_fn.interpolator,
                               normalize=self.cost_fn.normalize)

        T = T_init
        J_scaling = None
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, D_query, W_ref_query)
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
