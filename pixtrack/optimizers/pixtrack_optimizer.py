import torch
from pixloc.pixlib.models.learned_optimizer import LearnedOptimizer


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
