import torch


class StochasticFlip(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        super(StochasticFlip, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    # Apply stochastic flipping
                    # w <- w - lr * grad
                    update_direction = -torch.sign(p.grad)
                    can_flip = (p != update_direction)
                    flip_rate = torch.min(lr * p.grad.abs(), torch.tensor(1.0))
                    # print(can_flip)
                    flip_mask = torch.rand_like(flip_rate) < flip_rate
                    flipped = torch.logical_and(flip_mask, can_flip)
                    # print(flipped)

                    p[flipped] = -p[flipped]
