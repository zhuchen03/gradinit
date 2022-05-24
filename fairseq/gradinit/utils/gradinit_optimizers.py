import torch
import math
import pdb

class RescaleAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 min_scale=0, max_scale=float('inf'), grad_clip=0, amsgrad=False, grad_acc_steps=1,
                 per_elem_step=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, min_scale=min_scale, max_scale=max_scale,
                        grad_clip=grad_clip, grad_acc_steps=grad_acc_steps, per_elem_step=per_elem_step)
        super(RescaleAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RescaleAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def per_elem_step(self, p, group):
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # if group['weight_decay'] != 0:
        #     grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        p.data.addcdiv_(-step_size, exp_avg, denom)

    @torch.no_grad()
    def step(self, closure=None, is_constraint=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_list = []
        idx, counter = 11, 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['per_elem_step']:
                    self.per_elem_step(p, group)
                    continue
                # State initialization
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    state['total_iters'] = 0 # used for gradient accumulation
                    state['cons_step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    state['cons_exp_avg'] = 0
                    state['grad'] = 0
                    state['cons_exp_avg_sq'] = 0


                curr_norm = p.data.norm().item()
                if state['init_norm'] == 0 or curr_norm == 0:
                    # pdb.set_trace()
                    continue # typical for biases

                grad = torch.sum(p.grad * p.data).item() * state['init_norm'] / curr_norm
                grad_list.append(grad)
                counter += 1

                if group['grad_clip'] > 0:
                    grad = max(min(grad, group['grad_clip']), -group['grad_clip'])
                # Perform stepweight decay
                beta1, beta2 = group['betas']
                if is_constraint:
                    state['cons_step'] += 1
                    state['cons_exp_avg'] = state['cons_exp_avg'] * beta1 + grad * (1 - beta1)
                    # state['cons_exp_avg_sq'] = state['cons_exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)

                    steps = state['cons_step']
                    exp_avg = state['cons_exp_avg']
                    state['cons_exp_avg_sq'] = state['cons_exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)
                    exp_avg_sq = state['cons_exp_avg_sq']
                else:
                    state['step'] += 1
                    state['exp_avg'] = state['exp_avg'] * beta1 + grad * (1 - beta1)

                    steps = state['step']
                    exp_avg = state['exp_avg']

                    state['exp_avg_sq'] = state['exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)
                    exp_avg_sq = state['exp_avg_sq']
                lr = group['lr']

                bias_correction1 = 1 - beta1 ** steps
                bias_correction2 = 1 - beta2 ** steps

                # Decay the first and second moment running average coefficient
                denom = math.sqrt(exp_avg_sq / bias_correction2) + group['eps']
                step_size = lr / bias_correction1


                # update the parameter
                state['alpha'] = max(state['alpha'] - step_size * exp_avg / denom, group['min_scale'])
                state['alpha'] = min(state['alpha'], group['max_scale'])
                p.data.mul_(state['alpha'] * state['init_norm'] / curr_norm)

        return loss
