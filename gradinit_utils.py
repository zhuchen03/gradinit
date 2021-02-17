import torch
from torch import nn
from gradinit_optimizers import RescaleAdam
from gradinit_modules import GradInitConv2d, GradInitLinear, GradInitBatchNorm2d, GradInitBias, GradInitScale
import numpy as np
import os
import pdb


def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, GradInitConv2d) or isinstance(m, GradInitLinear) or isinstance(m, GradInitBatchNorm2d):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, GradInitScale):
            param_list.append(m.weight)
        elif isinstance(m, GradInitBias):
            param_list.append(m.bias)

    return param_list


def take_opt_step(net, grad_list, opt='sgd', trust=0.1):
    """Take the initial step of the chosen optimizer.
    """
    assert opt.lower() in ['adam', 'sgd']

    idx = 0
    for n, m in net.named_modules():
        if isinstance(m, GradInitConv2d) or isinstance(m, GradInitLinear) or isinstance(m, GradInitBatchNorm2d):
            grad = grad_list[idx]
            if opt.lower() == 'sgd':
                m.opt_weight = m.weight - trust * grad.detach()
            elif opt.lower() == 'adam':
                m.opt_weight = m.weight - trust * grad.sign()
            idx += 1

            if m.bias is not None:
                grad = grad_list[idx]
                if opt.lower() == 'sgd':
                    m.opt_bias = m.bias - trust * grad.detach()
                elif opt.lower() == 'adam':
                    m.opt_bias = m.bias - trust * grad.sign().detach()
                idx += 1
            else:
                m.opt_bias = None
        elif isinstance(m, GradInitScale):
            tr = trust
            grad = grad_list[idx]
            if opt.lower() == 'sgd':
                m.opt_weight = m.weight - tr * grad.detach()
            elif opt.lower() == 'adam':
                m.opt_weight = m.weight - tr * grad.sign().detach()
            idx += 1
            # print(n, m.opt_weight.sum())
        elif isinstance(m, GradInitBias):
            tr = trust
            grad = grad_list[idx]
            if opt.lower() == 'sgd':
                m.opt_bias = m.bias - tr * grad.detach()
            elif opt.lower() == 'adam':
                m.opt_bias = m.bias - tr * grad.sign().detach()
            idx += 1


def get_scale_stats(model, optimizer):
    stat_dict = {}
    # all_s_list = [p.norm().item() for n, p in model.named_parameters() if 'bias' not in n]
    all_s_list = []
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_max'] = max(all_s_list)
    stat_dict['s_min'] = min(all_s_list)
    stat_dict['s_mean'] = np.mean(all_s_list)
    all_s_list = []
    for n, p in model.named_parameters():
        if 'bias' not in n:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_weight_max'] = max(all_s_list)
    stat_dict['s_weight_min'] = min(all_s_list)
    stat_dict['s_weight_mean'] = np.mean(all_s_list)

    return stat_dict


def get_batch(data_iter, data_loader):
    try:
        inputs, targets = next(data_iter)
    except:
        data_iter = iter(data_loader)
        inputs, targets = next(data_iter)
    inputs, targets = inputs.cuda(), targets.cuda()
    return data_iter, inputs, targets


def gradinit(net, dataloader, args):
    if args.gradinit_resume:
        print("Resuming GradInit model from {}".format(args.gradinit_resume))
        sdict = torch.load(args.gradinit_resume)
        net.load_state_dict(sdict)
        return

    if isinstance(net, torch.nn.DataParallel):
        net_top = net.module
    else:
        net_top = net

    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr}],
                                     grad_clip=args.gradinit_grad_clip)

    criterion = nn.CrossEntropyLoss()

    net_top.gradinit(True) # This will switch the BNs in to training mode
    net.eval() # This further shuts down dropout, if any.
    # get all the parameters by order
    params_list = get_ordered_params(net)

    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums, total_sums_gnorm = 0, 0
    cs_count = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    while True:
        eta = args.gradinit_eta

        # continue
        # get the first half of the minibatch
        data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)

        # Get the second half of the data.
        data_iter, init_inputs_1, init_targets_1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([init_inputs_0, init_inputs_1])
        init_targets = torch.cat([init_targets_0, init_targets_1])
        # compute the gradient and take one step
        outputs = net(init_inputs)
        init_loss = criterion(outputs, init_targets)
        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        # Compute the loss w.r.t. the optimizer
        if args.gradinit_alg.lower() == 'adam':
            # grad-update inner product
            gnorm = sum([g.abs().sum() for g in all_grads])
            loss_grads = all_grads
        else:
            gnorm_sq = sum([g.square().sum() for g in all_grads])
            gnorm = gnorm_sq.sqrt()
            if args.gradinit_normalize_grad:
                loss_grads = [g / gnorm for g in all_grads]
            else:
                loss_grads = all_grads

        total_gnorm += gnorm.item()
        total_sums_gnorm += 1
        if gnorm.item() > args.gradinit_gamma:
            # project back into the gradient norm constraint
            optimizer.zero_grad()
            gnorm.backward()
            optimizer.step(is_constraint=True)

            cs_count += 1
        else:
            # take one optimization step
            take_opt_step(net, loss_grads, opt=args.gradinit_alg, trust=eta)

            total_l0 += init_loss.item()

            data_iter, inputs_2, targets_2 = get_batch(data_iter, dataloader)
            if args.batch_no_overlap:
                # sample a new batch for the half
                data_iter, init_inputs_0, init_targets_0 = get_batch(data_iter, dataloader)
            updated_inputs = torch.cat([init_inputs_0, inputs_2])
            updated_targets = torch.cat([init_targets_0, targets_2])

            # compute loss using the updated network
            net_top.opt_mode(True)
            updated_outputs = net(updated_inputs)
            net_top.opt_mode(False)
            updated_loss = criterion(updated_outputs, updated_targets)

            # If eta is larger, we should expect obj_loss to be even smaller.
            obj_loss = updated_loss / eta

            optimizer.zero_grad()
            obj_loss.backward()
            optimizer.step(is_constraint=False)
            total_l1 += updated_loss.item()

            total_loss += obj_loss.item()
            total_sums += 1

        total_iters += 1
        if (total_sums_gnorm > 0 and total_sums_gnorm % 10 == 0) or total_iters == args.gradinit_iters or total_iters == args.gradinit_iters:
            stat_dict = get_scale_stats(net, optimizer)
            print_str = "Iter {}, obj iters {}, eta {:.3e}, constraint count {} loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e}), " \
                        "total gnorm: {:.3e} ({:.3e})\t".format(
                total_sums_gnorm, total_sums, eta, cs_count,
                float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1,
                float(gnorm), total_gnorm / total_sums_gnorm)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == args.gradinit_iters:
            break

    net_top.gradinit(False)
    if not os.path.exists('chks'):
        os.makedirs('chks')
    torch.save(net.state_dict(), 'chks/{}_init.pth'.format(args.expname))


def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss, params, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]), params,
                             create_graph=True)
    out = sum([((g - p) / (g + eps * (2 * (g >= 0).float() - 1).detach())
               - 1).abs().sum() for g, p in zip(grad, prod)])

    gnorm = sum([(g**2).sum().item() for g in grad])
    return out / sum([p.data.numel() for p in params]), gnorm


def metainit(net, dataloader, args, experiment=None):

    if args.gradinit_resume:
        print("Resuming metainit model from {}".format(args.gradinit_resume))
        sdict = torch.load(args.gradinit_resume)
        net.load_state_dict(sdict)
        return

    if isinstance(net, torch.nn.DataParallel):
        net_top = net.module
    else:
        net_top = net

    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]


    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                                      {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr}],
                                     grad_clip=args.gradinit_grad_clip)

    criterion = nn.CrossEntropyLoss()

    net_top.gradinit(True)
    net.eval()
    # get all the parameters by order
    params_list = get_ordered_params(net)

    total_gq_loss = 0
    total_gnorm = 0
    for ite, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        gq, gnorm = gradient_quotient(loss, params_list, eps=1e-5)
        gq.backward()

        total_gq_loss += gq.item()
        total_gnorm += gnorm
        optimizer.step()

        if ite % 10 == 0 or ite == args.gradinit_iters - 1 or ite == len(dataloader) - 1:
            stat_dict = get_scale_stats(net, optimizer)
            print_str = "Iter {}, gq {:.3e} ({:.3e}), gnorm {:.3e} ({:.3e}), loss {:.3e}\t".format(
                ite, gnorm, total_gnorm / (ite + 1), gq.item(), total_gq_loss / (ite + 1), loss.item())

            if experiment is not None:
                experiment.log_metric("gq", gq.item(), ite)
                experiment.log_metric("init_loss", loss.item(), ite)
                experiment.log_metric("gnorm", gnorm, ite)
                for key, val in stat_dict.items():
                    experiment.log_metric(key, val, ite)
            # torch.save(net.state_dict(), 'chks/{}_init.pth'.format(args.expname))

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

    net_top.gradinit(False)
    if not os.path.exists('chks'):
        os.makedirs('chks')
    torch.save(net.state_dict(), 'chks/{}_init.pth'.format(args.expname))
