import torch
from torch import nn
import numpy as np
import os
from fairseq.data import iterators
from fairseq.logging import progress_bar
from fairseq.modules import MultiheadAttention
try:
    from fairseq.modules.layer_norm import FusedLayerNorm
    valid_nonattention_modules = (nn.LayerNorm, FusedLayerNorm, nn.Linear, nn.Embedding)
except:
    valid_nonattention_modules = (nn.LayerNorm, nn.Linear, nn.Embedding)

from .gradinit_optimizers import RescaleAdam
import sys

def get_ordered_params(net, model_cfg):
    param_list = []
    for n, m in net.named_modules():
        if 'decoder.output_projection' in n and model_cfg.share_decoder_input_output_embed:
            continue
        elif isinstance(m, valid_nonattention_modules):
            param_list.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, MultiheadAttention):
            pass
    return param_list


def set_param(module, name, alg, eta, grad):
    weight = getattr(module, name)
    # remove this parameter from parameter list
    del module._parameters[name]

    # compute the update steps according to the optimizers
    if alg.lower() == 'sgd':
        gstep = eta * grad
    elif alg.lower() == 'adam':
        gstep = eta * grad.sign()
    else:
        raise RuntimeError("Optimization algorithm {} not defined!".format(alg))

    # add the updated parameter as the new parameter
    module.register_parameter(name + '_prev', weight)

    # recompute weight before every forward()
    updated_weight = weight - gstep.data
    setattr(module, name, updated_weight)


def take_opt_step(net, grad_list, model_cfg, alg='sgd', eta=0.1):
    """Take the initial step of the chosen optimizer.
    """
    assert alg.lower() in ['adam', 'sgd']

    idx = 0
    for n, m in net.named_modules():
        if 'decoder.output_projection' in n and model_cfg.share_decoder_input_output_embed:
            continue
        elif isinstance(m, valid_nonattention_modules):
            set_param(m, 'weight', alg, eta, grad_list[idx])
            idx += 1
            if hasattr(m, 'bias') and m.bias is not None:
                set_param(m, 'bias', alg, eta, grad_list[idx])
                idx += 1
        elif isinstance(m, MultiheadAttention):
            # legacy issue: already considered by nn.Linear
            assert 'q_proj_weight' not in m._parameters
        else:
            # pay attention to learned position embedding in the future
            pass
    # return torch.nn.DataParallel(net.module).cuda()


def recover_params(net, model_cfg):
    """Reset the weights to the original values without the gradient step
    """

    def recover_param_(module, name):
        delattr(module, name)
        setattr(module, name, getattr(module, name + '_prev'))
        del module._parameters[name + '_prev']

    for n, m in net.named_modules():
        if 'decoder.output_projection' in n and model_cfg.share_decoder_input_output_embed:
            continue
        elif isinstance(m, valid_nonattention_modules):
            recover_param_(m, 'weight')
            if hasattr(m, 'bias') and m.bias is not None:
                recover_param_(m, 'bias')
        elif isinstance(m, MultiheadAttention):
            pass

    # return torch.nn.DataParallel(net.module).cuda()


def set_bn_modes(net):
    """Switch the BN layers into training mode, but does not track running stats.
    """
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.track_running_stats = False


def recover_bn_modes(net):
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


def get_scale_stats(model, optimizer):
    stat_dict = {}
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


def get_intersecting_samples(all_samples):
    """An easy hack to fairseq to get overlapping batches. Implemented for translation task."""
    n_samples = all_samples['id'].numel()
    n_per_div = int(n_samples / 3)
    init_samples, updated_samples = {}, {}

    init_samples['id'], updated_samples['id'] = all_samples['id'][:int(2*n_per_div)], all_samples['id'][n_per_div:]
    init_samples['nsentences'], updated_samples['nsentences'] = 2 * n_per_div, n_samples - n_per_div

    init_samples['net_input'] = {k: v[:int(2*n_per_div)] for k, v in all_samples['net_input'].items()}
    updated_samples['net_input'] = {k: v[n_per_div:] for k, v in all_samples['net_input'].items()}

    init_samples['target'], updated_samples['target'] = all_samples['target'][:int(2*n_per_div)], all_samples['target'][n_per_div:]
    init_samples['ntokens'] = (init_samples['target'] != 1).sum().item()
    updated_samples['ntokens'] = (updated_samples['target'] != 1).sum().item()

    if init_samples['ntokens'] > 4000 or updated_samples['ntokens'] > 4000:
        print(f"mismatched number of tokens: {init_samples['ntokens']}, {updated_samples['ntokens']}, "
              f"check if the difference is too large.")
    # assert (all_samples['target'] != 1).sum().item() == all_samples['ntokens']

    return init_samples, updated_samples


def get_train_iter(trainer, cfg):
    epoch_itr = trainer.get_train_iterator(
        epoch=1, load_dataset=True, max_tokens=cfg.gradinit.gradinit_max_tokens
    )
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=True,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        default_log_format='simple',
    )
    return progress, itr


def gradinit_proc(trainer, cfg):
    model_cfg = cfg.model
    args = cfg.gradinit
    if args.resume_init_chk:
        print("Resuming GradInit model from {}".format(args.resume_init_chk))
        sdict = torch.load(args.resume_init_chk)
        trainer.model.load_state_dict(sdict)
        return

    bias_params = [p for n, p in trainer.model.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in trainer.model.named_parameters() if 'weight' in n and 'layer_norm' not in n]
    ln_params = [p for n, p in trainer.model.named_parameters() if 'weight' in n and 'layer_norm' in n]

    ln_lr = args.gradinit_lr if args.gradinit_ln_lr < 0 else args.gradinit_ln_lr
    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                             {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr},
                             {'params': ln_params, 'min_scale': args.gradinit_min_scale,
                              'max_scale': args.gradinit_max_scale, 'lr': ln_lr}],
                             grad_clip=args.gradinit_grad_clip, betas=(args.gradinit_beta1, args.gradinit_beta2))

    trainer.model.eval() # This further shuts down dropout, if any.
    if cfg.common.fp16:
        trainer.model.float()

    # set_bn_modes(trainer.model) # Probably don't have BN for language models.

    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums, total_sums_gnorm = 0, 0
    cs_count = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    # get all the parameters by order
    params_list = get_ordered_params(trainer.model, model_cfg)
    eta = args.gradinit_eta
    while True:
        progress, itr = get_train_iter(trainer, cfg)
        for ns, samples in enumerate(progress):
            trainer.model.set_num_updates(0)
            trainer.model.eval()

            for i, sample in enumerate(samples):
                # second return value is the indicator of whether the sample is a dummy batch
                sample, _ = trainer._prepare_sample(sample)
                # pdb.set_trace()
                assert i < 1
            init_samples, updated_samples = get_intersecting_samples(sample)
            init_loss, init_sample_size, logging_output = trainer.criterion(trainer.model, init_samples)
            init_loss = init_loss / init_sample_size

            # compute the gradient and take one step
            all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)
            # get the first half of the minibatch

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
                take_opt_step(trainer.model, loss_grads, model_cfg, alg=args.gradinit_alg, eta=eta)

                total_l0 += init_loss.item()

                # compute loss using the updated network
                updated_loss, sample_size_update, logging_output = trainer.criterion(trainer.model, updated_samples)
                # init_loss, sample_size_init, logging_output = trainer.criterion(trainer.model, init_sample)
                updated_loss = updated_loss / sample_size_update

                # If eta is larger, we should expect obj_loss to be even smaller.
                obj_loss = updated_loss / eta

                recover_params(trainer.model, model_cfg)
                optimizer.zero_grad()
                obj_loss.backward()
                optimizer.step(is_constraint=False)
                total_l1 += updated_loss.item()

                total_loss += obj_loss.item()
                total_sums += 1

            total_iters += 1
            if (total_sums_gnorm > 0 and total_sums_gnorm % 10 == 0) or total_iters == args.gradinit_iters:
                stat_dict = get_scale_stats(trainer.model, optimizer)
                print_str = "Iter {}/{}, obj iters {}, eta {:.3e}, constraint count {} loss: {:.3e} ({:.3e}), init loss: {:.3e} ({:.3e}), update loss {:.3e} ({:.3e}), " \
                            "total gnorm: {:.3e} ({:.3e})\t".format(
                    total_iters, args.gradinit_iters, total_sums, eta, cs_count,
                    float(obj_loss), total_loss / total_sums if total_sums > 0 else -1,
                    float(init_loss), total_l0 / total_sums if total_sums > 0 else -1,
                    float(updated_loss), total_l1 / total_sums if total_sums > 0 else -1,
                    float(gnorm), total_gnorm / total_sums_gnorm)

                for key, val in stat_dict.items():
                    print_str += "{}: {:.2e}\t".format(key, val)
                print(print_str)
                sys.stdout.flush()

                ln_alphas, v_alphas, out_alphas = [], [], []
                fc1, fc2 = [], []
                for n, p in trainer.model.named_parameters():
                    if 'weight' in n:
                        alpha = optimizer.state[p]['alpha']
                        if 'layer_norm' in n:
                            ln_alphas.append(alpha)
                        elif 'out_proj' in n:
                            out_alphas.append(alpha)
                        elif 'v_proj' in n:
                            v_alphas.append(alpha)
                        elif 'ffn' in n:
                            if 'fc1' in n:
                                fc1.append(alpha)
                            else:
                                fc2.append(alpha)

                ln_avg = np.mean(ln_alphas)
                ln_scales = ', '.join([f'{a:.2f}' for a in ln_alphas])
                print(f'LN scales ({ln_avg:.2f}): {ln_scales}')
                alpha_avg = np.mean(out_alphas)
                out_alphas = ', '.join([f'{a:.2f}' for a in out_alphas])
                print(f'out_proj scales ({alpha_avg:.2f}): {out_alphas}')
                v_avg = np.mean(v_alphas)
                v_alphas = ', '.join([f'{a:.2f}' for a in v_alphas])
                print(f'v_proj scales ({v_avg:.2f}): {v_alphas}')
                print('\n')

            if total_iters == args.gradinit_iters:
                break
        if total_iters == args.gradinit_iters:
            break

    if cfg.common.fp16:
        trainer.model.half()

    # recover_bn_modes(net) # BN is probably not used for language models
    if not os.path.exists('chks'):
        os.makedirs('chks')
    torch.save(trainer.model.state_dict(), 'chks/{}_init.pth'.format(args.expname))

