import torch


def fgsm(input, epsilon, data_grad):
    pert_out = input + epsilon * data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    return pert_out


def pgdlinf(adversarial_input, input, eta, eps, data_grad):
    pert_out = adversarial_input.detach() + eta * data_grad.sign()
    pert_out = torch.clamp(pert_out - input, min=-eps, max=+eps)
    pert_out = torch.clamp(input + pert_out, min=0, max=1).detach()
    return pert_out


def pgdl2(adversarial_input, input, eta, eps, eps_for_div, data_grad):
    grad_norm = torch.norm(data_grad.view(1, -1), p=2, dim=1) + eps_for_div
    data_grad = data_grad / grad_norm.view(1,1,1,1)

    pert_out = adversarial_input.detach() + eta * data_grad
    pert_out = pert_out - input
    pert_out_norm = torch.norm(pert_out.view(1, -1), p=2, dim=1)

    factor = eps / pert_out_norm
    factor = torch.min(factor, torch.ones_like(pert_out_norm))

    pert_out = pert_out * factor.view(-1, 1, 1, 1)

    pert_out = torch.clamp(input + pert_out, min=0, max=1).detach()
    return pert_out
