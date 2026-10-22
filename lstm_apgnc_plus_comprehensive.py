import argparse
import copy
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = ("apgnc_plus", "lstm_apgnc_plus")


def parse_number_list(text, value_type):
    if not text.strip():
        return []
    return [value_type(item.strip()) for item in text.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="lstm_apgnc_plus_comprehensive_results",
    )
    parser.add_argument("--device", type=str, default="auto")

    # Each seed creates a new A, B and independent pair of models.
    parser.add_argument(
        "--seeds",
        type=str,
        default="2025,2026,2027",
    )

    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--m-list",
        type=str,
        default="200,250,300",
    )
    parser.add_argument("--layers", type=int, default=16)

    # Both models are trained only on this distribution.
    parser.add_argument(
        "--train-nonzero-prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)

    # Generalization evaluation.
    parser.add_argument(
        "--test-prob-list",
        type=str,
        default="0.02,0.05,0.10,0.15,0.20",
    )
    parser.add_argument(
        "--snr-list",
        type=str,
        default="40,30,20,10",
    )
    parser.add_argument(
        "--amplitude-list",
        type=str,
        default="rademacher,laplace",
    )

    # Same analytical B used by both models in each experiment.
    parser.add_argument("--b-epochs", type=int, default=2000)
    parser.add_argument("--b-lr", type=float, default=3e-3)
    parser.add_argument("--b-top-s", type=int, default=50)
    parser.add_argument("--b-fro-weight", type=float, default=1e-3)
    parser.add_argument("--reuse-b", action="store_true")

    # Both models independently train for exactly this many epochs.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--layer-loss-weight", type=float, default=0.05)
    parser.add_argument(
    "--adapt-regularization",
    type=float,
    default=3e-3,
)

    # ALePOM MCP parameters.
    parser.add_argument("--theta-init", type=float, default=0.10)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--eta-ratio-init", type=float, default=0.50)

    # APGnc+ Algorithm 4.
    parser.add_argument("--beta-init", type=float, default=0.50)
    parser.add_argument("--beta-scale", type=float, default=0.50)

    # LSTM parameter controller.
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--theta-adapt-scale", type=float, default=0.50)
    parser.add_argument("--q-adapt-scale", type=float, default=1.00)
    parser.add_argument("--gamma-adapt-scale", type=float, default=0.35)

    parser.add_argument("--support-threshold", type=float, default=1e-4)
    parser.add_argument("--success-nmse", type=float, default=1e-4)
    parser.add_argument("--timing-repeats", type=int, default=3)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(False)


def get_device(name):
    if name != "auto":
        return torch.device(name)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def inverse_softplus(value):
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def inverse_sigmoid(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def make_sensing_matrix(m, n, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    A = torch.randn(
        m,
        n,
        generator=generator,
    ) / math.sqrt(m)

    A = A / A.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-12)

    return A.to(device)


def sample_amplitudes(count, n, distribution, generator):
    if distribution == "gaussian":
        return torch.randn(
            count,
            n,
            generator=generator,
        )

    if distribution == "rademacher":
        values = torch.randint(
            0,
            2,
            (count, n),
            generator=generator,
        ).float()

        return 2.0 * values - 1.0

    if distribution == "laplace":
        uniform = torch.rand(
            count,
            n,
            generator=generator,
        ).clamp(1e-6, 1.0 - 1e-6)

        centered = uniform - 0.5
        scale = 1.0 / math.sqrt(2.0)

        return (
            -scale
            * torch.sign(centered)
            * torch.log1p(-2.0 * centered.abs())
        )

    raise ValueError(
        f"Unknown amplitude distribution: {distribution}"
    )


def add_snr_noise(clean_y, snr_db, generator, device):
    if math.isinf(snr_db):
        return clean_y

    noise = torch.randn(
        clean_y.shape,
        generator=generator,
    ).to(device)

    signal_norm = clean_y.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    noise_norm = noise.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    desired_noise_norm = (
        signal_norm
        / (10.0 ** (snr_db / 20.0))
    )

    return (
        clean_y
        + noise
        * desired_noise_norm
        / noise_norm
    )


def make_dataset(
    A,
    count,
    nonzero_prob,
    seed,
    device,
    amplitude="gaussian",
    snr_db=math.inf,
):
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    n = A.shape[1]

    support = (
        torch.rand(
            count,
            n,
            generator=generator,
        ) < nonzero_prob
    )

    values = sample_amplitudes(
        count,
        n,
        amplitude,
        generator,
    )

    x = values * support

    empty = support.sum(dim=1) == 0

    if empty.any():
        rows = torch.where(empty)[0]

        columns = torch.randint(
            0,
            n,
            (len(rows),),
            generator=generator,
        )

        x[rows, columns] = 1.0

    x = x.to(device)
    clean_y = x @ A.T

    y = add_snr_noise(
        clean_y,
        snr_db,
        generator,
        device,
    )

    return x, y


def normalize_B(B, A):
    diagonal = torch.sum(
        B * A.T,
        dim=1,
        keepdim=True,
    )

    signs = torch.where(
        diagonal >= 0,
        torch.ones_like(diagonal),
        -torch.ones_like(diagonal),
    )

    diagonal = torch.where(
        diagonal.abs() < 1e-8,
        signs * 1e-8,
        diagonal,
    )

    return B / diagonal


def pretrain_B(A, args, output_dir, seed, m):
    path = output_dir / f"B_m{m}_seed{seed}.pt"

    if args.reuse_b and path.exists():
        checkpoint = torch.load(
            path,
            map_location=A.device,
        )

        B = checkpoint["B"]

        if tuple(B.shape) != (args.n, m):
            raise ValueError(
                f"Saved B shape {tuple(B.shape)} "
                f"does not match {(args.n, m)}"
            )

        print(f"Loaded B from {path}")

        return normalize_B(
            B.to(A.device),
            A,
        ).detach()

    initial_B = (
        A.T
        / A.square()
        .sum(dim=0, keepdim=True)
        .T
        .clamp_min(1e-12)
    )

    parameter = nn.Parameter(
        initial_B.detach().clone()
    )

    optimizer = torch.optim.Adam(
        [parameter],
        lr=args.b_lr,
    )

    identity = torch.eye(
        args.n,
        device=A.device,
        dtype=A.dtype,
    )

    top_s = min(
        max(args.b_top_s, 1),
        args.n - 1,
    )

    best_loss = float("inf")
    best_B = None

    for epoch in range(args.b_epochs):
        optimizer.zero_grad(set_to_none=True)

        B = normalize_B(parameter, A)

        G = B @ A - identity
        G = G - torch.diag_embed(
            torch.diagonal(G)
        )

        largest = torch.topk(
            G.abs(),
            k=top_s,
            dim=1,
            largest=True,
            sorted=False,
        ).values

        coherence_loss = (
            largest.square()
            .sum(dim=1)
            .mean()
        )

        frobenius_loss = G.square().mean()

        loss = (
            coherence_loss
            + args.b_fro_weight * frobenius_loss
        )

        loss.backward()
        optimizer.step()

        value = float(loss.detach())

        if value < best_loss:
            best_loss = value
            best_B = normalize_B(
                parameter.detach(),
                A,
            ).clone()

        interval = max(args.b_epochs // 10, 1)

        if (epoch + 1) % interval == 0:
            print(
                f"[B m={m} seed={seed}] "
                f"epoch={epoch + 1:05d}/{args.b_epochs:05d}, "
                f"loss={value:.6e}"
            )

    torch.save(
        {
            "B": best_B.cpu(),
            "best_loss": best_loss,
            "m": m,
            "n": args.n,
            "seed": seed,
        },
        path,
    )

    return best_B.detach()


def expand_parameter(value, batch_size):
    if value.ndim == 0:
        return value.expand(batch_size, 1)

    if value.ndim == 1:
        return value.reshape(-1, 1)

    return value


def mcp_parameters(raw_theta, raw_q, raw_gamma):
    theta = F.softplus(raw_theta) + 1e-8
    gamma = F.softplus(raw_gamma) + 1e-8

    q = (
        torch.sigmoid(raw_q)
        * (1.0 - 1e-4)
    )

    eta = q / (2.0 * theta)

    return theta, q, eta, gamma


def mcp_prox(u, theta, q):
    theta = expand_parameter(
        theta,
        u.shape[0],
    )

    q = expand_parameter(
        q,
        u.shape[0],
    ).clamp(
        min=1e-5,
        max=1.0 - 1e-5,
    )

    absolute = u.abs()
    upper = theta / q

    middle = (
        u.sign()
        * (absolute - theta)
        / (1.0 - q).clamp_min(1e-6)
    )

    return torch.where(
        absolute <= theta,
        torch.zeros_like(u),
        torch.where(
            absolute < upper,
            middle,
            u,
        ),
    )


def mcp_penalty(x, theta, q):
    theta = expand_parameter(
        theta,
        x.shape[0],
    )

    q = expand_parameter(
        q,
        x.shape[0],
    ).clamp(
        min=1e-5,
        max=1.0 - 1e-5,
    )

    eta = q / (
        2.0 * theta
    ).clamp_min(1e-8)

    upper = 1.0 / (
        2.0 * eta
    ).clamp_min(1e-8)

    low = (
        x.abs() - eta * x.square()
    )

    high = torch.ones_like(x) / (
        4.0 * eta
    ).clamp_min(1e-8)

    return torch.where(
        x.abs() <= upper,
        low,
        high,
    ).sum(dim=1)


def layer_objective(
    x,
    measurement,
    A,
    theta,
    q,
    gamma,
):
    batch_size = x.shape[0]

    theta = expand_parameter(
        theta,
        batch_size,
    )

    gamma = expand_parameter(
        gamma,
        batch_size,
    )

    residual = x @ A.T - measurement

    data_term = (
        0.5 * residual.square().sum(dim=1)
    )

    regularization = (
        (theta / gamma).squeeze(1)
        * mcp_penalty(x, theta, q)
    )

    return data_term + regularization


def normalized_mse(prediction, target):
    error = (
        prediction - target
    ).square().sum(dim=1)

    energy = (
        target.square()
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    return (error / energy).mean()


class LSTMParameterAdapter(nn.Module):
    """
    Per-sample recurrent controller.

    It reads the recovery state at every unfolded layer and outputs
    bounded offsets for theta, q=2*theta*eta and gamma.
    """

    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.lstm_hidden
        self.cell = nn.LSTMCell(
            input_size=8,
            hidden_size=self.hidden_size,
        )

        self.head = nn.Linear(
            self.hidden_size,
            3,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

        # Zero output means the initial LSTM model exactly uses
        # the same parameters as baseline APGnc+.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, state):
        hidden, cell = self.cell(
            features,
            state,
        )

        logits = self.head(hidden)

        offsets = (
            0.9
            * torch.tanh(
                logits / 1.5
            )
        )

        return offsets, (hidden, cell)


class RecoveryModel(nn.Module):
    def __init__(self, A, B, args, method):
        super().__init__()

        if method not in METHODS:
            raise ValueError(
                f"Unknown method: {method}"
            )

        if not 0.0 < args.beta_init <= 1.0:
            raise ValueError(
                "beta-init must be in (0, 1]."
            )

        if not 0.0 < args.beta_scale < 1.0:
            raise ValueError(
                "beta-scale must be in (0, 1)."
            )

        self.method = method
        self.layers = args.layers

        self.beta_init = args.beta_init
        self.beta_scale = args.beta_scale

        self.theta_adapt_scale = (
            args.theta_adapt_scale
        )
        self.q_adapt_scale = (
            args.q_adapt_scale
        )
        self.gamma_adapt_scale = (
            args.gamma_adapt_scale
        )

        self.register_buffer(
            "A",
            A.detach().clone(),
        )

        self.register_buffer(
            "B",
            B.detach().clone(),
        )

        self.raw_theta = nn.Parameter(
            torch.full(
                (args.layers,),
                inverse_softplus(
                    args.theta_init
                ),
            )
        )

        self.raw_q = nn.Parameter(
            torch.full(
                (args.layers,),
                inverse_sigmoid(
                    args.eta_ratio_init
                ),
            )
        )

        self.raw_gamma = nn.Parameter(
            torch.full(
                (args.layers,),
                inverse_softplus(
                    args.gamma_init
                ),
            )
        )

        if method == "lstm_apgnc_plus":
            self.lstm = LSTMParameterAdapter(
                args
            )
        else:
            self.lstm = None

    def base_parameters(self, layer):
        return mcp_parameters(
            self.raw_theta[layer],
            self.raw_q[layer],
            self.raw_gamma[layer],
        )

    def make_state_features(
        self,
        iterate,
        previous_iterate,
        residual,
        measurement,
        base_theta,
        adaptive_beta,
        layer,
    ):
        epsilon = 1e-8
        n = iterate.shape[1]

        residual_ratio = (
            residual.norm(
                dim=1,
                keepdim=True,
            )
            / measurement.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(epsilon)
        )

        change_ratio = (
            (iterate - previous_iterate).norm(
                dim=1,
                keepdim=True,
            )
            / iterate.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(epsilon)
        )

        l1_l2 = (
            iterate.abs()
            .sum(dim=1, keepdim=True)
            / (
                math.sqrt(n)
                * iterate.norm(
                    dim=1,
                    keepdim=True,
                ).clamp_min(epsilon)
            )
        ).clamp(
            min=0.0,
            max=1.0,
        )

        active_ratio = (
            iterate.abs() > base_theta
        ).float().mean(
            dim=1,
            keepdim=True,
        )

        rms = iterate.square().mean(
            dim=1,
            keepdim=True,
        ).sqrt().clamp_min(epsilon)

        mean_abs_ratio = (
            iterate.abs().mean(
                dim=1,
                keepdim=True,
            ) / rms
        ).clamp(
            min=0.0,
            max=2.0,
        )

        max_abs_ratio = (
            iterate.abs().amax(
                dim=1,
                keepdim=True,
            ) / rms
        ).clamp(
            min=0.0,
            max=20.0,
        ) / 20.0

        layer_feature = torch.full(
            (iterate.shape[0], 1),
            float(layer)
            / max(self.layers - 1, 1),
            device=iterate.device,
            dtype=iterate.dtype,
        )

        features = torch.cat(
            [
                torch.log1p(residual_ratio),
                torch.log1p(change_ratio),
                l1_l2,
                active_ratio,
                mean_abs_ratio,
                max_abs_ratio,
                adaptive_beta,
                layer_feature,
            ],
            dim=1,
        )

        # The recurrent controller receives state information but does
        # not create unstable high-order dependencies through states.
        return features.detach()

    def instance_parameters(
        self,
        layer,
        iterate,
        previous_iterate,
        residual,
        measurement,
        adaptive_beta,
        recurrent_state,
    ):
        (
            base_theta,
            base_q,
            base_eta,
            base_gamma,
        ) = self.base_parameters(layer)

        batch_size = iterate.shape[0]

        if self.lstm is None:
            theta = base_theta.expand(
                batch_size,
                1,
            )

            q = base_q.expand(
                batch_size,
                1,
            )

            eta = base_eta.expand(
                batch_size,
                1,
            )

            gamma = base_gamma.expand(
                batch_size,
                1,
            )

            offsets = torch.zeros(
                batch_size,
                3,
                device=iterate.device,
                dtype=iterate.dtype,
            )

            return (
                theta,
                q,
                eta,
                gamma,
                offsets,
                recurrent_state,
            )

        features = self.make_state_features(
            iterate=iterate,
            previous_iterate=previous_iterate,
            residual=residual,
            measurement=measurement,
            base_theta=base_theta,
            adaptive_beta=adaptive_beta,
            layer=layer,
        )

        offsets, recurrent_state = self.lstm(
            features,
            recurrent_state,
        )

        theta = (
            base_theta
            * torch.exp(
                self.theta_adapt_scale
                * offsets[:, 0:1]
            )
        )

        q = torch.sigmoid(
            torch.logit(
                base_q.clamp(
                    1e-5,
                    1.0 - 1e-5,
                )
            )
            + self.q_adapt_scale
            * offsets[:, 1:2]
        ).clamp(
            min=1e-5,
            max=1.0 - 1e-5,
        )

        gamma = (
            base_gamma
            * torch.exp(
                self.gamma_adapt_scale
                * offsets[:, 2:3]
            )
        )

        eta = q / (
            2.0 * theta
        ).clamp_min(1e-8)

        return (
            theta,
            q,
            eta,
            gamma,
            offsets,
            recurrent_state,
        )

    def forward(
        self,
        measurement,
        return_all=False,
        return_auxiliary=False,
    ):
        batch_size = measurement.shape[0]

        zero = torch.zeros(
            batch_size,
            self.A.shape[1],
            device=measurement.device,
            dtype=measurement.dtype,
        )

        iterate = zero
        previous_iterate = zero
        previous_proximal = zero

        adaptive_beta = torch.full(
            (batch_size, 1),
            self.beta_init,
            device=measurement.device,
            dtype=measurement.dtype,
        )

        if self.lstm is not None:
            hidden = torch.zeros(
                batch_size,
                self.lstm.hidden_size,
                device=measurement.device,
                dtype=measurement.dtype,
            )

            cell = torch.zeros_like(hidden)
            recurrent_state = (hidden, cell)
        else:
            recurrent_state = None

        outputs = []
        acceptance_history = []
        beta_history = []
        offset_history = []

        adaptation_regularizer = torch.zeros(
            (),
            device=measurement.device,
        )

        for layer in range(self.layers):
            residual = (
                iterate @ self.A.T
                - measurement
            )

            (
                theta,
                q,
                eta,
                gamma,
                offsets,
                recurrent_state,
            ) = self.instance_parameters(
                layer=layer,
                iterate=iterate,
                previous_iterate=previous_iterate,
                residual=residual,
                measurement=measurement,
                adaptive_beta=adaptive_beta,
                recurrent_state=recurrent_state,
            )

            proximal_input = (
                iterate
                - gamma * (residual @ self.B.T)
            )

            proximal = mcp_prox(
                proximal_input,
                theta,
                q,
            )

            current_beta = adaptive_beta

            extrapolated = (
                proximal
                + current_beta
                * (
                    proximal
                    - previous_proximal
                )
            )

            proximal_value = layer_objective(
                proximal,
                measurement,
                self.A,
                theta,
                q,
                gamma,
            )

            extrapolated_value = layer_objective(
                extrapolated,
                measurement,
                self.A,
                theta,
                q,
                gamma,
            )

            choose_proximal = (
                proximal_value
                <= extrapolated_value
            ).detach()

            accept_extrapolation = (
                ~choose_proximal
            )

            next_iterate = torch.where(
                choose_proximal.unsqueeze(1),
                proximal,
                extrapolated,
            )

            reduced_beta = (
                self.beta_scale
                * adaptive_beta
            )

            increased_beta = torch.clamp(
                adaptive_beta
                / self.beta_scale,
                max=1.0,
            )

            adaptive_beta = torch.where(
                choose_proximal.unsqueeze(1),
                reduced_beta,
                increased_beta,
            ).detach()

            outputs.append(proximal)
            acceptance_history.append(
                accept_extrapolation.float()
            )
            beta_history.append(
                current_beta.squeeze(1).detach()
            )
            offset_history.append(offsets)

            adaptation_regularizer = (
                adaptation_regularizer
                + offsets.square().mean()
            )

            previous_iterate = iterate
            previous_proximal = proximal
            iterate = next_iterate

        outputs = torch.stack(
            outputs,
            dim=1,
        )

        acceptances = torch.stack(
            acceptance_history,
            dim=1,
        )

        betas = torch.stack(
            beta_history,
            dim=1,
        )

        offsets = torch.stack(
            offset_history,
            dim=1,
        )

        adaptation_regularizer = (
            adaptation_regularizer
            / self.layers
        )

        if return_all and return_auxiliary:
            return (
                proximal,
                outputs,
                adaptation_regularizer,
                acceptances,
                betas,
                offsets,
            )

        if return_all:
            return proximal, outputs

        if return_auxiliary:
            return (
                proximal,
                adaptation_regularizer,
                acceptances,
                betas,
                offsets,
            )

        return proximal


@torch.no_grad()
def validation_nmse(
    model,
    x_val,
    y_val,
    args,
):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for start in range(
        0,
        len(x_val),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(x_val),
        )

        target = x_val[start:end]
        prediction = model(
            y_val[start:end]
        )

        batch_loss = normalized_mse(
            prediction,
            target,
        )

        total_loss += (
            float(batch_loss)
            * len(target)
        )

        total_count += len(target)

    return total_loss / max(total_count, 1)


def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    args,
    training_seed,
):
    base_parameters = []
    controller_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("lstm."):
            controller_parameters.append(
                parameter
            )
        else:
            base_parameters.append(
                parameter
            )

    parameter_groups = [
        {
            "params": base_parameters,
            "lr": args.lr,
        }
    ]

    if controller_parameters:
        parameter_groups.append(
            {
                "params": controller_parameters,
                "lr": args.lr * 0.5,
            }
        )

    optimizer = torch.optim.Adam(
        parameter_groups,
        weight_decay=args.weight_decay,
    )

    print(
        f"[optimizer] base_lr={args.lr:.3e}, "
        f"controller_lr={args.lr * 0.5:.3e}, "
        f"base_parameters="
        f"{sum(p.numel() for p in base_parameters)}, "
        f"controller_parameters="
        f"{sum(p.numel() for p in controller_parameters)}"
    )

    best_validation = validation_nmse(
        model,
        x_val,
        y_val,
        args,
    )
    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    history = []

    for epoch in range(args.epochs):
        model.train()

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(
            training_seed + epoch
        )

        permutation = torch.randperm(
            len(x_train),
            generator=generator,
        )

        total_final = 0.0
        total_layer = 0.0
        total_regularizer = 0.0
        total_objective = 0.0
        total_count = 0

        for start in range(
            0,
            len(x_train),
            args.batch_size,
        ):
            indices = permutation[
                start:start + args.batch_size
            ].to(x_train.device)

            target = x_train.index_select(
                0,
                indices,
            )

            measurement = y_train.index_select(
                0,
                indices,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                prediction,
                layer_outputs,
                adaptation_regularizer,
                _,
                _,
                _,
            ) = model(
                measurement,
                return_all=True,
                return_auxiliary=True,
            )

            final_loss = normalized_mse(
                prediction,
                target,
            )

            layer_loss = torch.zeros(
                (),
                device=target.device,
            )

            for layer in range(
                layer_outputs.shape[1]
            ):
                layer_loss = (
                    layer_loss
                    + normalized_mse(
                        layer_outputs[
                            :, layer, :
                        ],
                        target,
                    )
                )

            layer_loss = (
                layer_loss
                / layer_outputs.shape[1]
            )

            objective = (
                final_loss
                + args.layer_loss_weight
                * layer_loss
                + args.adapt_regularization
                * adaptation_regularizer
            )

            if not torch.isfinite(objective):
                raise RuntimeError(
                    f"Non-finite loss in {model.method}. "
                    "Reduce --lr or adaptation scales."
                )

            objective.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            optimizer.step()

            batch_size = len(target)

            total_final += (
                float(final_loss.detach())
                * batch_size
            )

            total_layer += (
                float(layer_loss.detach())
                * batch_size
            )

            total_regularizer += (
                float(
                    adaptation_regularizer.detach()
                )
                * batch_size
            )

            total_objective += (
                float(objective.detach())
                * batch_size
            )

            total_count += batch_size

        validation = validation_nmse(
            model,
            x_val,
            y_val,
            args,
        )

        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

        record = {
            "epoch": epoch + 1,
            "train_final_nmse": (
                total_final
                / max(total_count, 1)
            ),
            "train_layer_nmse": (
                total_layer
                / max(total_count, 1)
            ),
            "adaptation_regularizer": (
                total_regularizer
                / max(total_count, 1)
            ),
            "train_objective": (
                total_objective
                / max(total_count, 1)
            ),
            "validation_nmse": validation,
            "best_validation_nmse": (
                best_validation
            ),
            "best_epoch": best_epoch,
        }

        history.append(record)

        print(
            f"[{model.method:17s}] "
            f"epoch={epoch + 1:03d}/"
            f"{args.epochs:03d}, "
            f"train_final="
            f"{record['train_final_nmse']:.6e}, "
            f"objective="
            f"{record['train_objective']:.6e}, "
            f"val={validation:.6e}, "
            f"best={best_validation:.6e} "
            f"(epoch {best_epoch})"
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"[{model.method}] completed "
        f"{args.epochs} epochs; "
        f"restored epoch {best_epoch}, "
        f"val={best_validation:.6e}"
    )

    return history


def compute_metrics(
    prediction,
    target,
    args,
):
    error_energy = (
        prediction - target
    ).square().sum(dim=1)

    signal_energy = (
        target.square()
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    sample_nmse = (
        error_energy / signal_energy
    )

    total_nmse = float(
        error_energy.sum()
        / signal_energy.sum()
    )

    ser = 10.0 * math.log10(
        float(
            signal_energy.sum()
            / error_energy.sum().clamp_min(
                1e-20
            )
        )
    )

    true_support = (
        target.abs()
        > args.support_threshold
    )

    predicted_support = (
        prediction.abs()
        > args.support_threshold
    )

    tp = (
        true_support & predicted_support
    ).sum(dim=1).float()

    fp = (
        (~true_support)
        & predicted_support
    ).sum(dim=1).float()

    fn = (
        true_support
        & (~predicted_support)
    ).sum(dim=1).float()

    precision = tp / (
        tp + fp
    ).clamp_min(1.0)

    recall = tp / (
        tp + fn
    ).clamp_min(1.0)

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision + recall
        ).clamp_min(1e-12)
    )

    exact_support = (
        true_support
        == predicted_support
    ).all(dim=1).float()

    success = (
        sample_nmse
        <= args.success_nmse
    ).float()

    return {
        "ser_db": ser,
        "nmse": total_nmse,
        "nmse_db": (
            10.0
            * math.log10(
                max(total_nmse, 1e-20)
            )
        ),
        "support_f1": float(
            f1.mean()
        ),
        "exact_support_rate": float(
            exact_support.mean()
        ),
        "success_rate": float(
            success.mean()
        ),
    }


@torch.no_grad()
def evaluate_model(
    model,
    x_test,
    y_test,
    args,
):
    model.eval()

    predictions = []

    layer_errors = torch.zeros(
        args.layers,
        dtype=torch.float64,
    )

    offset_sum = torch.zeros(
        args.layers,
        3,
        dtype=torch.float64,
    )

    offset_abs_sum = torch.zeros(
        args.layers,
        3,
        dtype=torch.float64,
    )

    acceptance_sum = torch.zeros(
        args.layers,
        dtype=torch.float64,
    )

    beta_sum = torch.zeros(
        args.layers,
        dtype=torch.float64,
    )

    signal_energy = 0.0
    total_samples = 0

    warmup_count = min(
        args.batch_size,
        len(y_test),
    )

    _ = model(
        y_test[:warmup_count]
    )

    runtime_values = []

    for _ in range(
        max(args.timing_repeats, 1)
    ):
        if x_test.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for start in range(
            0,
            len(y_test),
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                len(y_test),
            )

            _ = model(y_test[start:end])

        if x_test.device.type == "cuda":
            torch.cuda.synchronize()

        runtime_values.append(
            time.perf_counter() - start_time
        )

    for start in range(
        0,
        len(x_test),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(x_test),
        )

        target = x_test[start:end]
        measurement = y_test[start:end]

        (
            prediction,
            outputs,
            _,
            acceptances,
            beta_history,
            offsets,
        ) = model(
            measurement,
            return_all=True,
            return_auxiliary=True,
        )

        predictions.append(prediction)

        signal_energy += float(
            target.square().sum()
        )

        for layer in range(args.layers):
            layer_errors[layer] += float(
                (
                    outputs[:, layer, :]
                    - target
                ).square().sum()
            )

        acceptance_sum += (
            acceptances
            .sum(dim=0)
            .cpu()
            .double()
        )

        beta_sum += (
            beta_history
            .sum(dim=0)
            .cpu()
            .double()
        )

        offset_sum += (
            offsets
            .sum(dim=0)
            .cpu()
            .double()
        )

        offset_abs_sum += (
            offsets
            .abs()
            .sum(dim=0)
            .cpu()
            .double()
        )

        total_samples += len(target)

    prediction = torch.cat(
        predictions,
        dim=0,
    )

    result = compute_metrics(
        prediction,
        x_test,
        args,
    )

    runtime = float(
        np.median(runtime_values)
    )

    result["runtime_seconds"] = runtime

    result["milliseconds_per_sample"] = (
        runtime * 1000.0
        / len(x_test)
    )

    result["trainable_parameters"] = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    result["layer_ser"] = [
        10.0 * math.log10(
            signal_energy
            / max(
                float(layer_errors[layer]),
                1e-20,
            )
        )
        for layer in range(args.layers)
    ]

    result["extrapolation_rate_by_layer"] = (
        acceptance_sum
        / max(total_samples, 1)
    ).tolist()

    result["mean_beta_by_layer"] = (
        beta_sum
        / max(total_samples, 1)
    ).tolist()

    result["mean_offsets_by_layer"] = (
        offset_sum
        / max(total_samples, 1)
    ).tolist()

    result["mean_abs_offsets_by_layer"] = (
        offset_abs_sum
        / max(total_samples, 1)
    ).tolist()

    return result


def paired_comparison(
    models,
    x_test,
    y_test,
):
    """
    Evaluate paired sample-level NMSE. This is more informative than
    comparing only aggregate SER.
    """
    sample_nmse = {}

    for method in METHODS:
        model = models[method]
        model.eval()
        values = []

        with torch.no_grad():
            for start in range(
                0,
                len(x_test),
                256,
            ):
                end = min(
                    start + 256,
                    len(x_test),
                )

                target = x_test[start:end]
                prediction = model(
                    y_test[start:end]
                )

                error = (
                    prediction - target
                ).square().sum(dim=1)

                energy = target.square().sum(
                    dim=1
                ).clamp_min(1e-12)

                values.append(error / energy)

        sample_nmse[method] = torch.cat(
            values,
            dim=0,
        )

    baseline = sample_nmse["apgnc_plus"]
    lstm = sample_nmse["lstm_apgnc_plus"]

    return {
        "lstm_win_rate_over_apgnc_plus": float(
            (lstm < baseline).float().mean()
        ),
        "lstm_tie_rate_over_apgnc_plus": float(
            torch.isclose(
                lstm,
                baseline,
                rtol=1e-6,
                atol=1e-12,
            ).float().mean()
        ),
        "median_lstm_over_baseline_nmse_ratio": float(
            torch.median(
                lstm
                / baseline.clamp_min(1e-20)
            )
        ),
    }


def build_test_conditions(args):
    probabilities = parse_number_list(
        args.test_prob_list,
        float,
    )

    snrs = []

    for item in args.snr_list.split(","):
        item = item.strip().lower()

        if item == "inf":
            snrs.append(math.inf)
        else:
            snrs.append(float(item))

    amplitudes = [
        item.strip()
        for item in args.amplitude_list.split(",")
        if item.strip()
    ]

    conditions = [
        {
            "family": "in_distribution",
            "label": "train_distribution",
            "probability": args.train_nonzero_prob,
            "snr_db": math.inf,
            "amplitude": "gaussian",
        }
    ]

    for probability in probabilities:
        if abs(
            probability
            - args.train_nonzero_prob
        ) < 1e-12:
            continue

        conditions.append(
            {
                "family": "sparsity",
                "label": f"p={probability:.3f}",
                "probability": probability,
                "snr_db": math.inf,
                "amplitude": "gaussian",
            }
        )

    for snr_db in snrs:
        conditions.append(
            {
                "family": "snr",
                "label": f"snr={snr_db:g}",
                "probability": args.train_nonzero_prob,
                "snr_db": snr_db,
                "amplitude": "gaussian",
            }
        )

    for amplitude in amplitudes:
        conditions.append(
            {
                "family": "amplitude",
                "label": amplitude,
                "probability": args.train_nonzero_prob,
                "snr_db": math.inf,
                "amplitude": amplitude,
            }
        )

    return conditions


def mean_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(values.mean())

    if len(values) > 1:
        std = float(
            values.std(ddof=1)
        )

        ci95 = float(
            1.96 * std / math.sqrt(len(values))
        )
    else:
        std = 0.0
        ci95 = 0.0

    return mean, std, ci95


def save_aggregate(
    records,
    output_dir,
):
    with open(
        output_dir / "all_runs.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "seed",
            "m",
            "family",
            "condition",
            "probability",
            "snr_db",
            "amplitude",
            "method",
            "ser_db",
            "nmse",
            "support_f1",
            "exact_support_rate",
            "success_rate",
            "runtime_seconds",
            "milliseconds_per_sample",
            "lstm_win_rate",
            "median_lstm_baseline_nmse_ratio",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(list)

    for record in records:
        key = (
            record["m"],
            record["family"],
            record["condition"],
            record["method"],
        )

        grouped[key].append(record)

    aggregate = []

    for key, rows in grouped.items():
        m, family, condition, method = key

        row = {
            "m": m,
            "family": family,
            "condition": condition,
            "method": method,
            "runs": len(rows),
        }

        for metric in (
            "ser_db",
            "nmse",
            "support_f1",
            "exact_support_rate",
            "success_rate",
            "runtime_seconds",
            "milliseconds_per_sample",
        ):
            mean, std, ci95 = mean_std(
                [item[metric] for item in rows]
            )

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95

        aggregate.append(row)

    aggregate.sort(
        key=lambda item: (
            item["m"],
            item["family"],
            item["condition"],
            item["method"],
        )
    )

    with open(
        output_dir / "aggregate.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                aggregate[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(aggregate)

    with open(
        output_dir / "aggregate.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aggregate,
            file,
            indent=2,
        )

    make_plots(
        aggregate,
        output_dir,
    )


def make_plots(
    aggregate,
    output_dir,
):
    styles = {
        "apgnc_plus": {
            "color": "#1f77b4",
            "marker": "o",
            "label": "ALePOM-APGnc+",
        },
        "lstm_apgnc_plus": {
            "color": "#d62728",
            "marker": "s",
            "label": "LSTM-ALePOM-APGnc+",
        },
    }

    for m in sorted(
        {row["m"] for row in aggregate}
    ):
        for family in (
            "sparsity",
            "snr",
            "amplitude",
        ):
            rows = [
                row
                for row in aggregate
                if (
                    row["m"] == m
                    and row["family"] == family
                )
            ]

            if not rows:
                continue

            conditions = []

            for row in rows:
                if row["condition"] not in conditions:
                    conditions.append(
                        row["condition"]
                    )

            plt.figure(
                figsize=(8.0, 5.2)
            )

            x = np.arange(
                len(conditions)
            )

            for method in METHODS:
                values = {
                    row["condition"]: row
                    for row in rows
                    if row["method"] == method
                }

                means = [
                    values[condition][
                        "ser_db_mean"
                    ]
                    for condition in conditions
                ]

                stds = [
                    values[condition][
                        "ser_db_std"
                    ]
                    for condition in conditions
                ]

                style = styles[method]

                plt.errorbar(
                    x,
                    means,
                    yerr=stds,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    capsize=3,
                )

            plt.xticks(
                x,
                conditions,
                rotation=25,
            )

            plt.xlabel(family)
            plt.ylabel("SER (dB)")
            plt.title(
                f"{family}: M={m}"
            )
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                output_dir
                / f"ser_{family}_m{m}.png",
                dpi=200,
            )

            plt.close()


def save_checkpoint(
    output_dir,
    method,
    model,
    history,
    args,
    seed,
    m,
):
    torch.save(
        {
            "method": method,
            "model_state": {
                key: value.detach().cpu()
                for key, value
                in model.state_dict().items()
            },
            "history": history,
            "seed": seed,
            "m": m,
            "config": vars(args),
        },
        output_dir
        / f"{method}_m{m}_seed{seed}.pt",
    )


def run_trial(
    args,
    seed,
    m,
    device,
    output_dir,
    conditions,
):
    print("\n" + "=" * 88)
    print(
        f"Training pair: seed={seed}, "
        f"M={m}, N={args.n}"
    )
    print("=" * 88)

    set_seed(seed)

    A = make_sensing_matrix(
        m,
        args.n,
        seed,
        device,
    )

    B = pretrain_B(
        A,
        args,
        output_dir,
        seed,
        m,
    ).to(device)

    B = normalize_B(B, A).detach()
    B.requires_grad_(False)

    print(
        "max |diag(BA)-1| = "
        f"{float((torch.diagonal(B @ A) - 1.0).abs().max()):.6e}"
    )

    x_train, y_train = make_dataset(
        A,
        args.n_train,
        args.train_nonzero_prob,
        seed + 1,
        device,
        amplitude="gaussian",
        snr_db=math.inf,
    )

    x_val, y_val = make_dataset(
        A,
        args.n_val,
        args.train_nonzero_prob,
        seed + 2,
        device,
        amplitude="gaussian",
        snr_db=math.inf,
    )

    models = {}
    histories = {}

    for method in METHODS:
        set_seed(seed)

        model = RecoveryModel(
            A,
            B,
            args,
            method,
        ).to(device)

        # Scalar parameter initialization is explicitly identical.
        if method == "lstm_apgnc_plus":
            with torch.no_grad():
                model.raw_theta.copy_(
                    models["apgnc_plus"].raw_theta
                )
                model.raw_q.copy_(
                    models["apgnc_plus"].raw_q
                )
                model.raw_gamma.copy_(
                    models["apgnc_plus"].raw_gamma
                )

        print(f"\nTraining {method}")

        histories[method] = train_model(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            args,
            training_seed=seed,
        )

        models[method] = model

        save_checkpoint(
            output_dir,
            method,
            model,
            histories[method],
            args,
            seed,
            m,
        )

    records = []
    layer_records = []
    dynamics_records = []

    for condition_index, condition in enumerate(
        conditions
    ):
        test_seed = (
            seed
            + 100000
            + condition_index * 100
            + m
        )

        x_test, y_test = make_dataset(
            A,
            args.n_test,
            condition["probability"],
            test_seed,
            device,
            amplitude=condition["amplitude"],
            snr_db=condition["snr_db"],
        )

        result_by_method = {}

        for method in METHODS:
            result_by_method[method] = (
                evaluate_model(
                    models[method],
                    x_test,
                    y_test,
                    args,
                )
            )

        paired = paired_comparison(
            models,
            x_test,
            y_test,
        )

        for method in METHODS:
            result = result_by_method[method]

            records.append(
                {
                    "seed": seed,
                    "m": m,
                    "family": condition["family"],
                    "condition": condition["label"],
                    "probability": condition[
                        "probability"
                    ],
                    "snr_db": (
                        "inf"
                        if math.isinf(
                            condition["snr_db"]
                        )
                        else condition["snr_db"]
                    ),
                    "amplitude": condition[
                        "amplitude"
                    ],
                    "method": method,
                    "ser_db": result["ser_db"],
                    "nmse": result["nmse"],
                    "support_f1": result[
                        "support_f1"
                    ],
                    "exact_support_rate": result[
                        "exact_support_rate"
                    ],
                    "success_rate": result[
                        "success_rate"
                    ],
                    "runtime_seconds": result[
                        "runtime_seconds"
                    ],
                    "milliseconds_per_sample": result[
                        "milliseconds_per_sample"
                    ],
                    "lstm_win_rate": (
                        paired[
                            "lstm_win_rate_over_apgnc_plus"
                        ]
                        if method == "lstm_apgnc_plus"
                        else ""
                    ),
                    "median_lstm_baseline_nmse_ratio": (
                        paired[
                            "median_lstm_over_baseline_nmse_ratio"
                        ]
                        if method == "lstm_apgnc_plus"
                        else ""
                    ),
                }
            )

            for layer, ser in enumerate(
                result["layer_ser"]
            ):
                layer_records.append(
                    {
                        "seed": seed,
                        "m": m,
                        "family": condition["family"],
                        "condition": condition["label"],
                        "method": method,
                        "layer": layer + 1,
                        "ser_db": ser,
                    }
                )

            for layer in range(args.layers):
                dynamics_records.append(
                    {
                        "seed": seed,
                        "m": m,
                        "family": condition["family"],
                        "condition": condition["label"],
                        "method": method,
                        "layer": layer + 1,
                        "extrapolation_rate": result[
                            "extrapolation_rate_by_layer"
                        ][layer],
                        "mean_beta": result[
                            "mean_beta_by_layer"
                        ][layer],
                        "mean_theta_offset": result[
                            "mean_offsets_by_layer"
                        ][layer][0],
                        "mean_q_offset": result[
                            "mean_offsets_by_layer"
                        ][layer][1],
                        "mean_gamma_offset": result[
                            "mean_offsets_by_layer"
                        ][layer][2],
                        "mean_abs_theta_offset": result[
                            "mean_abs_offsets_by_layer"
                        ][layer][0],
                        "mean_abs_q_offset": result[
                            "mean_abs_offsets_by_layer"
                        ][layer][1],
                        "mean_abs_gamma_offset": result[
                            "mean_abs_offsets_by_layer"
                        ][layer][2],
                    }
                )

        print(
            f"[test] seed={seed}, M={m}, "
            f"{condition['label']}: "
            f"base="
            f"{result_by_method['apgnc_plus']['ser_db']:.3f} dB, "
            f"LSTM="
            f"{result_by_method['lstm_apgnc_plus']['ser_db']:.3f} dB, "
            f"gain="
            f"{result_by_method['lstm_apgnc_plus']['ser_db'] - result_by_method['apgnc_plus']['ser_db']:+.3f} dB, "
            f"win="
            f"{paired['lstm_win_rate_over_apgnc_plus']:.3f}"
        )

    del models

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return records, layer_records, dynamics_records


def main():
    args = parse_args()

    seeds = parse_number_list(
        args.seeds,
        int,
    )

    m_values = parse_number_list(
        args.m_list,
        int,
    )

    if not seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if not m_values:
        raise ValueError(
            "At least one M value is required."
        )

    device = get_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
        )

    conditions = build_test_conditions(
        args
    )

    all_records = []
    all_layer_records = []
    all_dynamics_records = []

    for m in m_values:
        if m >= args.n:
            raise ValueError(
                f"M={m} must be less than N={args.n}"
            )

        for seed in seeds:
            (
                records,
                layer_records,
                dynamics_records,
            ) = run_trial(
                args,
                seed,
                m,
                device,
                output_dir,
                conditions,
            )

            all_records.extend(records)
            all_layer_records.extend(
                layer_records
            )
            all_dynamics_records.extend(
                dynamics_records
            )

    save_aggregate(
        all_records,
        output_dir,
    )

    with open(
        output_dir / "layer_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "seed",
            "m",
            "family",
            "condition",
            "method",
            "layer",
            "ser_db",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_layer_records
        )

    with open(
        output_dir / "lstm_dynamics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = [
            "seed",
            "m",
            "family",
            "condition",
            "method",
            "layer",
            "extrapolation_rate",
            "mean_beta",
            "mean_theta_offset",
            "mean_q_offset",
            "mean_gamma_offset",
            "mean_abs_theta_offset",
            "mean_abs_q_offset",
            "mean_abs_gamma_offset",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_dynamics_records
        )

    print("\nCompleted LSTM comprehensive experiment.")
    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )
    print("Key outputs:")
    print("  aggregate.csv")
    print("  all_runs.csv")
    print("  layer_results.csv")
    print("  lstm_dynamics.csv")
    print("  ser_sparsity_m*.png")
    print("  ser_snr_m*.png")
    print("  ser_amplitude_m*.png")


if __name__ == "__main__":
    main()