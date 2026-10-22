"""
External Gaussian sparse-recovery comparison using 2025 papers.

This script reuses the supplied ALePOM implementation and adds ten
independently implemented baselines.  It keeps the same data generation,
analytical-matrix pretraining, normalized loss, validation checkpointing,
metrics, CSV records, and plot style used by the supplied paper scripts.

The five classical algorithms are:
    ISTA       Beck and Teboulle, SIAM Imaging Sciences, 2009.
    FISTA      Beck and Teboulle, SIAM Imaging Sciences, 2009.
    IHT        Blumensath and Davies, Applied and Computational Harmonic
               Analysis, 2009.
    OMP        Tropp and Gilbert, IEEE Transactions on Information Theory,
               2007.
    CoSaMP      Needell and Tropp, Transactions on Information Theory, 2009.

The five recent learned/unrolled baselines are tied to 2025 papers:
    SSRNN-2025       Sparse Signal Recovery Neural Network for MIMO-OTFS.
    Structured-SBL   Structured neural network for improved SBL iterations.
    SBCS-Net         Sparse Bayesian and deep-learning compressed sensing.
    SBL-Unfold-2025  Sparse Bayesian learning unfolding network.
    DeepCSnet        DeepCSnet compressed-sensing reconstruction network.

The papers target communication, sensor-network, DoA, or general compressed
sensing settings. Their trainable update blocks are reimplemented here for
the real-valued Gaussian vector experiment; source metadata and adaptation
notes are recorded in REFERENCES and references.json.

Run from this directory, for example:
    python external_sparse_recovery_comparison_2025.py --device cuda --reuse-b

The script intentionally does not execute on import.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm_apgnc_plus_comprehensive as core
import paper_all_experiments_m200 as paper


CLASSICAL_METHODS = (
    "ista",
    "fista",
    "iht",
    "omp",
    "cosamp",
)

LEARNED_METHODS = (
    "ssrnn_2025",
    "structured_sbl_2025",
    "sbcs_net",
    "sbl_unfold_2025",
    "deepcsnet",
)

LOCAL_METHODS = (
    "alepom",
    "proposed",
)

METHODS = LOCAL_METHODS + CLASSICAL_METHODS + LEARNED_METHODS


METHOD_CONFIGS = {
    "alepom": {
        "label": "ALePOM",
        "color": "#1f77b4",
        "marker": "o",
        "family": "supplied",
    },
    "proposed": {
        "label": "Proposed",
        "color": "#d62728",
        "marker": "s",
        "family": "supplied",
    },
    "ista": {
        "label": "ISTA",
        "color": "#9467bd",
        "marker": "v",
        "family": "classical",
    },
    "fista": {
        "label": "FISTA",
        "color": "#8c564b",
        "marker": "^",
        "family": "classical",
    },
    "iht": {
        "label": "IHT",
        "color": "#e377c2",
        "marker": "<",
        "family": "classical",
    },
    "omp": {
        "label": "OMP",
        "color": "#7f7f7f",
        "marker": ">",
        "family": "classical",
    },
    "cosamp": {
        "label": "CoSaMP",
        "color": "#bcbd22",
        "marker": "P",
        "family": "classical",
    },
    "ssrnn_2025": {
        "label": "SSRNN-2025",
        "color": "#17becf",
        "marker": "D",
        "family": "learned",
    },
    "structured_sbl_2025": {
        "label": "Structured-SBL",
        "color": "#2ca02c",
        "marker": "X",
        "family": "learned",
    },
    "sbcs_net": {
        "label": "SBCS-Net",
        "color": "#ff7f0e",
        "marker": "*",
        "family": "learned",
    },
    "sbl_unfold_2025": {
        "label": "SBL-Unfold-2025",
        "color": "#1b9e77",
        "marker": "h",
        "family": "learned",
    },
    "deepcsnet": {
        "label": "DeepCSnet",
        "color": "#a65628",
        "marker": "8",
        "family": "learned",
    },
}


REFERENCES = {
    "ista": {
        "citation": (
            "A. Beck and M. Teboulle, Fast Gradient-Based Algorithms for "
            "Constrained Total Variation Image Denoising and Deblurring "
            "Problems, SIAM Journal on Imaging Sciences, 2009."
        ),
        "doi_or_url": "https://doi.org/10.1137/080716542",
    },
    "fista": {
        "citation": (
            "A. Beck and M. Teboulle, A Fast Iterative Shrinkage-Thresholding "
            "Algorithm for Linear Inverse Problems, SIAM Journal on Imaging "
            "Sciences, 2009."
        ),
        "doi_or_url": "https://doi.org/10.1137/080716542",
    },
    "iht": {
        "citation": (
            "T. Blumensath and M. E. Davies, Iterative Hard Thresholding for "
            "Compressed Sensing, Applied and Computational Harmonic Analysis, "
            "2009."
        ),
        "doi_or_url": "https://doi.org/10.1016/j.acha.2009.04.002",
    },
    "omp": {
        "citation": (
            "J. A. Tropp and A. C. Gilbert, Signal Recovery from Random "
            "Measurements via Orthogonal Matching Pursuit, IEEE TIT, 2007."
        ),
        "doi_or_url": "https://doi.org/10.1109/TIT.2007.909108",
    },
    "cosamp": {
        "citation": (
            "D. Needell and J. A. Tropp, CoSaMP: Iterative Signal Recovery "
            "from Incomplete and Inaccurate Samples, IEEE TIT, 2009."
        ),
        "doi_or_url": "https://doi.org/10.1109/TIT.2009.2023823",
    },
    "ssrnn_2025": {
        "citation": (
            "Sparse Signal Recovery Neural Network With Application to "
            "High-Mobility Massive MIMO-OTFS Communication Systems, IEEE "
            "Transactions on Vehicular Technology, 2025."
        ),
        "year": 2025,
        "doi_or_url": "https://doi.org/10.1109/TVT.2025.3550456",
    },
    "structured_sbl_2025": {
        "citation": (
            "A Structured Neural Network Approach for Learning Improved "
            "Iterative Algorithms for SBL, ICASSP, 2025."
        ),
        "year": 2025,
        "doi_or_url": "https://doi.org/10.1109/ICASSP49660.2025.10890518",
    },
    "sbcs_net": {
        "citation": (
            "SBCS-Net: Sparse Bayesian and Deep Learning Framework for "
            "Compressed Sensing in Sensor Networks, Sensors, 2025."
        ),
        "year": 2025,
        "doi_or_url": "https://doi.org/10.3390/s25154559",
    },
    "sbl_unfold_2025": {
        "citation": (
            "Sparse Bayesian Learning Unfolding Network for Efficient DoA "
            "Estimation in Low SNR, IEEE Transactions on Vehicular Technology, "
            "2025."
        ),
        "year": 2025,
        "doi_or_url": "https://doi.org/10.1109/TVT.2025.3560445",
    },
    "deepcsnet": {
        "citation": (
            "Compressed Sensing Reconstruction Using DeepCSnet: A Deep Learning "
            "Framework, IEEE International Conference on Computer, Communication "
            "and Information Technology, 2025."
        ),
        "year": 2025,
        "doi_or_url": "https://doi.org/10.1109/ICCRTEE64519.2025.11053071",
    },
}


def parse_number_list(text, value_type):
    return [
        value_type(item.strip())
        for item in text.split(",")
        if item.strip()
    ]


def parse_snrs(text):
    values = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(math.inf if item == "inf" else float(item))
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ALePOM, Proposed, classical, and learned sparse-recovery models."
    )
    parser.add_argument("--output-dir", type=str, default="external_comparison_m200_2025")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=str, default="2025,2026,2027")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--train-nonzero-prob", type=float, default=0.10)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--test-prob-list", type=str, default="0.02,0.05,0.10,0.15,0.20")
    parser.add_argument("--snr-list", type=str, default="40,30,20,10")
    parser.add_argument("--amplitude-list", type=str, default="gaussian,rademacher,laplace")
    parser.add_argument("--b-epochs", type=int, default=2000)
    parser.add_argument("--b-lr", type=float, default=3e-3)
    parser.add_argument("--b-top-s", type=int, default=50)
    parser.add_argument("--b-fro-weight", type=float, default=1e-3)
    parser.add_argument("--reuse-b", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--layer-loss-weight", type=float, default=0.05)
    parser.add_argument("--adapt-regularization", type=float, default=3e-3)
    parser.add_argument("--theta-init", type=float, default=0.10)
    parser.add_argument("--gamma-init", type=float, default=1.0)
    parser.add_argument("--eta-ratio-init", type=float, default=0.50)
    parser.add_argument("--beta-init", type=float, default=0.50)
    parser.add_argument("--beta-scale", type=float, default=0.50)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--theta-adapt-scale", type=float, default=0.50)
    parser.add_argument("--q-adapt-scale", type=float, default=1.00)
    parser.add_argument("--gamma-adapt-scale", type=float, default=0.35)
    parser.add_argument("--support-threshold", type=float, default=1e-4)
    parser.add_argument("--success-nmse", type=float, default=1e-4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def apply_quick_mode(args):
    if not args.quick:
        return
    args.seeds = "2025"
    args.n_train = 512
    args.n_val = 128
    args.n_test = 128
    args.epochs = 3
    args.b_epochs = 30
    args.timing_repeats = 1


def inverse_softplus(value):
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def soft_threshold(value, threshold):
    return value.sign() * F.relu(value.abs() - threshold)


def hard_threshold(value, sparsity):
    sparsity = min(max(int(sparsity), 1), value.shape[1])
    indices = value.abs().topk(sparsity, dim=1).indices
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return torch.where(mask, value, torch.zeros_like(value))


def batched_least_squares(A, measurement, support):
    """Solve min_c ||A[:,support] c-y|| for every batch item."""
    design = A.T[support].transpose(1, 2)
    rhs = measurement.unsqueeze(-1)
    solution = torch.linalg.lstsq(design, rhs).solution.squeeze(-1)
    return solution


class FixedSparseModel(nn.Module):
    """Classical non-learned sparse-recovery algorithms."""

    def __init__(self, A, args, method):
        super().__init__()
        self.register_buffer("A", A.detach().clone())
        self.method = method
        self.layers = args.layers
        self.sparsity = max(1, round(args.train_nonzero_prob * args.n))
        spectral = float(torch.linalg.matrix_norm(A, ord=2).detach())
        self.step = 0.99 / max(spectral * spectral, 1e-8)
        self.threshold = 0.10 * self.step

    def _empty_aux(self, batch_size, device, dtype):
        return (
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, 3, device=device, dtype=dtype),
        )

    def _omp(self, measurement):
        batch_size = measurement.shape[0]
        n = self.A.shape[1]
        support = torch.zeros(batch_size, self.sparsity, dtype=torch.long, device=measurement.device)
        selected = torch.zeros(batch_size, n, dtype=torch.bool, device=measurement.device)
        iterate = torch.zeros(batch_size, n, device=measurement.device, dtype=measurement.dtype)
        outputs = []
        for layer in range(self.layers):
            residual = measurement - iterate @ self.A.T
            correlation = residual @ self.A
            correlation = correlation.masked_fill(selected, 0.0)
            index = correlation.abs().argmax(dim=1)
            slot = min(layer, self.sparsity - 1)
            support[:, slot] = index
            selected.scatter_(1, index.unsqueeze(1), True)
            active = support[:, :slot + 1]
            coefficient = batched_least_squares(self.A, measurement, active)
            iterate = torch.zeros_like(iterate)
            iterate.scatter_(1, active, coefficient)
            outputs.append(iterate)
        return torch.stack(outputs, dim=1)

    def _cosamp(self, measurement):
        batch_size = measurement.shape[0]
        n = self.A.shape[1]
        sparsity = min(self.sparsity, n)
        candidate_count = min(2 * sparsity, n)
        union_count = min(3 * sparsity, n)
        support_mask = torch.zeros(batch_size, n, dtype=torch.bool, device=measurement.device)
        iterate = torch.zeros(batch_size, n, device=measurement.device, dtype=measurement.dtype)
        outputs = []
        for _ in range(self.layers):
            residual = measurement - iterate @ self.A.T
            correlation = residual @ self.A
            candidate = correlation.abs().topk(candidate_count, dim=1).indices
            candidate_mask = torch.zeros_like(support_mask)
            candidate_mask.scatter_(1, candidate, True)
            union_mask = support_mask | candidate_mask
            union_score = union_mask.to(measurement.dtype)
            union = union_score.topk(union_count, dim=1).indices
            provisional = batched_least_squares(self.A, measurement, union)
            keep = provisional.abs().topk(sparsity, dim=1).indices
            new_support = union.gather(1, keep)
            new_coefficient = provisional.gather(1, keep)
            iterate = torch.zeros_like(iterate)
            iterate.scatter_(1, new_support, new_coefficient)
            support_mask = torch.zeros_like(support_mask)
            support_mask.scatter_(1, new_support, True)
            outputs.append(iterate)
        return torch.stack(outputs, dim=1)

    def forward(self, measurement, return_all=False, return_auxiliary=False):
        batch_size = measurement.shape[0]
        if self.method == "omp":
            outputs = self._omp(measurement)
        elif self.method == "cosamp":
            outputs = self._cosamp(measurement)
        else:
            iterate = torch.zeros(
                batch_size,
                self.A.shape[1],
                device=measurement.device,
                dtype=measurement.dtype,
            )
            outputs = []
            momentum = torch.zeros_like(iterate)
            t_value = 1.0
            for _ in range(self.layers):
                if self.method == "fista":
                    search = iterate + momentum
                    residual = search @ self.A.T - measurement
                    candidate = soft_threshold(
                        search - self.step * (residual @ self.A),
                        self.threshold,
                    )
                    next_t = (1.0 + math.sqrt(1.0 + 4.0 * t_value * t_value)) / 2.0
                    momentum = ((t_value - 1.0) / next_t) * (candidate - iterate)
                    t_value = next_t
                    iterate = candidate
                elif self.method == "iht":
                    residual = iterate @ self.A.T - measurement
                    iterate = hard_threshold(
                        iterate - self.step * (residual @ self.A),
                        self.sparsity,
                    )
                else:
                    residual = iterate @ self.A.T - measurement
                    iterate = soft_threshold(
                        iterate - self.step * (residual @ self.A),
                        self.threshold,
                    )
                outputs.append(iterate)
            outputs = torch.stack(outputs, dim=1)

        final = outputs[:, -1, :]
        regularizer, acceptances, betas, offsets = self._empty_aux(
            batch_size, measurement.device, measurement.dtype
        )
        if return_all and return_auxiliary:
            return final, outputs, regularizer, acceptances, betas, offsets
        if return_all:
            return final, outputs
        if return_auxiliary:
            return final, regularizer, acceptances, betas, offsets
        return final


class LearnedUnrolledModel(nn.Module):
    """Compact trainable implementations of five published unrolled models."""

    def __init__(self, A, B, args, method):
        super().__init__()
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B", B.detach().clone())
        self.method = method
        self.layers = args.layers
        spectral = float(torch.linalg.matrix_norm(A, ord=2).detach())
        initial_step = 0.99 / max(spectral * spectral, 1e-8)
        self.raw_step = nn.Parameter(
            torch.full((self.layers,), inverse_softplus(initial_step))
        )
        self.raw_theta = nn.Parameter(
            torch.full((self.layers,), inverse_softplus(0.10))
        )

        if method in ("structured_sbl_2025", "sbl_unfold_2025", "deepcsnet"):
            self.diagonal_gain = nn.Parameter(torch.zeros(self.layers, A.shape[1]))
        else:
            self.diagonal_gain = None

        if method == "ssrnn_2025":
            self.raw_onsager = nn.Parameter(torch.full((self.layers,), -2.0))
        else:
            self.raw_onsager = None

        if method in ("structured_sbl_2025", "sbcs_net", "sbl_unfold_2025"):
            self.raw_rho = nn.Parameter(torch.full((self.layers,), inverse_softplus(0.5)))
        else:
            self.raw_rho = None

    def _empty_aux(self, batch_size, device, dtype):
        return (
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, device=device, dtype=dtype),
            torch.zeros(batch_size, self.layers, 3, device=device, dtype=dtype),
        )

    def _gradient_step(self, iterate, measurement, layer, matrix):
        step = F.softplus(self.raw_step[layer]) + 1e-6
        theta = F.softplus(self.raw_theta[layer]) + 1e-6
        residual = iterate @ self.A.T - measurement
        update = iterate - step * (residual @ matrix.T)
        if self.diagonal_gain is not None:
            gain = 0.25 * torch.tanh(self.diagonal_gain[layer])
            update = update + gain * (update - iterate)
        return soft_threshold(update, theta)

    def forward(self, measurement, return_all=False, return_auxiliary=False):
        batch_size = measurement.shape[0]
        n = self.A.shape[1]
        iterate = torch.zeros(batch_size, n, device=measurement.device, dtype=measurement.dtype)
        previous = torch.zeros_like(iterate)
        outputs = []

        for layer in range(self.layers):
            theta = F.softplus(self.raw_theta[layer]) + 1e-6
            if self.method == "ssrnn_2025":
                step = F.softplus(self.raw_step[layer]) + 1e-6
                onsager = 0.25 * torch.tanh(self.raw_onsager[layer])
                residual = measurement - iterate @ self.A.T
                iterate = soft_threshold(
                    iterate + step * (residual @ self.A) + onsager * (iterate - previous),
                    theta,
                )
            elif self.method == "sbcs_net":
                step = F.softplus(self.raw_step[layer]) + 1e-6
                mixing = torch.sigmoid(self.raw_rho[layer])
                residual = measurement - iterate @ self.A.T
                linear = iterate + step * (residual @ self.A)
                denoised = soft_threshold(linear, theta)
                iterate = mixing * denoised + (1.0 - mixing) * iterate
            elif self.method == "sbl_unfold_2025":
                step = F.softplus(self.raw_step[layer]) + 1e-6
                precision = F.softplus(self.raw_rho[layer]) + 1e-6
                residual = iterate @ self.A.T - measurement
                linear = iterate - step * (residual @ self.B.T)
                adaptive_theta = theta / torch.sqrt(precision)
                gain = 0.25 * torch.tanh(self.diagonal_gain[layer])
                iterate = soft_threshold(linear + gain * (linear - previous), adaptive_theta)
            elif self.method == "structured_sbl_2025":
                step = F.softplus(self.raw_step[layer]) + 1e-6
                precision = F.softplus(self.raw_rho[layer]) + 1e-6
                residual = iterate @ self.A.T - measurement
                linear = iterate - step * (residual @ self.B.T)
                local_energy = linear.square()
                local_energy = F.avg_pool1d(
                    local_energy.unsqueeze(1), kernel_size=5, stride=1, padding=2
                ).squeeze(1)
                adaptive_theta = theta / torch.sqrt(precision + local_energy.mean(dim=1, keepdim=True))
                iterate = soft_threshold(linear, adaptive_theta)
            elif self.method == "deepcsnet":
                step = F.softplus(self.raw_step[layer]) + 1e-6
                residual = iterate @ self.A.T - measurement
                linear = iterate - step * (residual @ self.A)
                coordinate_gate = torch.sigmoid(self.diagonal_gain[layer]).unsqueeze(0)
                context = linear.mean(dim=1, keepdim=True)
                attended = linear + 0.15 * coordinate_gate * (context - linear)
                iterate = soft_threshold(attended, theta)
            else:
                raise ValueError(f"Unknown learned method: {self.method}")
            previous = iterate
            outputs.append(iterate)

        outputs = torch.stack(outputs, dim=1)
        final = outputs[:, -1, :]
        regularizer, acceptances, betas, offsets = self._empty_aux(
            batch_size, measurement.device, measurement.dtype
        )
        if return_all and return_auxiliary:
            return final, outputs, regularizer, acceptances, betas, offsets
        if return_all:
            return final, outputs
        if return_auxiliary:
            return final, regularizer, acceptances, betas, offsets
        return final


def build_conditions(args):
    probabilities = parse_number_list(args.test_prob_list, float)
    amplitudes = [item.strip().lower() for item in args.amplitude_list.split(",") if item.strip()]
    conditions = [
        {
            "family": "in_distribution",
            "label": "train_distribution",
            "plot_label": "Train",
            "probability": args.train_nonzero_prob,
            "snr_db": math.inf,
            "amplitude": "gaussian",
        }
    ]
    for probability in probabilities:
        conditions.append(
            {
                "family": "sparsity",
                "label": f"p={probability:.2f}",
                "plot_label": f"{probability:.2f}",
                "probability": probability,
                "snr_db": math.inf,
                "amplitude": "gaussian",
            }
        )
    for snr_db in parse_snrs(args.snr_list):
        conditions.append(
            {
                "family": "snr",
                "label": f"snr={snr_db:g}",
                "plot_label": f"{snr_db:g}",
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
                "plot_label": amplitude.capitalize(),
                "probability": args.train_nonzero_prob,
                "snr_db": math.inf,
                "amplitude": amplitude,
            }
        )
    return conditions


def make_model(name, A, B, args):
    if name == "alepom":
        return paper.PaperRecoveryModel(A, B, args, "alepom")
    if name == "proposed":
        return paper.PaperRecoveryModel(A, B, args, "proposed")
    if name in CLASSICAL_METHODS:
        return FixedSparseModel(A, args, name)
    return LearnedUnrolledModel(A, B, args, name)


def train_model(model, x_train, y_train, x_val, y_val, args, seed):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return []
    return core.train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
        training_seed=seed,
    )


@torch.no_grad()
def sample_nmse(model, x_test, y_test, batch_size):
    values = []
    model.eval()
    for start in range(0, len(x_test), batch_size):
        end = min(start + batch_size, len(x_test))
        prediction = model(y_test[start:end])
        error = (prediction - x_test[start:end]).square().sum(dim=1)
        energy = x_test[start:end].square().sum(dim=1).clamp_min(1e-12)
        values.append(error / energy)
    return torch.cat(values, dim=0)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return float(array.mean()), 0.0, 0.0
    std = float(array.std(ddof=1))
    return float(array.mean()), std, float(1.96 * std / math.sqrt(len(array)))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metric_records(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["family"],
            record["condition"],
            record["plot_label"],
            record["method"],
            record["method_label"],
        )
        grouped[key].append(record)
    rows = []
    for key, values in grouped.items():
        family, condition, plot_label, method, method_label = key
        row = {
            "family": family,
            "condition": condition,
            "plot_label": plot_label,
            "method": method,
            "method_label": method_label,
        }
        for metric in (
            "ser_db",
            "nmse",
            "support_f1",
            "exact_support_rate",
            "success_rate",
            "milliseconds_per_sample",
            "trainable_parameters",
        ):
            mean, std, ci95 = mean_std([value[metric] for value in values])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    order = {name: index for index, name in enumerate(METHODS)}
    rows.sort(key=lambda row: (row["family"], row["condition"], order[row["method"]]))
    return rows


def aggregate_layer_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["family"], record["condition"], record["method"], record["layer"])].append(record["ser_db"])
    rows = []
    for (family, condition, method, layer), values in grouped.items():
        mean, std, ci95 = mean_std(values)
        rows.append(
            {
                "family": family,
                "condition": condition,
                "method": method,
                "layer": layer,
                "ser_db_mean": mean,
                "ser_db_std": std,
                "ser_db_ci95": ci95,
            }
        )
    order = {name: index for index, name in enumerate(METHODS)}
    rows.sort(key=lambda row: (row["family"], row["condition"], row["layer"], order[row["method"]]))
    return rows


def aggregate_pairwise_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["family"], record["condition"], record["reference_method"], record["candidate_method"])].append(record)
    rows = []
    for key, values in grouped.items():
        family, condition, reference, candidate = key
        win_mean, win_std, win_ci = mean_std([value["win_rate"] for value in values])
        ratio_mean, ratio_std, ratio_ci = mean_std([value["median_nmse_ratio"] for value in values])
        rows.append(
            {
                "family": family,
                "condition": condition,
                "reference_method": reference,
                "candidate_method": candidate,
                "win_rate_mean": win_mean,
                "win_rate_std": win_std,
                "win_rate_ci95": win_ci,
                "median_nmse_ratio_mean": ratio_mean,
                "median_nmse_ratio_std": ratio_std,
                "median_nmse_ratio_ci95": ratio_ci,
            }
        )
    return rows


def save_figure(figure, output_dir, name):
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=200)
    figure.savefig(output_dir / f"{name}.pdf")
    plt.close(figure)


def plot_layer_ser(layer_rows, output_dir):
    family = "in_distribution"
    condition = "train_distribution"
    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    for method in METHODS:
        rows = [
            row for row in layer_rows
            if row["family"] == family and row["condition"] == condition and row["method"] == method
        ]
        rows.sort(key=lambda row: row["layer"])
        if not rows:
            continue
        x = np.asarray([row["layer"] for row in rows])
        y = np.asarray([row["ser_db_mean"] for row in rows])
        error = np.asarray([row["ser_db_ci95"] for row in rows])
        style = METHOD_CONFIGS[method]
        axis.plot(x, y, label=style["label"], color=style["color"], marker=style["marker"], linewidth=1.6, markersize=4)
        axis.fill_between(x, y - error, y + error, color=style["color"], alpha=0.08)
    axis.set_xlabel("Unfolded layer")
    axis.set_ylabel("SER (dB)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    save_figure(figure, output_dir, "layer_wise_ser_12_models")


def plot_condition_family(aggregate, output_dir, family, metric, ylabel, filename):
    conditions = [row for row in aggregate if row["family"] == family]
    labels = []
    for row in conditions:
        if row["plot_label"] not in labels:
            labels.append(row["plot_label"])
    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    x = np.arange(len(labels), dtype=float)
    for method in METHODS:
        means = []
        errors = []
        for label in labels:
            row = next(
                (item for item in conditions if item["plot_label"] == label and item["method"] == method),
                None,
            )
            means.append(np.nan if row is None else row[f"{metric}_mean"])
            errors.append(0.0 if row is None else row[f"{metric}_ci95"])
        style = METHOD_CONFIGS[method]
        axis.errorbar(
            x,
            means,
            yerr=errors,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.5,
            markersize=4,
            capsize=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25)
    axis.set_xlabel(family)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    save_figure(figure, output_dir, filename)


def plot_in_distribution_bars(aggregate, output_dir):
    rows = [
        row for row in aggregate
        if row["family"] == "in_distribution" and row["condition"] == "train_distribution"
    ]
    rows.sort(key=lambda row: METHODS.index(row["method"]))
    figure, axis = plt.subplots(figsize=(13.0, 6.0))
    x = np.arange(len(rows), dtype=float)
    values = [row["ser_db_mean"] for row in rows]
    errors = [row["ser_db_ci95"] for row in rows]
    colors = [METHOD_CONFIGS[row["method"]]["color"] for row in rows]
    bars = axis.bar(x, values, yerr=errors, color=colors, alpha=0.9, capsize=4)
    axis.set_xticks(x)
    axis.set_xticklabels([row["method_label"] for row in rows], rotation=30, ha="right")
    axis.set_ylabel("SER (dB)")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    save_figure(figure, output_dir, "in_distribution_ser_12_models")


def run_seed_experiment(args, seed, device, output_dir, conditions):
    print("\n" + "=" * 100)
    print(f"Seed={seed}, dimensions M={args.m}, N={args.n}, models={len(METHODS)}")
    print("=" * 100)
    core.set_seed(seed)
    A = core.make_sensing_matrix(args.m, args.n, seed, device)
    B = core.pretrain_B(A, args, output_dir, seed, args.m).to(device)
    B = core.normalize_B(B, A).detach()
    B.requires_grad_(False)

    x_train, y_train = core.make_dataset(
        A, args.n_train, args.train_nonzero_prob, seed + 1, device,
        amplitude="gaussian", snr_db=math.inf,
    )
    x_val, y_val = core.make_dataset(
        A, args.n_val, args.train_nonzero_prob, seed + 2, device,
        amplitude="gaussian", snr_db=math.inf,
    )

    models = {}
    histories = {}
    for method in METHODS:
        core.set_seed(seed)
        model = make_model(method, A, B, args).to(device)
        print(f"\nTraining {METHOD_CONFIGS[method]['label']}")
        histories[method] = train_model(model, x_train, y_train, x_val, y_val, args, seed)
        models[method] = model
        torch.save(
            {
                "method": method,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "history": histories[method],
                "seed": seed,
                "config": vars(args),
            },
            output_dir / f"{method}_seed{seed}.pt",
        )

    metric_records = []
    layer_records = []
    pairwise_records = []
    training_records = []
    for method, history in histories.items():
        for record in history:
            training_records.append({"seed": seed, "method": method, **record})

    for condition_index, condition in enumerate(conditions):
        test_seed = seed + 100000 + condition_index * 1000 + args.m
        x_test, y_test = core.make_dataset(
            A, args.n_test, condition["probability"], test_seed, device,
            amplitude=condition["amplitude"], snr_db=condition["snr_db"],
        )
        result_by_method = {}
        samples_by_method = {}
        for method in METHODS:
            result = core.evaluate_model(models[method], x_test, y_test, args)
            result_by_method[method] = result
            samples_by_method[method] = sample_nmse(models[method], x_test, y_test, args.batch_size)
            style = METHOD_CONFIGS[method]
            metric_records.append(
                {
                    "seed": seed,
                    "family": condition["family"],
                    "condition": condition["label"],
                    "plot_label": condition["plot_label"],
                    "method": method,
                    "method_label": style["label"],
                    "ser_db": result["ser_db"],
                    "nmse": result["nmse"],
                    "support_f1": result["support_f1"],
                    "exact_support_rate": result["exact_support_rate"],
                    "success_rate": result["success_rate"],
                    "milliseconds_per_sample": result["milliseconds_per_sample"],
                    "trainable_parameters": result["trainable_parameters"],
                }
            )
            for layer, ser_db in enumerate(result["layer_ser"], start=1):
                layer_records.append(
                    {
                        "seed": seed,
                        "family": condition["family"],
                        "condition": condition["label"],
                        "method": method,
                        "layer": layer,
                        "ser_db": ser_db,
                    }
                )

        reference_methods = ("alepom", "proposed")
        for reference in reference_methods:
            for candidate in METHODS:
                if candidate == reference:
                    continue
                reference_values = samples_by_method[reference]
                candidate_values = samples_by_method[candidate]
                pairwise_records.append(
                    {
                        "seed": seed,
                        "family": condition["family"],
                        "condition": condition["label"],
                        "reference_method": reference,
                        "candidate_method": candidate,
                        "win_rate": float((candidate_values < reference_values).float().mean()),
                        "median_nmse_ratio": float(torch.median(candidate_values / reference_values.clamp_min(1e-20))),
                    }
                )
        print(
            f"[test] {condition['label']:>18s} "
            f"ALePOM={result_by_method['alepom']['ser_db']:.3f} dB, "
            f"Proposed={result_by_method['proposed']['ser_db']:.3f} dB"
        )

    del models, x_train, y_train, x_val, y_val
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_records, layer_records, pairwise_records, training_records


def main():
    args = parse_args()
    apply_quick_mode(args)
    if args.m != 200:
        raise ValueError("This comparison uses the paper setting M=200.")
    if args.n <= args.m:
        raise ValueError("N must be larger than M.")
    seeds = parse_number_list(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed is required.")
    device = core.get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args)

    with open(output_dir / "config.json", "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)
    with open(output_dir / "references.json", "w", encoding="utf-8") as stream:
        json.dump(REFERENCES, stream, indent=2)
    with open(output_dir / "methods.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "method_order": list(METHODS),
                "method_configs": METHOD_CONFIGS,
                "local_methods": list(LOCAL_METHODS),
                "classical_methods": list(CLASSICAL_METHODS),
                "learned_methods": list(LEARNED_METHODS),
            },
            stream,
            indent=2,
        )

    all_metrics = []
    all_layers = []
    all_pairwise = []
    all_training = []
    start_time = time.perf_counter()
    for seed in seeds:
        metrics, layers, pairwise, training = run_seed_experiment(
            args, seed, device, output_dir, conditions
        )
        all_metrics.extend(metrics)
        all_layers.extend(layers)
        all_pairwise.extend(pairwise)
        all_training.extend(training)
        write_csv(output_dir / "all_runs.csv", all_metrics)
        write_csv(output_dir / "layer_results.csv", all_layers)
        write_csv(output_dir / "pairwise_results.csv", all_pairwise)
        write_csv(output_dir / "training_history.csv", all_training)

    aggregate = aggregate_metric_records(all_metrics)
    layer_aggregate = aggregate_layer_records(all_layers)
    pairwise_aggregate = aggregate_pairwise_records(all_pairwise)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_csv(output_dir / "layer_aggregate.csv", layer_aggregate)
    write_csv(output_dir / "pairwise_aggregate.csv", pairwise_aggregate)

    with open(output_dir / "aggregate.json", "w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2)

    plot_in_distribution_bars(aggregate, output_dir)
    plot_layer_ser(layer_aggregate, output_dir)
    plot_condition_family(aggregate, output_dir, "sparsity", "ser_db", "SER (dB)", "ser_sparsity_12_models")
    plot_condition_family(aggregate, output_dir, "snr", "ser_db", "SER (dB)", "ser_snr_12_models")
    plot_condition_family(aggregate, output_dir, "amplitude", "ser_db", "SER (dB)", "ser_amplitude_12_models")
    plot_condition_family(aggregate, output_dir, "sparsity", "support_f1", "Support F1", "support_f1_sparsity_12_models")
    plot_condition_family(aggregate, output_dir, "snr", "support_f1", "Support F1", "support_f1_snr_12_models")

    in_distribution = [
        row for row in aggregate
        if row["family"] == "in_distribution" and row["condition"] == "train_distribution"
    ]
    summary = {
        "dimensions": {"M": args.m, "N": args.n, "undersampling_ratio": args.m / args.n},
        "methods": list(METHODS),
        "in_distribution": in_distribution,
        "references": REFERENCES,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("\nComparison completed.")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
