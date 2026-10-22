# paper_all_experiments_m200.py
#
# Place this file in the same directory as:
#     lstm_apgnc_plus_comprehensive.py
#
# Run:
#     python paper_all_experiments_m200.py
#
# Optional:
#     python paper_all_experiments_m200.py \
#         --device cuda \
#         --reuse-b
#
# Quick debugging:
#     python paper_all_experiments_m200.py \
#         --quick \
#         --device cuda
#
# This script trains four ablation models:
#   1. alepom
#   2. apgnc_plus_alepom
#   3. lstm_alepom
#   4. proposed
#
# Fixed experimental dimensions:
#     N = 500
#     M = 200

import argparse

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import lstm_apgnc_plus_comprehensive as core


METHOD_CONFIGS = {
    "alepom": {
        "use_momentum": False,
        "use_lstm": False,
        "label": "ALePOM",
        "color": "#1f77b4",
        "marker": "o",
    },
    "apgnc_plus_alepom": {
        "use_momentum": True,
        "use_lstm": False,
        "label": "APGnc+-ALePOM",
        "color": "#ff7f0e",
        "marker": "^",
    },
    "lstm_alepom": {
        "use_momentum": False,
        "use_lstm": True,
        "label": "LSTM-ALePOM",
        "color": "#2ca02c",
        "marker": "D",
    },
    "proposed": {
        "use_momentum": True,
        "use_lstm": True,
        "label": "Proposed",
        "color": "#d62728",
        "marker": "s",
    },
}

METHODS = tuple(METHOD_CONFIGS.keys())


def parse_number_list(text, value_type):
    if not text.strip():
        return []

    return [
        value_type(item.strip())
        for item in text.split(",")
        if item.strip()
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Complete M=200 paper experiments for "
            "ALePOM, APGnc+-ALePOM, LSTM-ALePOM, "
            "and the complete proposed model."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_experiments_m200",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="2025,2026,2027",
    )

    # Fixed sparse recovery dimensions.
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--layers", type=int, default=16)

    # Training distribution.
    parser.add_argument(
        "--train-nonzero-prob",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--n-val",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
    )

    # Generalization experiments.
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
        default="gaussian,rademacher,laplace",
    )

    # Analytical B.
    parser.add_argument(
        "--b-epochs",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--b-lr",
        type=float,
        default=3e-3,
    )

    parser.add_argument(
        "--b-top-s",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--b-fro-weight",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--reuse-b",
        action="store_true",
    )

    # Recovery model training.
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--layer-loss-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--adapt-regularization",
        type=float,
        default=3e-3,
    )

    # MCP parameters.
    parser.add_argument(
        "--theta-init",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--gamma-init",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eta-ratio-init",
        type=float,
        default=0.50,
    )

    # APGnc+ parameters.
    parser.add_argument(
        "--beta-init",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--beta-scale",
        type=float,
        default=0.50,
    )

    # LSTM controller.
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--theta-adapt-scale",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--q-adapt-scale",
        type=float,
        default=1.00,
    )

    parser.add_argument(
        "--gamma-adapt-scale",
        type=float,
        default=0.35,
    )

    # Evaluation.
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--success-nmse",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
    )

    # Convenient debugging mode.
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a small dataset and fewer epochs to verify "
            "that the full pipeline runs correctly."
        ),
    )

    return parser.parse_args()


def apply_quick_mode(args):
    if not args.quick:
        return

    print("\n" + "!" * 80)
    print("QUICK MODE ENABLED")
    print("This mode is only for debugging and must not be used")
    print("to produce the final paper results.")
    print("!" * 80 + "\n")

    args.seeds = "2025"
    args.n_train = 512
    args.n_val = 128
    args.n_test = 128
    args.epochs = 3
    args.b_epochs = 30
    args.timing_repeats = 1


def parse_snrs(text):
    snrs = []

    for item in text.split(","):
        item = item.strip().lower()

        if not item:
            continue

        if item == "inf":
            snrs.append(math.inf)
        else:
            snrs.append(float(item))

    return snrs


def build_conditions(args):
    """
    Build all test conditions required by the paper.

    The same in-distribution condition is included separately
    from the sparsity generalization family, so the sparsity
    plot contains p=0.10 in the correct sequence.
    """
    probabilities = parse_number_list(
        args.test_prob_list,
        float,
    )

    snrs = parse_snrs(args.snr_list)

    amplitudes = [
        item.strip().lower()
        for item in args.amplitude_list.split(",")
        if item.strip()
    ]

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

    for snr_db in snrs:
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


class PaperRecoveryModel(core.RecoveryModel):
    """
    Unified model used for the complete 2x2 ablation:

                    No LSTM       LSTM
        no APGnc+   ALePOM        LSTM-ALePOM
        APGnc+      APGnc+        Proposed

    The underlying MCP parameterization, analytical matrix,
    LSTM features, and loss functions are inherited from the
    original implementation.
    """

    def __init__(
        self,
        A,
        B,
        args,
        method,
    ):
        if method not in METHOD_CONFIGS:
            raise ValueError(
                f"Unknown paper method: {method}"
            )

        config = METHOD_CONFIGS[method]

        # The original base class only distinguishes whether
        # an LSTM is constructed. We use its LSTM model when
        # required and override the momentum behavior below.
        parent_method = (
            "lstm_apgnc_plus"
            if config["use_lstm"]
            else "apgnc_plus"
        )

        super().__init__(
            A=A,
            B=B,
            args=args,
            method=parent_method,
        )

        self.method = method
        self.use_momentum = config["use_momentum"]
        self.use_lstm_controller = config["use_lstm"]

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

        # z^k in the paper.
        iterate = zero

        # z^{k-1}, used in the LSTM change feature.
        previous_iterate = zero

        # p^{k-1}, used in APGnc+ extrapolation.
        previous_proximal = zero

        if self.use_momentum:
            adaptive_beta = torch.full(
                (batch_size, 1),
                self.beta_init,
                device=measurement.device,
                dtype=measurement.dtype,
            )
        else:
            adaptive_beta = torch.zeros(
                batch_size,
                1,
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
            dtype=measurement.dtype,
        )

        proximal = zero

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

            proximal = core.mcp_prox(
                proximal_input,
                theta,
                q,
            )

            current_beta = adaptive_beta

            if self.use_momentum:
                extrapolated = (
                    proximal
                    + current_beta
                    * (
                        proximal
                        - previous_proximal
                    )
                )

                proximal_value = core.layer_objective(
                    proximal,
                    measurement,
                    self.A,
                    theta,
                    q,
                    gamma,
                )

                extrapolated_value = core.layer_objective(
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

            else:
                # Standard ALePOM behavior.
                next_iterate = proximal

                accept_extrapolation = torch.zeros(
                    batch_size,
                    device=measurement.device,
                    dtype=torch.bool,
                )

                adaptive_beta = torch.zeros_like(
                    adaptive_beta
                )

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


BASE_PARAMETER_NAMES = (
    "raw_theta",
    "raw_q",
    "raw_gamma",
)


def capture_base_parameters(model):
    """
    Capture an immutable snapshot of the common trainable
    ALePOM parameters before any model is trained.
    """
    return {
        name: getattr(model, name).detach().clone()
        for name in BASE_PARAMETER_NAMES
    }


def synchronize_base_parameters(
    target_model,
    initial_base_parameters,
):
    """
    Restore the common theta, q, and gamma parameters from
    the pre-training snapshot.
    """
    with torch.no_grad():
        for name in BASE_PARAMETER_NAMES:
            target_parameter = getattr(
                target_model,
                name,
            )

            source_parameter = (
                initial_base_parameters[name]
                .to(
                    device=target_parameter.device,
                    dtype=target_parameter.dtype,
                )
            )

            target_parameter.copy_(
                source_parameter
            )


def verify_base_parameters(
    model,
    initial_base_parameters,
):
    """
    Fail immediately if a model does not start from the
    common base-parameter initialization.
    """
    for name in BASE_PARAMETER_NAMES:
        current = getattr(
            model,
            name,
        ).detach()

        expected = (
            initial_base_parameters[name]
            .to(
                device=current.device,
                dtype=current.dtype,
            )
        )

        if not torch.equal(
            current,
            expected,
        ):
            max_error = float(
                (current - expected)
                .abs()
                .max()
            )

            raise RuntimeError(
                f"Initialization mismatch for {name}: "
                f"max absolute error={max_error:.6e}"
            )

@torch.no_grad()
def sample_nmse_values(
    model,
    x_test,
    y_test,
    batch_size,
):
    model.eval()
    values = []

    for start in range(
        0,
        len(x_test),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(x_test),
        )

        target = x_test[start:end]
        prediction = model(
            y_test[start:end]
        )

        error = (
            prediction - target
        ).square().sum(dim=1)

        energy = (
            target.square()
            .sum(dim=1)
            .clamp_min(1e-12)
        )

        values.append(
            (error / energy).detach().cpu()
        )

    return torch.cat(
        values,
        dim=0,
    )


def compute_pairwise_records(
    models,
    x_test,
    y_test,
    args,
    metadata,
):
    """
    Compare every model against:
      1. ALePOM
      2. APGnc+-ALePOM

    The second comparison directly quantifies the contribution
    of the LSTM to the APGnc+-enhanced baseline.
    """
    sample_values = {}

    for method in METHODS:
        sample_values[method] = (
            sample_nmse_values(
                models[method],
                x_test,
                y_test,
                args.batch_size,
            )
        )

    records = []

    reference_methods = (
        "alepom",
        "apgnc_plus_alepom",
    )

    for reference in reference_methods:
        baseline = sample_values[reference]

        for method in METHODS:
            if method == reference:
                continue

            candidate = sample_values[method]

            tie = torch.isclose(
                candidate,
                baseline,
                rtol=1e-6,
                atol=1e-12,
            )

            win = (
                candidate < baseline
            ) & (~tie)

            loss = (
                candidate > baseline
            ) & (~tie)

            ratio = (
                candidate
                / baseline.clamp_min(1e-20)
            )

            record = dict(metadata)

            record.update(
                {
                    "reference_method": reference,
                    "candidate_method": method,
                    "win_rate": float(
                        win.float().mean()
                    ),
                    "tie_rate": float(
                        tie.float().mean()
                    ),
                    "loss_rate": float(
                        loss.float().mean()
                    ),
                    "median_nmse_ratio": float(
                        torch.median(ratio)
                    ),
                    "mean_nmse_ratio": float(
                        ratio.mean()
                    ),
                }
            )

            records.append(record)

    return records


def save_model_checkpoint(
    model,
    history,
    args,
    output_dir,
    seed,
):
    path = (
        output_dir
        / "checkpoints"
        / (
            f"{model.method}"
            f"_m{args.m}"
            f"_seed{seed}.pt"
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "method": model.method,
            "model_state": {
                key: value.detach().cpu()
                for key, value
                in model.state_dict().items()
            },
            "history": history,
            "seed": seed,
            "m": args.m,
            "n": args.n,
            "config": vars(args),
        },
        path,
    )


def run_seed_experiment(
    args,
    seed,
    device,
    output_dir,
    conditions,
):
    print("\n" + "=" * 96)
    print(
        f"Paper experiment: seed={seed}, "
        f"M={args.m}, N={args.n}, "
        f"M/N={args.m / args.n:.3f}"
    )
    print("=" * 96)

    core.set_seed(seed)

    A = core.make_sensing_matrix(
        args.m,
        args.n,
        seed,
        device,
    )

    B = core.pretrain_B(
        A,
        args,
        output_dir,
        seed,
        args.m,
    ).to(device)

    B = core.normalize_B(
        B,
        A,
    ).detach()

    B.requires_grad_(False)

    diagonal_error = float(
        (
            torch.diagonal(B @ A)
            - 1.0
        ).abs().max()
    )

    print(
        "max |diag(BA)-1| = "
        f"{diagonal_error:.6e}"
    )

    x_train, y_train = core.make_dataset(
        A=A,
        count=args.n_train,
        nonzero_prob=args.train_nonzero_prob,
        seed=seed + 1,
        device=device,
        amplitude="gaussian",
        snr_db=math.inf,
    )

    x_val, y_val = core.make_dataset(
        A=A,
        count=args.n_val,
        nonzero_prob=args.train_nonzero_prob,
        seed=seed + 2,
        device=device,
        amplitude="gaussian",
        snr_db=math.inf,
    )

    models = {}
    histories = {}

    # Immutable snapshot shared by all four ablation models.
    # It is captured before the first model is trained.
    initial_base_parameters = None

    training_records = []

    for method in METHODS:
        print("\n" + "-" * 80)
        print(
            f"Training {METHOD_CONFIGS[method]['label']}"
        )
        print("-" * 80)

        # Reset model initialization for every ablation method.
        core.set_seed(seed)

        model = PaperRecoveryModel(
            A=A,
            B=B,
            args=args,
            method=method,
        ).to(device)

        if initial_base_parameters is None:
            # Capture the common initialization before the first
            # model is trained.
            initial_base_parameters = (
                capture_base_parameters(model)
            )
        else:
            synchronize_base_parameters(
                target_model=model,
                initial_base_parameters=(
                    initial_base_parameters
                ),
            )

        verify_base_parameters(
            model=model,
            initial_base_parameters=(
                initial_base_parameters
            ),
        )

        print(
            "[initialization] "
            f"{METHOD_CONFIGS[method]['label']}: "
            "common base parameters verified"
        )

        # Reset the RNG after model construction so that all
        # ablation models use the same minibatch order.
        core.set_seed(seed)

        history = core.train_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            args=args,
            training_seed=seed,
        )

        models[method] = model
        histories[method] = history

        save_model_checkpoint(
            model=model,
            history=history,
            args=args,
            output_dir=output_dir,
            seed=seed,
        )

        for record in history:
            training_records.append(
                {
                    "seed": seed,
                    "m": args.m,
                    "method": method,
                    **record,
                }
            )

    metric_records = []
    layer_records = []
    dynamics_records = []
    pairwise_records = []    
    
    

    for condition_index, condition in enumerate(
        conditions
    ):
        test_seed = (
            seed
            + 100000
            + 1000 * condition_index
            + args.m
        )

        x_test, y_test = core.make_dataset(
            A=A,
            count=args.n_test,
            nonzero_prob=condition[
                "probability"
            ],
            seed=test_seed,
            device=device,
            amplitude=condition[
                "amplitude"
            ],
            snr_db=condition[
                "snr_db"
            ],
        )

        result_by_method = {}

        for method in METHODS:
            result = core.evaluate_model(
                model=models[method],
                x_test=x_test,
                y_test=y_test,
                args=args,
            )

            result_by_method[method] = result

            metric_records.append(
                {
                    "seed": seed,
                    "m": args.m,
                    "n": args.n,
                    "undersampling_ratio": (
                        args.m / args.n
                    ),
                    "family": condition[
                        "family"
                    ],
                    "condition": condition[
                        "label"
                    ],
                    "plot_label": condition[
                        "plot_label"
                    ],
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
                    "method_label": (
                        METHOD_CONFIGS[
                            method
                        ]["label"]
                    ),
                    "ser_db": result[
                        "ser_db"
                    ],
                    "nmse": result[
                        "nmse"
                    ],
                    "nmse_db": result[
                        "nmse_db"
                    ],
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
                    "milliseconds_per_sample": (
                        result[
                            "milliseconds_per_sample"
                        ]
                    ),
                    "trainable_parameters": (
                        result[
                            "trainable_parameters"
                        ]
                    ),
                }
            )

            for layer, ser_db in enumerate(
                result["layer_ser"],
                start=1,
            ):
                layer_records.append(
                    {
                        "seed": seed,
                        "m": args.m,
                        "family": condition[
                            "family"
                        ],
                        "condition": condition[
                            "label"
                        ],
                        "method": method,
                        "layer": layer,
                        "ser_db": ser_db,
                    }
                )

            for layer in range(args.layers):
                dynamics_records.append(
                    {
                        "seed": seed,
                        "m": args.m,
                        "family": condition[
                            "family"
                        ],
                        "condition": condition[
                            "label"
                        ],
                        "method": method,
                        "layer": layer + 1,
                        "extrapolation_rate": (
                            result[
                                "extrapolation_rate_by_layer"
                            ][layer]
                        ),
                        "mean_beta": (
                            result[
                                "mean_beta_by_layer"
                            ][layer]
                        ),
                        "mean_theta_offset": (
                            result[
                                "mean_offsets_by_layer"
                            ][layer][0]
                        ),
                        "mean_q_offset": (
                            result[
                                "mean_offsets_by_layer"
                            ][layer][1]
                        ),
                        "mean_gamma_offset": (
                            result[
                                "mean_offsets_by_layer"
                            ][layer][2]
                        ),
                        "mean_abs_theta_offset": (
                            result[
                                "mean_abs_offsets_by_layer"
                            ][layer][0]
                        ),
                        "mean_abs_q_offset": (
                            result[
                                "mean_abs_offsets_by_layer"
                            ][layer][1]
                        ),
                        "mean_abs_gamma_offset": (
                            result[
                                "mean_abs_offsets_by_layer"
                            ][layer][2]
                        ),
                    }
                )

        pairwise_metadata = {
            "seed": seed,
            "m": args.m,
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
        }

        pairwise_records.extend(
            compute_pairwise_records(
                models=models,
                x_test=x_test,
                y_test=y_test,
                args=args,
                metadata=pairwise_metadata,
            )
        )

        base_ser = result_by_method[
            "alepom"
        ]["ser_db"]

        momentum_ser = result_by_method[
            "apgnc_plus_alepom"
        ]["ser_db"]

        proposed_ser = result_by_method[
            "proposed"
        ]["ser_db"]

        print(
            f"[test] seed={seed}, "
            f"condition={condition['label']:>18s}, "
            f"ALePOM={base_ser:8.3f} dB, "
            f"APGnc+={momentum_ser:8.3f} dB, "
            f"Proposed={proposed_ser:8.3f} dB, "
            f"momentum gain="
            f"{momentum_ser - base_ser:+7.3f} dB, "
            f"LSTM gain="
            f"{proposed_ser - momentum_ser:+7.3f} dB"
        )

    del models
    del initial_base_parameters
    del x_train
    del y_train
    del x_val
    del y_val

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "metrics": metric_records,
        "layers": layer_records,
        "dynamics": dynamics_records,
        "pairwise": pairwise_records,
        "training": training_records,
    }


def write_csv(path, records):
    if not records:
        print(
            f"Warning: no records for {path}"
        )
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(records[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(records)


def mean_std_ci(values):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(array.mean())

    if len(array) > 1:
        std = float(
            array.std(ddof=1)
        )

        ci95 = float(
            1.96
            * std
            / math.sqrt(len(array))
        )
    else:
        std = 0.0
        ci95 = 0.0

    return mean, std, ci95


def aggregate_metric_records(records):
    grouped = defaultdict(list)

    for record in records:
        key = (
            record["m"],
            record["family"],
            record["condition"],
            record["plot_label"],
            record["probability"],
            record["snr_db"],
            record["amplitude"],
            record["method"],
            record["method_label"],
        )

        grouped[key].append(record)

    aggregate = []

    metric_names = (
        "ser_db",
        "nmse",
        "nmse_db",
        "support_f1",
        "exact_support_rate",
        "success_rate",
        "runtime_seconds",
        "milliseconds_per_sample",
        "trainable_parameters",
    )

    for key, rows in grouped.items():
        (
            m,
            family,
            condition,
            plot_label,
            probability,
            snr_db,
            amplitude,
            method,
            method_label,
        ) = key

        output = {
            "m": m,
            "family": family,
            "condition": condition,
            "plot_label": plot_label,
            "probability": probability,
            "snr_db": snr_db,
            "amplitude": amplitude,
            "method": method,
            "method_label": method_label,
            "runs": len(rows),
        }

        for metric in metric_names:
            mean, std, ci95 = mean_std_ci(
                [
                    float(row[metric])
                    for row in rows
                ]
            )

            output[
                f"{metric}_mean"
            ] = mean

            output[
                f"{metric}_std"
            ] = std

            output[
                f"{metric}_ci95"
            ] = ci95

        aggregate.append(output)

    family_order = {
        "in_distribution": 0,
        "sparsity": 1,
        "snr": 2,
        "amplitude": 3,
    }

    method_order = {
        method: index
        for index, method in enumerate(
            METHODS
        )
    }

    aggregate.sort(
        key=lambda row: (
            family_order.get(
                row["family"],
                99,
            ),
            str(row["condition"]),
            method_order[row["method"]],
        )
    )

    return aggregate


def aggregate_pairwise_records(records):
    grouped = defaultdict(list)

    for record in records:
        key = (
            record["m"],
            record["family"],
            record["condition"],
            record["reference_method"],
            record["candidate_method"],
        )

        grouped[key].append(record)

    aggregate = []

    metrics = (
        "win_rate",
        "tie_rate",
        "loss_rate",
        "median_nmse_ratio",
        "mean_nmse_ratio",
    )

    for key, rows in grouped.items():
        (
            m,
            family,
            condition,
            reference,
            candidate,
        ) = key

        output = {
            "m": m,
            "family": family,
            "condition": condition,
            "reference_method": reference,
            "candidate_method": candidate,
            "runs": len(rows),
        }

        for metric in metrics:
            mean, std, ci95 = mean_std_ci(
                [
                    float(row[metric])
                    for row in rows
                ]
            )

            output[
                f"{metric}_mean"
            ] = mean

            output[
                f"{metric}_std"
            ] = std

            output[
                f"{metric}_ci95"
            ] = ci95

        aggregate.append(output)

    aggregate.sort(
        key=lambda row: (
            row["family"],
            row["condition"],
            row["reference_method"],
            row["candidate_method"],
        )
    )

    return aggregate


def aggregate_layer_records(records):
    grouped = defaultdict(list)

    for record in records:
        key = (
            record["family"],
            record["condition"],
            record["method"],
            int(record["layer"]),
        )

        grouped[key].append(
            float(record["ser_db"])
        )

    aggregate = []

    for key, values in grouped.items():
        family, condition, method, layer = key

        mean, std, ci95 = mean_std_ci(
            values
        )

        aggregate.append(
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

    aggregate.sort(
        key=lambda row: (
            row["family"],
            row["condition"],
            row["method"],
            row["layer"],
        )
    )

    return aggregate


def aggregate_dynamics_records(records):
    grouped = defaultdict(list)

    for record in records:
        key = (
            record["family"],
            record["condition"],
            record["method"],
            int(record["layer"]),
        )

        grouped[key].append(record)

    aggregate = []

    metrics = (
        "extrapolation_rate",
        "mean_beta",
        "mean_theta_offset",
        "mean_q_offset",
        "mean_gamma_offset",
        "mean_abs_theta_offset",
        "mean_abs_q_offset",
        "mean_abs_gamma_offset",
    )

    for key, rows in grouped.items():
        family, condition, method, layer = key

        output = {
            "family": family,
            "condition": condition,
            "method": method,
            "layer": layer,
            "runs": len(rows),
        }

        for metric in metrics:
            mean, std, ci95 = mean_std_ci(
                [
                    float(row[metric])
                    for row in rows
                ]
            )

            output[
                f"{metric}_mean"
            ] = mean

            output[
                f"{metric}_std"
            ] = std

            output[
                f"{metric}_ci95"
            ] = ci95

        aggregate.append(output)

    aggregate.sort(
        key=lambda row: (
            row["family"],
            row["condition"],
            row["method"],
            row["layer"],
        )
    )

    return aggregate


def aggregate_training_records(records):
    grouped = defaultdict(list)

    for record in records:
        key = (
            record["method"],
            int(record["epoch"]),
        )

        grouped[key].append(record)

    aggregate = []

    metrics = (
        "train_final_nmse",
        "train_layer_nmse",
        "adaptation_regularizer",
        "train_objective",
        "validation_nmse",
        "best_validation_nmse",
    )

    for key, rows in grouped.items():
        method, epoch = key

        output = {
            "method": method,
            "epoch": epoch,
            "runs": len(rows),
        }

        for metric in metrics:
            mean, std, ci95 = mean_std_ci(
                [
                    float(row[metric])
                    for row in rows
                ]
            )

            output[
                f"{metric}_mean"
            ] = mean

            output[
                f"{metric}_std"
            ] = std

            output[
                f"{metric}_ci95"
            ] = ci95

        aggregate.append(output)

    aggregate.sort(
        key=lambda row: (
            row["method"],
            row["epoch"],
        )
    )

    return aggregate


def save_figure(
    figure,
    output_dir,
    filename,
):
    figure_dir = output_dir / "figures"

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        figure_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        figure_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def get_aggregate_row(
    aggregate,
    family,
    condition,
    method,
):
    for row in aggregate:
        if (
            row["family"] == family
            and row["condition"] == condition
            and row["method"] == method
        ):
            return row

    return None


def plot_condition_family(
    aggregate,
    conditions,
    family,
    metric,
    ylabel,
    output_dir,
    filename,
):
    family_conditions = [
        condition
        for condition in conditions
        if condition["family"] == family
    ]

    if not family_conditions:
        return

    figure, axis = plt.subplots(
        figsize=(8.2, 5.3)
    )

    x = np.arange(
        len(family_conditions)
    )

    for method in METHODS:
        means = []
        ci95s = []

        for condition in family_conditions:
            row = get_aggregate_row(
                aggregate,
                family,
                condition["label"],
                method,
            )

            if row is None:
                means.append(np.nan)
                ci95s.append(0.0)
            else:
                means.append(
                    row[f"{metric}_mean"]
                )

                ci95s.append(
                    row[f"{metric}_ci95"]
                )

        config = METHOD_CONFIGS[method]

        axis.errorbar(
            x,
            means,
            yerr=ci95s,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linewidth=2.0,
            markersize=6,
            capsize=3,
        )

    axis.set_xticks(x)

    axis.set_xticklabels(
        [
            condition["plot_label"]
            for condition in family_conditions
        ]
    )

    if family == "sparsity":
        axis.set_xlabel(
            "Nonzero probability"
        )
    elif family == "snr":
        axis.set_xlabel(
            "Measurement SNR (dB)"
        )
    else:
        axis.set_xlabel(
            "Nonzero-amplitude distribution"
        )

    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(
        frameon=True,
        fontsize=9,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        filename,
    )


def plot_in_distribution_bars(
    aggregate,
    output_dir,
):
    rows = {
        method: get_aggregate_row(
            aggregate,
            "in_distribution",
            "train_distribution",
            method,
        )
        for method in METHODS
    }

    if any(
        row is None
        for row in rows.values()
    ):
        return

    labels = [
        METHOD_CONFIGS[method]["label"]
        for method in METHODS
    ]

    colors = [
        METHOD_CONFIGS[method]["color"]
        for method in METHODS
    ]

    means = [
        rows[method]["ser_db_mean"]
        for method in METHODS
    ]

    errors = [
        rows[method]["ser_db_ci95"]
        for method in METHODS
    ]

    figure, axis = plt.subplots(
        figsize=(8.2, 5.2)
    )

    bars = axis.bar(
        np.arange(len(METHODS)),
        means,
        yerr=errors,
        color=colors,
        capsize=4,
        alpha=0.90,
    )

    axis.set_xticks(
        np.arange(len(METHODS))
    )

    axis.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )

    axis.set_ylabel("SER (dB)")
    axis.set_title(
        "In-distribution recovery: "
        "M=200, N=500"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        means,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "in_distribution_ser",
    )


def plot_layer_ser(
    layer_aggregate,
    output_dir,
):
    rows = [
        row
        for row in layer_aggregate
        if (
            row["family"]
            == "in_distribution"
            and row["condition"]
            == "train_distribution"
        )
    ]

    if not rows:
        return

    figure, axis = plt.subplots(
        figsize=(8.2, 5.3)
    )

    for method in METHODS:
        method_rows = sorted(
            [
                row
                for row in rows
                if row["method"] == method
            ],
            key=lambda row: row["layer"],
        )

        if not method_rows:
            continue

        config = METHOD_CONFIGS[method]

        layers = [
            row["layer"]
            for row in method_rows
        ]

        means = [
            row["ser_db_mean"]
            for row in method_rows
        ]

        ci95s = [
            row["ser_db_ci95"]
            for row in method_rows
        ]

        axis.plot(
            layers,
            means,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linewidth=2.0,
            markersize=4,
        )

        lower = (
            np.asarray(means)
            - np.asarray(ci95s)
        )

        upper = (
            np.asarray(means)
            + np.asarray(ci95s)
        )

        axis.fill_between(
            layers,
            lower,
            upper,
            color=config["color"],
            alpha=0.12,
        )

    axis.set_xlabel("Unfolded layer")
    axis.set_ylabel("SER (dB)")
    axis.set_xticks(
        range(
            1,
            max(row["layer"] for row in rows)
            + 1,
        )
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "layer_wise_ser",
    )


def plot_training_curves(
    training_aggregate,
    output_dir,
):
    if not training_aggregate:
        return

    figure, axis = plt.subplots(
        figsize=(8.2, 5.3)
    )

    for method in METHODS:
        rows = sorted(
            [
                row
                for row in training_aggregate
                if row["method"] == method
            ],
            key=lambda row: row["epoch"],
        )

        if not rows:
            continue

        config = METHOD_CONFIGS[method]

        epochs = [
            row["epoch"]
            for row in rows
        ]

        validation = [
            row["validation_nmse_mean"]
            for row in rows
        ]

        axis.semilogy(
            epochs,
            validation,
            label=config["label"],
            color=config["color"],
            linewidth=2.0,
        )

    axis.set_xlabel("Training epoch")
    axis.set_ylabel("Validation NMSE")
    axis.grid(
        alpha=0.25,
        which="both",
    )

    axis.legend(fontsize=9)
    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "training_validation_nmse",
    )


def plot_momentum_dynamics(
    dynamics_aggregate,
    output_dir,
):
    rows = [
        row
        for row in dynamics_aggregate
        if (
            row["family"]
            == "in_distribution"
            and row["condition"]
            == "train_distribution"
            and METHOD_CONFIGS[
                row["method"]
            ]["use_momentum"]
        )
    ]

    if not rows:
        return

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.6),
    )

    for method in (
        "apgnc_plus_alepom",
        "proposed",
    ):
        method_rows = sorted(
            [
                row
                for row in rows
                if row["method"] == method
            ],
            key=lambda row: row["layer"],
        )

        if not method_rows:
            continue

        config = METHOD_CONFIGS[method]

        layers = [
            row["layer"]
            for row in method_rows
        ]

        acceptance = [
            row[
                "extrapolation_rate_mean"
            ]
            for row in method_rows
        ]

        beta = [
            row["mean_beta_mean"]
            for row in method_rows
        ]

        axes[0].plot(
            layers,
            acceptance,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linewidth=2.0,
        )

        axes[1].plot(
            layers,
            beta,
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            linewidth=2.0,
        )

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel(
        "Extrapolation acceptance rate"
    )

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(
        r"Mean momentum $\beta_k$"
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "momentum_dynamics",
    )


def plot_lstm_offsets(
    dynamics_aggregate,
    output_dir,
):
    rows = [
        row
        for row in dynamics_aggregate
        if (
            row["family"]
            == "in_distribution"
            and row["condition"]
            == "train_distribution"
            and METHOD_CONFIGS[
                row["method"]
            ]["use_lstm"]
        )
    ]

    if not rows:
        return

    offset_metrics = (
        (
            "mean_abs_theta_offset_mean",
            r"$|\delta_{\theta}|$",
            "#9467bd",
        ),
        (
            "mean_abs_q_offset_mean",
            r"$|\delta_q|$",
            "#8c564b",
        ),
        (
            "mean_abs_gamma_offset_mean",
            r"$|\delta_{\gamma}|$",
            "#17becf",
        ),
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.6),
        sharey=True,
    )

    lstm_methods = (
        "lstm_alepom",
        "proposed",
    )

    for axis, method in zip(
        axes,
        lstm_methods,
    ):
        method_rows = sorted(
            [
                row
                for row in rows
                if row["method"] == method
            ],
            key=lambda row: row["layer"],
        )

        if not method_rows:
            continue

        layers = [
            row["layer"]
            for row in method_rows
        ]

        for metric, label, color in (
            offset_metrics
        ):
            axis.plot(
                layers,
                [
                    row[metric]
                    for row in method_rows
                ],
                label=label,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
            )

        axis.set_title(
            METHOD_CONFIGS[
                method
            ]["label"]
        )

        axis.set_xlabel("Layer")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    axes[0].set_ylabel(
        "Mean absolute LSTM offset"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "lstm_offset_dynamics",
    )


def plot_ablation_ser(
    aggregate,
    output_dir,
):
    rows = []

    for method in METHODS:
        row = get_aggregate_row(
            aggregate,
            "in_distribution",
            "train_distribution",
            method,
        )

        if row is not None:
            rows.append(row)

    if len(rows) != len(METHODS):
        return

    values = [
        row["ser_db_mean"]
        for row in rows
    ]

    errors = [
        row["ser_db_ci95"]
        for row in rows
    ]

    labels = [
        METHOD_CONFIGS[
            row["method"]
        ]["label"]
        for row in rows
    ]

    colors = [
        METHOD_CONFIGS[
            row["method"]
        ]["color"]
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(8.2, 5.2)
    )

    x = np.arange(len(rows))

    bars = axis.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=colors,
        alpha=0.9,
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )

    axis.set_ylabel("SER (dB)")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "component_ablation_ser",
    )


def make_latex_tables(
    aggregate,
    pairwise_aggregate,
    output_dir,
):
    """
    Generate a LaTeX file that can be copied directly into
    the paper after the final experiments are complete.
    """
    rows = {
        method: get_aggregate_row(
            aggregate,
            "in_distribution",
            "train_distribution",
            method,
        )
        for method in METHODS
    }

    path = output_dir / "paper_tables.tex"

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "% Automatically generated paper tables.\n"
        )

        file.write(
            "% Review all values before submission.\n\n"
        )

        file.write(
            "\\begin{table*}[t]\n"
        )

        file.write(
            "\\centering\n"
        )

        file.write(
            "\\caption{In-distribution sparse recovery "
            "performance for $M=200$ and $N=500$.}\n"
        )

        file.write(
            "\\label{tab:generated_in_distribution}\n"
        )

        file.write(
            "\\begin{tabular}{lcccccc}\n"
        )

        file.write(
            "\\toprule\n"
        )

        file.write(
            "Method & SER (dB) & NMSE & Support F1 & "
            "Exact Support & Success Rate & ms/Sample \\\\\n"
        )

        file.write(
            "\\midrule\n"
        )

        for method in METHODS:
            row = rows[method]

            if row is None:
                continue

            label = (
                METHOD_CONFIGS[
                    method
                ]["label"]
                .replace("+", "{+}")
            )

            file.write(
                f"{label} & "
                f"${row['ser_db_mean']:.3f}"
                f"\\pm"
                f"{row['ser_db_std']:.3f}$ & "
                f"${row['nmse_mean']:.3e}"
                f"\\pm"
                f"{row['nmse_std']:.3e}$ & "
                f"${row['support_f1_mean']:.4f}"
                f"\\pm"
                f"{row['support_f1_std']:.4f}$ & "
                f"${row['exact_support_rate_mean']:.4f}"
                f"\\pm"
                f"{row['exact_support_rate_std']:.4f}$ & "
                f"${row['success_rate_mean']:.4f}"
                f"\\pm"
                f"{row['success_rate_std']:.4f}$ & "
                f"${row['milliseconds_per_sample_mean']:.4f}"
                f"\\pm"
                f"{row['milliseconds_per_sample_std']:.4f}$"
                " \\\\\n"
            )

        file.write(
            "\\bottomrule\n"
        )

        file.write(
            "\\end{tabular}\n"
        )

        file.write(
            "\\end{table*}\n\n"
        )

        # Automatically provide the two principal gains.
        alepom = rows.get("alepom")
        momentum = rows.get(
            "apgnc_plus_alepom"
        )
        proposed = rows.get("proposed")

        if (
            alepom is not None
            and momentum is not None
            and proposed is not None
        ):
            momentum_gain = (
                momentum["ser_db_mean"]
                - alepom["ser_db_mean"]
            )

            lstm_gain = (
                proposed["ser_db_mean"]
                - momentum["ser_db_mean"]
            )

            total_gain = (
                proposed["ser_db_mean"]
                - alepom["ser_db_mean"]
            )

            file.write(
                "% Automatically calculated SER gains:\n"
            )

            file.write(
                f"% APGnc+ over ALePOM: "
                f"{momentum_gain:+.4f} dB\n"
            )

            file.write(
                f"% Proposed over APGnc+-ALePOM: "
                f"{lstm_gain:+.4f} dB\n"
            )

            file.write(
                f"% Proposed over ALePOM: "
                f"{total_gain:+.4f} dB\n"
            )

        target_pairwise = [
            row
            for row in pairwise_aggregate
            if (
                row["family"]
                == "in_distribution"
                and row["condition"]
                == "train_distribution"
                and row[
                    "reference_method"
                ]
                == "apgnc_plus_alepom"
                and row[
                    "candidate_method"
                ]
                == "proposed"
            )
        ]

        if target_pairwise:
            row = target_pairwise[0]

            file.write(
                "\n% Proposed versus APGnc+-ALePOM:\n"
            )

            file.write(
                f"% Win rate: "
                f"{row['win_rate_mean']:.6f}\n"
            )

            file.write(
                f"% Median NMSE ratio: "
                f"{row['median_nmse_ratio_mean']:.6f}\n"
            )


def generate_summary(
    aggregate,
    pairwise_aggregate,
    output_dir,
):
    rows = {
        method: get_aggregate_row(
            aggregate,
            "in_distribution",
            "train_distribution",
            method,
        )
        for method in METHODS
    }

    summary = {
        "dimensions": {
            "M": 200,
            "N": 500,
            "undersampling_ratio": 0.4,
        },
        "in_distribution": {},
    }

    for method, row in rows.items():
        if row is None:
            continue

        summary[
            "in_distribution"
        ][method] = row

    if (
        rows.get("alepom") is not None
        and rows.get(
            "apgnc_plus_alepom"
        ) is not None
        and rows.get("proposed") is not None
    ):
        summary["ser_gains_db"] = {
            "apgnc_plus_over_alepom": (
                rows[
                    "apgnc_plus_alepom"
                ]["ser_db_mean"]
                - rows[
                    "alepom"
                ]["ser_db_mean"]
            ),
            "proposed_over_apgnc_plus": (
                rows[
                    "proposed"
                ]["ser_db_mean"]
                - rows[
                    "apgnc_plus_alepom"
                ]["ser_db_mean"]
            ),
            "proposed_over_alepom": (
                rows[
                    "proposed"
                ]["ser_db_mean"]
                - rows[
                    "alepom"
                ]["ser_db_mean"]
            ),
        }

    summary["paired_comparisons"] = (
        pairwise_aggregate
    )

    with open(
        output_dir / "paper_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("MAIN IN-DISTRIBUTION RESULTS")
    print("=" * 90)

    for method in METHODS:
        row = rows.get(method)

        if row is None:
            continue

        print(
            f"{METHOD_CONFIGS[method]['label']:24s} "
            f"SER={row['ser_db_mean']:10.4f} "
            f"+/- {row['ser_db_std']:.4f} dB, "
            f"NMSE={row['nmse_mean']:.4e}, "
            f"F1={row['support_f1_mean']:.4f}, "
            f"success={row['success_rate_mean']:.4f}, "
            f"time={row['milliseconds_per_sample_mean']:.4f} "
            f"ms/sample"
        )

    if "ser_gains_db" in summary:
        gains = summary["ser_gains_db"]

        print("\nPrincipal SER gains:")

        print(
            "  APGnc+ over ALePOM: "
            f"{gains['apgnc_plus_over_alepom']:+.4f} dB"
        )

        print(
            "  Proposed over APGnc+-ALePOM: "
            f"{gains['proposed_over_apgnc_plus']:+.4f} dB"
        )

        print(
            "  Proposed over ALePOM: "
            f"{gains['proposed_over_alepom']:+.4f} dB"
        )


def main():
    args = parse_args()
    apply_quick_mode(args)

    if args.m != 200:
        raise ValueError(
            "This paper experiment script is designed "
            "for M=200. Please use --m 200."
        )

    if args.n != 500:
        print(
            "Warning: the paper setting uses N=500, "
            f"but N={args.n} was supplied."
        )

    if args.m >= args.n:
        raise ValueError(
            "M must be less than N."
        )

    seeds = parse_number_list(
        args.seeds,
        int,
    )

    if not seeds:
        raise ValueError(
            "At least one random seed is required."
        )

    device = core.get_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )

    print(
        f"Dimensions: M={args.m}, "
        f"N={args.n}, "
        f"M/N={args.m / args.n:.3f}"
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

    conditions = build_conditions(
        args
    )

    with open(
        output_dir / "conditions.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            conditions,
            file,
            indent=2,
        )

    all_metrics = []
    all_layers = []
    all_dynamics = []
    all_pairwise = []
    all_training = []

    start_time = time.perf_counter()

    for seed in seeds:
        result = run_seed_experiment(
            args=args,
            seed=seed,
            device=device,
            output_dir=output_dir,
            conditions=conditions,
        )

        all_metrics.extend(
            result["metrics"]
        )

        all_layers.extend(
            result["layers"]
        )

        all_dynamics.extend(
            result["dynamics"]
        )

        all_pairwise.extend(
            result["pairwise"]
        )

        all_training.extend(
            result["training"]
        )

        # Save intermediate records after each seed so that
        # completed runs are not lost if a later run stops.
        write_csv(
            output_dir / "all_runs.csv",
            all_metrics,
        )

        write_csv(
            output_dir / "layer_results.csv",
            all_layers,
        )

        write_csv(
            output_dir / "dynamics.csv",
            all_dynamics,
        )

        write_csv(
            output_dir / "pairwise_results.csv",
            all_pairwise,
        )

        write_csv(
            output_dir / "training_history.csv",
            all_training,
        )

    aggregate = aggregate_metric_records(
        all_metrics
    )

    pairwise_aggregate = (
        aggregate_pairwise_records(
            all_pairwise
        )
    )

    layer_aggregate = (
        aggregate_layer_records(
            all_layers
        )
    )

    dynamics_aggregate = (
        aggregate_dynamics_records(
            all_dynamics
        )
    )

    training_aggregate = (
        aggregate_training_records(
            all_training
        )
    )

    write_csv(
        output_dir / "aggregate.csv",
        aggregate,
    )

    write_csv(
        output_dir / "pairwise_aggregate.csv",
        pairwise_aggregate,
    )

    write_csv(
        output_dir / "layer_aggregate.csv",
        layer_aggregate,
    )

    write_csv(
        output_dir / "dynamics_aggregate.csv",
        dynamics_aggregate,
    )

    write_csv(
        output_dir / "training_aggregate.csv",
        training_aggregate,
    )

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

    # Main paper figures.
    plot_in_distribution_bars(
        aggregate,
        output_dir,
    )

    plot_ablation_ser(
        aggregate,
        output_dir,
    )

    plot_layer_ser(
        layer_aggregate,
        output_dir,
    )

    plot_training_curves(
        training_aggregate,
        output_dir,
    )

    plot_condition_family(
        aggregate=aggregate,
        conditions=conditions,
        family="sparsity",
        metric="ser_db",
        ylabel="SER (dB)",
        output_dir=output_dir,
        filename="ser_sparsity_m200",
    )

    plot_condition_family(
        aggregate=aggregate,
        conditions=conditions,
        family="snr",
        metric="ser_db",
        ylabel="SER (dB)",
        output_dir=output_dir,
        filename="ser_snr_m200",
    )

    plot_condition_family(
        aggregate=aggregate,
        conditions=conditions,
        family="amplitude",
        metric="ser_db",
        ylabel="SER (dB)",
        output_dir=output_dir,
        filename="ser_amplitude_m200",
    )

    plot_condition_family(
        aggregate=aggregate,
        conditions=conditions,
        family="sparsity",
        metric="support_f1",
        ylabel="Support F1",
        output_dir=output_dir,
        filename="support_f1_sparsity_m200",
    )

    plot_condition_family(
        aggregate=aggregate,
        conditions=conditions,
        family="snr",
        metric="support_f1",
        ylabel="Support F1",
        output_dir=output_dir,
        filename="support_f1_snr_m200",
    )

    plot_momentum_dynamics(
        dynamics_aggregate,
        output_dir,
    )

    plot_lstm_offsets(
        dynamics_aggregate,
        output_dir,
    )

    make_latex_tables(
        aggregate=aggregate,
        pairwise_aggregate=pairwise_aggregate,
        output_dir=output_dir,
    )

    generate_summary(
        aggregate=aggregate,
        pairwise_aggregate=pairwise_aggregate,
        output_dir=output_dir,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    with open(
        output_dir / "total_runtime.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Total runtime: {elapsed:.3f} seconds\n"
        )

        file.write(
            f"Total runtime: {elapsed / 3600.0:.3f} hours\n"
        )

    print("\n" + "=" * 96)
    print("ALL PAPER EXPERIMENTS COMPLETED")
    print("=" * 96)

    print(
        f"Total runtime: "
        f"{elapsed / 3600.0:.3f} hours"
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

    print("\nMain data files:")

    for name in (
        "aggregate.csv",
        "all_runs.csv",
        "pairwise_aggregate.csv",
        "layer_aggregate.csv",
        "dynamics_aggregate.csv",
        "training_aggregate.csv",
        "paper_summary.json",
        "paper_tables.tex",
    ):
        print(f"  {name}")

    print("\nMain paper figures:")

    for name in (
        "in_distribution_ser.png/pdf",
        "component_ablation_ser.png/pdf",
        "layer_wise_ser.png/pdf",
        "ser_sparsity_m200.png/pdf",
        "ser_snr_m200.png/pdf",
        "ser_amplitude_m200.png/pdf",
        "support_f1_sparsity_m200.png/pdf",
        "support_f1_snr_m200.png/pdf",
        "momentum_dynamics.png/pdf",
        "lstm_offset_dynamics.png/pdf",
        "training_validation_nmse.png/pdf",
    ):
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()