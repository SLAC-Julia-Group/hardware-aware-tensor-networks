# QiML — Quantum-Inspired Machine Learning for Anomaly Detection

Tensor network based anomaly detection targeting the 40 MHz LHC trigger challenge.

Reference: [arXiv:2603.26604](https://arxiv.org/pdf/2603.26604.pdf)

---

## Structure

```
QiML/
├── tn4ml/          # Core tensor network ML framework (submodule)
├── cascaded_tn/    # Cascaded SMPO encoder package
└── Training/       # Training notebooks and utilities
└── HLS/            # The C++ HLS code
```

### `tn4ml` - https://github.com/bsc-quantic/tn4ml

General-purpose tensor network ML library providing MPS, MPO, and SMPO primitives, embeddings, loss functions, and training infrastructure (JAX + Quimb).
Reference: [arXiv:2502.13090](https://arxiv.org/pdf/2502.13090.pdf)

### `cascaded_tn`

Builds multi-layer cascaded encoders from Spaced Matrix Product Operators (SMPOs). Each layer compresses an input MPS to fewer sites; layers are chained to form a deep encoder. Includes a fixed-point quantization testbed for hardware deployment.

### `Training/`

Notebooks for training and evaluation on the 40 MHz dataset:

- `Training_SMPO_19_1` — single-layer SMPO, 19→1
- `Training_CSMPO_19_2_1` — cascaded SMPO, 19→2→1
- `Training_CSMPO_19_7_1` — cascaded SMPO, 19→7→1

---

## Installation

See `Install.md` for the creation of the python environment.
Required:

- cuda libraries for GPU access
- tn4ml (and related dependencies)
- torch (for lazy loading)
