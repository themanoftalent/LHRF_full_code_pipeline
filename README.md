# LHRF: Lightweight Hybrid Representation Learning Framework

MATLAB research-code package for colorectal lesion classification, accompanying the LHRF experimental study.

## Framework Overview

LHRF is a modular, lightweight representation-learning framework for colonoscopy image classification. Rather than relying on computationally intensive end-to-end deep learning backbones, LHRF progressively refines a compact local visual representation.

The pipeline integrates:
1. **SURF & Bag-of-Visual-Words (BoVW):** Extracts and encodes local visual descriptors into fixed-length histograms.
2. **Stacked Autoencoder (SAE):** Compresses the standardized BoVW features into a 50-dimensional nonlinear latent representation.
3. **Triplet Embedding (TE):** Refines the latent geometry to promote intra-class compactness and inter-class separation.
4. **Hierarchical Clustering (HC):** (Optional) Augments the features with structural latent-space organization.
5. **FC-ResNet:** A fully connected residual network with projection shortcuts for final classification.

Two configurations are evaluated:
- **LHRF-Core:** BoVW → SAE → TE → FC-ResNet
- **LHRF-HC:** BoVW → SAE → TE → HC → FC-ResNet

## Contents

- `matlab/run_lhrf.m`: Main MATLAB experimental runner.
- `matlab/run_lhrf_author_supplied_original.m`: Baseline version of the runner retained alongside the main runner.
- `examples/manifest_template.csv`: Input schema.
- `tests/`: Static checks and a MATLAB smoke-test script.
- `docs/MANUSCRIPT_ALIGNMENT.md`: Implementation notes and mapping between code and manuscript.

## Usage

Requires MATLAB R2022b or compatible, Computer Vision Toolbox, Deep Learning Toolbox, and Statistics and Machine Learning Toolbox.

```matlab
addpath('matlab');
% Run LHRF-Core
resultsCore = run_lhrf('manifest.csv', 'results_core', false);

% Run LHRF-HC (with clustering augmentation)
resultsHC = run_lhrf('manifest.csv', 'results_hc', true);