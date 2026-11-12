# Manuscript Alignment

Reference: *A Lightweight Hybrid Representation Learning Framework for Colorectal Lesion Classification in Colonoscopy Images*, accompanying manuscript.

## Main Architecture

- BoVW vocabulary: 150 words.
- SAE: 150 → 100 → 50 → 100 → 150, with ReLU activations.
- Triplet margin: 0.2; balanced mini-batches of 32.
- Optional Ward hierarchical clustering with two clusters.
- FC-ResNet hidden widths: 300, 200, 100, 70.
- Image resizing: 224 × 224.
- L2: 0.001; learning rate: 0.001.
- Sparsity penalty weight 4 and target 0.05.
- A dedicated 15% tuning partition is excluded before evaluation.
- CVC-ClinicDB: sequence grouping; CVC-ColonDB: source grouping; Kvasir mixed task: stratified image-level evaluation.

## Scope of This Release

The MATLAB runner is based on the author-supplied MATLAB file included in this package. This package preserves an original snapshot and provides a corrected runner.

The corrected runner implements the manuscript's SAE sparsity penalty, dense multiscale SURF grid, historical tuning exclusion, baseline suite, and the analyses described in the manuscript. The pipeline follows the configuration listed above.

All metrics in this repository are computed by running the provided code.