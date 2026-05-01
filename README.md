# CMKL: Continual Multimodal Knowledge Graph Learner

Anonymous code release accompanying the NeurIPS 2026 main-track submission **CMKL: Modality-Aware Continual Learning for Evolving Biomedical Knowledge Graphs**. This repository contains the full CMKL model, training procedure, and ablation infrastructure. CMKL is evaluated on the PrimeKG-CL benchmark, which is released as a separate downloadable archive.

## What is CMKL?

CMKL is a continual learning framework for biomedical knowledge graphs that natively encodes structural, textual, and molecular signal and protects previously learned knowledge through modality-aware regularization and replay.

**Architecture (4 components):**
1. **Modality-specific encoders.** R-GCN over the graph (structural), frozen BiomedBERT projected to the shared dim (textual), Morgan-fingerprint MLP (molecular).
2. **Mixture-of-Experts (MoE) fusion.** A learned per-entity router produces modality weights; the fused embedding is the additive combination. This lets the router suppress an unreachable modality (e.g., frozen text) without forcing it through a learned bottleneck that would destroy its pretrained geometry.
3. **Decoder.** DistMult by default; the codebase also includes ComplEx, RotatE, and TransE decoders.
4. **Continual learning mechanisms.** Per-encoder EWC (separate Fisher matrices per modality) and a K-means-diverse multimodal replay buffer (replay triples carry all available modality features).

**Headline results on PrimeKG-CL (10 tasks, 5 seeds):**
- *Biomedical relationship prediction:* CMKL AP $= 0.062 \pm 0.010$ (matches the strongest structural CL methods within seed noise; significantly above LKGE 0.039 and Joint Training 0.047).
- *Biomedical entity classification:* CMKL AP $= 0.591 \pm 0.005$, +60% over the strongest structural baseline (Joint 0.370), with near-zero forgetting.
- *Methodological finding:* the greedy modality problem is encoder-level, not fusion-level — a frozen text encoder can post a higher per-task ceiling (text-only AP 0.136) than any trainable fusion, but is unreachable by margin-ranking gradients.

## Repository layout

```
cmkl/
├── src/
│   ├── models/           # CMKL: encoders.py, fusion.py (MoE / cross-attn / concat),
│   │                     # decoders.py, cmkl.py (top-level model), ogm_ge.py
│   ├── continual/        # modality_ewc.py (per-encoder Fisher), multimodal_replay.py
│   │                     # (K-means-diverse buffer), distillation.py
│   ├── data/             # benchmark loaders (operates on the released PrimeKG-CL archive)
│   ├── baselines/        # shared training infrastructure (_base.py) + structural baselines
│   │                     # used as same-decoder comparison points (Naive, Joint, EWC, ER)
│   ├── evaluation/       # filtered MRR, AP/AF/BWT/REM, modality-specific forgetting
│   └── utils/
├── scripts/              # run_cmkl.py (main entry), run_ablations.py, generate_paper_b_figures.py,
│                         # precompute_features.py, merge_seed_results.py
├── slurm/                # SLURM job scripts (run_cmkl.sh, run_cmkl_ablation.sh, run_cmkl_sf.sh)
├── configs/              # YAML configs (cmkl.yaml, ablation configs)
├── requirements.txt
└── environment.yml
```

## Quick start

```bash
# 1. Set up the environment
conda env create -f environment.yml
conda activate mcgl

# 2. Download the PrimeKG-CL benchmark (released as a separate archive)
#    Unpack to ./data/benchmark/

# 3. Pre-compute multimodal features (one-time, ~30 min)
python scripts/precompute_features.py --data_root data/benchmark

# 4. Train CMKL on the 10-task continual sequence (5 seeds)
python scripts/run_cmkl.py \
    --config configs/cmkl.yaml \
    --seeds 42 123 456 789 1024 \
    --data_root data/benchmark \
    --output results/

# 5. Aggregate seeds and produce paper figures
python scripts/merge_seed_results.py
python scripts/generate_paper_b_figures.py
```

## Reproducing the paper's tables and figures

| Paper result | Script |
|---|---|
| Table 1 (LP main) | `slurm/run_cmkl.sh` |
| Table 2 (NC main) | `slurm/run_cmkl.sh --task nc` |
| Figure 2 (per-task) | `scripts/generate_paper_b_figures.py` |
| Figure 3 (fusion ablation) | `slurm/run_cmkl_sf.sh` (score-level fusion variants) |
| Figure 4 (modality forgetting) | `scripts/generate_paper_b_figures.py` |
| Supplement: CL component ablation | `slurm/run_cmkl_ablation.sh` |

## Hyperparameters (DistMult default)

- Embedding dim: 256 · R-GCN: 2 layers, 30 basis matrices · BiomedBERT projection: 768 → 256 (frozen) · Morgan MLP: 1024 → 256 → 256
- Per-encoder EWC: $\lambda_s{=}10$, $\lambda_t{=}5$, $\lambda_m{=}1$, $\lambda_f{=}5$
- Replay: 1,000-triple K-means-diverse buffer, replayed with all available modality features
- Optimizer: Adam, lr $10^{-3}$, batch 512, 100 epochs/task
- Seeds: $\{42, 123, 456, 789, 1024\}$

## Decoder scope

CMKL is deliberately scoped to bilinear (DistMult) decoders. A real-valued additive MoE composes cleanly with bilinear scoring but does not respect the rotational metric a RotatE decoder is built to exploit. Our four-generation attempt to extend CMKL to RotatE/ComplEx plateaued at AP $\approx 0.025$/$0.021$ (vs. baseline EWC-RotatE 0.088, EWC-ComplEx 0.029); we frame this as a precise statement of where the recipe applies and a concrete design problem for future work (decoder-aware fusion: complex-valued routing, near-identity initialization in the rotational metric).

## License

Code is released under the MIT license. See `LICENSE`.

## Anonymous submission notice

This is an anonymous code release for double-blind peer review. Author identities, institutional affiliations, and external links to author repositories have been redacted. The code released here reproduces all numbers reported in the paper.
