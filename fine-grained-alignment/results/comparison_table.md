# Fine-Grained Patch Alignment: Baseline vs. Trained Comparison

## Quantitative Results Table

| Evaluation Metric | Baseline (Untrained) | Trained Projector | Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Align($V, W$) Mean IoU** | 0.1069 | **0.2051** | +0.0982 | **+91.9%** |
| **Align($V, W$) Acc @ IoU $\ge 0.50$** | 1.29% | **11.29%** | +10.00% | **+875.2%** |
| **On-Target Cosine Similarity** | -0.0170 | **0.6307** | +0.6477 | Major Alignment Gain |
| **Background Cosine Similarity** | -0.0203 | **0.4232** | 0.4435 | Well-separated foreground |
| **Von Neumann Entropy $S(\rho)$** | 2.0169 | **1.5675** | -0.4494 | Semantic structuring |
| **Entropy Compression Ratio** | 0.9794 | **0.7612** | -0.2182 | Structured manifold |
| **Matching Pursuit GT Match Acc** | 0.00% | **60.97%** | +60.97% | Dramatic concept alignment |
| **Matching Pursuit Top-1 Word Sim** | -0.0716 | **0.8486** | +0.9202 | Coherent word correlation |
| **Matching Pursuit Residual Ratio** | 0.9889 | **0.3718** | -0.6171 | Better dictionary coverage |

---

## Qualitative Matching Pursuit Comparisons

Demonstrates the shift from garbled, uninterpretable baseline patch representations to coherent semantic representations:

### Example 1: Ground Truth Object `truck`
- **Baseline (Untrained):** Top-1 Word: `wine glass` (cos=-0.0655) | Decomposed Atoms: `wine glass (-0.77), refrigerator (+0.87), vase (-0.86)`
- **Trained Projector:** Top-1 Word: `truck` (cos=0.9152) | Decomposed Atoms: `truck (+10.49), person (+2.51), traffic light (+1.34)`
- **Assessment:** ✓ Direct Concept Alignment

### Example 2: Ground Truth Object `person`
- **Baseline (Untrained):** Top-1 Word: `horse` (cos=-0.0879) | Decomposed Atoms: `horse (-1.42), umbrella (+1.63), boat (-1.22)`
- **Trained Projector:** Top-1 Word: `person` (cos=0.7835) | Decomposed Atoms: `person (+8.71), cow (+3.26), tie (+2.45)`
- **Assessment:** ✓ Direct Concept Alignment

### Example 3: Ground Truth Object `person`
- **Baseline (Untrained):** Top-1 Word: `grass` (cos=-0.0733) | Decomposed Atoms: `grass (-0.89), elephant (+0.95), sky (-0.94)`
- **Trained Projector:** Top-1 Word: `person` (cos=0.9895) | Decomposed Atoms: `person (+21.73), teddy bear (-1.40), handbag (-0.73)`
- **Assessment:** ✓ Direct Concept Alignment

### Example 4: Ground Truth Object `tennis racket`
- **Baseline (Untrained):** Top-1 Word: `white` (cos=-0.1006) | Decomposed Atoms: `white (-1.50), mouse (+1.08), book (-1.34)`
- **Trained Projector:** Top-1 Word: `tennis racket` (cos=0.9427) | Decomposed Atoms: `tennis racket (+22.47), frisbee (+2.80), remote (-1.73)`
- **Assessment:** ✓ Direct Concept Alignment

### Example 5: Ground Truth Object `person`
- **Baseline (Untrained):** Top-1 Word: `white` (cos=-0.0806) | Decomposed Atoms: `white (-0.90), elephant (+1.14), train (-0.96)`
- **Trained Projector:** Top-1 Word: `person` (cos=0.9629) | Decomposed Atoms: `person (+12.54), baseball glove (+2.31), person (-0.82)`
- **Assessment:** ✓ Direct Concept Alignment

