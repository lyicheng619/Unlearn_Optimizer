<div align='center'>
 
# Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning
<!-- 
[![Venue: ICML 2025](https://img.shields.io/badge/Venue-ICML%202025-green)](https://icml.cc/virtual/2025/poster/43469)
[![preprint](https://img.shields.io/badge/arXiv-2502.05374-B31B1B)](https://arxiv.org/abs/2502.05374)
[![collection](https://img.shields.io/badge/HuggingFace-Collection-yellow)](https://huggingface.co/collections/OPTML-Group/smooth-unlearned-model-67a92bb04d402b6ca3b2fb01)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/Unlearn-Smooth?tab=MIT-1-ov-file)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Smooth)](https://github.com/OPTML-Group/Unlearn-Smooth) -->

</div>

## Abstract
Large language model (LLM) unlearning aims to surgically remove the influence of undesired data or knowledge from an existing model while preserving its utility on unrelated tasks. This paradigm has shown promise in addressing privacy and safety concerns. However, recent findings reveal that unlearning effects are often *fragile*: post-unlearning manipulations such as weight quantization or fine-tuning can quickly neutralize the intended forgetting. Prior efforts to improve robustness primarily reformulate unlearning objectives by explicitly assuming the role of vulnerability sources. In this work, we take a different perspective by investigating the role of the *optimizer*, independent of unlearning objectives and formulations, in shaping unlearning robustness. We show that the “*grade*” of the optimizer, defined by the level of information it exploits — ranging from zeroth-order (gradient-free) to first-order (gradient-based) to second-order (Hessian-based) — is tightly linked to the resilience of unlearning. Surprisingly, we find that downgrading the optimizer, such as using zeroth-order methods or compressed-gradient variants (e.g., gradient sign-based optimizers), often leads to stronger robustness. While these optimizers produce noisier and less precise updates, they encourage convergence to harder-to-disturb basins in the loss landscape, thereby resisting post-training perturbations. By connecting zeroth-order methods with randomized smoothing, we further highlight their natural advantage for robust unlearning. Motivated by these insights, we propose a *hybrid optimizer* that combines first-order and zeroth-order updates, preserving unlearning efficacy while enhancing robustness. Extensive experiments on the MUSE and WMDP benchmarks, across multiple LLM unlearning algorithms, validate that our approach achieves more resilient forgetting without sacrificing unlearning quality.

## Cite This Work
```
@article{lang2025downgrade,
  title={Downgrade to Upgrade: Optimizer Simplification Enhances Robustness in LLM Unlearning},
  author={Lang, Yicheng and Zhang, Yihua and Fan, Chongyu and Wang, Changsheng and Jia, Jinghan and Liu, Sijia},
  journal={arXiv preprint arXiv:2510.00761},
  year={2025}
}
```