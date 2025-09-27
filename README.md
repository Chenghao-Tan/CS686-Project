# CS686-Project
Project: Testing Logit-Based Metrics of LLM on Out-of-Domain Knowledges: Impact of Gradient Dynamics and RAG/ICL Methods

## Introduction
Although epistemic uncertainty (EU)\[1\] is not an objective directly optimised during model training, the original EU paper proposed EU as a measure of uncertainty driven by knowledge gaps, explained by gradient dynamics. However, that paper offered mostly theoretical analysis, no experiments were conducted to verify this gradient dynamics phenomenon, and few are conducted to establish the reliability of EU/logit-based score in practice. Moreover, contemporary retrieval-augmented generation (RAG) methods achieve substantial accuracy gains without invoking any gradient dynamics. And there exists paper that argues the similiarity between RAG/In-context-learning and gradient-based training\[2\]. Thus, how would EU/logit-based score work in non-gradient-dynamics scenario is also interesting. In summary, the key research questions are:
* Is EU/logit-based metrics directly related to the accuracy of the model output? (Reliability)
* Does gradient dynamics truly affect EU/logit-based metrics in the proposed way? (Explainability)
* How will logits change when applying methods without gradient dynamics, like RAG? Does it still affect metrics like EU in a similar way to gradient dynamics? (Generalization)

## Content
See the [**PDF**](Testing_Logit_based_Metrics_On_Out_of_domain_Knowledges_Without_Gradient_Dynamics.pdf) for the report.

Code is not well cleaned, but since the built [**dataset**](ds-shuffledchoices.jsonl) is open-sourced, it should be easy to reproduce the results.

## References
1. Huan Ma et al. *Estimating LLM Uncertainty with Evidence*. 2025. arXiv: 2502.00290 \[cs.CL\]. URL: [https://arxiv.org/abs/2502.00290](https://arxiv.org/abs/2502.00290).
2. Johannes von Oswald et al. *Transformers learn in-context by gradient descent*. 2023. arXiv: 2212.07677 \[cs.LG\]. URL: [https://arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677).
