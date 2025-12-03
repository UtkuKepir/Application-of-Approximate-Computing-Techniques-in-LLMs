# Application-of-Approximate-Computing-Techniques-in-LLMs

LLMs have recently achieved state-of-the-art performance in a wide range of natural language processing tasks, but their rapid growth in size has introduced severe challenges in terms of computational cost, memory consumption, and energy efficiency. This makes their deployment on resourceconstrained environments increasingly difficult, and has motivated research into approximation strategies that trade exactness for efficiency. The first half of this thesis presents an extensive survey of approximate computing methods for transformer-based architectures, focusing on techniques such as quantization, pruning, low-rank approximation (LoRA), stochastic perturbations, and stochastic memory masking. Alongside the survey, a benchmarking framework was developed to evaluate these approaches in a consistent and comparable manner. The framework integrates support for multiple datasets, including Alpaca, Databricks-Dolly-15k, and AgentInstruct, and provides metrics such as BLEU score, ROUGE-L score, F1 score, SBERT similarity, inference time, output size, model size and perplexity. Experiments were conducted on two representative models, LLaMA-3.2-1B-Instruct and Gemma-3-1B-Instruct, to investigate the efficiency–accuracy trade-offs of different approximation methods. The second half of this thesis focuses on combining multiple approximation methods to further reduce computational overhead while preserving task performance. In particular, the work investigates the integration of LoRA with other methods to minimize the number of trainable parameters and improve training efficiency. This stage of the work emphasizes the importance of evaluating approximation techniques not only in isolation but also in combination, highlighting scenarios in which hybrid approaches achieve better efficiency–accuracy trade-offs than single methods. Overall, this thesis provides a systematic exploration of approximation strategies for LLMs and their impact on both training and inference. The results demonstrate that lightweight approaches such as LoRA and quantization achieve substantial reductions in memory usage and computational load with minimal performance degradation, while more aggressive approximations require careful tuning to maintain robustness.

# Inference Results for the Dolly-15k Dataset  
**Fine-tuned with Alpaca Dataset**

| Model           | Approximation Technique     | BLEU | ROUGE-L | SBERT | Inf. Time (s) | Out. Size (KB) | Model Size (MB) |
|-----------------|-----------------------------|------|---------|--------|----------------|----------------|------------------|
| **Baseline**     |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | —                           | 0.58 | 0.67    | 0.88   | 2.2            | 163.8          | 2,858            |
| Gemma-3-1B       | —                           | 0.25 | 0.46    | 0.68   | 7.9            | 121.3          | 2,483            |
| **Fine-tuned**   |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned                      | 0.62 | 0.69    | 0.90   | 2.0            | 164.9          | 3,370            |
| Gemma-3-1B       | Pruned                      | 0.26 | 0.46    | 0.69   | 8.0            | 122.4          | 2,878            |
| LLaMA-3.2-1B     | INT8                        | 0.86 | 0.90    | 0.94   | 1.3            | 154.3          | 1,681            |
| Gemma-3-1B       | INT8                        | 0.25 | 0.46    | 0.68   | 15.3           | 125.5          | 1,532            |
| LLaMA-3.2-1B     | INT4                        | 0.36 | 0.48    | 0.81   | 7.6            | 184.9          | 1,092            |
| Gemma-3-1B       | INT4                        | 0.24 | 0.42    | 0.77   | 15.0           | 199.4          | 1,055            |
| LLaMA-3.2-1B     | NF4                         | 0.47 | 0.56    | 0.83   | 8.1            | 187.7          | 1,164            |
| Gemma-3-1B       | NF4                         | 0.23 | 0.46    | 0.73   | 19.5           | 163.5          | 1,112            |
| LLaMA-3.2-1B     | LoRA                        | 0.55 | 0.65    | 0.89   | 2.7            | 162.8          | 2,859            |
| Gemma-3-1B       | LoRA                        | 0.25 | 0.45    | 0.66   | 8.9            | 116.2          | 2,484            |
| LLaMA-3.2-1B     | Perturbed                   | 0.60 | 0.70    | 0.89   | 1.9            | 159.8          | 2,858            |
| Gemma-3-1B       | Perturbed                   | 0.21 | 0.40    | 0.65   | 8.1            | 119.4          | 2,483            |
| LLaMA-3.2-1B     | Mem-masked                  | 0.58 | 0.66    | 0.87   | 2.2            | 158.0          | 2,858            |
| Gemma-3-1B       | Mem-masked                  | 0.22 | 0.43    | 0.67   | 8.1            | 129.8          | 2,483            |
| **Fine-tuned (with LoRA)** |                  |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned + LoRA               | 0.57 | 0.66    | 0.87   | 2.7            | 165.2          | 3,371            |
| Gemma-3-1B       | Pruned + LoRA               | 0.23 | 0.41    | 0.63   | 9.2            | 163.7          | 2,879            |
| LLaMA-3.2-1B     | INT8 + LoRA                 | 0.75 | 0.80    | 0.91   | 4.1            | 141.5          | 1,682            |
| Gemma-3-1B       | INT8 + LoRA                 | 0.21 | 0.37    | 0.73   | 20.2           | 187.9          | 1,533            |
| LLaMA-3.2-1B     | INT4 + LoRA                 | 0.32 | 0.44    | 0.83   | 10.5           | 169.6          | 1,092            |
| Gemma-3-1B       | INT4 + LoRA                 | 0.32 | 0.53    | 0.84   | 20.1           | 222.8          | 1,056            |
| LLaMA-3.2-1B     | NF4 + LoRA                  | 0.46 | 0.56    | 0.83   | 12.1           | 173.4          | 1,164            |
| Gemma-3-1B       | NF4 + LoRA                  | 0.23 | 0.42    | 0.75   | 20.6           | 172.5          | 1,113            |
| LLaMA-3.2-1B     | Perturbed + LoRA            | 0.55 | 0.65    | 0.87   | 2.7            | 165.9          | 2,859            |
| Gemma-3-1B       | Perturbed + LoRA            | 0.23 | 0.40    | 0.61   | 9.3            | 164.9          | 2,484            |
| LLaMA-3.2-1B     | Masked + LoRA               | 0.49 | 0.61    | 0.85   | 3.1            | 171.7          | 2,859            |
| Gemma-3-1B       | Masked + LoRA               | 0.21 | 0.39    | 0.61   | 9.4            | 173.9          | 2,484            |
| **Fine-tuned (Combinations)** |               |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | INT8 + Pruned + LoRA        | 0.71 | 0.77    | 0.88   | 4.1            | 169.9          | 1,682            |
| Gemma-3-1B       | INT8 + Pruned + LoRA        | 0.31 | 0.53    | 0.72   | 20.3           | 151.9          | 1,533            |
| LLaMA-3.2-1B     | INT4 + Pruned + LoRA        | 0.35 | 0.46    | 0.82   | 10.4           | 183.6          | 1,092            |
| Gemma-3-1B       | INT4 + Pruned + LoRA        | 0.26 | 0.45    | 0.82   | 20.2           | 193.3          | 1,056            |


# Inference Results for the Dolly-15k Dataset
**Fine-tuned with Agent Dataset**

| Model           | Approximation Technique     | BLEU | ROUGE-L | SBERT | Inf. Time (s) | Out. Size (KB) | Model Size (MB) |
|-----------------|-----------------------------|------|---------|--------|----------------|----------------|------------------|
| **Baseline**     |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | —                           | 0.54 | 0.64    | 0.86   | 2.2            | 152.4          | 2,858            |
| Gemma-3-1B       | —                           | 0.29 | 0.49    | 0.75   | 7.7            | 133.6          | 2,483            |
| **Fine-tuned**   |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned                      | 0.59 | 0.68    | 0.88   | 2.5            | 157.4          | 3,370            |
| Gemma-3-1B       | Pruned                      | 0.27 | 0.48    | 0.73   | 7.7            | 131.3          | 2,878            |
| LLaMA-3.2-1B     | INT8                        | 0.53 | 0.62    | 0.83   | 5.5            | 174.5          | 1,681            |
| Gemma-3-1B       | INT8                        | 0.27 | 0.47    | 0.74   | 14.7           | 129.7          | 1,532            |
| LLaMA-3.2-1B     | INT4                        | 0.34 | 0.44    | 0.77   | 7.7            | 190.9          | 1,092            |
| Gemma-3-1B       | INT4                        | 0.28 | 0.47    | 0.85   | 14.8           | 187.9          | 1,055            |
| LLaMA-3.2-1B     | NF4                         | 0.42 | 0.53    | 0.83   | 9.4            | 170.4          | 1,164            |
| Gemma-3-1B       | NF4                         | 0.22 | 0.44    | 0.69   | 18.6           | 176.9          | 1,112            |
| LLaMA-3.2-1B     | LoRA                        | 0.56 | 0.66    | 0.86   | 2.7            | 160.5          | 2,859            |
| Gemma-3-1B       | LoRA                        | 0.26 | 0.45    | 0.68   | 8.5            | 129.0          | 2,484            |
| LLaMA-3.2-1B     | Perturbed                   | 0.56 | 0.66    | 0.89   | 2.5            | 158.3          | 2,858            |
| Gemma-3-1B       | Perturbed                   | 0.26 | 0.46    | 0.73   | 7.7            | 130.5          | 2,483            |
| LLaMA-3.2-1B     | Mem-masked                  | 0.54 | 0.63    | 0.87   | 2.8            | 161.2          | 2,858            |
| Gemma-3-1B       | Mem-masked                  | 0.26 | 0.45    | 0.74   | 8.3            | 139.5          | 2,483            |
| **Fine-tuned (with LoRA)** |                  |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned + LoRA               | 0.54 | 0.63    | 0.83   | 2.8            | 175.6          | 3,371            |
| Gemma-3-1B       | Pruned + LoRA               | 0.29 | 0.46    | 0.68   | 8.8            | 154.0          | 2,879            |
| LLaMA-3.2-1B     | INT8 + LoRA                 | 0.55 | 0.63    | 0.85   | 8.0            | 175.9          | 1,682            |
| Gemma-3-1B       | INT8 + LoRA                 | 0.30 | 0.47    | 0.72   | 19.6           | 146.1          | 1,533            |
| LLaMA-3.2-1B     | INT4 + LoRA                 | 0.30 | 0.42    | 0.77   | 10.8           | 193.3          | 1,092            |
| Gemma-3-1B       | INT4 + LoRA                 | 0.26 | 0.45    | 0.83   | 19.7           | 176.6          | 1,056            |
| LLaMA-3.2-1B     | NF4 + LoRA                  | 0.43 | 0.54    | 0.85   | 12.0           | 153.9          | 1,164            |
| Gemma-3-1B       | NF4 + LoRA                  | 0.25 | 0.43    | 0.73   | 26.4           | 161.5          | 1,113            |
| LLaMA-3.2-1B     | Perturbed + LoRA            | 0.50 | 0.58    | 0.82   | 3.4            | 182.3          | 2,859            |
| Gemma-3-1B       | Perturbed + LoRA            | 0.32 | 0.52    | 0.74   | 8.8            | 157.8          | 2,484            |
| LLaMA-3.2-1B     | Masked + LoRA               | 0.49 | 0.58    | 0.82   | 3.3            | 178.8          | 2,859            |
| Gemma-3-1B       | Masked + LoRA               | 0.24 | 0.47    | 0.73   | 8.9            | 163.2          | 2,484            |
| **Fine-tuned (Combinations)** |               |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | INT8 + Pruned + LoRA        | 0.59 | 0.67    | 0.86   | 7.5            | 159.5          | 1,682            |
| Gemma-3-1B       | INT8 + Pruned + LoRA        | 0.25 | 0.43    | 0.69   | 19.3           | 164.4          | 1,533            |
| LLaMA-3.2-1B     | INT4 + Pruned + LoRA        | 0.31 | 0.42    | 0.84   | 10.6           | 165.8          | 1,092            |
| Gemma-3-1B       | INT4 + Pruned + LoRA        | 0.31 | 0.48    | 0.85   | 19.6           | 220.4          | 1,056            |

# Inference Results for the AgentInstruct Dataset
**Fine-tuned with Alpaca Dataset**
| Model           | Approximation Technique     | BLEU | ROUGE-L | SBERT | Inf. Time (s) | Out. Size (KB) | Model Size (MB) |
|-----------------|-----------------------------|------|---------|--------|----------------|----------------|------------------|
| **Baseline**     |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | —                           | 0.59 | 0.68    | 0.85   | 2.7            | 34.3           | 2,858            |
| Gemma-3-1B       | —                           | 0.26 | 0.56    | 0.75   | 8.1            | 36.7           | 2,483            |
| **Fine-tuned**   |                             |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned                      | 0.59 | 0.67    | 0.86   | 2.4            | 34.6           | 3,370            |
| Gemma-3-1B       | Pruned                      | 0.22 | 0.54    | 0.71   | 8.3            | 35.4           | 2,878            |
| LLaMA-3.2-1B     | INT8                        | 0.93 | 0.95    | 0.98   | 0.57           | 29.8           | 1,681            |
| Gemma-3-1B       | INT8                        | 0.30 | 0.58    | 0.82   | 15.4           | 41.9           | 1,532            |
| LLaMA-3.2-1B     | INT4                        | 0.23 | 0.37    | 0.71   | 7.5            | 39.8           | 1,092            |
| Gemma-3-1B       | INT4                        | 0.19 | 0.46    | 0.87   | 15.3           | 41.5           | 1,055            |
| LLaMA-3.2-1B     | NF4                         | 0.49 | 0.61    | 0.75   | 8.2            | 37.3           | 1,164            |
| Gemma-3-1B       | NF4                         | 0.17 | 0.39    | 0.71   | 19.2           | 37.9           | 1,112            |
| LLaMA-3.2-1B     | LoRA                        | 0.57 | 0.68    | 0.82   | 3.5            | 33.2           | 2,859            |
| Gemma-3-1B       | LoRA                        | 0.25 | 0.56    | 0.70   | 9.2            | 35.3           | 2,484            |
| LLaMA-3.2-1B     | Perturbed                   | 0.48 | 0.58    | 0.75   | 2.8            | 34.8           | 2,858            |
| Gemma-3-1B       | Perturbed                   | 0.22 | 0.53    | 0.73   | 8.4            | 36.6           | 2,483            |
| LLaMA-3.2-1B     | Mem-masked                  | 0.48 | 0.60    | 0.75   | 3.3            | 33.4           | 2,858            |
| Gemma-3-1B       | Mem-masked                  | 0.22 | 0.48    | 0.72   | 8.3            | 38.8           | 2,483            |
| **Fine-tuned (with LoRA)** |                 |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | Pruned + LoRA               | 0.44 | 0.57    | 0.75   | 3.8            | 34.9           | 3,371            |
| Gemma-3-1B       | Pruned + LoRA               | 0.25 | 0.58    | 0.73   | 9.3            | 35.2           | 2,879            |
| LLaMA-3.2-1B     | INT8 + LoRA                 | 0.86 | 0.88    | 0.95   | 1.8            | 31.6           | 1,682            |
| Gemma-3-1B       | INT8 + LoRA                 | 0.26 | 0.55    | 0.82   | 20.3           | 38.5           | 1,533            |
| LLaMA-3.2-1B     | INT4 + LoRA                 | 0.33 | 0.47    | 0.83   | 10.0           | 42.3           | 1,092            |
| Gemma-3-1B       | INT4 + LoRA                 | 0.19 | 0.41    | 0.84   | 20.3           | 40.2           | 1,056            |
| LLaMA-3.2-1B     | NF4 + LoRA                  | 0.51 | 0.59    | 0.83   | 11.5           | 39.3           | 1,164            |
| Gemma-3-1B       | NF4 + LoRA                  | 0.20 | 0.57    | 0.80   | 26.2           | 41.4           | 1,113            |
| LLaMA-3.2-1B     | Perturbed + LoRA            | 0.44 | 0.57    | 0.82   | 3.5            | 33.8           | 2,859            |
| Gemma-3-1B       | Perturbed + LoRA            | 0.31 | 0.64    | 0.73   | 9.3            | 34.9           | 2,484            |
| LLaMA-3.2-1B     | Masked + LoRA               | 0.47 | 0.61    | 0.81   | 3.5            | 32.8           | 2,859            |
| Gemma-3-1B       | Masked + LoRA               | 0.23 | 0.53    | 0.75   | 9.3            | 39.5           | 2,484            |
| **Fine-tuned (Combinations)** |              |      |         |        |                |                |                  |
| LLaMA-3.2-1B     | INT8 + Pruned + LoRA        | 0.72 | 0.77    | 0.88   | 4.9            | 35.3           | 1,682            |
| Gemma-3-1B       | INT8 + Pruned + LoRA        | 0.21 | 0.54    | 0.73   | 20.0           | 37.4           | 1,533            |
| LLaMA-3.2-1B     | INT4 + Pruned + LoRA        | 0.37 | 0.48    | 0.80   | 9.8            | 44.0           | 1,092            |
| Gemma-3-1B       | INT4 + Pruned + LoRA        | 0.22 | 0.40    | 0.87   | 20.0           | 44.4           | 1,056            |
