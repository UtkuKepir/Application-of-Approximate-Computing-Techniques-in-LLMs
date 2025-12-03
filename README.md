# Application-of-Approximate-Computing-Techniques-in-LLMs

Large Language Models (LLMs) have recently achieved state-of-the-art performance in a wide range of natural language processing tasks, but their rapid growth in size has introduced severe challenges in terms of computational cost, memory consumption, and energy efficiency. This makes their deployment on resourceconstrained environments increasingly difficult, and has motivated research into approximation strategies that trade exactness for efficiency. The first half of this thesis presents an extensive survey of approximate computing methods for transformer-based architectures, focusing on techniques such as quantization, pruning, low-rank approximation (LoRA), stochastic perturbations, and stochastic memory masking. Alongside the survey, a benchmarking framework was developed to evaluate these approaches in a consistent and comparable manner. The framework integrates support for multiple datasets, including Alpaca, Databricks-Dolly-15k, and AgentInstruct, and provides metrics such as BLEU score, ROUGE-L score, F1 score, SBERT similarity, inference time, output size, model size and perplexity. Experiments were conducted on two representative models, LLaMA-3.2-1B-Instruct and Gemma-3-1B-Instruct, to investigate the efficiency–accuracy trade-offs of different approximation methods. The second half of this thesis focuses on combining multiple approximation methods to further reduce computational overhead while preserving task performance. In particular, the work investigates the integration of LoRA with other methods to minimize the number of trainable parameters and improve training efficiency. This stage of the work emphasizes the importance of evaluating approximation techniques not only in isolation but also in combination, highlighting scenarios in which hybrid approaches achieve better efficiency–accuracy trade-offs than single methods. Overall, this thesis provides a systematic exploration of approximation strategies for LLMs and their impact on both training and inference. The results demonstrate that lightweight approaches such as LoRA and quantization achieve substantial reductions in memory usage and computational load with minimal performance degradation, while more aggressive approximations require careful tuning to maintain robustness.

\subsection{Inference Results for the Dolly-15k Dataset(Fine-tuned with Alpaca Dataset)}
\label{sec:alpaca_results}

\begin{longtable}{|l l c c c c c c|}
\hline
\textbf{Model} & \makecell{\textbf{Approximation} \\ \textbf{Technique}} & \textbf{\gls{BLEU}} & \textbf{\gls{ROUGE-L}} & \textbf{SBERT} & \makecell{\textbf{Inf.} \\ \textbf{Time} \\ \textbf{(s)}} & \makecell{\textbf{Out.} \\ \textbf{Size} \\ \textbf{(KB)}} & \makecell{\textbf{Model} \\ \textbf{Size} \\ \textbf{(MB)}} \\
\hline
\endfirsthead
\hline
\textbf{Model} & \makecell{\textbf{Approximation} \\ \textbf{Technique}} & \textbf{\gls{BLEU}} & \textbf{\gls{ROUGE-L}} & \textbf{SBERT} & \makecell{\textbf{Inf.} \\ \textbf{Time} \\ \textbf{(s)}} & \makecell{\textbf{Out.} \\ \textbf{Size} \\ \textbf{(KB)}} & \makecell{\textbf{Model} \\ \textbf{Size} \\ \textbf{(MB)}} \\
\hline
\endhead
\hline
\multicolumn{8}{r}{\textit{Continued on next page}} \\
\hline
\endfoot
\hline
\caption{Inference results for LLaMA-3.2-1B-Instruct and Gemma-3-1B-Instruct on the Dolly-15k dataset (Fine-tuned with Alpaca dataset).}
\endlastfoot
\multicolumn{8}{|l|}{\textbf{Baseline}} \\
\scriptsize{LLaMA-3.2-1B} & -----             & 0.58 & 0.67 & 0.88 & 2.2 & 163.8 & 2,858\\
\scriptsize{Gemma-3-1B}   & -----             & 0.25 & 0.46 & 0.68 & 7.9 & 121.3 & 2,483\\
\hline
\multicolumn{8}{|l|}{\textbf{Fine-tuned}} \\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Pruned}        & 0.62 & 0.69 & 0.90 & 2.0 & 164.9 & 3,370\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Pruned}        & 0.26 & 0.46 & 0.69 & 8.0 & 122.4 & 2,878\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT8}          & 0.86 & 0.90 & 0.94 & 1.3 & 154.3 & 1,681\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT8}          & 0.25 & 0.46 & 0.68 & 15.3 & 125.5 & 1,532\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT4}          & 0.36 & 0.48 & 0.81 & 7.6 & 184.9 & 1,092\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT4}          & 0.24 & 0.42 & 0.77 & 15.0 & 199.4 & 1,055\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{NF4}           & 0.47 & 0.56 & 0.83 & 8.1 & 187.7 & 1,164\\
\scriptsize{Gemma-3-1B}   & \scriptsize{NF4}           & 0.23 & 0.46 & 0.73 & 19.5 & 163.5 & 1,112\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{\gls{LoRA}}    & 0.55 & 0.65 & 0.89 & 2.7 & 162.8 & 2,859\\
\scriptsize{Gemma-3-1B}   & \scriptsize{\gls{LoRA}}    & 0.25 & 0.45 & 0.66 & 8.9 & 116.2 & 2,484\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Perturbed}     & 0.60 & 0.70 & 0.89 & 1.9 & 159.8 & 2,858\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Perturbed}     & 0.21 & 0.40 & 0.65 & 8.1 & 119.4 & 2,483\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Mem-masked}    & 0.58 & 0.66 & 0.87 & 2.2 & 158.0 & 2,858\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Mem-masked}    & 0.22 & 0.43 & 0.67 & 8.1 & 129.8 & 2,483\\
\hline
\multicolumn{8}{|l|}{\textbf{Fine-tuned}} \\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Pruned + \gls{LoRA}}        & 0.57 & 0.66 & 0.87 & 2.7 & 165.2 & 3,371\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Pruned + \gls{LoRA}}        & 0.23 & 0.41 & 0.63 & 9.2 & 163.7 & 2,879\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT8 + \gls{LoRA}}          & 0.75 & 0.80 & 0.91 & 4.1 & 141.5 & 1,682\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT8 + \gls{LoRA}}          & 0.21 & 0.37 & 0.73 & 20.2 & 187.9 & 1,533\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT4 + \gls{LoRA}}          & 0.32 & 0.44 & 0.83 & 10.5 & 169.6 & 1,092\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT4 + \gls{LoRA}}          & 0.32 & 0.53 & 0.84 & 20.1 & 222.8 & 1,056\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{NF4 + \gls{LoRA}}           & 0.46 & 0.56 & 0.83 & 12.1 & 173.4 & 1,164\\
\scriptsize{Gemma-3-1B}   & \scriptsize{NF4 + \gls{LoRA}}           & 0.23 & 0.42 & 0.75 & 20.6 & 172.5 & 1,113\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Perturbed + \gls{LoRA}}     & 0.55 & 0.65 & 0.87 & 2.7 & 165.9 & 2,859\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Perturbed + \gls{LoRA}}     & 0.23 & 0.40 & 0.61 & 9.3 & 164.9 & 2,484\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{Masked + \gls{LoRA}}        & 0.49 & 0.61 & 0.85 & 3.1 & 171.7 & 2,859\\
\scriptsize{Gemma-3-1B}   & \scriptsize{Masked + \gls{LoRA}}        & 0.21 & 0.39 & 0.61 & 9.4 & 173.9 & 2,484\\
\hline
\multicolumn{8}{|l|}{\textbf{Fine-tuned}} \\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT8+Pruned+\gls{LoRA}}     & 0.71 & 0.77 & 0.88 & 4.1 & 169.9 & 1,682\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT8+Pruned+\gls{LoRA}}     & 0.31 & 0.53 & 0.72 & 20.3 & 151.9 & 1,533\\
\scriptsize{LLaMA-3.2-1B} & \scriptsize{INT4+Pruned+\gls{LoRA}}     & 0.35 & 0.46 & 0.82 & 10.4 & 183.6 & 1,092\\
\scriptsize{Gemma-3-1B}   & \scriptsize{INT4+Pruned+\gls{LoRA}}     & 0.26 & 0.45 & 0.82 & 20.2 & 193.3 & 1,056\\
\end{longtable}
