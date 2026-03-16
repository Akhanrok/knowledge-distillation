# Knowledge Distillation for LSTM-based Sentiment Classification

**The goal** of this project was to explore how knowledge distillation techniques can compress a large BiLSTM teacher model into smaller LSTM student models while maintaining high sentiment classification accuracy. The study focused on comparing three KD strategies (logit-based, feature-based, and attention-based) in terms of model performance, parameter efficiency, inference latency, and memory usage.

**The primary objective** was to evaluate the effectiveness of different KD methods for transferring knowledge from a high-capacity BiLSTM teacher to smaller students. Specifically, the study investigates:

- How logit-based KD (soft label transfer) impacts student accuracy and recall.
- Whether feature-based KD (matching intermediate hidden states) improves student learning beyond logit alignment.
- The effect of attention-based KD (aligning importance patterns of hidden states) on overall performance and efficiency.
- Trade-offs between model size, inference latency, memory footprint, and classification metrics.

**The questions** addressed in this study were:
1. Can a small LSTM student achieve near-teacher performance using KD?
2. Which distillation method best preserves accuracy and F1-score while minimizing parameters?
3. How do hyperparameters such as temperature and alpha influence student performance in logit- and attention-based KD?
4. How do compression and runtime trade-offs differ among KD methods?

## Methodology
The project implemented three KD approaches, each wrapping a small LSTM student:

- **logit-based KD** (student learns from both its classification loss and the KL-divergence between its softmax outputs and the teacher’s soft logits, transferring predictive knowledge) [1];
- **feature KD** (student aligns its hidden representations with the teacher’s intermediate BiLSTM features via mean-squared error, combined with standard classification loss, enabling richer internal knowledge transfer) [2];
- **attention KD** (extends logit-based distillation by enforcing similarity between teacher and student attention patterns, aligning the relative importance of internal hidden states in addition to output predictions) [3].

The teacher model is a BiLSTM with embedding size 128, two bidirectional LSTM layers, and dense output.
The student models are lightweight LSTMs with embedding size 32 and a single LSTM layer.

Training and evaluation were performed on the IMDB sentiment dataset, using subsets of 10000 training and 2000 test examples for rapid prototyping. Labels were one-hot encoded, and sequences were padded to length 200.

**Distillation setup:**
1. teacher logits were precomputed for logit- and attention-based KD;
2. training datasets were batched and shuffled for efficient GPU utilization;
3. students were trained for 5 epochs with the Adam optimizer;
4. hyperparameter sweeps were performed for temperature (T=1,2,4,8) and alpha (α=0.3,0.5,0.7) in 2 top-performing models (temperature softens the teacher’s predicted probabilities, letting the student learn from class similarities rather than only hard labels. Alpha balances the student’s classification loss against the distillation loss, controlling the trade-off between learning from ground-truth labels and teacher guidance).

**Evaluation Metrics:**
- classification metrics: accuracy, precision, recall, F1-score;
- efficiency metrics: number of parameters, memory footprint (MB), inference latency (seconds), and compression ratio relative to teacher parameters.



## Results and Analysis

**Key observations:**
1. attention KD achieved the highest student accuracy (0.8135) and F1-score (0.8036), closely approaching teacher performance while using 6× fewer parameters;
2. logit and feature KD improved recall or precision selectively but underperformed in balanced F1 compared to Attention KD. Logit KD favored recall, while Feature KD boosted precision;
3. baseline student without KD performed poorly (Acc 0.564), demonstrating the importance of knowledge transfer;
4. attention KD consistently provided the best trade-off between performance and efficiency, with lower latency and memory footprint.

**Hyperparameter sweeps:**
- Temperature and alpha critically influenced performance:
  - attention KD performed best at T=4–8 and  α=0.3–0.5;
  - logit KD favored moderate temperature (T=4) and balanced alpha (α=0.5) for optimal F1;
- extreme alpha values often led to overemphasis on either student loss or distillation loss, reducing overall F1.

**Limitations:**
- small training and test subsets for rapid prototyping may underestimate real-world performance;
- only one teacher architecture (BiLSTM) was explored;
- sequence length was fixed; longer sequences may affect distillation efficacy;
- only a single dataset (IMDB) was used; generalization to other tasks requires further testing.

Despite the limitations of dataset size and a single teacher architecture, the study demonstrates that knowledge distillation can effectively compress large BiLSTM models into lightweight LSTMs while preserving strong sentiment classification performance. Even with small subsets of the IMDB dataset, attention-based KD consistently delivered near-teacher accuracy and F1 scores, demonstrating the practical potential of KD for deploying efficient models in resource-constrained environments. These results suggest that careful selection of distillation strategies and hyperparameters can achieve meaningful model compression without sacrificing predictive quality, providing a solid foundation for future research and real-world applications.

**Directions for future work:**
- test KD on larger and more diverse sentiment datasets;
- explore more student architectures (e.g., stacked LSTM layers, transformer-based students);
- investigate combined distillation strategies (logit + feature + attention);
-examine multi-teacher distillation for improved knowledge transfer;
-incorporate automated hyperparameter search for alpha and temperature.

## Liturature
[1] Hinton, Geoffrey & Dean, Jeff & Vinyals, Oriol & Rachmad, Yoesoep. (2014). Distilling the Knowledge in a Neural Network. 1-9. arXiv:1503.02531

[2] Romero, Adriana & Kahou, Samira Ebrahimi & Ballas, Nicolas & Chassang, Antoine & Rachmad, Yoesoep & Gatta, Carlo. (2014). FitNets: Hints for Thin Deep Nets. arxiv:1412.6550

[3] Zagoruyko, Sergey & Komodakis, Nikos. (2016). Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer. arXiv.1612.03928. 
