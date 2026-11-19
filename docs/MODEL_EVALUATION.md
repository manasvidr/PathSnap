\# PathSnap — Machine Learning Model Evaluation



\## 1. Objective



The objective of model evaluation is to identify the most effective classifier for matching noisy GPS observations to candidate road segments.



The experiments compare classification accuracy, macro F1 score, and the effects of sequence decoding.



\## 2. Dataset and Validation



The experiments use:



\- 80 training trips

\- 10 student-test trips

\- 10 reserved evaluator-test trips



Trip-based three-fold GroupKFold cross-validation was used to reduce leakage between training and validation trajectories.



The reserved evaluator-test data was not used during training.



\## 3. Models Evaluated



Five machine-learning classifiers were compared:



1\. Logistic Regression

2\. Gaussian Naive Bayes

3\. Decision Tree

4\. Random Forest

5\. Histogram Gradient Boosting



\## 4. Model Comparison



| Model | CV Macro F1 | Student-Test Top-1 Accuracy | Student-Test Candidate Macro F1 |

|---|---:|---:|---:|

| Logistic Regression | 0.8089 | 79.65% | 0.8037 |

| Gaussian Naive Bayes | 0.7372 | 77.15% | 0.7291 |

| Decision Tree | 0.8464 | 79.18% | 0.8402 |

| Random Forest | 0.8876 | 80.54% | 0.8777 |

| Histogram Gradient Boosting | \*\*0.8900\*\* | \*\*80.68%\*\* | \*\*0.8835\*\* |



Histogram Gradient Boosting achieved the highest cross-validation macro F1 score among the five models.



\## 5. Model Selection



Histogram Gradient Boosting was selected because it produced the strongest cross-validation macro F1 score.



Macro F1 was used to evaluate performance across candidate classes rather than relying only on overall accuracy.



\## 6. Hyperparameter Optimization



Optuna was used to tune Histogram Gradient Boosting.



The tuning process used eight trials with three-fold cross-validation.



The best recorded parameters were:



\- Learning rate: 0.10641583749108179

\- Maximum leaf nodes: 23

\- L2 regularization: 1.6150593059496638



The tuned model achieved approximately 0.8896 cross-validation macro F1.



This was slightly lower than the untuned baseline, so tuning did not demonstrate a measurable improvement in the recorded experiment.



\## 7. HMM/Viterbi Decoding



The complete local implementation also evaluates road-network-aware Viterbi decoding.



The decoder uses transition constraints to encourage continuity between consecutive road segments.



Recorded student-test averages:



| Metric | Classifier | Viterbi Decoding |

|---|---:|---:|

| Pointwise accuracy | 83.14% | 81.88% |

| Direct road continuity | 89.90% | 94.76% |



Viterbi decoding improved average continuity but reduced average pointwise accuracy.



It improved pointwise accuracy on six of the ten student-test trips and reduced it on four.



\## 8. Evaluation Limitations



The experiments use reference-route alignment to generate candidate-level pseudo-labels. These are not independently audited pointwise ground-truth labels.



The student-test results should not be interpreted as performance on the reserved evaluator-test dataset.



The GitHub repository should be checked to ensure that all final model-training and decoding modules are included before submission.



\## 9. Conclusion



Histogram Gradient Boosting provided the strongest cross-validation classification performance among the evaluated models.



Sequence decoding improved road continuity, demonstrating the trade-off between local classification accuracy and globally consistent route reconstruction.

