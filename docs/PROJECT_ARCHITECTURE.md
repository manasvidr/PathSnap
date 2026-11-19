\# PathSnap — Project Architecture



\## 1. Introduction



PathSnap is a GPS map-matching project developed for the UE24CS352A Machine Learning Hackathon.



The objective is to identify the most likely road segment corresponding to each noisy GPS observation and reconstruct a vehicle's probable route.



\## 2. Overall Workflow



The complete project follows these stages:



1\. Load GPS trajectory and road-network datasets.

2\. Validate data quality and inspect missing or invalid records.

3\. Analyze GPS observations and road-network characteristics.

4\. Generate nearby candidate road segments.

5\. Extract geometric and movement-related features.

6\. Train machine-learning classification models.

7\. Evaluate model performance using trip-based validation.

8\. Apply road-network-aware HMM/Viterbi decoding.

9\. Visualize predictions and reconstructed routes.



\## 3. Dataset Components



The dataset contains four main file types:



| File | Purpose |

|---|---|

| `.track` | GPS observations and timestamps |

| `.nodes` | Road-network node information |

| `.arcs` | Directed road segments |

| `.route` | Reference route information |



The experiment uses 80 training trips and 10 student-test trips. An additional 10 trips are reserved for evaluator testing.



\## 4. Data Preprocessing



The preprocessing stage loads GPS observations and road-network information, checks data quality, and prepares the records for candidate generation.



\## 5. Candidate Road Generation



Each GPS observation is associated with nearby road segments.



This reduces the search space and allows the classifier to compare plausible road candidates instead of evaluating every segment in the network.



\## 6. Feature Engineering



The completed experiments use 13 engineered features describing relationships between GPS observations and candidate road segments.



These features capture geometric and movement-related information.



\## 7. Machine-Learning Classification



The project compares five classification algorithms:



\- Logistic Regression

\- Gaussian Naive Bayes

\- Decision Tree

\- Random Forest

\- Histogram Gradient Boosting



Histogram Gradient Boosting achieved the highest recorded cross-validation macro F1 score.



\## 8. HMM/Viterbi Decoding



The complete local implementation evaluates Viterbi decoding to improve road-network continuity.



The decoder considers relationships between consecutive road segments instead of treating each GPS observation independently.



\## 9. Visualization



The project uses plots and interactive maps to examine GPS trajectories, candidate segments, and predicted routes.



\## 10. Repository Implementation



The GitHub repository contains data-processing utilities, candidate-generation scripts, tests, notebooks, results, and documentation.



The complete local implementation also includes additional machine-learning, decoding, and API modules. These should be synchronized with GitHub before final submission.



\## 11. Conclusion



PathSnap combines road-network geometry, supervised learning, and sequence decoding to address GPS map matching.



The architecture separates data preparation, prediction, evaluation, and visualization so that each component can be tested and improved independently.

