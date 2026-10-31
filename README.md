# CVD-STACK+: An Explainable, Causally Interpretable, and Robust Stacking Ensemble Framework with Invariance Validation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XAI: SHAP & LIME](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-green.svg)](https://shap.readthedocs.io/)
[![Causal: LiNGAM & DoWhy](https://img.shields.io/badge/Causal-LiNGAM%20%7C%20DoWhy-purple.svg)](https://lingam.readthedocs.io/)

---

## 📌 Executive Summary

**CVD-STACK+** is an end-to-end, clinically grounded machine learning and causal discovery framework designed for cardiovascular disease (CVD) risk prediction. Moving beyond traditional "black-box" classification, **CVD-STACK+** bridges high-precision predictive modeling with dual-layer explainability (XAI), game-theoretic causal topology, multi-hop counterfactual simulation, and rigorous scientific falsification.

Tested on a benchmark cohort of **69,000+ patient records**, the optimal **V2 Stacking Ensemble** achieves an **ROC-AUC of 0.842 ± 0.004** and an **Accuracy of 0.762 ± 0.003**, outperforming standard baseline classifiers while providing actionable, invariant diagnostic pathways for clinical decision support.

---

## 🚀 Key Framework Highlights

1. **Stratified Preprocessing & Physiological Engineering:** Computes derived hemodynamic metrics—Body Mass Index (BMI), Pulse Pressure (`ap_hi - ap_lo`), and Mean Arterial Pressure (MAP)—on a 65:35 stratified train-test split.
2. **Optuna-Tuned Stacking Tournament:** Evaluates 5 heterogeneous stacking variants (V1–V5). The optimal **V2 configuration** pairs diverse base learners (RF, KNN, GNB, DT) with a calibrated linear meta-learner (Logistic Regression).
3. **Dual-Layer Explainable AI (XAI):** Unifies global SHAP feature importance (identifying `ap_hi`, `MAP`, and `age` as primary drivers) with instance-level LIME patient profiles to resolve conflicting clinical indicators.
4. **Standard LiNGAM Causal Baseline:** Constructs a baseline Directed Acyclic Graph (DAG) on raw observational data, revealing a 3-tier clinical hierarchy: *Root Lifestyle/Demographics → Physiological Mediators → Terminal CVD Risk*.
5. **Novel Weighted SHAP-LiNGAM (WSL) Pipeline:** Projects game-theoretic Shapley attributions (ϕ) directly into DirectLiNGAM (`ϕ = B·ϕ + e`) to quantify attribution-space interaction dynamics, isolating **Synergistic** vs. **Compensatory** relationships.
6. **Multi-Hop Counterfactual "What-If" Engine:** Leverages formal *do*-calculus operators `do(X_j = x_j*)` to simulate cascading downstream risk abatement along discovered structural DAG paths (e.g., reducing Patient 4 risk from 88.59% to 16.59%).
7. **3-Tier Macro-Stressor Refutation:** Subjects the derived causal estimands to Random Common Cause insertion, Placebo Treatment permutations, and 80% Data Subset cross-validation to guarantee invariant resilience (p ≈ 1.0) against selection bias and confounding.

---

## 🛠️ Complete 11-Stage Methodology Breakdown

### Stage 1: Data Preprocessing & Hemodynamic Feature Engineering
* **Data Cleaning:** Handled missing values, duplicate records, and extreme physiological outliers across the 69,000-patient cohort.
* **Feature Scaling:** Applied `StandardScaler` to continuous numerical features to ensure zero mean and unit variance.
* **Hemodynamic Indicators:** Engineered three clinically validated compound attributes:
  * **Body Mass Index (BMI):** `weight / (height / 100)^2`
  * **Pulse Pressure (PP):** `ap_hi - ap_lo`
  * **Mean Arterial Pressure (MAP):** `ap_lo + (ap_hi - ap_lo) / 3`

---

### Stage 2: Stratified Data Splitting
* **Split Ratio:** Partitioned the preprocessed dataset into a **65:35 Train-Test ratio**.
* **Stratification:** Utilized stratified sampling on the target label (`cardio`) to preserve exact positive and negative class proportions across both training and unseen testing subsets, preventing class imbalance bias.

---

### Stage 3: Baseline Model Development
Constructed five standalone baseline machine learning classifiers to establish initial predictive benchmarks:
1. **Decision Tree (DT):** Rule-based hierarchical splitting.
2. **Logistic Regression (LR):** Calibrated linear baseline with sigmoidal output.
3. **Gaussian Naïve Bayes (GNB):** Probabilistic classifier assuming Gaussian feature distributions.
4. **Random Forest (RF):** De-correlated ensemble tree architecture with majority voting.
5. **K-Nearest Neighbors (KNN):** Distance-based instance classifier using Euclidean metrics.

---

### Stage 4: Optuna-Based Bayesian Hyperparameter Optimization
* Integrated the **Optuna** framework to execute Bayesian search optimization across the hyperparameter space of all baseline models.
* Automated parameter exploration maximized objective functions while mitigating cross-validation variance across individual models.

---

### Stage 5: Stacking Ensemble Tournament (V1–V5 Architectures)
Developed five heterogeneous stacking configurations combining diverse base-learner probability outputs into meta-features:
* **V1:** Base [`LR`, `DT`, `KNN`, `RF`] → Meta-Learner [`GNB`]
* **V2 (Optimal):** Base [`RF`, `KNN`, `GNB`, `DT`] → Meta-Learner [`LR`]
* **V3:** Base [`LR`, `DT`, `RF`, `GNB`] → Meta-Learner [`KNN`]
* **V4:** Base [`LR`, `DT`, `KNN`, `GNB`] → Meta-Learner [`RF`]
* **V5:** Base [`LR`, `KNN`, `GNB`, `RF`] → Meta-Learner [`DT`]

**Key Finding:** The **V2 configuration** with a linear Logistic Regression meta-learner produced the most stable decision boundaries, outperforming non-linear meta-aggregators (V4/V5) that exhibited higher variance.

---

### Stage 6: Model Evaluation & Clinical Validation
* **Validation Strategy:** Applied **Stratified 5-Fold Cross-Validation** to eliminate evaluation bias.
* **Performance Metrics:** Evaluated across five diagnostic criteria: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Optimal Result:** Stacking V2 achieved peak metrics across all folds (**ROC-AUC: 0.842 ± 0.004**, **Accuracy: 0.762 ± 0.003**).

---

### Stage 7: Dual-Layer Explainable AI (XAI) Engine
* **Global Interpretability (SHAP):** Generated SHAP summary and beeswarm plots to rank global risk drivers. Hemodynamic factors (`ap_hi`, `MAP`, `age`) emerged as top predictors of positive CVD diagnosis.
* **Local Interpretability (LIME):** Generated instance-level patient explanations to resolve conflicting clinical indicators (e.g., weighing protective factors like `ap_hi <= 120 mmHg` against risk factors like `age > 58`).

---

### Stage 8: Standard LiNGAM Causal Discovery Baseline
* Applied the Linear Non-Gaussian Acyclic Model (LiNGAM) directly to observational data to construct a baseline Directed Acyclic Graph (DAG) without medical priors.
* **Discovered 3-Tier Hierarchy:**
  1. **Root Nodes:** Lifestyle and demographic attributes (`alco`, `smoke`, `active`, `gluc`, `gender`).
  2. **Mediator Nodes:** Physical and chemical indicators (`weight`, `height`, `bmi`, `cholesterol`, `ap_hi`, `ap_lo`).
  3. **Terminal Sink:** CVD Risk Probability (`cvd_risk_prob`), confirming risk propagates through multi-hop intermediate pathways.

---

### Stage 9: Weighted SHAP-LiNGAM (WSL) Pipeline
* Projected post-prediction SHAP attribution matrices (`ϕ`) directly into the DirectLiNGAM algorithm to evaluate attribution-space causal topology (`ϕ = B·ϕ + e`).
* **Interaction Dynamics Isolated:**
  * **Synergistic Links:** Co-dependent features where the model amplifies predictive focus (e.g., `MAP → ap_hi`, Link Strength: `+1.897`).
  * **Compensatory Links:** Information trade-offs where the model suppresses redundant signals (e.g., `ap_lo → ap_hi`, Link Strength: `-1.261`).

---

### Stage 10: Structural Counterfactual "What-If" Engine
* Built an interactive clinical decision support module executing formal *do*-calculus interventions: `do(X_j = x_j*)`.
* Simulated multi-hop downstream risk cascading by updating descendant feature values through the discovered `B-matrix`:

```text
x_(i,cf) = x_(i,empirical) + Σ [ b_ij * (x_(j,cf) - x_(j,empirical)) ]
```
* **Clinical Demonstration:** Lowered Patient 4's baseline critical risk from **88.59%** down to **16.59%** via step-by-step hemodynamic, metabolic, and behavioral interventions.

---

### Stage 11: Causal Invariance Refutation & Sensitivity Testing
Validated structural graph stability against background noise, confounding, and data artifacts using three macro-stressors:
1. **Random Common Cause Refuter:** Added an unmeasured random covariate. Invariant effect estimates with pseudo `p ≈ 1.0` confirmed resilience to noise.
2. **Placebo Treatment Refuter:** Randomly permuted trigger values; complete collapse of causal effect to near-zero confirmed target specificity.
3. **Data Subset Refuter:** Evaluated on 80% cross-validated dataset slices; stable coefficients confirmed population-level invariance.
---

## 📊 Experimental Results

### 1. Stacking Ensemble Tournament Configurations

| Version | Base Learners | Meta-Learner | Core Characteristics |
| :---: | :--- | :---: | :--- |
| **V1** | LR, DT, KNN, RF | GNB | Probabilistic meta-aggregation |
| **V2 (★)** | **RF, KNN, GNB, DT** | **LR** | **Calibrated linear decision boundary (Optimal)** |
| **V3** | LR, DT, RF, GNB | KNN | Non-parametric instance-based meta-learning |
| **V4** | LR, DT, KNN, GNB | RF | Non-linear ensemble meta-learner (High variance) |
| **V5** | LR, KNN, GNB, RF | DT | Rule-based decision tree meta-learner |

---

### 2. Model Performance Comparison (Stratified 5-Fold Cross-Validation)

| Model / Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *Decision Tree (DT)* | 0.729 ± 0.005 | 0.729 ± 0.004 | 0.729 ± 0.005 | 0.728 ± 0.005 | 0.790 ± 0.007 |
| *Gaussian Naïve Bayes (GNB)* | 0.726 ± 0.005 | 0.728 ± 0.004 | 0.726 ± 0.005 | 0.724 ± 0.005 | 0.789 ± 0.007 |
| *K-Nearest Neighbors (KNN)* | 0.721 ± 0.005 | 0.731 ± 0.005 | 0.721 ± 0.005 | 0.718 ± 0.006 | 0.785 ± 0.006 |
| *Logistic Regression (LR)* | 0.732 ± 0.005 | 0.733 ± 0.005 | 0.732 ± 0.005 | 0.731 ± 0.005 | 0.797 ± 0.006 |
| *Random Forest (RF)* | 0.727 ± 0.004 | 0.729 ± 0.004 | 0.727 ± 0.004 | 0.726 ± 0.004 | 0.788 ± 0.006 |
| **Stacking V1** | 0.759 ± 0.005 | 0.759 ± 0.005 | 0.759 ± 0.005 | 0.758 ± 0.004 | 0.837 ± 0.005 |
| **Stacking V2 (Optimal)** | **0.762 ± 0.003** | **0.762 ± 0.003** | **0.762 ± 0.003** | **0.762 ± 0.003** | **0.842 ± 0.004** |
| **Stacking V3** | 0.746 ±
