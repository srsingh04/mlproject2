# ML4Science Project @ LPBS Lab 

This project compares the performance of several classification algorithms using **5-fold cross-validation** under different random seeds and preprocessing conditions.

The models evaluated are:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost

The main objective is to compare model performance and understand the impact of **feature scaling** on different classification algorithms.

---

## Model Evaluation

Model performance was evaluated using **5-fold cross-validation**.

For each model, the following metrics are reported:

- **Mean Accuracy** — average validation accuracy across the five folds.
- **Standard Deviation** — variation in accuracy across the folds. A lower value generally indicates more consistent performance.

---

## Results With Feature Scaling

### 5-Fold Cross-Validation — Random Seed 6

| Model | Mean Accuracy | Std. Deviation |
|---|---:|---:|
| Logistic Regression | 0.6441 | 0.1190 |
| Decision Tree Classifier | 0.7621 | 0.1782 |
| Random Forest Classifier | 0.7310 | 0.1337 |
| SVM | 0.6967 | 0.1398 |
| XGBoost | 0.7621 | 0.1782 |

With feature scaling, **Decision Tree** and **XGBoost** achieved the highest mean accuracy of approximately **76.2%**.

However, their relatively high standard deviation indicates noticeable variation across the validation folds.

---

## Results Without Feature Scaling

### 5-Fold Cross-Validation — Random Seed 6

| Model | Mean Accuracy | Std. Deviation |
|---|---:|---:|
| Logistic Regression | 0.7111 | 0.0846 |
| Decision Tree Classifier | **0.9421** | **0.0349** |
| Random Forest Classifier | 0.8847 | 0.0761 |
| SVM | 0.5792 | 0.0755 |
| XGBoost | **0.9421** | **0.0349** |

Without feature scaling, **Decision Tree** and **XGBoost** achieved the best performance, reaching approximately **94.2% mean accuracy**.

Random Forest also performed strongly with approximately **88.5% accuracy**.

---

## Impact of Feature Scaling

A comparison of the results shows that scaling affected the models differently.

| Model | With Scaling | Without Scaling |
|---|---:|---:|
| Logistic Regression | 64.4% | **71.1%** |
| Decision Tree | 76.2% | **94.2%** |
| Random Forest | 73.1% | **88.5%** |
| SVM | **69.7%** | 57.9% |
| XGBoost | 76.2% | **94.2%** |

The strongest performance in these experiments was obtained from the **unscaled dataset**.

### Tree-Based Models

Decision Trees, Random Forests, and XGBoost generally do not require feature scaling because their decisions are based on feature thresholds rather than distances between observations.

For example, a tree may learn a rule such as:

```text
feature_1 < 25
```

Changing the scale of that feature does not normally change the ordering of the observations.

Therefore, scaling is usually unnecessary for:

- Decision Trees
- Random Forests
- XGBoost and other gradient-boosted decision trees

In this experiment, these models also produced substantially better results on the unscaled data.

> Note: Standard scaling itself should not normally degrade a correctly implemented decision tree simply because the magnitude of a feature changes. Large differences in results may therefore also be influenced by factors such as preprocessing order, train/test splits, random-state handling, data leakage, or implementation details.

### SVM

SVMs are generally sensitive to feature scale because distances between observations influence the optimization process.

In this experiment:

- Without scaling: **57.9%**
- With scaling: **69.7%**

This improvement is consistent with the expectation that SVM often benefits from standardized features.

---

## Additional Cross-Validation Experiments

To evaluate the stability of the results, additional experiments were performed using different random seeds.

### Random Seed 42

| Model | Mean Accuracy | Std. Deviation |
|---|---:|---:|
| Logistic Regression | 0.6042 | 0.0876 |
| Decision Tree Classifier | **0.9421** | **0.0349** |
| Random Forest Classifier | 0.8977 | 0.0664 |
| SVM | 0.6967 | 0.1398 |
| XGBoost | **0.9421** | **0.0349** |

---

### Random Seed 26

| Model | Mean Accuracy | Std. Deviation |
|---|---:|---:|
| Logistic Regression | 0.6042 | 0.0876 |
| Decision Tree Classifier | **0.9421** | **0.0349** |
| Random Forest Classifier | 0.9082 | 0.0575 |
| SVM | 0.6967 | 0.1398 |
| XGBoost | **0.9421** | **0.0349** |

---

## Overall Comparison

Across the experiments, the tree-based models consistently achieved the strongest classification performance.

### Best Performing Models

**Decision Tree Classifier**

- Mean Accuracy: **94.21%**
- Standard Deviation: **3.49%**

**XGBoost**

- Mean Accuracy: **94.21%**
- Standard Deviation: **3.49%**

**Random Forest**

- Mean Accuracy: approximately **88–91%**
- Performance varied slightly depending on the random seed.

Decision Tree and XGBoost therefore produced the highest observed cross-validation accuracy in these experiments.

---

## Key Findings

1. **Decision Tree and XGBoost achieved the highest accuracy**, reaching approximately **94.2%**.
2. **Random Forest also performed strongly**, achieving approximately **88–91% accuracy**.
3. **SVM benefited from feature scaling**, increasing from approximately **57.9% to 69.7%** in the reported comparison.
4. Tree-based models do not generally require feature scaling.
5. Results should be checked across multiple random seeds to ensure conclusions are not dependent on one particular split.
6. Standard deviation should be considered alongside accuracy because it indicates how stable a model is across validation folds.

---

## Conclusion

Among the models tested, **Decision Tree and XGBoost produced the strongest results, with a mean cross-validation accuracy of approximately 94.2%**.

Random Forest was also competitive, achieving approximately **88–91% accuracy** depending on the experiment.

Feature scaling improved SVM performance but was unnecessary for the tree-based algorithms. Future experiments should use model-specific preprocessing pipelines and additional classification metrics to obtain a more complete assessment of model performance.
