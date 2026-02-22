# Student Grade Predictor 

A machine learning journey exploring student performance, from failed linear assumptions to successful grade classification.

---

## Executive Summary of Attempts

| Attempt | Approach | Key Metric | Status |
| :--- | :--- | :--- | :--- |
| **1** | Linear Regression | $R^2$: -0.0058 |  **Underfit** |
| **2** | Tree-Based Regression | $R^2$: ~0.0 |  **Data Leakage/Noise** |
| **3** | Random Forest Classifier | Accuracy: 61.2% |  **Success** |

---

## Attempt 1: Linear Regression Model
**Result:**  Failed

**Observations:**
* **Linearity Violation:** Features (e.g., Study Hours, Sleep) showed negligible correlation with the target variable, violating the core assumption of linear relationships.
* **Metric:** The $R^2$ score was **-0.0058**, indicating the model was worse than a horizontal line representing the mean.
* **Range Issue:** The model only predicted values in the narrow 68-70 range.

**Next Step:** Pivot to Feature Engineering to uncover non-linear relationships.

---

## Attempt 2: Supervised Learning (Tree-Based)
**Result:** Failed

**Observations:**
* **The "Deterministic" Trap:** The dataset uses a fixed weighted formula for `Final_Score`. Traditional regression struggles when the target is a direct sum of specific components (Assignments, Midterms).
* **Feature Importance:** Only `Assignments_Avg` and `Midterm_Score` showed predictive power. Auxiliary features like attendance and extracurriculars provided insufficient signal to explain deviations.

**Conclusion:** Raw score prediction is unsuitable here. The focus shifted to **Classification** (predicting the letter grade).

---

## Attempt 3: Student Grade Prediction (Current)
**Result:** Success

### Overview
This attempt transitions from predicting a continuous number to a 5-class classification problem (**A, B, C, D, F**). The goal was to evaluate if a model could learn patterns despite significant class imbalance.

### Dataset Characteristics
The data reflects a realistic, non-uniform distribution:
* **Grade A:** Very rare (~16 samples) - *Minority Class*
* **Grade C:** Majority class
* **Grade F:** Small sample size



### Model: Random Forest Classifier
* **Why:** Robust against outliers, handles non-linear structured data, and provides "Feature Importance" metrics.
* **Accuracy:** **61.2%**
![Heatmap](heatmap.png)

### Evaluation & Confusion Matrix


**Key Findings:**
1. **Adjacency Confusion:** The model often confuses A with B, or F with D. This is expected as these grades overlap heavily in the feature space.
2. **Class A Gap:** Due to the severe lack of "A" samples, the model rarely predicts it.
3. **Class F Bias:** Failed students are often misclassified as "D," suggesting the features for "F" are not distinct enough.

---

## Implementation Details

### User-Friendly Input
To make the model interactive and practical, we implemented a `get_average_input` function. 
* Instead of asking for a single daunting number, it allows users to input multiple scores (e.g., individual assignments) and automatically calculates the average.
* **Required Inputs:** Assignment scores, Project score, Midterm score, Attendance %, Study hours, Quiz Score **(optional)**, Age and Sleep hours.

### Optimization through Defaults
We identified that features like **Internet Access, Parent Education, ECA and Family Income** had negligible impact on this specific dataset's predictions. To improve User Experience (UX), these are set to **default values** so the user isn't fatigued by unnecessary questions.

---
## Feature Engineering & Derived Variables

The model utilizes both **raw input features** and **engineered (derived) features** during training.

While users provide primary academic and behavioral inputs, certain composite indicators (such as Participation Score and Stress Index) are computed internally. These derived values are necessary because:

- They were part of the training feature space.
- They cannot be directly provided by users.
- They capture structured behavioral patterns more effectively than raw inputs alone.

---

### 1. Participation Score (Derived Feature)

Although attendance, quiz scores, and study hours are used independently in the model, we also compute a composite **Participation Score** to better represent academic engagement.

Participation =  
`0.5 × Attendance + 0.3 × Quiz Average + 0.2 × Study Component`

Where:

`Study Component = clip((study_hours / 30) × 100, 0, 100)`

This derived metric:

- Preserves training data structure  
- Encodes engagement intensity  
- Reduces behavioral feature fragmentation  
- Improves predictive stability  

---

### 2. Stress Level Index (Derived Feature)

Midterm performance, study hours, and sleep hours are used directly in training. However, we additionally compute a structured **Stress Index** to align with the model’s training schema.

Components:

- Academic Stress  
  `100 − Midterm Average`

- Burnout Stress  
  `clip(((study_hours − 35) / 15) × 100, 0, 100)`

- Sleep Stress  
  `clip(((7 − sleep_hours) / 7) × 100, 0, 100)`

Combined Stress Score:

`0.5 × Academic Stress + 0.3 × Sleep Stress + 0.2 × Burnout Stress`

The result is scaled from **0–100 → 1–10**:

`Stress Level = round((combined / 100) × 9 + 1)`

This ensures:

- Compatibility with training feature expectations  
- Behavioral abstraction beyond raw metrics  
- Better generalization across lifestyle variations  

---

## Next Steps
1. **Synthetic Data (SMOTE):** Use oversampling techniques to create "synthetic" Grade A examples to improve recall for top students.

### Final Reflection:
This project demonstrates that Machine Learning is an iterative cycle of failure, analysis, and pivot. The jump from **61.2% accuracy** to a production-ready tool won't come from a "better" algorithm alone, but from the **Data Engineering** planned in the next steps—specifically using SMOTE to give a voice to the minority classes.

---

*This project is a learning journey exploring the nuances of data imbalance and the transition from regression to classification.*