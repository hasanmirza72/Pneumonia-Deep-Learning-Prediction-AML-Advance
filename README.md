# 🩻 Comparative Analysis of Deep Learning Architectures for Pneumonia Detection in Chest X-Rays
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Bioinformatics](https://img.shields.io/badge/Domain-Bioinformatics-27ae60?style=for-the-badge)

**Researcher:** Mirza Muhammad Hasan Ali

**Academic Institution:** University of Bologna

---

## 📝 1. Abstract
This research establishes an advanced deep learning pipeline for the automated classification of pneumonia from Chest X-ray (CXR) images. This project prioritizes **safety** by maximizing sensitivity to minimize False Negatives. We benchmark a custom **Baseline CNN** against an optimized **EfficientNet-B0** to demonstrate how clinical engineering—specifically contrast enhancement and class balancing—creates a superior diagnostic "ceiling".

---

## ⚠️ 2. Problem Statement
Pneumonia is a leading cause of mortality worldwide, necessitating early and accurate radiological detection. Standard AI models often fail in clinical settings due to:
* **Visual Ambiguity:** Subtle pulmonary infiltrates are often "invisible" in low-contrast raw X-rays.
* **Class Imbalance:** Datasets are typically skewed toward infected cases, causing models to ignore "Normal" lung features.
* **Safety Risks:** A standard "high accuracy" model might still miss critical cases (False Negatives), leading to delayed treatment.

This project solves these challenges by establishing a **Safety-First** diagnostic logic that achieves a **99.0% Recall**.

---

## 📂 3. Dataset: ChestXRay2017 (UCSD)
* **Total Images:** 5,856 pediatric Chest X-ray images.
* **Categories:** 2 (Normal vs. Pneumonia).
* **Clinical Site:** Guangzhou Women and Children’s Medical Center.

### Key Challenge: Imbalance
The training data is imbalanced with a roughly $1:3$ ratio of Normal to Pneumonia cases. Without mitigation, models develop a "majority bias" that compromises diagnostic reliability.

---

## 🛠️ 4. Methodology: The Clinical Pipeline

### 📊 4.1 Stratified Experimental Rigor
We utilized a custom modular script (`00_dataset_splitter.py`) to execute a **stratified 80/20 train-to-validation split**. This ensures that both clinical classes are represented in identical proportions in both sets, preventing data leakage and ensuring fair evaluation.

### ⚖️ 4.2 Handling Class Imbalance
* **WeightedRandomSampler:** Implemented in `data_loader.py` to ensure the model encounters "Normal" samples as frequently as "Pneumonia" during every epoch.
* **Clinical Loss Weighting:** The `CrossEntropyLoss` is penalized using inverse frequency weights, forcing the model to respect the rarity of healthy samples.

### 💡 4.3 Radiological Feature Enhancement (CLAHE)
Standard image resizing often loses critical diagnostic detail. We implemented **Contrast Limited Adaptive Histogram Equalization (CLAHE)**:
* **Mechanism:** Isolates the luminance channel to sharpen contrast within the lung fields.
* **Value:** This highlights pulmonary opacities that indicate infection, simulating a radiologist's high-definition workstation.

![Figure 1: Radiology Enhancement](./Visuals/radiology_enhancement_comparison.png)  
> **Figure 1:** Comparison of raw clinical data versus the CLAHE-enhanced input used in the advanced pipeline.

---

## 🧠 5. Architectural Comparative Study

### 📉 5.1 The Baseline: Standard 3-Layer CNN
A custom 3-layer sequential CNN was developed to establish a performance floor (`baseline_model.py`). This baseline was intentionally fed **unoptimized raw data** (No CLAHE, No Weighted Sampler) to demonstrate how standard architectures struggle with clinical noise and class imbalance.

![Figure 2: Baseline Curves](./Visuals/baseline_learning_curves.png)  
> **Figure 2:** Baseline training history showing significant performance instability on raw data.

### 📈 5.2 The Advanced: EfficientNet-B0 Pipeline
The primary architecture utilizes an **EfficientNet-B0** backbone with **Transfer Learning**. It features a custom diagnostic head with **BatchNorm1d** for stability and **Dual Dropout** to prevent overfitting, treating depth, width, and resolution as a joint optimization problem.

![Figure 3: Advanced Curves](./Visuals/full_training_performance.png)  
> **Figure 3:** Advanced model convergence showing stable diagnostic accuracy and loss reduction history.

---

## 🔬 6. Clinical Performance Audit

### 🏆 6.1 Final Results & Clinical Gain
The final audit on unseen test data revealed a massive gain in diagnostic safety. The Advanced model reduced the risk of missed diagnoses by **84%**, identifying 21 additional patients that the baseline model failed to recognize.

![Figure 4: Final Comparison](./Visuals/comparison_between_baseline_&_advance_model.png)  
> **Figure 4:** Side-by-side performance benchmarking highlighting the gains in Sensitivity and MCC across architectures.

### ✅ 6.2 High-Sensitivity Scorecard
The optimized pipeline achieved a **99.0% Recall**. This high-sensitivity threshold ensures that nearly every patient with pneumonia is correctly flagged for medical intervention.

![Figure 5: Final Scorecard](./Visuals/final_test_scorecard.png)  
> **Figure 5:** Final diagnostic scorecard achieving the "Gold Standard" safety threshold on unseen data.

---

## 🔍 7. Interpretability & Validation

### 🗺️ 7.1 Biological Interpretability (Grad-CAM)
To ensure the model focuses on biological symptoms rather than image noise, we implemented **Grad-CAM** localization. Heatmaps confirm that model attention is correctly localized within the **pulmonary parenchyma**.

![Figure 6: Grad-CAM Analysis](./Visuals/gradcam_clinical_analysis.png)  
> **Figure 6:** Heatmap visualization confirming model focus on biologically relevant lung regions.

### 🖼️ 7.2 Error Analysis & Gallery
A systematic visual audit was used to identify "borderline" clinical ambiguities. A general gallery audit further verified that high-confidence predictions align perfectly with actual clinical labels.

![Figure 7: Error Analysis](./Visuals/misclassification_report.png)  
> **Figure 7:** Visual audit of misclassified cases identifying borderline clinical ambiguities.

---

## 🏁 8. Conclusion
This project successfully transitions from a naive AI approach to a robust, clinically-aware diagnostic pipeline. The combination of **Radiology CLAHE**, **Weighted Samplers**, and **EfficientNet-B0** establishes a framework that is significantly safer for hospital use than standard CNNs. By prioritizing **Recall (99.0%)** and **MCC (0.646)**, this study provides a reliable decision-support tool for modern cardiovascular and respiratory bioinformatics.
