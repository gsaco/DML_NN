# 🧠 Neural Networks & Double Machine Learning

[![R](https://img.shields.io/badge/R-4.5.1-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.x-9558B2?style=flat&logo=julia&logoColor=white)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Assignment 5 - Group 1**  
> Advanced econometric analysis combining neural network fundamentals with Double/Debiased Machine Learning (DML) for causal inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Project Components](#project-components)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Results](#results)
- [Contributors](#contributors)

---

## 🎯 Overview

This repository explores two fundamental questions in modern econometric machine learning:

1. **Neural Network Basics**: How do different activation functions and architectures affect model performance in approximating nonlinear functions?

2. **Double Machine Learning**: How can we combine machine learning prediction power with rigorous causal inference to estimate treatment effects while avoiding overfitting bias?

We implement these methods in **R**, **Python**, and **Julia**, using the Pennsylvania Reemployment Bonus dataset to estimate the causal effect of unemployment insurance bonuses on unemployment duration.

---

## 📁 Repository Structure

```
DML_NN/
├── R/
│   ├── scripts/
│   │   ├── Question1_NN_Basics.ipynb      # Neural network fundamentals
│   │   └── Question2_DML.ipynb            # Double Machine Learning implementation
│   └── output/                             # Results and visualizations
│       ├── dml_results_r.csv
│       ├── nn_activation_metrics_r.csv
│       └── *.png (plots)
├── Python/
│   ├── scripts/
│   │   ├── Question1_NN_Basics.ipynb
│   │   └── Question2_DML.ipynb
│   └── output/                             # Python results
│       ├── dml_comparison.csv
│       ├── dml_crossfit_results.csv
│       └── dml_nocrossfit_results.csv
├── Julia/
│   ├── scripts/                            # Julia implementations
│   │   ├── Question1_NN_Basics.ipynb
│   │   └── Question2_DML.ipynb
│   └── output/                             # Julia results
├── input/
│   └── penn_jae.csv                        # Pennsylvania Reemployment dataset
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## 🔬 Project Components

### Question 1: Neural Network Fundamentals

**Objective**: Compare activation functions and learning configurations for sine wave approximation.

**Experiments**:
- 🧪 **Activation Functions**: ReLU, Sigmoid, Tanh
- 🏗️ **Architectures**: Small (1 layer, 50 units), Medium (2 layers), Large (3 layers, 100+100+50 units)
- 📊 **Learning Rates**: 0.01, 0.001, 0.0001
- 🎯 **Task**: Approximate f(x) = sin(x) over [0, 4π]

**Key Components**:
- Data generation and train/test split
- Custom neural network training with Keras
- In-sample vs. out-of-sample RMSE comparison
- Visualization of predictions vs. true function

---

### Question 2: Double Machine Learning (DML)

**Objective**: Estimate causal treatment effects while avoiding regularization bias and overfitting.

**Method**: Partially Linear Model with Cross-Fitting
```
Y = θD + g(X) + ε     (outcome equation)
D = m(X) + η          (treatment equation)
```

**ML Algorithms Tested**:
- 📈 **OLS** (Ordinary Least Squares)
- 🎯 **Lasso** (L1 regularization)
- 🌳 **Random Forest** (ensemble trees)
- 🧠 **Neural Networks** (deep learning)

**Experiments**:
- ✅ Cross-fitting (K=2 folds) for valid inference
- ❌ No cross-fitting (overfitting control)
- 🔄 48 model combinations (4 algorithms × 4 algorithms × 3 settings)

---

## 🏆 Key Findings

### Neural Network Results

| Activation | In-Sample RMSE | Out-Sample RMSE | R² |
|-----------|----------------|-----------------|-----|
| **Sigmoid** ✨ | 0.0466 | 0.0598 | 0.909 |
| ReLU | 0.0532 | 0.0642 | 0.898 |
| Tanh | 0.0498 | 0.0615 | 0.903 |

**Winner**: **Sigmoid activation** with 3-layer architecture and learning rate 0.001

**Insights**:
- Sigmoid performs best for smooth function approximation
- Larger networks (3 layers) capture complexity better
- Lower learning rates (0.001) provide stable convergence

---

### Double Machine Learning Results

| Method | Treatment Effect (θ) | 95% CI | RMSE Outcome | RMSE Treatment |
|--------|---------------------|--------|--------------|----------------|
| **Cross-Fitting** | -0.080 to -0.096 | [-0.28, 0.10] | 0.49-0.52 | 0.36-0.40 |
| No Cross-Fitting | -0.095 to -0.120 | Biased | 0.40-0.49 | 0.31-0.34 |

**Key Insights**:

1. **Treatment Effect**: Receiving unemployment bonus **reduces unemployment duration by ~8-11%**

2. **Cross-Fitting Advantage**: 
   - Provides valid confidence intervals
   - Prevents overfitting in treatment effect estimation
   - RMSE 3-18% **higher** (correctly penalizes complexity)

3. **Best Models**: Neural Networks and Random Forest for both outcome and treatment prediction

4. **Neyman Orthogonality**: DML debiases ML predictions, combining prediction power with causal validity

---

## 🛠️ Installation

### R Requirements

```r
install.packages(c(
  "tidyverse",
  "keras3",
  "glmnet",
  "randomForest",
  "caret",
  "gridExtra"
))
```

### Python Requirements

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy statsmodels tensorflow keras jupyterlab
```

### Julia Requirements

```julia
using Pkg
Pkg.add(["DataFrames", "CSV", "GLM", "MLJ", "Plots"])
```

---

## 🚀 Usage

### Running the Notebooks

1. **Clone the repository**:
```bash
git clone https://github.com/gsaco/DML_NN.git
cd DML_NN
```

2. **Choose your language** (R, Python, or Julia):
```bash
cd R/scripts/
# or
cd Python/scripts/
# or
cd Julia/scripts/
```

3. **Open Jupyter notebooks**:
```bash
jupyter lab
```

4. **Execute cells sequentially** in:
   - `Question1_NN_Basics.ipynb` for neural network experiments
   - `Question2_DML.ipynb` for DML causal analysis

---

## 💻 Technologies

### Languages
- **R 4.5.1**: Primary implementation with tidyverse + keras3
- **Python 3.x**: Alternative implementation with scikit-learn + TensorFlow
- **Julia 1.x**: High-performance alternative

### Key Libraries

**R**:
- `keras3` - Neural network training
- `glmnet` - Lasso regularization
- `randomForest` - Ensemble methods
- `caret` - ML framework
- `tidyverse` - Data manipulation

**Python**:
- `tensorflow/keras` - Deep learning
- `scikit-learn` - ML algorithms
- `statsmodels` - Statistical models
- `pandas` - Data analysis

**Statistical Methods**:
- Double/Debiased Machine Learning (DML)
- Cross-fitting for valid inference
- Neyman orthogonality conditions
- Partially Linear Models

---

## 📊 Results

All results are saved in respective `output/` folders:

- 📈 **Activation function comparisons** (PNG plots)
- 📉 **Learning rate experiments** (CSV tables)
- 🎯 **DML treatment effect estimates** (CSV results)
- 📊 **Cross-fitting vs. no cross-fitting comparisons**
- 🖼️ **Visualization plots** for predictions and residuals

**Output Files**:
- `nn_activation_metrics_r.csv` - Neural network performance metrics
- `dml_results_r.csv` - DML treatment effect estimates
- `dml_comparison.csv` - Cross-fitting vs. no cross-fitting comparison
- Various PNG plots for visual analysis

---

## 👥 Contributors

**Group 1** 

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Pennsylvania Reemployment Bonus dataset from [Journal of Applied Econometrics Data Archive](https://journaldata.zbw.eu/)
- Double Machine Learning methodology from Chernozhukov et al. (2018)
- Neural network fundamentals from Goodfellow, Bengio, and Courville (2016)

---

## 📚 References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal, 21(1), C1-C68.

2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT press.

3. Mullainathan, S., & Spiess, J. (2017). *Machine learning: an applied econometric approach*. Journal of Economic Perspectives, 31(2), 87-106.

---

<div align="center">
  
**[⬆ Back to Top](#-neural-networks--double-machine-learning)**

Made with 🧠 and ❤️ by Group 1

</div>
