# Math4AI Final Capstone

## From Linear Scores to a Single Hidden Layer: A Mathematical Study of Simple Learning Systems

---

**Organization:** National AI Center - AI Academy  
**Deadline:** March 30, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Questions and Motivation](#2-core-questions-and-motivation)
3. [Installation and Setup](#3-installation-and-setup)
4. [Project Structure](#4-project-structure)
5. [Technical Architecture](#5-technical-architecture)
6. [Implementation Details](#6-implementation-details)
7. [Datasets and Data](#7-datasets-and-data)
8. [Experiments and Results](#8-experiments-and-results)
9. [Mathematical Analysis and Derivatives](#9-mathematical-analysis-and-derivatives)
10. [Graphics and Visualization](#10-graphics-and-visualization)
11. [Evaluation Protocols](#11-evaluation-protocols)
12. [Team and Responsibilities](#12-team-and-responsibilities)
13. [Comprehensive Checklists](#13-comprehensive-checklists)
14. [Frequently Asked Questions](#15-frequently-asked-questions)

---

# 1. Project Overview

This capstone project is the final phase of the Math4AI program organized by **National AI Center - AI Academy**.

---

# 2. Core Questions and Motivation

## Scientific Question

The central question of this project is:

> **Question:** When does a one-hidden-layer non-linear classifier improve upon a linear decision boundary, and when is the added complexity unnecessary?

This question has practical significance because choosing the most complex model for every problem is not efficient. Sometimes a simple linear model is sufficient, while other times non-linear capability is essential.

## Hypotheses and Expectations

**Hypothesis 1:** On linearly separable data, both Softmax and NN should achieve the same performance, as NN's additional capacity is not activated here.

**Hypothesis 2:** On non-linearly separable data (e.g., Moons), NN gains significant advantage.

**Hypothesis 3:** As hidden width increases, the model can learn more complex boundaries, but very large width can lead to overfitting.


---

# 3. Installation and Setup

## System Requirements

| Requirement | Minimal | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| RAM | 4 GB | 8 GB |
| Storage | 1 GB | 2 GB |

Can work in any OS.

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Project
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# With pip
pip install -r necessity.txt

# Or with uv
uv sync
```

### 4. Run Verification Checks

Verify the installation is correct:

```bash
python main.py --experiment check
```

Expected output:
```
CHECKING IMPLEMENTATIONS...
[1/6] Checking DataLoader... [OK]
[2/6] Checking SoftmaxRegression... [OK]
[3/6] Checking SoftmaxRegression backward... [OK]
[4/6] Checking OneHiddenLayerNN... [OK]
[5/6] Checking OneHiddenLayerNN backward... [OK]
[6/6] Checking Trainer... [OK]

[OK] ALL IMPLEMENTATIONS COMPLETE!
```

## Running Experiments

### Run All Experiments Together

```bash
python main.py --experiment all
```

This command will automatically execute:
- Linear Gaussian experiment
- Moons experiment
- Digits experiment
- Capacity ablation
- Optimizer comparison
- Failure case analysis
- Track A
- 3D visualization Track A (optional)
- Track B (optional, can be modified in arguments)

### Run Individual Experiments

```bash
# Linear Gaussian only
python main.py --experiment linear_gaussian

# Moons only
python main.py --experiment moons

# Digits only
python main.py --experiment digits

# Ablations only
python main.py --experiment ablations
```

---

# 4. Project Structure

## Directory Structure

```
Project/
│
├── main.py                         # Main experiment script
├── pyproject.toml                  # Python project configuration
├── necessity.txt                   # Python package requirements
├── .python-version                # Python version
├── uv.lock                        # uv lock file
│
└── starter_pack/                   # Starter pack
    │
    ├── README.md                   # Project documentation
    │
    ├── src/                        # Main code directory
    │   ├── __init__.py             # Package init
    │   ├── models.py               # Softmax and NN models
    │   ├── optimizers.py           # SGD, Momentum, Adam
    │   ├── trainer.py               # Training loop
    │   ├── evaluation.py            # Evaluation tools
    │   ├── visualization.py         # Plot functions
    │   ├── data_loader.py          # Data loader
    │   └── logging_utils.py        # Logging tools
    │
    ├── data/                       # Dataset files
    │   ├── digits_data.npz         # Digits dataset
    │   ├── digits_split_indices.npz # Split indices
    │   ├── linear_gaussian.npz     # Linear synthetic dataset
    │   └── moons.npz               # Non-linear synthetic dataset
    │
    ├── scripts/                    # Helper scripts
    │   ├── generate_synthetic.py   # Synthetic data generator
    │   └── make_digits_split.py    # Split indices generator
    │
    ├── figures/                    # Output plots
    │
    ├── results/                    # Results directory
    │   ├── tables/                 # Tables
    │   ├── metrics/                # Metrics
    │   ├── statistics/             # Statistics
    │   └── logs/                   # Log files
    │
    ├── slides/                     # Presentation materials
    │
    └── report/                     # Report templates
```

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Coordination and execution of all experiments |
| `models.py` | Softmax and NN architectures |
| `optimizers.py` | Three optimization algorithms |
| `trainer.py` | Training loop and checkpointing |
| `evaluation.py` | Metrics and sanity checks |
| `visualization.py` | Plot and diagram functions |
| `data_loader.py` | Dataset loading |
| `logging_utils.py` | Logging utilities |

---

# 5. Technical Architecture

## Softmax Regression Architecture

Softmax Regression is a linear classifier with the following architecture:

```
Input (x) ──────► Linear Transform (Wx + b) ──────► Softmax ──────► Probability (p)
   d-dim                                                 k-classes     Σp = 1
```

**The Softmax Definition:**
$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \text{where} \quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}$$

**The Forward Pass Equation:**
$$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$$
$$\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$$

---

https://sebastianraschka.com/images/faq/softmax_regression/1.png




### Mathematical Description

**Forward Pass:**
```
Z = X @ W.T + b              # X: (n, d), W: (k, d), b: (k,)
                             # Z: (n, k) - raw scores (logits)

P = softmax(Z)               # P[i,j] = exp(Z[i,j]) / Σexp(Z[i,:])
                             # P: (n, k) - probability distribution
```

**Loss Function:**
```
L = -Σ Y[i] * log(P[i])      # Cross-entropy loss
```

## One-Hidden-Layer Neural Network Architecture

This architecture is a two-layer neural network: one hidden layer and one output layer.

```
Input (x) ──► Linear1 (W₁, b₁) ──► tanh ──► Linear2 (W₂, b₂) ──► Softmax ──► Probability
   d-dim        (d → h)              h-dim       (h → k)           k-classes
```

### Mathematical Description

**Forward Pass:**
```
Z₁ = X @ W₁.T + b₁           # Affine transformation
                              # X: (n, d), W₁: (h, d), b₁: (h,)
                              # Z₁: (n, h)

H = tanh(Z₁)                  # Hidden activations
                              # H: (n, h), range: (-1, 1)

Z₂ = H @ W₂.T + b₂            # Output transformation
                              # W₂: (k, h), b₂: (k,)
                              # Z₂: (n, k)

P = softmax(Z₂)                # Final probabilities
```

## Parameter Count Comparison

| Model | Parameters | Formula |
|-------|------------|---------|
| Softmax | k×d + k | W: (k,d), b: (k,) |
| NN (h=32, d=64, k=10) | h×d + h + k×h + k | 32×64 + 32 + 10×32 + 10 = 2218 |

NN's capacity is directly proportional to the number of parameters.

---

# 6. Implementation Details

## Models.py

This file contains two main classes: `SoftmaxRegression` and `OneHiddenLayerNN`.

### SoftmaxRegression Class

```python
class SoftmaxRegression:
    def __init__(self, input_dim, num_classes, learning_rate=0.05, reg_lambda=1e-4):
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros((num_classes,))
    
    def forward(self, X):
        logits = X @ self.W.T + self.b
        probabilities = softmax_stable(logits)
        return logits, probabilities
    
    def backward(self, X, Y, P):
        n = X.shape[0]
        dL_dS = (P - Y) / n
        grad_W = dL_dS.T @ X
        grad_b = np.sum(dL_dS, axis=0)
        return grad_W, grad_b
```

### OneHiddenLayerNN Class

```python
class OneHiddenLayerNN:
    def __init__(self, input_dim, hidden_dim, num_classes, 
                 learning_rate=0.05, reg_lambda=1e-4):
        # He initialization
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(num_classes, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros((num_classes,))
    
    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = tanh(Z1)
        S = H @ self.W2.T + self.b2
        P = softmax_stable(S)
        return {'Z1': Z1, 'H': H, 'S': S, 'P': P}
    
    def backward(self, X, Y, cache):
        dL_dS = (cache['P'] - Y) / n
        grad_W2 = dL_dS.T @ cache['H']
        grad_b2 = np.sum(dL_dS, axis=0)
        
        dL_dH = dL_dS @ self.W2
        dL_dZ1 = dL_dH * (1 - cache['H']**2)
        grad_W1 = dL_dZ1.T @ X
        grad_b1 = np.sum(dL_dZ1, axis=0)
        
        return grad_W1, grad_b1, grad_W2, grad_b2
```

## Optimizers.py

Three optimization algorithm implementations:

### SGD (Stochastic Gradient Descent)

```python
class SGD(Optimizer):
    def step(self, model, grads):
        if hasattr(model, 'W') and 'W' in grads:
            model.W = model.W - self.learning_rate * grads['W']
            model.b = model.b - self.learning_rate * grads['b']
```

### Momentum

```python
class Momentum(Optimizer):
    def step(self, model, grads):
        for param_name, grad in grads.items():
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(grad)
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] + grad
            )
            setattr(model, param_name, 
                   getattr(model, param_name) - self.learning_rate * self.velocity[param_name])
```

### Adam

```python
class Adam(Optimizer):
    def step(self, model, grads):
        self.t += 1
        for param_name, grad in grads.items():
            # First moment
            self.m[param_name] = self.beta1 * self.m[param_name] + (1-self.beta1) * grad
            # Second moment
            self.v[param_name] = self.beta2 * self.v[param_name] + (1-self.beta2) * (grad**2)
            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[param_name] / (1 - self.beta2**self.t)
            # Update
            setattr(model, param_name,
                   getattr(model, param_name) - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps))
```

## Trainer.py

Training loop structures:

```python
class Trainer:
    def train(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            # 1. Train epoch (minibatch loop)
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            
            # 2. Validation
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # 3. Checkpoint (save best model)
            if val_loss < self.best_val_loss:
                self.best_params = self.model.get_params()
                self.best_val_loss = val_loss
```

## Evaluation.py

Evaluation metrics:

```python
class Evaluator:
    def compute_metrics(self, model, X, y):
        P = self.predict_proba(model, X)
        y_pred = np.argmax(P, axis=1)
        
        return {
            'accuracy': np.mean(y_pred == y),
            'cross_entropy': -np.mean(np.log(P[np.arange(len(y)), y] + 1e-9)),
            'confidence': np.mean(np.max(P, axis=1)),
            'entropy': -np.mean(np.sum(P * np.log(P + 1e-9), axis=1))
        }
```

---

# 7. Datasets and Data

## Linear Gaussian Dataset

This synthetic dataset represents linearly separable data.

**Characteristics:**
- Number of samples: 400 (240 train / 80 val / 80 test)
- Feature dimension: 2
- Classes: 2
- Distribution: Each class follows Gaussian distribution

**Generation process:**
```python
# Class 0: μ = [-1, -1], σ = 0.5
# Class 1: μ = [1, 1], σ = 0.5
```

**Expected result:**
Both Softmax and NN should achieve ~100% accuracy since this data is linearly separable.

## Moons Dataset

A non-linear dataset in the shape of two crescents.

**Characteristics:**
- Number of samples: 400 (240 train / 80 val / 80 test)
- Feature dimension: 2
- Classes: 2
- Shape: Two circular curves

**Expected result:**
- Softmax: ~80% accuracy (limited)
- NN (h=32): ~95%+ accuracy

## Digits Dataset

Scikit-learn's handwritten digits dataset.

**Characteristics:**
- Number of samples: 1797
- Split: 1074 train / 355 val / 368 test
- Feature dimension: 64 (8×8 pixels)
- Classes: 10 (digits 0-9)

**Data preparation:**
```python
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data          # (1797, 64)
y = digits.target       # (1797,)
```

---

# 8. Experiments and Results

## Core Experiments

### 1. Linear Gaussian

**Objective:** Compare both models' performance on linear data.

**Methodology:**
1. Train both models with same hyperparameters
2. Plot decision boundaries
3. Record accuracy and loss metrics

**Expected result:**
| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Softmax | ~100% | ~0.01 |
| NN (h=32) | ~100% | ~0.01 |

**Interpretation:** Both models achieve the same performance, demonstrating that additional non-linearity is unnecessary for linearly separable data.

### 2. Moons

**Objective:** Demonstrate NN's advantage on non-linear data.

**Methodology:**
1. Train both models on Moons dataset
2. Visualize decision boundaries
3. Observe how models draw different boundaries

**Expected result:**
| Model | Test Accuracy | Decision Boundary |
|-------|--------------|-------------------|
| Softmax | ~80% | Linear (straight) |
| NN (h=32) | ~95%+ | Non-linear (curved) |

**Interpretation:** Softmax can only draw straight lines, so it cannot fit the curvature of moons. NN learns curved boundaries through tanh activation.

### 3. Digits

**Objective:** Compare models on 10-class classification.

**Methodology:**
1. Train both models
2. Conduct repeated evaluation with 5 different seeds
3. Perform confusion matrix analysis

**Expected result:**
| Model | Accuracy (mean ± std) | 95% CI |
|-------|----------------------|--------|
| Softmax | ~87% ± 2% | [85%, 89%] |
| NN (h=32) | ~95% ± 1% | [94%, 96%] |

**Confused digits:** Most commonly confused pairs are usually 1↔7, 3↔8, 4↔9.

## Ablation Studies

### Capacity Ablation (Moons)

**Objective:** Learn how hidden width affects classification capability.

**Methodology:**
Change hidden width to {2, 8, 32} on Moons dataset and observe each one's decision boundary.

**Results:**
| Hidden Width | Decision Boundary | Accuracy | Overfitting? |
|-------------|-------------------|----------|-------------|
| 2 | Very simple, almost linear | ~85% | No |
| 8 | Medium complexity | ~92% | No |
| 32 | Very complex, curved | ~96% | No (due to regularization) |

**Finding:** As hidden width increases, the model can learn more complex patterns.

### Optimizer Ablation (Digits)

**Objective:** Compare convergence speed of different optimizers.

**Configuration:**
| Optimizer | Learning Rate | Momentum | β₁ | β₂ |
|-----------|--------------|----------|-----|-----|
| SGD | 0.05 | - | - | - |
| Momentum | 0.05 | 0.9 | - | - |
| Adam | 0.001 | - | 0.9 | 0.999 |

**Results:**
| Optimizer | Convergence Speed | Final Accuracy | Final Loss |
|-----------|----------------|---------------|------------|
| SGD | Slowest | ~95% | ~0.15 |
| Momentum | Medium | ~95% | ~0.15 |
| Adam | Fastest | ~95% | ~0.15 |

**Finding:** While final performance is similar, Adam converges in fewer epochs.

### Failure Case Analysis

**Objective:** Analyze the situation where the model fails.

**Scenario:** Moons dataset with hidden width = 1.

**Result:**
| Metric | Value |
|--------|-------|
| Accuracy | ~75% |
| Loss | ~0.6 |

**Analysis:**
- Hidden width = 1 means there is only one hidden neuron
- A single neuron can only learn one linear function
- Since Moons is non-linear, the model fails

**Lesson:** Without sufficient capacity, the model cannot learn complex patterns.

---

# 9. Mathematical Analysis and Derivatives

## Softmax Function

Softmax converts raw scores (logits) to probabilities:

$$P_j = \frac{e^{s_j}}{\sum_{l=1}^{k} e^{s_l}}$$

**Properties:**
1. Outputs are positive: $P_j > 0$
2. Sum to one: $\sum_j P_j = 1$
3. Smooths maximum selection

**Numerical stability problem:**

If $s_j$ is very large, $e^{s_j}$ can overflow:

```python
# Stable implementation
def softmax_stable(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
```

## Cross-Entropy Loss

Cross-entropy measures the "distance" between two distributions:

$$L_{CE} = -\sum_{i} y_i \log(p_i)$$

Where $y_i$ is the true probability (one-hot) and $p_i$ is the predicted probability.

## Backpropagation Derivatives

### Softmax + Cross-Entropy Derivative

Due to a special mathematical property:

$$\frac{\partial L}{\partial s_j} = p_j - y_j$$

This is the simplified form of "softmax derivative" and "cross-entropy derivative".

### NN Backpropagation Step-by-Step

**Step 1:** Output sensitivity

$$\frac{\partial L}{\partial S} = \frac{1}{n}(P - Y)$$

**Step 2:** Gradients for W₂ and b₂

$$\frac{\partial L}{\partial W_2} = \left(\frac{\partial L}{\partial S}\right)^T \cdot H$$

$$\frac{\partial L}{\partial b_2} = \sum_i \frac{\partial L}{\partial S_i}$$

**Step 3:** Backpropagate to hidden layer

First, gradient with respect to H:

$$\frac{\partial L}{\partial H} = \frac{\partial L}{\partial S} \cdot W_2^T$$

Then, gradient with respect to Z₁ (chain rule + tanh derivative):

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial H} \odot (1 - H^2)$$

**Step 4:** Gradients for W₁ and b₁

$$\frac{\partial L}{\partial W_1} = \left(\frac{\partial L}{\partial Z_1}\right)^T \cdot X$$

$$\frac{\partial L}{\partial b_1} = \sum_i \frac{\partial L}{\partial Z_{1i}}$$

## Tanh Activation

**Formula:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Properties:**
- Range: $(-1, 1)$
- Zero-centered
- Less vanishing gradient

**Derivative:**
$$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$$

This property makes backpropagation computationally efficient, as $\tanh(Z)$ has already been computed.

## L2 Regularization

Total loss:

$$L_{total} = L_{CE} + \frac{\lambda}{2}\sum_\theta \theta^2$$

Gradient:

$$\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{CE}}{\partial W} + \lambda W$$

This creates a "weight decay" effect - pulls weights toward zero.

---

# 10. Graphics and Visualization

## Decision Boundary Plot

This plot visualizes the boundary learned by the model.

**How it works:**
1. Create a 200×200 = 40,000 point grid
2. Predict class for each point
3. Color points by class

**Interpretation:**
- **Linear boundary:** Characteristic of Softmax
- **Curved boundary:** Characteristic of NN

## Training Dynamics Plot

Changes in loss and accuracy over epochs.

**How to read:**
- Train and validation curves should be close
- If train decreases but validation increases → overfitting
- If both are high → underfitting

## Confusion Matrix

Number of correct and incorrect predictions for each class.

**Interpretation:**
- Diagonal: Correct predictions
- Off-diagonal: Confused classes

## Confidence vs Accuracy Plot (Track B)

Checks the model's self-confidence level.

**Good calibration:** If model is 80% confident, it should be 80% accurate.

---

# 11. Evaluation Protocols

## Sanity Checks

### 1. Gradient Check

Compares analytical gradients with numerical gradients:

```python
def gradient_check(model, X, Y, epsilon=1e-5):
    analytical_grad = model.backward(X, Y, cache)
    numerical_grad = compute_numerical_grad(model, X, Y, epsilon)
    
    relative_error = |analytical - numerical| / (|analytical| + |numerical|)
    return relative_error < 1e-5
```

### 2. Probability Sum Check

Verifies that softmax outputs sum to one:

```python
def check_probability_sum(P):
    sums = np.sum(P, axis=1)
    return np.allclose(sums, 1.0)
```

### 3. NaN/Inf Check

No NaN or Inf in model outputs:

```python
def check_nan_inf(model, X):
    cache = model.forward(X)
    return not (np.any(np.isnan(cache)) or np.any(np.isinf(cache)))
```

## Repeated Seed Evaluation

Repeated evaluation with 5 different seeds to ensure statistical significance:

**Protocol:**
1. For each seed, train model from scratch
2. Record test accuracy and loss
3. Calculate mean and standard deviation

**95% Confidence Interval:**
$$CI = \mu \pm t_{0.975, n-1} \cdot \frac{\sigma}{\sqrt{n}}$$

Where $t_{0.975, 4} = 2.776$ (t-critical value for n=5).

---

# 12. Team and Responsibilities

This project is executed as a team. Below are the team members and their areas of responsibility.

## Team Members

| Team Member | Role | Area of Responsibility |
|------------|------|------------------------|
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |

## Role Descriptions

| Role | Description | Key Responsibilities |
|------|-------------|---------------------|
| **Softmax Engineer** | Softmax regression model implementation | `models.py` forward/backward, loss function |
| **Neural Network Engineer** | NN model implementation | `models.py` OneHiddenLayerNN, backpropagation |
| **Experiment Lead** | Experiment planning and execution | `main.py`, `trainer.py`, scheduling |
| **Visualization Expert** | Graphics and diagram preparation | `visualization.py`, figures |
| **Report Author** | Written report preparation | PDF report, documentation |
| **Presentation Lead** | Technical presentation preparation | Slides, Q&A preparation |

## GitHub Branch Strategy

```
main (production branch)
│
├── feature/softmax-implementation
├── feature/neural-network
├── feature/optimizers
├── feature/training-loop
├── feature/visualization
├── feature/experiments
├── feature/report
├── feature/slides
│
└── (each branch managed by respective team member)
```

---

# 13. Comprehensive Checklists

## Starter Pack Integrity Check

- [ ] `starter_pack/data/digits_data.npz` file exists
- [ ] `starter_pack/data/digits_split_indices.npz` file exists
- [ ] `starter_pack/data/linear_gaussian.npz` file exists
- [ ] `starter_pack/data/moons.npz` file exists
- [ ] `scripts/generate_synthetic.py` script works
- [ ] `scripts/make_digits_split.py` script works
- [ ] No model implementation code exists in starter pack
- [ ] All data file shapes are correct

## Setup Checklist

Follow these steps when setting up the system.

- [ ] Python 3.10+ installed
- [ ] Virtual environment created (recommended)
- [ ] Dependencies installed (`pip install -r necessity.txt`)
- [ ] `python main.py --experiment check` runs successfully
- [ ] All src files exist as skeletons
- [ ] All .npz data files are readable
- [ ] Git repository properly initialized
- [ ] Feature branches created

## Implementation Checklist

### Softmax Regression
- [ ] `forward()` method correctly computes softmax
- [ ] `backward()` method correctly computes gradients
- [ ] L2 regularization is applied
- [ ] Gradient check passes
- [ ] Probability sum check passes

### Neural Network
- [ ] `forward()` method performs complete forward pass
- [ ] `backward()` method performs backpropagation
- [ ] Tanh activation is correctly computed
- [ ] Tanh derivative is correctly computed (`1 - H²`)
- [ ] He initialization is used
- [ ] Gradient check passes

### Optimizers
- [ ] SGD works correctly
- [ ] Momentum accumulates velocity
- [ ] Adam performs bias correction
- [ ] Each optimizer has reset() method

### Trainer
- [ ] Minibatch creation is correct
- [ ] One-hot encoding is correct
- [ ] Checkpointing saves best model
- [ ] Training history is recorded
- [ ] Validation monitoring works

## Experiment Checklist

### Linear Gaussian
- [ ] Decision boundary plot created
- [ ] Both models trained
- [ ] Results recorded

### Moons
- [ ] Decision boundary plot created
- [ ] Both models trained
- [ ] NN's curved boundary demonstrated

### Digits
- [ ] Both models trained
- [ ] Confusion matrix created
- [ ] 5-seed repeated evaluation performed
- [ ] 95% CI calculated

### Capacity Ablation
- [ ] Hidden width {2, 8, 32} compared
- [ ] Decision boundaries compared
- [ ] Accuracy table created

### Optimizer Study
- [ ] SGD, Momentum, Adam compared
- [ ] Convergence speeds compared
- [ ] Training curves plotted

### Failure Case Analysis
- [ ] One failure case analyzed
- [ ] Cause explained
- [ ] Results drawn


# 14. Frequently Asked Questions

## Q: Which track should I choose?

- **Track A (PCA/SVD):** If you are interested in mathematical PCA analysis
- **Track B (Confidence):** If you are interested in model calibration and reliability

---

## Q: Why tanh, not ReLU?

**A:** The required activation is tanh. This choice is deliberate and offers the following advantages:

1. **Zero-centered:** Tanh outputs are in $(-1, 1)$ range and centered around zero, providing more stable gradient flow.

2. **Less vanishing gradient:** ReLU has zero gradient when $z < 0$, causing "dead neurons" problem. Tanh still has gradient present.

3. **More stable gradient flow:** The smooth nature of tanh makes optimization more stable.

4. **Mathematical simplicity:** Tanh derivative is easily computed as `1 - tanh²(z)`, making backpropagation efficient.

---

## Q: What if overfitting occurs?

**A:** When overfitting occurs, apply the following methods:

1. **Increase L2 Regularization:** Increase the `reg_lambda` parameter (e.g., 1e-4 → 1e-3). This pulls weights toward zero.

2. **Apply Early Stopping:** If validation loss starts increasing, stop training and restore the best checkpoint.

3. **Reduce model capacity:** Decrease hidden width (e.g., 32 → 16).

4. **Collect more data:** Apply data augmentation techniques or add new data.

---

## Q: Why does Adam converge faster than SGD?

**A:** Adam converges faster because it applies individual learning rates for each parameter:

**Adam's advantages:**
- **First moment (momentum):** Retains direction of past gradients, causing acceleration in the correct direction.
- **Second moment (adaptive lr):** Each parameter has its own learning rate:
  - Small lr for large gradients (for stability)
  - Large lr for small gradients (for speed)
- **Bias correction:** Since momentum and velocity start at zero in early steps, this correction accelerates convergence in initial epochs.

**SGD's limitations:**
- Uses the same learning rate for all parameters
- Oscillates in narrow valleys
- Requires more epochs

---

## Q: How to choose hidden width?

**A:** Hidden width selection is done through capacity ablation technique:

**Step 1: Start with small width**
For example, start with width = 2 and observe what the model learns.

**Step 2: Increase as accuracy improves**
Increase width as {2, 4, 8, 16, 32, 64}. Observe accuracy at each step.

**Step 3: Find the overfitting point**
If training accuracy increases but validation accuracy decreases, this indicates overfitting. Stop at this point.

**Practical recommendations:**
- Simple problems: width = 8-16 is sufficient
- Medium complexity: width = 32 is optimal
- Very complex problems: width = 64-128 (but overfitting risk)

**General rule:** Choose the smallest width that still achieves good performance.

---

## Q: How many epochs should I train?

**A:** The number of epochs depends on problem and model complexity:

**General recommendations:**
- **With early stopping:** The best method - train until validation loss stops improving
- **Fixed epochs:** 200 epochs is sufficient in most cases
- **Monitor:** Check validation metrics every 10 epochs

**Overfitting signs:**
- Training loss decreases but validation loss increases
- Training accuracy increases but validation accuracy decreases

---

## Q: What are sanity checks for?

**A:** Sanity checks ensure implementation correctness:

1. **Gradient Check:** Compares analytical gradients with numerical gradients. Relative error should be < 1e-5.

2. **Probability Sum Check:** Softmax outputs should sum to 1. If not, there is numerical instability.

3. **NaN/Inf Check:** No NaN or Inf in model outputs. This indicates gradient explosion or other problems.

---

---

# 🇦🇿 AZƏRBAYCAN DİLİNDƏ

---

# 1. Layihəyə Baxış

Bu capstone layihəsi **National AI Center - AI Academy** tərəfindən təşkil olunan Math4AI proqramının final mərhələsidir. Layihənin əsas məqsədi tələbələrə xətti və qeyri-xətti klassifikasiya üsullarını dərin riyazi anlayışla birlikdə praktiki olaraq tətbiq etmə bacarığı qazandırmaqdır.

## Layihənin Məqsədləri

Bu layihə aşağıdakı bacarıqları inkişaf etdirməyi hədəfləyir:

1. **Riyazi Anlayış:** Xətti cəbr, ehtimal nəzəriyyəsi və optimizasiya üsullarının maşın öyrənməsinə tətbiqini başa düşmək
2. **Proqramlaşdırma Bacarığı:** NumPy ilə sıfırdan neural şəbəkələr qurmaq və train etmək
3. **Elmi Metodologiya:** Eksperimental dizayn, nəticələrin təhlili və interpretasiya bacarığı
4. **Ünsiyyət:** Texniki hesabat yazmaq və elmi təqdimat hazırlamaq

## Nə Öyrənəcəksiniz?

Bu layihəni tamamladıqdan sonra siz aşağıdakı mövzuları dərindən başa düşəcəksiniz:

- Softmax funksiyası və cross-entropy itkisi necə işləyir?
- Backpropagation alqoritmi riyazi olaraq necə işləyir?
- Tərs funksiya teoremi (derivative chain rule) necə tətbiq olunur?
- Tanh aktivasiyası niyə məhz seçilir?
- Hidden layer width klassifikasiya qabiliyyətinə necə təsir edir?
- Fərqli optimizasiya alqoritmləri (SGD, Momentum, Adam) necə fərqlənir?

---

# 2. Əsas Suallar və Motivasiya

## Elmi Sual

Bu layihənin mərkəzində aşağıdakı sual dayanır:

> **Sual:** Bir gizli qatlı qeyri-xətti klassifikator xətti qərar qaydasına nə vaxt yaxşılaşdırır və nə vaxt əlavə mürəkkəblik lazımsızdır?

Bu sual praktik əhəmiyyətə malikdir, çünki real dünyada hər problem üçün ən mürəkkəb model seçmək effektiv deyil. Bəzən sadə xətti model kifayətdir, bəzən isə qeyri-xətti qabiliyyət vacibdir.

## Hipotezlər və Gözləntilər

**Hipotez 1:** Xətti ayrıla bilən verilənlərdə həm Softmax, həm də NN eyni performans göstərməlidir, çünki NN-in əlavə tutumluğu burada aktivləşmir.

**Hipotez 2:** Qeyri-xətti ayrıla bilən verilənlərdə (məsələn, Moons) NN əhəmiyyətli üstünlük əldə edir.

**Hipotez 3:** Hidden width artdıqca model daha mürəkkəb sərhədləri öyrənə bilir, lakin çox böyük width overfitting-ə səbəb ola bilər.

## Real Dünya Əlaqəsi

Bu tədqiqat praktiki ssenarilərdə birbaşa tətbiq tapır:

- **Email spam kateqoriyalama:** Əgər spam və normal email-lər xətti ayrıla bilirsə, sadə model kifayətdir
- **Səs tanıma:** Mürəkkəb akustik nümunələr üçün çoxqatlı şəbəkələr lazımdır
- **Tibbi diaqnostika:** Xətti olmayan əlaqələri aşkar etmək üçün qeyri-xətti modellər üstünlük təşkil edir

---

# 3. Quraşdırma və İşə Salma

## Sistem Tələbləri

| Tələb | Minimal | Tövsiyə olunan |
|-------|---------|----------------|
| Python | 3.10+ | 3.12+ |
| RAM | 4 GB | 8 GB |
| Yaddaş | 1 GB | 2 GB |
| Əməliyyat sistemi | Windows/Linux/macOS | Windows/Linux/macOS |

## Addım-Adım Quraşdırma

### 1. Repository-ni Klonlayın

```bash
git clone <repository-url>
cd Project
```

### 2. Virtual Mühit Yaradın (Tövsiyə olunur)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependencies Quraşdırın

```bash
# pip ilə
pip install -r necessity.txt

# və ya uv ilə
uv sync
```

### 4. Yoxlamaları İşə Salın

Quraşdırmanın düzgün olduğunu yoxlayın:

```bash
python main.py --experiment check
```

Gözlənilən nəticə:
```
CHECKING IMPLEMENTATIONS...
[1/6] Checking DataLoader... [OK]
[2/6] Checking SoftmaxRegression... [OK]
[3/6] Checking SoftmaxRegression backward... [OK]
[4/6] Checking OneHiddenLayerNN... [OK]
[5/6] Checking OneHiddenLayerNN backward... [OK]
[6/6] Checking Trainer... [OK]

[OK] ALL IMPLEMENTATIONS COMPLETE!
```

## Eksperimentləri İşə Salma

### Bütün Eksperimentləri Bir Yerdə İşə Salın

```bash
python main.py --experiment all
```

Bu əmr aşağıdakıları avtomatik olaraq icra edəcək:
- Linear Gaussian eksperimenti
- Moons eksperimenti
- Digits eksperimenti
- Capacity ablasiya
- Optimizer müqayisəsi
- Uğursuzluq halı təhlili

### Ayrı-Ayrı Eksperimentlər

```bash
# Yalnız Linear Gaussian
python main.py --experiment linear_gaussian

# Yalnız Moons
python main.py --experiment moons

# Yalnız Digits
python main.py --experiment digits

# Yalnız Ablasiyalar
python main.py --experiment ablations
```

---

# 4. Layihə Strukturu

## Qovluq Strukturu

```
Project/
│
├── main.py                         # Əsas eksperiment skripti
├── pyproject.toml                  # Python layihə konfiqurasiyası
├── necessity.txt                   # Python paket tələbləri
├── .python-version                # Python versiyası
├── uv.lock                        # uv lock faylı
│
└── starter_pack/                   # Başlanğıc paket
    │
    ├── README.md                   # Layihə dokumentasiyası
    │
    ├── src/                        # Əsas kod qovluğu
    │   ├── __init__.py             # Package init
    │   ├── models.py               # Softmax və NN modelləri
    │   ├── optimizers.py           # SGD, Momentum, Adam
    │   ├── trainer.py               # Training loop
    │   ├── evaluation.py            # Qiymətləndirmə alətləri
    │   ├── visualization.py         # Qrafik funksiyaları
    │   ├── data_loader.py          # Verilənlər yükləyicisi
    │   └── logging_utils.py        # Logging alətləri
    │
    ├── data/                       # Dataset faylları
    │   ├── digits_data.npz         # Rəqəmlər dataseti
    │   ├── digits_split_indices.npz # Split göstərişləri
    │   ├── linear_gaussian.npz     # Xətti sintektik dataset
    │   └── moons.npz               # Qeyri-xətti sintektik dataset
    │
    ├── scripts/                    # Köməkçi skriptlər
    │   ├── generate_synthetic.py   # Sintektik verilən yaradıcısı
    │   └── make_digits_split.py    # Split indices generator
    │
    ├── figures/                    # Çıxış qrafikləri
    │
    ├── results/                    # Nəticələr qovluğu
    │   ├── tables/                 # Cədvəllər
    │   ├── metrics/                # Metriklər
    │   ├── statistics/             # Statistikalar
    │   └── logs/                   # Log faylları
    │
    ├── slides/                     # Prezentasiya materialları
    │
    └── report/                     # Hesabat şablonları
```

## Fayl Təsvirləri

| Fayl | Funksiya |
|------|----------|
| `main.py` | Bütün eksperimentlərin əlaqələndirilməsi və icrası |
| `models.py` | Softmax və NN arxitekturaları |
| `optimizers.py` | Üç optimizasiya alqoritmi |
| `trainer.py` | Training loop və checkpointing |
| `evaluation.py` | Metriklər və sanity checks |
| `visualization.py` | Qrafik və diaqram funksiyaları |
| `data_loader.py` | Dataset yükləmə |
| `logging_utils.py` | Nəticələrin saxlanması |

---

# 5. Texniki Arxitektura

## Softmax Regression Arxitekturası

Softmax Regression - bu xətti klassifikatordur. Onun arxitekturası aşağıdakı kimidir:

```
Input (x) ──────► Linear Transform (Wx + b) ──────► Softmax ──────► Probability (p)
   d-dim                                                 k-sinif      Σp = 1
```

### Riyazi Təsvir

**Forward Pass:**
```
Z = X @ W.T + b              # X: (n, d), W: (k, d), b: (k,)
                             # Z: (n, k) - raw scores (logits)

P = softmax(Z)               # P[i,j] = exp(Z[i,j]) / Σexp(Z[i,:])
                             # P: (n, k) - probability distribution
```

**Loss Function:**
```
L = -Σ Y[i] * log(P[i])      # Cross-entropy loss
```

## One-Hidden-Layer Neural Network Arxitekturası

Bu arxitektura iki qatlı neural şəbəkədir: bir gizli qat və bir output qat.

```
Input (x) ──► Linear1 (W₁, b₁) ──► tanh ──► Linear2 (W₂, b₂) ──► Softmax ──► Probability
   d-dim        (d → h)              h-dim       (h → k)           k-sinif
```

### Riyazi Təsvir

**Forward Pass:**
```
Z₁ = X @ W₁.T + b₁           # Affine transformasiya
                              # X: (n, d), W₁: (h, d), b₁: (h,)
                              # Z₁: (n, h)

H = tanh(Z₁)                  # Gizli aktivasiyalar
                              # H: (n, h), range: (-1, 1)

Z₂ = H @ W₂.T + b₂            # Output transformasiya
                              # W₂: (k, h), b₂: (k,)
                              # Z₂: (n, k)

P = softmax(Z₂)                # Final probabilities
```

## Parametr Sayı Müqayisəsi

| Model | Parametrlər | Formula |
|-------|-------------|---------|
| Softmax | k×d + k | W: (k,d), b: (k,) |
| NN (h=32, d=64, k=10) | h×d + h + k×h + k | 32×64 + 32 + 10×32 + 10 = 2218 |

NN-in tutumluğu (capacity) parametrlərin sayı ilə düz mütənasibdir.

---

# 6. İmplementasiya Detalları

## Models.py

Bu fayl iki əsas class ehtiva edir: `SoftmaxRegression` və `OneHiddenLayerNN`.

### SoftmaxRegression Class

```python
class SoftmaxRegression:
    def __init__(self, input_dim, num_classes, learning_rate=0.05, reg_lambda=1e-4):
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros((num_classes,))
    
    def forward(self, X):
        logits = X @ self.W.T + self.b
        probabilities = softmax_stable(logits)
        return logits, probabilities
    
    def backward(self, X, Y, P):
        n = X.shape[0]
        dL_dS = (P - Y) / n
        grad_W = dL_dS.T @ X
        grad_b = np.sum(dL_dS, axis=0)
        return grad_W, grad_b
```

### OneHiddenLayerNN Class

```python
class OneHiddenLayerNN:
    def __init__(self, input_dim, hidden_dim, num_classes, 
                 learning_rate=0.05, reg_lambda=1e-4):
        # He initialization
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(num_classes, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.b2 = np.zeros((num_classes,))
    
    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = tanh(Z1)
        S = H @ self.W2.T + self.b2
        P = softmax_stable(S)
        return {'Z1': Z1, 'H': H, 'S': S, 'P': P}
    
    def backward(self, X, Y, cache):
        dL_dS = (cache['P'] - Y) / n
        grad_W2 = dL_dS.T @ cache['H']
        grad_b2 = np.sum(dL_dS, axis=0)
        
        dL_dH = dL_dS @ self.W2
        dL_dZ1 = dL_dH * (1 - cache['H']**2)
        grad_W1 = dL_dZ1.T @ X
        grad_b1 = np.sum(dL_dZ1, axis=0)
        
        return grad_W1, grad_b1, grad_W2, grad_b2
```

## Optimizers.py

Üç optimizasiya alqoritmi implementasıyası:

### SGD (Stochastic Gradient Descent)

```python
class SGD(Optimizer):
    def step(self, model, grads):
        if hasattr(model, 'W') and 'W' in grads:
            model.W = model.W - self.learning_rate * grads['W']
            model.b = model.b - self.learning_rate * grads['b']
```

### Momentum

```python
class Momentum(Optimizer):
    def step(self, model, grads):
        for param_name, grad in grads.items():
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(grad)
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] + grad
            )
            setattr(model, param_name, 
                   getattr(model, param_name) - self.learning_rate * self.velocity[param_name])
```

### Adam

```python
class Adam(Optimizer):
    def step(self, model, grads):
        self.t += 1
        for param_name, grad in grads.items():
            # First moment
            self.m[param_name] = self.beta1 * self.m[param_name] + (1-self.beta1) * grad
            # Second moment
            self.v[param_name] = self.beta2 * self.v[param_name] + (1-self.beta2) * (grad**2)
            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[param_name] / (1 - self.beta2**self.t)
            # Update
            setattr(model, param_name,
                   getattr(model, param_name) - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps))
```

## Trainer.py

Training loop-ları aşağıdakı struktura malikdir:

```python
class Trainer:
    def train(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            # 1. Train epoch (minibatch loop)
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            
            # 2. Validation
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # 3. Checkpoint (ən yaxşı modeli saxla)
            if val_loss < self.best_val_loss:
                self.best_params = self.model.get_params()
                self.best_val_loss = val_loss
```

## Evaluation.py

Qiymətləndirmə metrikləri:

```python
class Evaluator:
    def compute_metrics(self, model, X, y):
        P = self.predict_proba(model, X)
        y_pred = np.argmax(P, axis=1)
        
        return {
            'accuracy': np.mean(y_pred == y),
            'cross_entropy': -np.mean(np.log(P[np.arange(len(y)), y] + 1e-9)),
            'confidence': np.mean(np.max(P, axis=1)),
            'entropy': -np.mean(np.sum(P * np.log(P + 1e-9), axis=1))
        }
```

---

# 7. Dataset-lər və Verilənlər

## Linear Gaussian Dataset

Bu sintektik dataset xətti ayrıla bilən verilənləri təmsil edir.

**Xüsusiyyətləri:**
- Nümunə sayı: 400 (240 train / 80 val / 80 test)
- Xüsusiyyət ölçüsü: 2
- Siniflər: 2
- Paylanma: Hər sinif qauss paylanmasına malikdir

**Yaradılma prosesi:**
```python
# Class 0: μ = [-1, -1], σ = 0.5
# Class 1: μ = [1, 1], σ = 0.5
```

**Gözlənilən nəticə:**
Həm Softmax, həm də NN ~100% accuracy əldə etməlidir, çünki bu verilənlər xətti ayrıla biləndir.

## Moons Dataset

İki aypara formasında olan qeyri-xətti dataset.

**Xüsusiyyətləri:**
- Nümunə sayı: 400 (240 train / 80 val / 80 test)
- Xüsusiyyət ölçüsü: 2
- Siniflər: 2
- Forma: İki dəyirmi əyri

**Gözlənilən nəticə:**
- Softmax: ~80% accuracy (məhdud)
- NN (h=32): ~95%+ accuracy

## Digits Dataset

Scikit-learn-in əl yazısı rəqəmlər dataseti.

**Xüsusiyyətləri:**
- Nümunə sayı: 1797
- Bölmə: 1074 train / 355 val / 368 test
- Xüsusiyyət ölçüsü: 64 (8×8 piksel)
- Siniflər: 10 (rəqəmlər 0-9)

**Verilənlərin hazırlanması:**
```python
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data          # (1797, 64)
y = digits.target       # (1797,)
```

---

# 8. Eksperimentlər və Nəticələr

## Əsas Eksperimentlər

### 1. Linear Gaussian

**Məqsəd:** Xətti verilənlərdə hər iki modelin performansını müqayisə etmək.

**Metodologiya:**
1. Hər iki modeli eyni hyperparametrlərlə train edin
2. Decision boundary qrafiklərini çəkin
3. Accuracy və loss metriklərini qeyd edin

**Gözlənilən nəticə:**
| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Softmax | ~100% | ~0.01 |
| NN (h=32) | ~100% | ~0.01 |

**Nəticə interpretasiyası:** Hər iki model eyni performans göstərir, bu onu göstərir ki, xətti ayrıla bilən verilənlərdə əlavə qeyri-xəttilik lazım deyil.

### 2. Moons

**Məqsəd:** Qeyri-xətti verilənlərdə NN-in üstünlüyünü nümayiş etdirmək.

**Metodologiya:**
1. Hər iki modeli Moons datasetində train edin
2. Decision boundary-ləri vizualizasiya edin
3. Modellərin fərqli sərhədlər çəkməsini müşahidə edin

**Gözlənilən nəticə:**
| Model | Test Accuracy | Decision Boundary |
|-------|--------------|-------------------|
| Softmax | ~80% | Xətti (düz) |
| NN (h=32) | ~95%+ | Qeyri-xətti (əyri) |

**Nəticə interpretasiyası:** Softmax yalnız düz xətt çəkə bildiyi üçün moons-un əyriliyinə uyğunlaşa bilmir. NN isə tanh aktivasiyası sayəsində əyri sərhədlər öyrənir.

### 3. Digits

**Məqsəd:** 10 sinifli klassifikasiyada modellərin müqayisəsi.

**Metodologiya:**
1. Hər iki modeli train edin
2. 5 fərqli seed ilə təkrar qiymətləndirmə aparın
3. Confusion matrix analizi edin

**Gözlənilən nəticə:**
| Model | Accuracy (mean ± std) | 95% CI |
|-------|----------------------|--------|
| Softmax | ~87% ± 2% | [85%, 89%] |
| NN (h=32) | ~95% ± 1% | [94%, 96%] |

**Qarışdırılan rəqəmlər:** Ən çox qarışdırılan cütlər adətən 1↔7, 3↔8, 4↔9 olur.

## Ablasiya Tədqiqatları

### Capacity Ablasiyası (Moons)

**Məqsəd:** Hidden width-in klassifikasiya qabiliyyətinə təsirini öyrənmək.

**Metodologiya:**
Moons datasetində hidden width-i {2, 8, 32} olaraq dəyişdirin və hər birinin decision boundary-sini müşahidə edin.

**Nəticələr:**
| Hidden Width | Decision Boundary | Accuracy | Overfitting? |
|-------------|-------------------|----------|-------------|
| 2 | Çox sadə, demək olar xətti | ~85% | Yox |
| 8 | Orta mürəkkəblik | ~92% | Yox |
| 32 | Çox mürəkkəb, əyri | ~96% | Yox (regularization sayəsində) |

**Tapıntı:** Hidden width artdıqca model daha mürəkkəb nümunələri öyrənə bilir.

### Optimizer Ablasiyası (Digits)

**Məqsəd:** Fərqli optimizatorların yığılma sürətini müqayisə etmək.

**Konfiqurasiya:**
| Optimizer | Learning Rate | Momentum | β₁ | β₂ |
|-----------|--------------|----------|-----|-----|
| SGD | 0.05 | - | - | - |
| Momentum | 0.05 | 0.9 | - | - |
| Adam | 0.001 | - | 0.9 | 0.999 |

**Nəticələr:**
| Optimizer | Yığılma sürəti | Final Accuracy | Final Loss |
|-----------|----------------|---------------|------------|
| SGD | Ən yavaş | ~95% | ~0.15 |
| Momentum | Orta | ~95% | ~0.15 |
| Adam | Ən sürətli | ~95% | ~0.15 |

**Tapıntı:** Final performans bənzər olsa da, Adam daha az epoch-da yığılır.

### Uğursuzluq Halı Təhlili

**Məqsəd:** Modelin uğursuz olduğu şəraiti təhlil etmək.

**Ssenari:** Hidden width = 1 ilə Moons dataseti.

**Nəticə:**
| Metric | Dəyər |
|--------|-------|
| Accuracy | ~75% |
| Loss | ~0.6 |

**Təhlil:**
- Hidden width = 1 deməkdir ki, yalnız bir hidden neuron var
- Bir neuron yalnız bir xətti funksiya öyrənə bilər
- Moons isə qeyri-xəttidir, buna görə də model uğursuz olur

**Dərs:** Kifayət qədər tutumluq (capacity) olmadan model mürəkkəb nümunələri öyrənə bilmir.

---

# 9. Riyazi Analiz və Törəmələr

## Softmax Funksiyası

Softmax funksiyası raw score-ları (logits) ehtimallara çevirir:

$$P_j = \frac{e^{s_j}}{\sum_{l=1}^{k} e^{s_l}}$$

**Xüsusiyyətləri:**
1. Çıxışlar müsbətdir: $P_j > 0$
2. Cəmi birdir: $\sum_j P_j = 1$
3. Maksimum seçimini smooth edir

**Numerik stabillik problemi:**

Əgər $s_j$ çox böyükdürsə, $e^{s_j}$ overflow edə bilər:

```python
# Stabil implementasiya
def softmax_stable(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
```

## Cross-Entropy Loss

Cross-entropy iki paylanma arasındakı "məsafəni" ölçür:

$$L_{CE} = -\sum_{i} y_i \log(p_i)$$

Burada $y_i$ həqiqi ehtimal (one-hot) və $p_i$ proqnozlaşdırılan ehtimaldır.

## Backpropagation Törəmələri

### Softmax + Cross-Entropy Törəməsi

Xüsusi bir riyazi xassə sayəsində:

$$\frac{\partial L}{\partial s_j} = p_j - y_j$$

Bu, "softmax derivative" və "cross-entropy derivative"-nin sadələşmiş formasıdır.

### NN Backpropagation Addım-Adım

**Addım 1:** Output sensitivity

$$\frac{\partial L}{\partial S} = \frac{1}{n}(P - Y)$$

**Addım 2:** W₂ və b₂ üçün gradient

$$\frac{\partial L}{\partial W_2} = \left(\frac{\partial L}{\partial S}\right)^T \cdot H$$

$$\frac{\partial L}{\partial b_2} = \sum_i \frac{\partial L}{\partial S_i}$$

**Addım 3:** Hidden layer-ə backpropagate

Əvvəlcə H-ya görə gradient:

$$\frac{\partial L}{\partial H} = \frac{\partial L}{\partial S} \cdot W_2^T$$

Sonra Z₁-ə görə gradient (chain rule + tanh derivative):

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial H} \odot (1 - H^2)$$

**Addım 4:** W₁ və b₁ üçün gradient

$$\frac{\partial L}{\partial W_1} = \left(\frac{\partial L}{\partial Z_1}\right)^T \cdot X$$

$$\frac{\partial L}{\partial b_1} = \sum_i \frac{\partial L}{\partial Z_{1i}}$$

## Tanh Aktivasiyası

**Formula:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Xüsusiyyətləri:**
- Range: $(-1, 1)$
- Sıfır mərkəzli (zero-centered)
- Daha az yox olan gradient (less vanishing gradient)

**Derivative:**
$$\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$$

Bu xassə sayəsində backpropagation səmərəli hesablanır, çünki $\tanh(Z)$ artıq hesablanmışdır.

## L2 Regularization

Total loss:

$$L_{total} = L_{CE} + \frac{\lambda}{2}\sum_\theta \theta^2$$

Gradient:

$$\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{CE}}{\partial W} + \lambda W$$

Bu "weight decay" effekti yaradır - ağırlıkları sıfıra doğru çəkir.

---

# 10. Qrafik və Vizualizasiya

## Decision Boundary Plot

Bu qrafik modelin öyrəndiyi sərhədi vizualizasiya edir.

**Necə işləyir:**
1. 200×200 = 40,000 nöqtəlik grid yaradılır
2. Hər nöqtə üçün model proqnozlaşdırır
3. Nöqtələr sinifə görə rənglənir

**Interpretasiya:**
- **Xətti sərhəd:** Softmax üçün xarakterik
- **Əyri sərhəd:** NN üçün xarakterik

## Training Dynamics Plot

Loss və accuracy-nin epoch-lar üzrə dəyişməsi.

**Necə oxunur:**
- Train və validation əyriləri yaxın olmalıdır
- Əgər train azalır amma validation artırsa → overfitting
- Əgər hər ikisi yüksəkdir → underfitting

## Confusion Matrix

Hər sinif üçün doğru və səhv proqnozların sayı.

**Interpretasiya:**
- Diagonal: Doğru proqnozlar
- Off-diagonal: Qarışdırılan siniflər

## Confidence vs Accuracy Plot (Track B)

Modelin özünə güvənə bilmə dərəcəsini yoxlayır.

**Yaxşı calibration:** Əgər model 80% güvənirirsə, 80% dəqiq olmalıdır.

---

# 11. Qiymətləndirmə Protokolları

## Sanity Checks

### 1. Gradient Check

Analytical gradient-ləri numerical gradient-lərlə müqayisə edir:

```python
def gradient_check(model, X, Y, epsilon=1e-5):
    analytical_grad = model.backward(X, Y, cache)
    numerical_grad = compute_numerical_grad(model, X, Y, epsilon)
    
    relative_error = |analytical - numerical| / (|analytical| + |numerical|)
    return relative_error < 1e-5
```

### 2. Probability Sum Check

Softmax çıxışlarının vahid cəmə bərabər olduğunu yoxlayır:

```python
def check_probability_sum(P):
    sums = np.sum(P, axis=1)
    return np.allclose(sums, 1.0)
```

### 3. NaN/Inf Check

Model çıxışlarında NaN və ya Inf yoxdur:

```python
def check_nan_inf(model, X):
    cache = model.forward(X)
    return not (np.any(np.isnan(cache)) or np.any(np.isinf(cache)))
```

## Repeated Seed Evaluation

Statistik əhəmiyyəti təmin etmək üçün 5 fərqli seed ilə təkrar qiymətləndirmə:

**Protokol:**
1. Hər seed üçün modeli sıfırdan train edin
2. Test accuracy və loss qeyd edin
3. Orta və standart deviasiya hesablayın

**95% Confidence Interval:**
$$CI = \mu \pm t_{0.975, n-1} \cdot \frac{\sigma}{\sqrt{n}}$$

Burada $t_{0.975, 4} = 2.776$ (n=5 üçün t-critical dəyəri).

---

# 12. Komanda və Məsuliyyətlər

Bu layihə komanda şəklində icra olunur. Aşağıda komanda üzvləri və onların məsuliyyət sahələri qeyd olunmuşdur.

## Komanda Üzvləri

| Komanda Üzvü | Rol | Məsuliyyət Sahəsi |
|--------------|-----|-------------------|
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |
| _________________ | _________________ | _______________________________________ |

## Rol Təsvirləri

| Rol | Təsvir | Əsas Vəzifələr |
|-----|--------|-----------------|
| **Softmax Mühəndisi** | Softmax regression modelinin implementasıyası | `models.py` forward/backward, loss function |
| **Neural Şəbəkə Mühəndisi** | NN modelinin implementasıyası | `models.py` OneHiddenLayerNN, backpropagation |
| **Eksperiment Rəhbəri** | Eksperimentlərin planlaşdırılması və icrası | `main.py`, `trainer.py`, scheduling |
| **Vizualizasiya Mütəxəssisi** | Qrafiklər və diaqramların hazırlanması | `visualization.py`, figures |
| **Report Müəllifi** | Yazılı hesabatın hazırlanması | PDF report, documentation |
| **Prezentasiya Rəhbəri** | Texniki təqdimatın hazırlanması | Slides, Q&A preparation |

## GitHub Branch Strategiyası

```
main (production branch)
│
├── feature/softmax-implementation
├── feature/neural-network
├── feature/optimizers
├── feature/training-loop
├── feature/visualization
├── feature/experiments
├── feature/report
├── feature/slides
│
└── (hər branch müvafiq komanda üzvü tərəfindən idarə olunur)
```

---

# 13. Yoxlama Siyahıları

## Starter Pack Yoxlaması

Layihəyə başlamazdan əvvəl aşağıdakıları yoxlayın.

- [ ] `starter_pack/data/digits_data.npz` faylı mövcuddur
- [ ] `starter_pack/data/digits_split_indices.npz` faylı mövcuddur
- [ ] `starter_pack/data/linear_gaussian.npz` faylı mövcuddur
- [ ] `starter_pack/data/moons.npz` faylı mövcuddur
- [ ] `scripts/generate_synthetic.py` skripti işləyir
- [ ] `scripts/make_digits_split.py` skripti işləyir
- [ ] Starter pack-də heç bir model implementasıyası yoxdur
- [ ] Bütün data fayllarının forması (shape) düzgündür

## Quraşdırma Yoxlaması

Sistemi quraşdırarkən aşağıdakı addımları izləyin.

- [ ] Python 3.10+ quraşdırılıb
- [ ] Virtual mühit yaradılıb ( tövsiyə olunur)
- [ ] Dependencies quraşdırılıb (`pip install -r necessity.txt`)
- [ ] `python main.py --experiment check` əmri uğurla icra olunur
- [ ] Bütün src faylları skeleton olaraq mövcuddur
- [ ] data qovluğundakı bütün .npz faylları oxunur
- [ ] Git repository düzgün init olunub
- [ ] Feature branch-lər yaradılıb

## İmplementasiya Yoxlaması

### Softmax Regression
- [ ] `forward()` metodu düzgün softmax hesablayır
- [ ] `backward()` metodu düzgün gradient hesablayır
- [ ] L2 regularization tətbiq olunur
- [ ] Gradient check uğurla keçir
- [ ] Probability sum yoxlaması uğurla keçir

### Neural Network
- [ ] `forward()` metodu tam forward pass edir
- [ ] `backward()` metodu backpropagation edir
- [ ] Tanh aktivasiyası düzgün hesablanır
- [ ] Tanh derivative düzgün hesablanır (`1 - H²`)
- [ ] He initialization istifadə olunur
- [ ] Gradient check uğurla keçir

### Optimizers
- [ ] SGD düzgün çalışır
- [ ] Momentum velocity yığıb saxlayır
- [ ] Adam bias correction edir
- [ ] Hər optimizer reset() metoduna malikdir

### Trainer
- [ ] Minibatch yaradılması düzgündür
- [ ] One-hot encoding düzgündür
- [ ] Checkpointing ən yaxşı modeli saxlayır
- [ ] Training history qeyd olunur
- [ ] Validation monitoring işləyir

## Eksperiment Yoxlaması

### Linear Gaussian
- [ ] Decision boundary plot hazırlanıb
- [ ] Hər iki model train olunub
- [ ] Nəticələr qeyd olunub

### Moons
- [ ] Decision boundary plot hazırlanıb
- [ ] Hər iki model train olunub
- [ ] NN-in curved boundary öyrəndiyi göstərilib

### Digits
- [ ] Hər iki model train olunub
- [ ] Confusion matrix hazırlanıb
- [ ] 5 seed ilə repeated evaluation aparılıb
- [ ] 95% CI hesablanıb

### Capacity Ablation
- [ ] Hidden width {2, 8, 32} müqayisə olunub
- [ ] Decision boundary-lər müqayisə olunub
- [ ] Accuracy cədvəli hazırlanıb

### Optimizer Study
- [ ] SGD, Momentum, Adam müqayisə olunub
- [ ] Convergence sürətləri müqayisə olunub
- [ ] Training curves qrafikləri hazırlanıb

### Failure Case Analysis
- [ ] Bir uğursuzluq halı təhlil olunub
- [ ] Səbəb izah olunub
- [ ] Qərar çıxarılmışdır

# 14. Tez-tez Verilən Suallar

## S: Hansı track-i seçildi?


- **Track A (PCA/SVD):** Əgər riyazi PCA analizi ilə maraqlanırsınızsa
- **Track B (Confidence):** Əgər model calibration və etibarlılıq ilə maraqlanırsınızsa

---

## S: Niyə tanh, ReLU deyil?

**Cavab:** Tələb olunan aktivasiya tanh-dır. Bu seçim təsadüfi deyil və aşağıdakı üstünlüklərə malikdir:

1. **Sıfır mərkəzli (zero-centered):** Tanh çıxışı $(-1, 1)$ aralığındadır və sıfır ətrafında mərkəzləşir. Bu, gradient axınını daha sabit edir.

2. **Daha az yox olan gradient (less vanishing gradient):** ReLU-da $z < 0$ olduqda gradient sıfırdır, bu da "ölü neyronlar" probleminə səbəb olur. Tanh-da isə gradient hələ də mövcuddur.

3. **Daha sabit gradient axını:** Tanh funksiyası hamar (smooth) olduğu üçün optimizasiya prosesi daha sabit baş verir.

4. **Riyazi sadəlik:** Tanh derivative-i `1 - tanh²(z)` asanlıqla hesablanır, bu da backpropagation-u səmərəli edir.

---

## S: Overfitting baş verərsə nə edim?

**Cavab:** Overfitting baş verdikdə aşağıdakı üsulları tətbiq edin:

1. **L2 Regularization artırın:** `reg_lambda` parametrini artırın (məsələn, 1e-4 → 1e-3). Bu, ağırlıkları sıfıra doğru çəkir.

2. **Early Stopping tətbiq edin:** Əgər validation loss artmağa başlayırsa, training-i dayandırın və ən yaxşı checkpoint-i bərpa edin.

3. **Model tutumluluğunu azaldın:** Hidden width-i kiçildin (məsələn, 32 → 16).

4. **Daha çox data toplayın:** Data augmentation texnikaları tətbiq edin və ya yeni data əlavə edin.

---

## S: Niyə Adam SGD-dən daha sürətli yığılır?

**Cavab:** Adam daha sürətli yığılır çünki o, hər parametrlər üçün fərdi learning rate tətbiq edir:

**Adam-ın üstünlükləri:**
- **Birinci moment (momentum):** Keçmiş gradient-lərin istiqamətini saxlayır, bu da düz istiqamətdə sürətlənməyə səbəb olur.
- **İkinci moment (adaptive lr):** Hər parametrlərin özünəməxsus learning rate-i var:
  - Böyük gradient-lər üçün kiçik lr (sabitlik üçün)
  - Kiçik gradient-lər üçün böyük lr (sürət üçün)
- **Bias correction:** İlk addımlarda momentum və velocity sıfır olduğu üçün bu düzəliş ilk dövrlərdə yığılmanı sürətləndirir.

**SGD-nin məhdudiyyətləri:**
- Bütün parametrlər üçün eyni learning rate istifadə edir
- Dar dərələrdə (narrow valleys) oscillasiya edir
- Daha çox epoch tələb edir

---

## S: Hidden width necə seçilir?

**Cavab:** Hidden width seçimi capacity ablation texnikası ilə aparılır:

**Addım 1: Kiçik width-dən başlayın**
Məsələn, width = 2 ilə başlayın və modelin nə öyrəndiyini müşahidə edin.

**Addım 2: Accuracy artdıqca artırın**
Width-i {2, 4, 8, 16, 32, 64} olaraq artırın. Hər addımda accuracy-ni müşahidə edin.

**Addım 3: Overfitting nöqtəsini tapın**
Əgər training accuracy artır amma validation accuracy azalırsa, bu overfitting əlamətidir. Bu nöqtədə dayanın.

**Praktik tövsiyə:**
- Sadə problemlər üçün: width = 8-16 kifayətdir
- Orta mürəkkəblik: width = 32 ən yaxşı seçimdir
- Çox mürəkkəb problemlər: width = 64-128 (lakin overfitting riski var)

**Ümumi qayda:** Ən kiçik width seçin ki, hələ də yaxşı performance göstərsin.

---

## S: Neçə epoch train etməliyəm?

**Cavab:** Epoch sayı problem və model mürəkkəbliyindən asılıdır:

**Ümumi tövsiyələr:**
- **Early stopping ilə:** Ən yaxşı üsul - validation loss dayanana qədər train edin
- **Fixed epoch:** 200 epoch əksər hallarda kifayətdir
- **Monitor edin:** Hər 10 epoch-da validation metriklərini yoxlayın

**Overfitting əlamətləri:**
- Training loss azalır, amma validation loss artır
- Training accuracy artır, amma validation accuracy azalır

---

## S: Sanity check-lər nə üçündür?

**Cavab:** Sanity check-lər implementasıyanın düzgünlüyünü təmin edir:

1. **Gradient Check:** Analytical gradient-ləri numerical gradient-lərlə müqayisə edir. Relative error < 1e-5 olmalıdır.

2. **Probability Sum Check:** Softmax çıxışlarının cəmi 1 olmalıdır. Əgər deyilsə, numerical instability var.

3. **NaN/Inf Check:** Model çıxışlarında NaN və ya Inf yoxdur. Bu, gradient explosion və ya digər problemləri göstərir.

---

</div>
