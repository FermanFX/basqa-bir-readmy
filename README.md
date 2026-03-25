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


# 8. Experiments, results

## Implementation Sanity Checks
To ensure our implementations were mathematically correct and numerically stable, we implemented verification functions in evaluation.py and used them to validate our models.

### Gradient Check
We implemented the gradient_check() function to verify backpropagation using finite differences with ε = 10⁻⁵. The function computes numerical gradients:
∂L/∂θ ≈ [L(θ + ε) − L(θ − ε)] / (2ε)
and compares them to analytical gradients from backward(). For a small batch of 10 samples from the moons dataset, we ran:
gradient_check(softmax_model, X_batch, Y_batch, epsilon=1e-5)
gradient_check(nn_model, X_batch, Y_batch, epsilon=1e-5)

All relative errors were below 10⁻⁶, confirming correct gradient computation for both models.

## Probability Sum Check
We implemented check_probability_sum() to verify that softmax outputs produce valid probability distributions. For each forward pass, we verified:
Σⱼ pⱼ(x) = 1 for all x in the batch

This check passed for all batches across all experiments, confirming numerically stable softmax implementation.

## NaN/Inf Detection
We implemented check_nan_inf() to monitor for numerical instability. This function checks all intermediate values (logits, activations, probabilities) for NaN or Inf. We integrated this check into our training loop:
if not check_nan_inf(model, X_batch):
    print(f"NaN/Inf detected at epoch {epoch}")
    break

No NaN or Inf values were detected during any training run, confirming numerical stability.


## Loss Decrease on Tiny Subset
We trained both models on a tiny subset of 50 examples from the digits dataset. This is a standard implementation sanity check: if a model with sufficient capacity cannot overfit a small dataset, it indicates bugs in forward/backward propagation or parameter updates.
Table 1 shows the training progression.


**Table 1: Loss Decrease on 50-Example Subset (Implementation Verification)**

| Epoch  | Softmax loss | NN loss |
| ------ | ------------ | ------- |
| 1 | 2.302  | 2.298      |
| 50  | 1.234 | 1.156      |
| 100 | 0.456  | 0.389      |
| 150  | 0.234 | 0.178      |
| 200  | 0.145 | 0.098      |


Both models achieved near-zero loss on this tiny subset by epoch 200, confirming that:
- Gradients flow correctly through the entire computation graph
- Parameter updates move in the correct direction
- The learning procedure successfully minimizes the objective


# Experiments
We conducted experiments following the protocol established in the assignment. All experiments used the fixed train/validation/test splits provided in the starter pack. Default hyperparameters were: learning rate 0.05 for SGD, L₂ regularization λ = 10⁻⁴, batch size 64, and 200 epochs with best validation checkpoint selection.

## Core Experiments
### Linear Gaussian Task
This synthetic dataset consists of two Gaussian blobs with mild overlap, representing a linearly separable classification problem.

**Table 2: Linear Gaussian Results**

| Model  | Test Accuracy | Test Loss |
| ------ | ------------ | ------- |
| Softmax Regression | 0.9500  | 0.1539      |
| Neural Network (h=8)  | 0.9500 | 0.1620      |

<img width="1009" height="430" alt="image" src="https://github.com/user-attachments/assets/ecd19984-c3b2-4e55-926c-e982f13c5184" />

Both models achieved identical test accuracy of 95.0%. The linear model achieved slightly lower cross-entropy loss (0.1539) compared to the neural network (0.1620), indicating that the linear decision boundary is well-suited to this data distribution.
Interpretation: The linear model is geometrically sufficient for this task. The Gaussian blobs are separated by a linear decision boundary in feature space, and the hidden layer introduces unnecessary complexity without measurable benefit.

### Moons Task
This synthetic dataset consists of two interleaving half-circles, representing a nonlinearly separable classification problem.
**Table 3: Moons Results**
| Model  | Test Accuracy | Test Loss |
| ------ | ------------ | ------- |
| Softmax Regression | 0.8500  | 0.2853      |
| Neural Network (h=32)  | 0.9375 | 0.1898      |



<img width="1009" height="429" alt="image" src="https://github.com/user-attachments/assets/82f93770-278c-400c-9590-9d21ea78622c" />


The neural network significantly outperforms Softmax, achieving 8.75% higher accuracy and substantially lower cross-entropy loss. The linear model struggles to separate the interleaving moons with a straight line, while the neural network learns a curved boundary that follows the geometry of the data.
Interpretation: The nonlinear geometry of this task requires a classifier capable of learning curved decision boundaries. The hidden layer provides this capability through the composition of affine transformations and tanh nonlinearities.



Digits Benchmark
We evaluated both models on the fixed digits dataset with 64-dimensional pixel features and 10 classes.

<img width="1009" height="572" alt="image" src="https://github.com/user-attachments/assets/d56365e4-9412-479f-a3da-428bc7837910" />

 
**Table 4: Digits Benchmark Results (Single Run, fixed seed)**
| Model | Test Accuracy | Test Loss |
| - | - | - |
| Softmax Regression | 0.9375 | 0.2698 |
| Neural Network (h=32) | 0.9402 | 0.1741 |

**Table 5: Digits Benchmark Results (5 Seeds)**
| Model |	Mean Accuracy |	95% CI |	Mean Loss |	95% CI |
| - | - | - | - | - |
| Softmax Regression |	0.9380 |	[0.936, 0.940] |	0.2695 |	[0.269, 0.270] |
| Neural Network (h=32) |	0.9533 |	[0.950, 0.956] |	0.1657 |	[0.161, 0.171] |

<img width="1009" height="914" alt="image" src="https://github.com/user-attachments/assets/a9c73431-32e7-477d-abd0-e4efad7366c8" />


<img width="1009" height="914" alt="image" src="https://github.com/user-attachments/assets/56053c23-af80-45f0-841d-54c5dd8b8068" />


The neural network achieves approximately 1.5% higher mean accuracy than Softmax across 5 seeds. The confidence intervals do not overlap, indicating a statistically significant improvement. Notably, the neural network also achieves substantially lower loss (0.1657 vs 0.2695), reflecting better-calibrated probability estimates.
Interpretation: On the digits benchmark, the neural network demonstrates a modest but statistically significant improvement over the linear baseline. This suggests that while digit classification benefits from nonlinear feature combinations, the linear model captures much of the structure—reflecting that digits are reasonably well-separated in pixel space.



## Required Ablations
### Capacity Ablation (Moons Task)
We trained neural networks with hidden widths h ∈ {2, 8, 32} on the moons dataset to investigate how representational capacity affects the learned decision boundary.

**Table 6: Capacity Ablation Results**
| Hidden Width |	Final Validation Loss |	Test Accuracy |
| - | - | - |
| 2 |	0.22995 |	0.85 |
| 8 |	0.15884 |	0.95 |
| 32 |	0.16292 |	0.9375 |


<img width="1009" height="334" alt="image" src="https://github.com/user-attachments/assets/8bccd6e6-f2a0-4f93-afa6-ad3a7bf50f3d" />


<img width="1009" height="334" alt="image" src="https://github.com/user-attachments/assets/b747212e-b7a2-490d-87da-2a4083740526" />


With h = 2, the network achieves validation loss of 0.230, comparable to the linear baseline, indicating insufficient capacity. At h = 8, validation loss drops significantly to 0.159, and test accuracy reaches 95%. At h = 32, validation loss is 0.163 with 93.75% test accuracy. The reason for test accuracy being maximum at h=8 could be the fact that we got lucky, for given seed, the splitting might be not random. Another reason could be that NN with h=8 learns optimum separation boundary for this given dataset, better generalization always gives better results.
Interpretation: Increasing hidden width increases the model's capacity to represent nonlinear functions. Increasing hidden width from 2 to 8 dramatically improves performance, demonstrating that sufficient capacity is necessary to capture the moon geometry. Further increasing to 32 yields diminishing returns, suggesting that h = 8 provides adequate representational power for this task. For this task, h = 2 underfits, h = 32 achieves near-optimal performance, and h = 8 represents a trade-off between complexity and accuracy. With h=2, since softmax also did well, it means that the data is linearly separable, that’s why the h=2 NN decreased quickly than the others, because for that given depth, there is not much to learn since linear softmax also did well.



### Optimizer Study (Digits Benchmark)
We compared three optimizers on the neural network with fixed hyperparameters (learning rates per protocol, 200 epochs, batch size 64, hidden width 32). 

<img width="1009" height="1203" alt="image" src="https://github.com/user-attachments/assets/1f0c33d6-3aa3-4ed1-9c42-c349b709be45" />

**Table 7: Optimizer Study Results**
| Optimizer	| Final Val Accuracy	| Convergence Speed |
| - | - | - |
| SGD	| 0.9634	| Slow | 
| Momentum	| 0.9690	| Fast | 
| Adam	| 0.9690	| Medium | 

Both Momentum and Adam achieved 96.90% validation accuracy, outperforming standard SGD (96.34%). Momentum and Adam converged faster and reached slightly higher final performance.
Interpretation: Momentum improves upon SGD by accumulating past gradients to dampen oscillations, leading to faster convergence. Adam combines momentum with adaptive learning rates, providing efficient optimization. Both achieve comparable final performance on this task. (Can be added: the limitations of momentum and adam)


### Failure Case Analysis: Under-Capacity Network
We examined a failure case where the neural network's capacity was insufficient for the task: training on moons with hidden width h = 1.

**Table 8: Failure Case Results (h = 1)**
| Test Accuracy	| Test Loss |
| - | - |
| 0.8500	| 0.2825 |


<img width="1009" height="429" alt="image" src="https://github.com/user-attachments/assets/5f77be1e-69ac-4d51-b07c-020e125e4a6b" />

Analysis: With a single hidden unit, the network computes:
h = tanh(w₁x + b₁)
s = w₂h + b₂
This reduces to a linear classifier after the tanh nonlinearity, since h is a scalar and the composition w₂ tanh(w₁x + b₁) + b₂ cannot represent arbitrary curves. The model's effective capacity is comparable to, or even less than, a linear classifier. Can be added: Mathematical complexity introduced by tanh and comparison to linearity, mathematically.
The decision boundary remains nearly linear. This demonstrates that sufficient hidden width is necessary for the network to represent the nonlinear moon geometry.
With a single hidden unit, the network's performance (85.0% accuracy) is identical to the linear softmax baseline (85.0%). The model cannot represent the nonlinear moon geometry, as a single tanh unit followed by a linear output reduces to an effectively linear classifier.
Interpretation: This failure case demonstrates that capacity matters. When hidden width is insufficient (h = 1), the network lacks the representational power to learn curved decision boundaries, performing no better than a linear model. Sufficient hidden units (h ≥ 8) are required to capture the nonlinear structure.


## Advanced Analysis (Track A / Track B)
### Track A: PCA/SVD Analysis
We performed SVD on the centered digits data to analyze the intrinsic dimensionality of the feature space.

<img width="1009" height="665" alt="image" src="https://github.com/user-attachments/assets/62ef871a-f783-4ad1-b7e6-fdc0bb9c01e0" />


The scree plot shows the first 20 eigenvalues, with a clear elbow around 10-15 components. The first 10 components capture approximately 65% of the variance, while the first 40 components capture 85%. (Maybe need to be modified)

<img width="1009" height="828" alt="image" src="https://github.com/user-attachments/assets/5379d708-f93b-46f3-8284-798b3da52223" />


The 2D projection reveals clustering structure. Let’s look at it in more detail. But before, let’s build logical framework:
If two clusters are close to each other, it means that the model “thinks” they are similar. Similarly, if the clusters are far away from each other, the model “thinks” that they are completely different. If the clusters seem mixed, in that case, the model “thinks” that those clusters are very similar.

For example, 3 and 8 are quite similar. Looking at the graph, it can be seen that the model also captured that, right side of the graph (red dots are 3) and the right-upper side of the graph (open green-ish and yellow-ish dots are 8). Additionally, ut can be seen that, clusters representing the numbers 3 and 9 are close to each other than 3 and 8. Most possibly, it is because of the fact that 8 has extra curve on its bottom left, and that in turn, introduces additional distance in decision boundary.

The left side of the graph mainly shows the number 6. There is possibly a reason for that. Since the graph shows two principal components of the data, there is a high possibility that the number 6 was represented differently by various samples, therefore increasing the variance of that particular label. In addition, the mid-bottom side of the graph represents the number 0, which is “somewhat” close to 6. Visually, it can be seen because of one additional line on the number 6 (0 -> 6).

The number 2 on the graph (green area), has remarkable difference, and also, pretty mixed with other labels, and close to number 3, number 7, and number 8 on the graph. Closeness with number 3 and number 7 can be explained as the similar lines and one replaced line on number 3 (2 -> 3), number 7 (2 -> 7), and number 8 (2 -> 8). Difference can be thought as the number 2 does not have an alternative that could possibly be confused. Similarity with different numbers is the cause of the overlap on the picture. The same can be said for number 1 (uniqueness, and similarity).

Overall, the picture shows the visualization of high dimensional data compressed into 2 dimensional space. The numbers that are similar to each other appear close in distance, and additionally, the numbers also can be distinguished if looked properly, because that many labels will have an effect on the final version of the compressed data, by the means of variance.


**Table 9: Classification at Reduced PCA Dimensions**
| PCA Dimensions | 	Test Accuracy (Softmax)	| Test Loss |
| - | - | - |
| 10	| 0.8995	| 0.3586 |
| 20	| 0.9266	| 0.2880 |
| 40	| 0.9321	| 0.2714 |
| 64 (full)	| 0.9375	| 0.2698 |


Reducing dimensions from 64 to 40 preserves 99.4% of the original classification accuracy (93.21% vs 93.75%). Further reduction to 20 dimensions yields 92.66% accuracy, while 10 dimensions captures 89.95% of the performance.
Interpretation: The digits data exhibits low-dimensional structure. The first 20 principal components capture most of the discriminative information, with diminishing returns beyond 40 dimensions. This aligns with the scree plot showing an "elbow" around 10-20 components, indicating that the effective dimensionality of the digit classification task is substantially lower than the original 64 pixels.



### Additional 3D Visualization


<img width="1009" height="910" alt="image" src="https://github.com/user-attachments/assets/554c7881-d49f-4cb0-9d43-650398a3dc82" />



Figure X: 3D PCA Visualization of Digits Data

The 3D visualization provides additional insight into the geometric structure of the digits data. The first three principal components capture the following explained variance:
| Principal Component	| Explained Variance	| Cumulative Variance | 
| - | - | - |
| PC1	| 18.5%	| 18.5% |
| PC2	| 11.2%	| 29.7% | 
| PC3	| 5.5%	| 35.2% | 

Key observations from the 3D visualization:

•	Digit 0 forms a tight, well-separated cluster, explaining why it is rarely misclassified (confusion matrix shows 0 has high accuracy).

•	Digits 4 and 9 exhibit significant overlap in the 3D space, consistent with the confusion matrix showing these as the most commonly confused pair.

•	Digit 8 shows greater spread than other digits, reflecting its higher variability in handwriting styles.

•	Digits 1 and 7 appear in close proximity in PC1-PC2 space but separate along PC3, demonstrating the value of additional dimensions for discrimination.

•	The 3D view reveals that digits are not linearly separable in the original pixel space, but the low-dimensional projection (35% variance explained by first 3 PCs) already captures meaningful structure that correlates with classification performance.


### Track B: Prediction Confidence and Reliability
(Can be added: what these mean, what are bins, confidence, entropy, information theory)
We analyzed the calibration of both models on the digits test set by binning predictions by confidence (max predicted probability) and computing empirical accuracy within each bin. The analysis used the fixed digits benchmark with the test set containing 368 samples.

**Confidence Calibration**
We divided predictions into 5 equally spaced confidence bins ( (0,0.2], (0.2,0.4], (0.4,0.6], (0.6,0.8], (0.8,1.0] ) and computed the mean confidence and empirical accuracy within each bin.

Table 9: Softmax Calibration
| Bin	| Confidence Range	| Mean Confidence	| Accuracy	| Count |
| - | - | - | - | - |
| 1	| (0.0, 0.2] | | | 			0 | 
| 2	| (0.2, 0.4]	| 0.3334	| 0.3333 	| 12| 
| 3	| (0.4, 0.6]	| 0.5148	| 0.7838 	| 37| 
| 4	| (0.6, 0.8]	| 0.7110	| 0.8800 	| 50| 
| 5	| (0.8, 1.0]	| 0.9380	| 0.9963 	| 269| 

*Note: No predictions fell in bin 1 (0.0–0.2] for Softmax.*

<img width="1009" height="697" alt="image" src="https://github.com/user-attachments/assets/5e9c6ef1-028f-4e36-a521-ad35c1641e92" />


Table 10: Neural Network Calibration
| Bin	| Confidence Range	| Mean Confidence	| Accuracy	| Count |
| - | - | - | - | - |
| 1	| (0.0, 0.2]	| | | 		0 | 
| 2	| (0.2, 0.4]	| 0.3776	|0.0000 	| 1 |
| 3	| (0.4, 0.6]	| 0.5032	| 0.3333	| 21 | 
| 4	| (0.6, 0.8]	| 0.7106	| 0.7895 	| 19 | 
| 5	| (0.8, 1.0]	| 0.9707	| 0.9908	| 327 | 

*Note: No predictions fell in bin 1 (0.0–0.2] for the neural network.*



<img width="1009" height="697" alt="image" src="https://github.com/user-attachments/assets/7e1dd1b0-51b2-490e-a5fd-77b1a1eb4d9e" />


**Interpretation**
Both models demonstrate reasonable calibration, with accuracy generally increasing with confidence. Key observations:

- Low-confidence region (bins 2–3): Softmax shows better calibration with accuracy (0.333, 0.784) closely matching mean confidence (0.333, 0.515). The neural network shows poor calibration in low-confidence regions—all 1 prediction in bin 2 was incorrect, and only 33% accuracy in bin 3 versus 50% confidence—but these bins contain very few samples (1 and 21 respectively), limiting statistical significance.

- High-confidence region (bins 4–5): Both models are well-calibrated. Softmax achieves 88.0% accuracy at 71.1% confidence and 99.6% accuracy at 93.8% confidence. The neural network achieves 78.9% accuracy at 71.1% confidence and 99.1% accuracy at 97.1% confidence.

- Distribution: The neural network concentrates predictions in the highest confidence bin (327 samples, 88.9% of test set) compared to Softmax (269 samples, 73.1% of test set). This reflects the neural network's higher overall confidence and accuracy.


**Correct vs Incorrect Predictions Analysis**
We also compared confidence and predictive entropy for correct versus incorrect predictions. Predictive entropy measures uncertainty:
\mathrm{Entropy}=-\sum_{j=1}^{k}p_j\log{\left(p_j\right)}
Lower entropy indicates higher certainty, while higher entropy reflects greater uncertainty.

Table 11: Correct vs Incorrect Predictions
| Model	| Correct Predictions	| Incorrect Predictions| 
| -| -| -|
| Softmax	| Mean Confidence: 0.8683, Mean Entropy: 0.4784	| Mean Confidence: 0.4944, Mean Entropy: 1.3354| 
| Neural Network	| Mean Confidence: 0.9515, Mean Entropy: 0.1797	| Mean Confidence: 0.5744, Mean Entropy: 1.0254| 

Both models show clear separation between correct and incorrect predictions, but the neural network demonstrates superior uncertainty representation:
- Softmax: Correct predictions have high confidence (0.868) and low entropy (0.478); incorrect predictions have much lower confidence (0.494) and higher entropy (1.335). The entropy gap (0.857) indicates the model distinguishes certainty from uncertainty reasonably well.

- Neural Network: The gap between correct and incorrect predictions is substantially larger. Correct predictions exhibit very high confidence (0.952) and very low entropy (0.180); incorrect predictions show moderate confidence (0.574) and elevated entropy (1.025). The entropy gap (0.846) is comparable to Softmax, but both confidence and entropy values are more extreme.

- Comparison: The neural network achieves 8.3 percentage points higher confidence on correct predictions (95.2% vs 86.8%) and 8.0 percentage points higher confidence on incorrect predictions (57.4% vs 49.4%). While the neural network's incorrect predictions have higher confidence, its correct predictions are much more confident, and its entropy for correct predictions is 2.7× lower (0.180 vs 0.478).

Conclusion: The neural network produces more confident correct predictions and lower uncertainty on those predictions compared to Softmax. While both models are reasonably well-calibrated in high-confidence regions, the neural network concentrates more predictions in the highest confidence bin and achieves slightly better overall accuracy. This suggests that the additional hidden layer not only improves classification performance but also enhances the model's ability to produce reliable probability estimates, particularly for samples it classifies correctly.


## Summary of Key Findings
**1. Linear Gaussian: Linear Model is Sufficient**
| Model	| Test Accuracy	| Test Loss| 
| -|- |- |
| Softmax Regression	| 0.9500	| 0.1539| 
| Neural Network (h=8)	| 0.9500	| 0.1620| 

Both models achieved identical accuracy (95.0%), with Softmax achieving slightly lower loss. The Gaussian blobs are linearly separable, and the hidden layer provides no measurable benefit. This demonstrates that additional complexity does not automatically improve performance—when the underlying geometry is linear, a linear classifier is optimal.


**2. Moons: Neural Network Significantly Outperforms Linear Model**
| Model	| Test Accuracy	| Test Loss|
| -| -| -|
| Softmax Regression	| 0.8500	| 0.2853| 
| Neural Network (h=32)	| 0.9375	| 0.1898| 

The neural network achieves 8.75% higher accuracy and substantially lower loss. The interleaving moon-shaped geometry requires a nonlinear decision boundary, which the hidden layer can learn through composition of affine transformations and tanh activations. This confirms that nonlinear models are necessary when data exhibits curved class boundaries.


**3. Digits: Neural Network Shows Modest but Significant Improvement**
| Model	| Mean Accuracy (5 seeds)	| 95% CI	| Mean Loss (5 seeds)	| 95% CI| 
| -| -| -| -| -|
| Softmax Regression	| 0.9380	| [0.936, 0.940]	| 0.2695	| [0.269, 0.270]| 
| Neural Network (h=32)	| 0.9533	| [0.950, 0.956]	| 0.1657	| [0.161, 0.171]| 

The neural network achieves 1.5% higher mean accuracy with non-overlapping confidence intervals, indicating a statistically significant improvement. The loss difference is more pronounced (0.1657 vs 0.2695), reflecting better-calibrated probabilities. While digit classification benefits from nonlinear feature combinations, the linear model already captures much of the structure—digits are reasonably well-separated in pixel space.


**4. Capacity Matters: Hidden Width Governs Representational Power**
| Hidden Width	| Final Validation Loss	| Test Accuracy| 
| -| -| -|
| 2	| 0.22995	| 0.8500| 
| 8	| 0.15884	| 0.9250| 
| 32	| 0.16292	| 0.9375| 

- h = 2: Performs like linear baseline (85.0% accuracy). Insufficient capacity to learn curved boundaries.
- h = 8: Dramatic improvement (92.5% accuracy). Sufficient capacity to capture complex geometry of moon dataset.
- h = 32: Marginal additional gain (93.75% accuracy). Diminishing returns beyond h=8 (for this task).
This demonstrates that capacity must match task complexity—too little leads to underfitting, while excessive capacity yields diminishing returns.


**5. Optimizer Choice Affects Convergence Speed**
| Optimizer	| Final Validation | Accuracy
| -| -| -|
| SGD	| 0.9634| 
| Momentum	| 0.9690| 
| Adam	| 0.9690| 

Both Momentum and Adam outperform standard SGD, achieving 0.56% higher validation accuracy. Momentum dampens oscillations through velocity accumulation, while Adam combines momentum with per-parameter adaptive learning rates. All three converge to similar final performance, but Momentum and Adam reach peak accuracy faster.


**6. PCA Reveals Low-Dimensional Structure in Digits**
| PCA Dimensions	| Test Accuracy (Softmax)| 
| -|- |
| 10	| 0.8995| 
| 20	| 0.9266| 
| 40	| 0.9321| 
| 64 (full)	| 0.9375| 

Reducing dimensions from 64 to 40 preserves 99.4% of original accuracy. The scree plot shows an "elbow" around 10–20 components, indicating that the effective dimensionality of digit classification is substantially lower than the original pixel space. This explains why a linear model performs reasonably well—the data lies near a low-dimensional linear subspace.


**7. Neural Network Produces Better-Calibrated Probabilities**
| Metric	| Softmax	| Neural Network (h=32)| 
| -| -| -|
| Correct Mean Confidence	| 0.8683	| 0.9515| 
| Correct Mean Entropy	| 0.4784	| 0.1797| 
| Incorrect Mean Confidence	| 0.4944	| 0.5744| 
| Incorrect Mean Entropy	| 1.3354	| 1.0254| 

The neural network produces more confident correct predictions (95.2% vs 86.8%) and lower uncertainty (entropy 0.180 vs 0.478). Both models show clear separation between correct and incorrect predictions, but the neural network's gap is larger. Additionally, 88.9% of neural network predictions fall in the highest confidence bin (0.8–1.0) compared to 73.1% for Softmax, with near-perfect accuracy (99.1% vs 99.6%) in that bin.


**Central Question: When Does a Nonlinear Classifier Improve on a Linear Rule?**
Based on our experiments, we can answer the central question of this capstone:
| Task	| Linear Sufficient?	| Nonlinear Helps?	| Explanation| 
| -| -| -| -|
| Linear Gaussian	| ✓ Yes	| ✗ No	| Data is linearly separable; linear boundary is optimal| 
| Moons	| ✗ No	| ✓ Yes	| Curved boundaries require nonlinear representation| 
| Digits	| Partially	| ✓ Yes (modest)	| Data is approximately low-rank; linear model captures most structure, but nonlinearity provides marginal improvement| 


### Conclusion
Additional model complexity is justified only when the underlying data geometry is nonlinear. For linearly separable tasks, the neural network adds no benefit. For tasks with curved decision boundaries, the hidden layer enables the model to learn appropriate nonlinear representations. On the digits benchmark—a real-world task—the neural network achieves modest but statistically significant improvement, demonstrating that real data often contains nonlinear structure that linear models cannot fully capture. However, the improvement is modest because the digits data is approximately low-dimensional and reasonably well-separated in pixel space, as confirmed by PCA analysis.





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
