# Multiclass Logistic Regression

Multiclass Logistic Regression is an extension of Binary Logistic Regression used when the target contains **more than two classes**.

For binary classification:

$$
y \in \{0,1\}
$$

For multiclass classification:

$$
y \in \{0,1,2,\dots,K-1\}
$$

where **K** represents the number of classes.

For the Iris dataset:

- Setosa
- Versicolor
- Virginica

Therefore:

$$
K = 3
$$

---

# 1. Binary vs Multiclass Logistic Regression

The main concepts change as follows:

| Binary Logistic Regression | Multiclass Logistic Regression |
|---|---|
| Weight vector $w$ | Weight matrix $W$ |
| Single bias $b$ | Bias vector $b$ |
| One score | One score per class |
| Sigmoid | Softmax |
| Binary Cross-Entropy | Categorical Cross-Entropy |
| Threshold (usually 0.5) | Argmax |

The main idea is that instead of predicting one probability, the model predicts a **probability distribution across all classes**.

---

# 2. Weight Matrix and Bias Vector

In Binary Logistic Regression, we have:

$$
z = Xw + b
$$

For multiclass classification, every class needs its own set of weights.

Therefore, instead of a weight vector, we use a **weight matrix**:

$$
W \in \mathbb{R}^{n \times K}
$$

where:

- $n$ = number of features
- $K$ = number of classes

For Iris:

- 4 features
- 3 classes

Therefore:

$$
W : (4,3)
$$

Conceptually:

```text
             Class 0    Class 1    Class 2

Feature 1      w          w          w
Feature 2      w          w          w
Feature 3      w          w          w
Feature 4      w          w          w
```

Each class also needs its own bias.

Therefore:

$$
b \in \mathbb{R}^{K}
$$

For Iris:

$$
b : (3,)
$$

---

# 3. Linear Scores

The linear equation becomes:

$$
\boxed{Z = XW + b}
$$

If:

$$
X : (m,n)
$$

and:

$$
W : (n,K)
$$

then:

$$
Z : (m,K)
$$

For Iris:

```text
X       = (m,4)
W       = (4,3)
b       = (3,)

Z       = (m,3)
```

Example:

```text
             Class 0    Class 1    Class 2

Sample 1       2.5        1.2       -0.5
Sample 2       0.3        3.1        1.4
Sample 3      -0.2        1.1        4.2
```

These values are called **scores or logits**.

They are not probabilities yet.

---

# 4. Softmax

Binary Logistic Regression uses **Sigmoid**.

Multiclass Logistic Regression uses **Softmax**.

Softmax converts the class scores into probabilities.

For class $k$:

$$
P(y=k|x)
=
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}
$$

Example:

```text
Scores:

[2.0, 1.0, 0.5]

        ↓ Softmax

Probabilities:

[0.63, 0.23, 0.14]
```

The probabilities for each sample always sum to:

$$
1
$$

For example:

$$
0.63 + 0.23 + 0.14 = 1
$$

Softmax operates **across the classes of each sample**.

If:

```text
Z.shape = (m,3)
```

then Softmax operates row-wise:

```python
axis=1
```

---

# 5. Numerically Stable Softmax

Directly calculating:

$$
e^z
$$

can cause numerical overflow when the scores become very large.

To avoid this, subtract the maximum score from each row before calculating the exponential:

$$
z_k - \max(z)
$$

In code:

```python
z = z - np.max(z, axis=1, keepdims=True)

exp_z = np.exp(z)

soft_max = exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

This improves **numerical stability** without changing the final Softmax probabilities.

---

# 6. One-Hot Encoding

For multiclass Cross-Entropy, the target classes are represented using **one-hot encoding**.

For three classes:

```text
Class 0 → [1, 0, 0]

Class 1 → [0, 1, 0]

Class 2 → [0, 0, 1]
```

Therefore, the target matrix has the shape:

$$
Y : (m,K)
$$

For Iris:

```text
Y.shape = (m,3)
```

Example:

```text
             Class 0    Class 1    Class 2

Sample 1        1          0          0
Sample 2        0          1          0
Sample 3        0          0          1
```

---

# 7. Categorical Cross-Entropy

Binary Logistic Regression uses **Binary Cross-Entropy**.

Multiclass Logistic Regression uses **Categorical Cross-Entropy**.

The cost function is:

$$
\boxed{
J =
-\frac{1}{m}
\sum_{i=1}^{m}
\sum_{k=1}^{K}
y_{ik}\log(\hat{y}_{ik})
}
$$

Because $Y$ is one-hot encoded, only the probability assigned to the **correct class** contributes to the loss.

Example:

```text
Actual:

[0, 1, 0]

Prediction:

[0.10, 0.80, 0.10]
```

The loss effectively becomes:

$$
-\log(0.80)
$$

Therefore:

```text
High probability for correct class → Low loss

Low probability for correct class  → High loss
```

---

# 8. Epsilon in Cross-Entropy

During implementation, a very small value called **epsilon** can be added:

```python
epsilon = 1e-15

loss = -y * np.log(soft_max + epsilon)

cost = np.sum(loss) / m
```

Epsilon prevents:

$$
\log(0)
$$

because:

$$
\log(0) = -\infty
$$

Therefore, epsilon is used for **numerical stability**.

It is not part of the theoretical Cross-Entropy formula.

---

# 9. Error

After Softmax, the predicted probability matrix is:

$$
\hat{Y}
$$

The error is:

$$
\boxed{E = \hat{Y} - Y}
$$

Its shape is:

$$
E : (m,K)
$$

For Iris:

```text
soft_max    = (m,3)
y           = (m,3)

error       = (m,3)
```

This error is then used to calculate the gradients.

---

# 10. Weight Gradient

The gradient for the weight matrix is:

$$
\boxed{
dW =
\frac{1}{m}X^T(\hat{Y}-Y)
}
$$

Dimension check:

$$
X^T : (n,m)
$$

$$
(\hat{Y}-Y) : (m,K)
$$

Therefore:

$$
(n,m)(m,K) = (n,K)
$$

For Iris:

$$
(4,m)(m,3) = (4,3)
$$

So:

```text
W.shape  = (4,3)

dW.shape = (4,3)
```

---

# 11. Bias Gradient

Since every class has its own bias, we also need one bias gradient for every class.

$$
db : (K,)
$$

In code:

```python
db = np.sum(soft_max - y, axis=0) / m
```

Here, `axis=0` means we sum the errors across all samples while keeping each class separate.

For three classes:

```text
db = [db_class0, db_class1, db_class2]
```

Therefore:

```text
b.shape  = (3,)

db.shape = (3,)
```

---

# 12. Gradient Descent

After calculating the gradients, update the parameters:

$$
\boxed{
W := W - \alpha dW
}
$$

$$
\boxed{
b := b - \alpha db
}
$$

where:

$$
\alpha = \text{learning rate}
$$

In code:

```python
self.weights -= self.lr * dw

self.base -= self.lr * db
```

The process is repeated for the specified number of iterations.

---

# 13. Prediction

After training, the model produces class probabilities.

Example:

```text
[0.05, 0.90, 0.05]
```

To make the final prediction, select the class with the highest probability:

$$
\boxed{
\hat{y} = \arg\max_k P(y=k|x)
}
$$

Example:

```text
[0.05, 0.90, 0.05]
          ↑

Predicted Class = 1
```

Unlike Binary Logistic Regression, a `0.5` threshold is not normally used.

---

# 14. Complete Training Flow

```text
Input X
   │
   ▼
Z = XW + b
   │
   ▼
Softmax
   │
   ▼
Predicted Probabilities Ŷ
   │
   ▼
Categorical Cross-Entropy
   │
   ▼
Error = Ŷ - Y
   │
   ├──────────────► dW = Xᵀ(Ŷ - Y) / m
   │
   └──────────────► db = Σ(Ŷ - Y) / m
                         │
                         ▼
                  Gradient Descent
                         │
                         ▼
                    Update W, b
                         │
                         ▼
                       Repeat
```

---

# 15. Important Dimensions

For:

- $m$ = number of samples
- $n$ = number of features
- $K$ = number of classes

| Variable | Shape | Meaning |
|---|---|---|
| $X$ | `(m,n)` | Input features |
| $W$ | `(n,K)` | Weights for all classes |
| $b$ | `(K,)` | One bias per class |
| $Z$ | `(m,K)` | Class scores |
| $\hat{Y}$ | `(m,K)` | Softmax probabilities |
| $Y$ | `(m,K)` | One-hot encoded labels |
| $dW$ | `(n,K)` | Weight gradients |
| $db$ | `(K,)` | Bias gradients |

---

# Major Mistakes in My First Implementation

## 1. Calculating Classes Separately

### Wrong

```python
for j in range(n_class):
    y_h = np.dot(X, self.weights[:,j]) + self.base[j]
```

This calculated the scores for one class at a time.

### Fix

Calculate the scores for **all classes together**:

```python
z = X @ self.weights + self.base
```

Now:

```text
z.shape = (m,K)
```

---

## 2. Applying Softmax Across the Entire Dataset

### Wrong

```python
z = z / np.sum(z)
```

This normalized the entire matrix together, causing different samples to compete with each other.

### Fix

Softmax must normalize the classes **inside each sample**:

```python
z = z - np.max(z, axis=1, keepdims=True)

exp_z = np.exp(z)

soft_max = exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

The important part is:

```python
axis=1
```

because Softmax operates across classes.

---

## 3. Incorrect Number of Samples

### Wrong

```python
m = len(X.shape[0])
```

`X.shape[0]` already returns the number of samples.

### Fix

```python
m = X.shape[0]
```

---

## 4. Incorrect Loss Calculation

### Wrong

```python
loss = []

loss[j] = -y[j] * np.log(soft_max[j])
```

This mixed class indexing with sample indexing and did not calculate the loss over the complete prediction matrix.

### Fix

Use the complete one-hot target and probability matrices:

```python
epsilon = 1e-15

loss = -y * np.log(soft_max + epsilon)

cost = np.sum(loss) / m
```

---

## 5. Incorrect Bias Gradient

### Wrong

```python
db = np.sum(soft_max - y) / m
```

This returned only one scalar value even though every class has its own bias.

### Fix

```python
db = np.sum(soft_max - y, axis=0) / m
```

Now:

```text
db.shape = (K,)
```

which matches the bias vector.

---

# Quick Cheat Sheet

```text
Linear Scores:

Z = XW + b


Probabilities:

Ŷ = Softmax(Z)


Cross-Entropy:

J = -(1/m) ΣΣ Y log(Ŷ)


Error:

E = Ŷ - Y


Weight Gradient:

dW = (1/m) XᵀE


Bias Gradient:

db = (1/m) ΣE


Gradient Descent:

W = W - αdW

b = b - αdb


Prediction:

class = argmax(Ŷ)
```

---

# Key Idea

The main difference to remember is:

> **Binary Logistic Regression uses Sigmoid to predict one probability, while Multiclass Logistic Regression uses Softmax to produce a probability distribution across all classes.**

The complete multiclass process is:

$$
\boxed{
X
\rightarrow
XW+b
\rightarrow
Softmax
\rightarrow
CrossEntropy
\rightarrow
Gradients
\rightarrow
GradientDescent
}
$$
