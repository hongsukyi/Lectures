# Lecture 03: Logistic Regression for Binary Classification

### 🔗 Start the Practice Session
👉 [Ready to run in Google Colab](https://colab.research.google.com/github/hongsukyi/Lectures/blob/main/AI_NLP_Course/notebook/Week03.ipynb)

## From a Linear Score to a Class Probability

This lecture predicts Pass/Fail from two input features: it computes a linear score $z$, turns $z$ into a probability with sigmoid, decides the class with a 0.5 threshold, and trains with `BCEWithLogitsLoss`.

## 1. The Example Dataset

Two features — study time (hours/week) and attendance (%) — predict a binary target: Fail = 0, Pass = 1.

```python
X = torch.tensor([
    [1.0, 55.0], [1.5, 60.0], [2.0, 58.0], [2.5, 65.0], [3.0, 62.0],
    [3.2, 66.0], [3.5, 70.0], [4.0, 75.0], [4.2, 68.0], [4.5, 62.0],
    [4.8, 78.0], [5.1, 88.0], [5.5, 80.0], [5.8, 90.0], [6.0, 84.0],
    [6.3, 86.0], [6.8, 92.0], [7.2, 94.0], [7.5, 89.0], [8.0, 96.0],
], dtype=torch.float32)

y = torch.tensor([
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=torch.float32).reshape(-1, 1)
```

![Binary Classification Data](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/01_binary_classification_data.png)

The two classes are roughly separable, but not perfectly, by a straight boundary.

## 2. Feature Standardization

Study time and attendance sit on very different numeric scales (roughly 1–8 vs. 55–96). Standardizing each feature to zero mean and unit variance makes training more stable.

```python
feature_mean = X.mean(dim=0, keepdim=True)
feature_std = X.std(dim=0, keepdim=True)
X_scaled = (X - feature_mean) / feature_std
```

## 3. Linear Score and Sigmoid

Logistic regression first computes a linear score $z=\mathbf{w}^\top\mathbf{x}+b$, then applies sigmoid: $p=\sigma(z)=\dfrac{1}{1+e^{-z}}$.

```python
demo_w = torch.tensor([[0.8, 1.2]])
demo_b = torch.tensor([0.1])

demo_z = X_scaled[:3] @ demo_w.T + demo_b
demo_p = torch.sigmoid(demo_z)
```

```python
z_values = torch.linspace(-6, 6, 300)
p_values = torch.sigmoid(z_values)
```

![Sigmoid Function](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/02_sigmoid_function.png)

The characteristic S-curve: $\sigma(0)=0.5$, approaching 0 for very negative $z$ and 1 for very positive $z$.

## 4. Logistic Regression as One Neuron

`nn.Linear(2, 1)` computes exactly the linear score: 2 inputs, 2 weights, 1 bias, 1 output logit. The probability is obtained *after* training with `torch.sigmoid(logit)`.

```python
torch.manual_seed(42)
model = nn.Linear(2, 1)

module_logits = model(X_scaled)
manual_logits = X_scaled @ model.weight.T + model.bias
# nn.Linear and manual calculation are equal: True
```

## 5. BCEWithLogitsLoss

PyTorch's `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in one numerically stable operation. During training, **raw logits** are passed in — not sigmoid outputs.

```python
loss_fn = nn.BCEWithLogitsLoss()

initial_logits = model(X_scaled)
initial_loss = loss_fn(initial_logits, y)
```

## 6. Training Logistic Regression

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 1000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(X_scaled)
    loss = loss_fn(logits, y)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
```

![Logistic Regression Training Loss](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/03_training_loss.png)

## 7. Probability, Class Prediction, and the Confusion Matrix

Probability $\geq 0.5$ → class 1; probability $< 0.5$ → class 0.

```python
model.eval()
with torch.no_grad():
    logits = model(X_scaled)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    training_accuracy = (predictions == y).float().mean()
# Training accuracy: 0.950
```

A confusion matrix can be built directly by iterating over (actual, predicted) pairs — rows are the actual class, columns are the predicted class:

```python
confusion = torch.zeros((2, 2), dtype=torch.int64)

for actual, predicted in zip(y.squeeze().to(torch.int64), predictions.squeeze().to(torch.int64)):
    confusion[actual, predicted] += 1
```

## 8. Plotting the Decision Boundary

The location where probability equals 0.5 is the decision boundary. A fine grid is built over the feature space, standardized the same way as the training data, and the model's probability is evaluated across the whole grid before contouring at the 0.5 level.

```python
hour_grid = torch.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 220)
attendance_grid = torch.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5, 220)
H, A = torch.meshgrid(hour_grid, attendance_grid, indexing="xy")

grid = torch.stack([H.reshape(-1), A.reshape(-1)], dim=1)
grid_scaled = (grid - feature_mean) / feature_std

with torch.no_grad():
    grid_probability = torch.sigmoid(model(grid_scaled)).reshape(H.shape)
```

![Learned Linear Decision Boundary](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/04_decision_boundary.png)

## 9. Predicting for a New Student

A new example is standardized using the **same** training mean/std, passed through the model to get a logit, converted to a probability with sigmoid, and thresholded at 0.5 for the final class.

```python
new_student = torch.tensor([[4.5, 80.0]])
new_student_scaled = (new_student - feature_mean) / feature_std

with torch.no_grad():
    new_logit = model(new_student_scaled)
    new_probability = torch.sigmoid(new_logit)
    new_prediction = int(new_probability.item() >= 0.5)
```

## 10. Reading the Learned Parameters

In the standardized feature space, a **positive** weight means that as the feature increases, the chance of class 1 increases; a **negative** weight means the opposite.

```python
learned_parameters = pd.DataFrame({
    "parameter": ["study_time weight", "attendance weight", "bias"],
    "value": [
        model.weight[0, 0].item(),
        model.weight[0, 1].item(),
        model.bias[0].item(),
    ],
})
```

## Practice Problem: Logistic Regression for Admission Prediction

This week's practice problem applies the same logistic regression pipeline from the lecture to a new binary classification task: predicting university admission decisions from two exam scores.

**Scenario**: A university admissions office wants to predict Admit (1) / Reject (0) from two exam scores.

| Math score | English score | Target |
|---|---|---|
| 40 | 45 | 0 |
| 45 | 50 | 0 |
| 50 | 48 | 0 |
| 52 | 55 | 0 |
| 55 | 58 | 0 |
| 58 | 60 | 0 |
| 60 | 65 | 1 |
| 65 | 68 | 0 |
| 68 | 72 | 1 |
| 70 | 75 | 1 |
| 72 | 78 | 1 |
| 75 | 80 | 1 |
| 78 | 85 | 1 |
| 82 | 88 | 1 |
| 85 | 90 | 1 |
| 90 | 92 | 1 |

**Task**: Using the same method as this lecture — standardize the two features, build `nn.Linear(2, 1)`, and train with `BCEWithLogitsLoss` and `SGD(lr=0.1)` for 1000 epochs — complete the following:

1. Train a logistic regression model on this data.
2. Plot the training loss curve.
3. Compute training accuracy and the confusion matrix.
4. Plot the decision boundary.
5. Predict the outcome for a new applicant with Math=63, English=70.

The core of the exercise is the same as in the lecture: a single linear layer produces a raw score, `BCEWithLogitsLoss` combines the sigmoid and binary cross-entropy internally for numerical stability, and the decision boundary is recovered by thresholding the predicted probability at 0.5.

![Practice: Training Loss](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/05_practice_training_loss.png)
![Practice: Decision Boundary](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week03/06_practice_decision_boundary.png)
