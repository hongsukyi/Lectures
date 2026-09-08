# Lecture 02: Linear Regression and Gradient Descent

### 🔗 Start the Practice Session
👉 [Ready to run in Google Colab](https://colab.research.google.com/github/hongsukyi/Lectures/blob/main/AI_NLP_Course/notebook/Week02.ipynb)

## How a Model Learns from Prediction Error

This lecture predicts a continuous number from a single input, using the model equation $\hat{y}=wx+b$, measuring error with Mean Squared Error (MSE), and computing gradients with PyTorch's `autograd` — no custom `class` or `def` is used.

## 1. The Example Problem

*Can we predict an exam score from study time?* The input $x$ is study hours, the target $y$ is exam score.

```python
hours = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.]).reshape(-1, 1)
scores = torch.tensor([45., 52., 58., 64., 69., 76., 82., 88.]).reshape(-1, 1)
```

![Study Time and Exam Score](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/01_study_time_vs_score.png)

The data shows a clear rising pattern: more study time is associated with a higher exam score.

## 2. Scaling the Input and Target

For a beginner exercise, scaling values into a small range helps gradient descent move smoothly. The meaning of the original data does not change — only the numeric range used during training.

```python
x = hours / hours.max()
y = scores / 100.0
```

## 3. Building a Linear Model

The model equation is $\hat{y}=wx+b$, where $w$ is the slope and $b$ is the intercept. Both are created with `requires_grad=True`, which tells PyTorch to track gradients for them.

```python
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# Matrix multiplication: [8, 1] @ [1, 1] -> [8, 1]
y_hat = x @ w + b
```

## 4. Computing the Mean Squared Error

$$\text{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$$

```python
errors = y_hat - y
squared_errors = errors ** 2
loss = squared_errors.mean()
```

## 5. One Gradient Step

Calling `loss.backward()` makes PyTorch compute $\partial L/\partial w$ and $\partial L/\partial b$. Moving in the **opposite** direction of the gradient decreases the loss. Gradients must be reset (`.grad.zero_()`) before the next computation, since PyTorch accumulates gradients by default.

```python
loss.backward()

learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

# Reset the gradients before the next computation
w.grad.zero_()
b.grad.zero_()
```

## 6. The Full Training Loop

Every epoch repeats five steps: (1) predict, (2) calculate loss, (3) backward, (4) update parameters, (5) reset gradients.

```python
w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.1; epochs = 800; loss_history = []
for epoch in range(epochs):
    # 1. Predict
    y_hat = x @ w + b

    loss = ((y_hat - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    loss_history.append(loss.item())
```

![Training Loss](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/02_training_loss.png)

The loss drops sharply within the first few dozen epochs and then flattens out near zero.

## 7. Reading the Learned Line

Since the model trained on scaled values, predictions must be converted back to the original scale (multiplying by 100) for plotting and interpretation.

```python
with torch.no_grad():
    predicted_scores = (x @ w + b) * 100.0

# Slope and intercept in the original units
score_per_hour = 100.0 * w.item() / hours.max().item()
score_intercept = 100.0 * b.item()
```

![Linear Regression Result](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/03_learned_line.png)

The learned line passes almost exactly through the data points, and the slope/intercept can be read back in original units — e.g., the approximate score increase per additional hour of study.

## 8. Predicting a New Value

A new input is scaled the same way the training data was scaled, passed through the learned model, and then converted back to the original units.

```python
new_hours = torch.tensor([[6.5]])
new_x = new_hours / hours.max()

with torch.no_grad():
    new_score = (new_x @ w + b) * 100.0
```

## 9. Comparing Learning Rates

- **Too small**: training is slow.
- **Just right**: loss decreases steadily.
- **Too large**: loss can grow or diverge.

```python
learning_rates = [0.01, 0.10, 0.80]
loss_by_lr = {}

for lr in learning_rates:
    temp_w = torch.tensor([[0.0]], requires_grad=True)
    temp_b = torch.tensor([0.0], requires_grad=True)
    temp_history = []
    for epoch in range(120):
        temp_prediction = x @ temp_w + temp_b
        temp_loss = ((temp_prediction - y) ** 2).mean()
        if not torch.isfinite(temp_loss) or temp_loss.item() > 1e12:
            break

        temp_loss.backward()
        with torch.no_grad():
            temp_w -= lr * temp_w.grad
            temp_b -= lr * temp_b.grad

        temp_w.grad.zero_();  temp_b.grad.zero_()
        temp_history.append(temp_loss.item())

    loss_by_lr[lr] = temp_history
```

![Learning Rate Comparison](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/04_learning_rate_comparison.png)

With `lr=0.8`, the loss diverges toward infinity within about 100 epochs (note the log scale). `lr=0.01` decreases too slowly to reach a low loss in 120 epochs, while `lr=0.1` reaches a much lower loss steadily and safely.

## Practice Problem: Linear Regression from Scratch

This week's practice problem steps away from the classification techniques covered so far and returns to the most fundamental regression setup. The goal is to implement gradient descent by hand using only PyTorch's `autograd`, training a linear model from scratch.

**Scenario**: A small coffee shop recorded the outdoor temperature (°C) and the number of iced coffees sold that day.

| Temperature (°C) | Iced coffees sold |
|---|---|
| 15 | 10 |
| 18 | 18 |
| 21 | 25 |
| 24 | 33 |
| 27 | 40 |
| 30 | 48 |
| 33 | 55 |
| 36 | 63 |

**Task**: Using the same approach as in the lecture, scale the data, build a linear model of the form $\hat{y}=wx+b$, and train it with gradient descent for 800 epochs at a learning rate of 0.1. Then complete the following:

1. Train a linear model to predict iced coffees sold from temperature.
2. Plot the training loss curve.
3. Plot the data together with the learned line.
4. Predict how many iced coffees will be sold at 29°C.

The core of the exercise is declaring `w` and `b` with `requires_grad=True`, computing the MSE loss at every epoch, calling `.backward()` to obtain the gradients, and then updating the parameters by hand inside a `torch.no_grad()` block. The point is to see the inner workings of gradient descent directly, without relying on a high-level API like `optimizer.step()`.

![Practice: Training Loss](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/05_practice_training_loss.png)
![Practice: Linear Regression Result](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week02/06_practice_learned_line.png)
