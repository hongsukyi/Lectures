# Lecture 04: Multi-Layer Perceptron for Nonlinear Binary Classification

### 🔗 Start the Practice Session
👉 [Ready to run in Google Colab](https://colab.research.google.com/github/hongsukyi/Lectures/blob/main/AI_NLP_Course/notebook/Week04.ipynb)

## From One Neuron to Hidden Layers

This lecture moves from a single sigmoid neuron (logistic regression) to a small neural network that can learn nonlinear patterns.

## 1. A Nonlinear Binary Dataset: Two Moons

`make_moons` creates two curved, crescent-shaped classes. A single straight line cannot separate them well.

```python
X_numpy, y_numpy = make_moons(
    n_samples=300,
    noise=0.20,
    random_state=42,
)

X_all = torch.tensor(X_numpy, dtype=torch.float32)
y_all = torch.tensor(y_numpy, dtype=torch.float32).reshape(-1, 1)
```

![Nonlinear Two-Moons Dataset](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/01_two_moons_dataset.png)

## 2. Train/Validation Split and Standardization

Standardization uses **only** the training set's mean and standard deviation — the validation set is transformed with those same training statistics, never refit on its own.

```python
X_train_numpy, X_val_numpy, y_train_numpy, y_val_numpy = train_test_split(
    X_numpy, y_numpy,
    test_size=0.25,
    random_state=42,
    stratify=y_numpy,
)

train_mean = X_train.mean(dim=0, keepdim=True)
train_std = X_train.std(dim=0, keepdim=True)

X_train_scaled = (X_train - train_mean) / train_std
X_val_scaled = (X_val - train_mean) / train_std
```

## 3. Building Two Models Side by Side

- **Logistic regression** (`nn.Linear(2, 1)`): no hidden layer, one linear boundary.
- **MLP** (`nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))`): one hidden layer with ReLU, able to form a nonlinear boundary.

```python
torch.manual_seed(42)
linear_model = nn.Linear(2, 1)

torch.manual_seed(42)
mlp_model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
```

## 4. ReLU Activation

$$\text{ReLU}(z)=\max(0,z)$$

```python
relu_input = torch.linspace(-4, 4, 200)
relu_output = torch.relu(relu_input)
```

![ReLU Activation](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/02_relu_activation.png)

## 5. Training Both Models

Both models use the identical training loop — forward pass, `BCEWithLogitsLoss`, backward, optimizer step — with different learning rates suited to each architecture:

```python
loss_fn = nn.BCEWithLogitsLoss()
linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.05)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.03)

epochs = 800
for epoch in range(epochs):
    linear_model.train()
    linear_optimizer.zero_grad()
    train_logits = linear_model(X_train_scaled)
    train_loss = loss_fn(train_logits, y_train)
    train_loss.backward()
    linear_optimizer.step()

    linear_model.eval()
    with torch.no_grad():
        val_logits = linear_model(X_val_scaled)
        val_loss = loss_fn(val_logits, y_val)
    # ... same loop shape repeated for mlp_model
```

## 6. Comparing Learning Curves and Accuracy

![Training and Validation Loss](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/03_train_val_loss.png)

The linear model's validation loss plateaus early, while the MLP's continues to drop — a first sign that the hidden layer captures structure the linear model cannot.

```python
metrics = pd.DataFrame({
    "model": ["Logistic regression", "MLP"],
    "train_accuracy": [linear_train_accuracy, mlp_train_accuracy],
    "validation_accuracy": [linear_val_accuracy, mlp_val_accuracy],
    "parameter_count": [linear_parameter_count, mlp_parameter_count],
})
```

## 7. Comparing Decision Boundaries

Both models' probabilities are computed over the same fine grid, standardized with the training mean/std, then plotted side by side.

```python
with torch.no_grad():
    linear_grid_probability = torch.sigmoid(linear_model(grid_scaled)).reshape(G1.shape)
    mlp_grid_probability = torch.sigmoid(mlp_model(grid_scaled)).reshape(G1.shape)
```

![Decision Boundaries: Logistic Regression vs. MLP](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/04_decision_boundaries.png)

Logistic regression → a straight boundary. MLP → a curved, nonlinear boundary that actually follows the two-moons shape.

## 8. What the Hidden Layer Learns

`mlp_model[0]` is the first linear layer, `mlp_model[1]` is ReLU. With a hidden size of 8, each input sample is transformed into an 8-dimensional hidden representation.

```python
mlp_model.eval()
with torch.no_grad():
    hidden_linear = mlp_model[0](X_train_scaled)
    hidden_features = mlp_model[1](hidden_linear)
```

![Two Learned Hidden Features](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/05_hidden_features.png)

Plotting two of the eight hidden features (colored by class) shows the hidden layer has reorganized the data into a more separable representation than the original two input features — many points collapse to 0 (ReLU's flat region), and the two classes spread apart along the remaining active directions.

## 9. Hidden Size and Capacity

Comparing hidden sizes (2, 8, 64) trained for the same number of epochs shows the capacity trade-off directly:

```python
hidden_sizes = [2, 8, 64]
capacity_results = []

for hidden_size in hidden_sizes:
    torch.manual_seed(42)
    temp_model = nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )
    temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.03)
    for epoch in range(500):
        temp_model.train()
        temp_optimizer.zero_grad()
        temp_loss = loss_fn(temp_model(X_train_scaled), y_train)
        temp_loss.backward()
        temp_optimizer.step()
```

- Too small a hidden size (2) → the model may underfit.
- Too large a hidden size (64) → training accuracy is high, but validation performance does not necessarily improve further.

## Practice Problem: MLP vs. Linear Model on Concentric Circles

This week's practice problem repeats the linear-vs-MLP comparison from the lecture, but replaces the two-moons dataset with a harder nonlinear pattern: two concentric circles, where a straight line cannot separate the classes at all.

**Scenario**: Instead of two moons, generate a dataset of two concentric circles with `sklearn.datasets.make_circles` (`n_samples=300`, `noise=0.10`, `factor=0.4`, `random_state=42`) — an inner ring (Class 1) surrounded by an outer ring (Class 0).

**Task**: Using the same method as this lecture, complete the following:

1. Generate the dataset and split it into training (75%) and validation (25%) sets with `train_test_split(..., stratify=...)`.
2. Standardize the features using only the training mean/std.
3. Build the same two models: `linear_model = nn.Linear(2, 1)` and `mlp_model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))`.
4. Train both for 800 epochs with `BCEWithLogitsLoss` (Adam, lr=0.05 for the linear model, lr=0.03 for the MLP).
5. Compare their training/validation accuracy.
6. Plot both decision boundaries side by side.

The key point of this exercise is to see the limitation of a purely linear model directly: `linear_model` can only draw a straight decision boundary, so it cannot separate two concentric rings no matter how it is trained, while `mlp_model`'s hidden ReLU layer lets it bend the boundary into a closed curve that follows the true class structure.

![Practice: Concentric Circles Dataset](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/06_practice_circles_dataset.png)
![Practice: Decision Boundaries](https://cdn.jsdelivr.net/gh/hongsukyi/Lectures@main/AI_NLP_Course/assets/week04/07_practice_decision_boundaries.png)
