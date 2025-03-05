import os
import pandas as pd

import time
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn

from datetime import datetime
from sklearn.metrics import mean_squared_error

from QLSTM import SequenceDataset, QShallowRegressionLSTM


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss


def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


def main():
    columns = ["Close"]
    df = pd.read_csv("./AAPL_2022-01-01_2023-01-01.csv")

    data = df.filter(columns)
    dataset = data.values

    # Splitting the data into train and test
    size = int(len(df) * 0.7)
    df_train = dataset[:size].copy()
    df_test = dataset[size:].copy()

    # Select the features
    df_train = pd.DataFrame(df_train, columns=columns)
    df_test = pd.DataFrame(df_test, columns=columns)

    features = df_train.columns
    target = 'Close'

    # Normalizing the data
    target_mean = df_train.mean()
    target_stdev = df_train.std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()
        df_train[c] = (df_train[c] - mean) / stdev

    for c in df_test.columns:
        mean = df_test[c].mean()
        stdev = df_test[c].std()
        df_test[c] = (df_test[c] - mean) / stdev

    torch.manual_seed(101)

    batch_size = 1
    sequence_length = 3

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))

    learning_rate = 0.01
    num_hidden_units = 16

    results = ""

    os.makedirs("results", exist_ok=True)

    for qubit in range(14, 21):
        quantumModel = QShallowRegressionLSTM(
            num_sensors=len(features),
            hidden_units=num_hidden_units,
            n_qubits=qubit,
            n_qlayers=1,
        )
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(quantumModel.parameters(), lr=learning_rate)

        quantum_loss_train = []
        quantum_loss_test = []

        test_loss = test_model(test_loader, quantumModel, loss_function)

        num_epochs = 50

        start = time.time()

        for ix_epoch in range(num_epochs):
            print(f"Epoch {ix_epoch}\n---------")
            train_loss = train_model(
                train_loader, quantumModel, loss_function, optimizer=optimizer
            )
            test_loss = test_model(test_loader, quantumModel, loss_function)
            epochs_end = time.time()
            print(f"Epoch time: {epochs_end - start}")
            quantum_loss_train.append(train_loss)
            quantum_loss_test.append(test_loss)

        end = time.time()

        # Save quantum loss train and test to a file
        with open(f"results/quantum_loss_qubits_{qubit}.csv", "w") as f:
            f.write("train_loss,test_loss\n")
            for i in range(num_epochs):
                f.write(f"{quantum_loss_train[i]},{quantum_loss_test[i]}\n")

        # Save the model
        torch.save(quantumModel.state_dict(), f"results/model_qubits_{qubit}.pt")

        quantumModel.eval()
        with torch.no_grad():
            train_eval_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )
            test_eval_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            ystar_col = "Model Forecast"
            df_train[ystar_col] = predict(train_eval_loader, quantumModel).numpy()
            df_test[ystar_col] = predict(test_eval_loader, quantumModel).numpy()

            train_rmse = math.sqrt(
                mean_squared_error(df_train["Close"], df_train["Model Forecast"])
            )
            test_rmse = math.sqrt(
                mean_squared_error(df_test["Close"], df_test["Model Forecast"])
            )
            print(f"Train RMSE: {train_rmse}")
            print(f"Test RMSE: {test_rmse}")

            # Calculate the accuracy of the model
            def accuracy(y, y_star):
                return np.mean(np.abs(y - y_star) < 0.1)

            train_accuracy = accuracy(df_train["Close"], df_train["Model Forecast"])
            test_accuracy = accuracy(df_test["Close"], df_test["Model Forecast"])
            print(f"Train accuracy: {train_accuracy}")
            print(f"Test accuracy: {test_accuracy}")

            results += f"Qubits: {qubit}\n"
            results += f"Train time: {end - start}\n"
            results += f"Train RMSE: {train_rmse}\n"
            results += f"Test RMSE: {test_rmse}\n"
            results += f"Train accuracy: {train_accuracy}\n"
            results += f"Test accuracy: {test_accuracy}\n"
            results += "\n"

            with open(f"results/result_qubits_{qubit}.txt", "w") as f:
                f.write(results)


if __name__ == "__main__":
    main()
