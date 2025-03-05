# Stock Price Prediction Using QLSTM

This is the offical implementation of the paper: \
**A Hybrid Quantum-Classical Model for Stock Price Prediction Using Quantum-Enhanced Long Short-Term Memory (Under Review)** \
Kimleang Kea, Dongmin Kim, Prof. Tae-Kyung Kim, and Prof. Youngsun Han

## Introduction
Stock price prediction is a challenging task due to the non-linear and non-stationary nature of the stock market. In this paper, we propose a hybrid quantum-classical model for stock price prediction using quantum-enhanced long short-term memory (QLSTM). The proposed model consists of a classical LSTM with variational quantum circuits (VQCs), which are used to enhance the LSTM's ability to capture the non-linear relationships in the stock price data. The VQCs are trained for 50 epochs using the Adam optimizer. The proposed model is evaluated on the Apple Inc dataset spanning from Jan 1st 2022 to Jan 1st 2023, and the results show that the hybrid quantum-classical model outperforms the classical LSTM model in terms of prediction accuracy and robustness to noise.

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Classical Training
To train the classical LSTM model, open the `stock_price_prediction_c.ipynb` file and run the cells.

### Quantum Training
To train the classical LSTM model, open the `stock_price_prediction_q.ipynb` file and run the cells.
- For noisy and actual data, you need to provide or overwrite your IBMQ token.
```python
from qiskit_ibm_provider import IBMProvider
IBMProvider.save_account('YOUR TOKEN', overwrite=True)
```

## Citation
TODO: Add citation

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://raw.githubusercontent.com/QCL-PKNU/SPP-QLSTM/main/LICENSE) file for details. \
All rights reserved at [PKNU QCL](http://quantum.pknu.ac.kr).
