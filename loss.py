import numpy as np

def mse(y_true, y_pred):
  return np.mean((y_pred - y_true) ** 2)
def mse_derivative(y_true, y_pred):
  return 2 * np.mean((y_pred - y_true))

if __name__ == '__main__':
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.5, 2.5, 2.8])

    print("MSE:", mse(y_true, y_pred))
    print("MSE Derivative:", mse_derivative(y_true, y_pred))
