import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize, RobustScaler

def preprocess_flux(flux_array):
    X = flux_array.reshape(1, -1)
    X = np.abs(np.fft.fft(X, axis=1))
    X = savgol_filter(X, 21, 4, deriv=0, axis=1)
    X = normalize(X)
    X = RobustScaler().fit_transform(X.T).T
    return np.expand_dims(X, axis=2)
