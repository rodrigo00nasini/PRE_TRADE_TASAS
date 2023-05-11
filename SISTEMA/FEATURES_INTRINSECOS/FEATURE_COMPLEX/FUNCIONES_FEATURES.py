import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.io.json import json_normalize
import numpy as np
import scipy 
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from scipy.stats import skew, kurtosis, entropy, iqr, median_abs_deviation
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from math import sqrt
import sklearn.metrics as metrics
import sklearn.metrics as sm
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from scipy.stats import entropy
from numpy.linalg import norm
from scipy.stats import wasserstein_distance
import pywt
from scipy.signal import stft
from skimage.transform import radon
from scipy.fftpack import dst
from sympy import symbols, integrate, exp
from scipy.stats import entropy
from math import log

def find_signals(time_series):
    start_index = None
    max_abs_val = None
    signals = []
    
    for i, value in enumerate(time_series):
        if start_index is None and value <= -0.06:
            start_index = i
            max_abs_val = value
        elif start_index is not None:
            if value < max_abs_val:
                max_abs_val = value
            
            if value >= 0.03:
                signals.append((start_index, i, max_abs_val))
                start_index = None
                max_abs_val = None
    
    df = pd.DataFrame(signals, columns=['Start index', 'End index', 'Maximum Abs Value'])
    df["Duracion_Trade"]=df["End index"]-df["Start index"]
    return df



def hurst_rescaled_range(data):
    n = len(data)
    mean_data = np.mean(data)
    deviations = data - mean_data
    cumulative_dev = np.cumsum(deviations)
    min_cumulative_dev, max_cumulative_dev = np.min(cumulative_dev), np.max(cumulative_dev)
    R = max_cumulative_dev - min_cumulative_dev
    S = np.std(data)
    return np.log(R / S) / np.log(n)

def rolling_hurst_exp(data, window):
    hurst_exp = np.zeros(len(data))

    for i in range(window, len(data)):
        window_data = data[i - window : i]

        try:
            H = hurst_rescaled_range(window_data)
            hurst_exp[i] = H
        except FloatingPointError:
            hurst_exp[i] = 0.5  # Default value for cases where the computation fails

    return hurst_exp

def compute_entropy(series):
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    return entropy(probabilities) / log(len(series))  # Normalize the entropy


def variance_ratio_test(series, lag=4):
    n = len(series)
    mu = series.diff().mean()
    m = ((n - 1) / lag) * np.sum((series.diff().dropna() - mu)**2) 
    y = np.sum(np.square(series.diff(lag).dropna() - lag*mu))
    return y / m

def rolling_variance_ratio_test(series, window, lag=2):
    return series.rolling(window).apply(variance_ratio_test, args=(lag,))




def scoring(trading_spread, short_w, long_w, distance_function):
    short = short_w
    long=long_w

    data=pd.DataFrame()
    data["Trading_Spread"]=trading_spread["spread"]
    data["HURST_LONG"]=rolling_hurst_exp(trading_spread["spread"], long)
    data["HURST_SHORT"]=rolling_hurst_exp(trading_spread["spread"], short)

    data["ENTROPY_LONG"] = trading_spread["spread"].rolling(long).apply(compute_entropy, raw=False)
    data["ENTROPY_SHORT"] = trading_spread["spread"].rolling(short).apply(compute_entropy, raw=False)

    data["VARIANCE_RATIO_LONG"]= rolling_variance_ratio_test( trading_spread["spread"], window=long)
    data["VARIANCE_RATIO_SHORT"]= rolling_variance_ratio_test( trading_spread["spread"], window=short)


    data["SCORE"] = 0
    #QUANTIL 50
    data.loc[data["HURST_SHORT"] > data["HURST_LONG"].quantile(0.50), "SCORE"] += 1
    data.loc[data["ENTROPY_SHORT"] > data["ENTROPY_LONG"].quantile(0.50), "SCORE"] += 1
    data.loc[data["VARIANCE_RATIO_SHORT"] > data["VARIANCE_RATIO_LONG"].quantile(0.50),"SCORE"] += 1
    
    #QUANTIL 65
    data.loc[data["HURST_SHORT"] > data["HURST_LONG"].quantile(0.65), "SCORE"] += 1
    data.loc[data["ENTROPY_SHORT"] > data["ENTROPY_LONG"].quantile(0.65), "SCORE"] +=1
    data.loc[data["VARIANCE_RATIO_SHORT"] > data["VARIANCE_RATIO_LONG"].quantile(0.65),"SCORE"] +=1

     #QUANTIL 80
    data.loc[data["HURST_SHORT"] > data["HURST_LONG"].quantile(0.80), "SCORE"] += 1
    data.loc[data["ENTROPY_SHORT"] > data["ENTROPY_LONG"].quantile(0.80), "SCORE"] += 1
    data.loc[data["VARIANCE_RATIO_SHORT"] > data["VARIANCE_RATIO_LONG"].quantile(0.80),"SCORE"] += 1

    return data


def find_signals(time_series):
    start_index = None
    max_abs_val = None
    signals = []
    
    for i, value in enumerate(time_series):
        if start_index is None and value <= -0.06:
            start_index = i
            max_abs_val = value
        elif start_index is not None:
            if value < max_abs_val:
                max_abs_val = value
            
            if value >= 0.03:
                signals.append((start_index, i, max_abs_val))
                start_index = None
                max_abs_val = None
    
    df = pd.DataFrame(signals, columns=['Start index', 'End index', 'Maximum Abs Value'])
    df["Duracion_Trade"]=df["End index"]-df["Start index"]
    return df


def scoring_vs_max_spread(ts_trading,scoring_ts,filtracion_n_trades):
    
    final_df=pd.DataFrame()
    final_df["Total_Spread"]=ts_trading
    final_df["Total_Score_Parametro_1_negativo"]=scoring_ts["SCORE"]

    time_series = final_df["Total_Spread"]
    result = find_signals(time_series)

    df_1 = result
    df_2 = final_df["Total_Score_Parametro_1_negativo"]
    indices = df_1["Start index"].values
    df_score_spread = df_2.loc[indices]

    df_score_filtrado=pd.DataFrame(df_score_spread)
    result["Score_Inicio_Trade"]=df_score_filtrado["Total_Score_Parametro_1_negativo"].values

    result_sorted = result.sort_values('Maximum Abs Value', ascending=True)
    result_sorted.dropna(inplace=True)
    result_filter=result_sorted[result_sorted["Duracion_Trade"]<filtracion_n_trades]
    return result_filter


def feature_importance(final_spread_score):
    x=final_spread_score['Maximum Abs Value']
    x = np.array(x).reshape(-1, 1)
    y=final_spread_score['Score_Inicio_Trade']


    sc=StandardScaler().fit(x)
    x_sc=sc.transform(x)

    lm=LinearRegression()
    lm.fit(x_sc, y)
    predictions=lm.predict(x_sc)
    correlation, _ = pearsonr(final_spread_score['Maximum Abs Value'], final_spread_score['Score_Inicio_Trade'])
    
    print("Metricas De Relacion Scoring y Maximum Abs")
    print("  ")
    print("Mean absolute error =", round(sm.mean_absolute_error(y, predictions), 2)) 
    print("Mean squared error =", round(sm.mean_squared_error(y, predictions), 2)) 
    print("Median absolute error =", round(sm.median_absolute_error(y, predictions), 2)) 
    print("Explain variance score =", round(sm.explained_variance_score(y, predictions), 2)) 
    print("R2 score =", round(sm.r2_score(y, predictions), 2))
    print("Coeficiente de Correlación coefficient =", round(correlation, 2))
