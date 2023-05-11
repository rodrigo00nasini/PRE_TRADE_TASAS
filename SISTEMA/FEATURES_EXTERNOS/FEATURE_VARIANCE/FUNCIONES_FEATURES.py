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

def find_signals(time_series):
    start_index = None
    max_abs_val = None
    signals = []
    
    for i, value in enumerate(time_series):
        if start_index is None and value < 0.0:
            start_index = i
            max_abs_val = value
        elif start_index is not None:
            if value < max_abs_val:
                max_abs_val = value
            
            if value > 0.0:
                signals.append((start_index, i, max_abs_val))
                start_index = None
                max_abs_val = None
    
    df = pd.DataFrame(signals, columns=['Start index', 'End index', 'Maximum Abs Value'])
    df["Duracion_Trade"]=df["End index"]-df["Start index"]
    return df



# PRIMER FEATURE VAR:
def varianza(data):
    s = pd.Series(data)
    variance = s.var()
    return variance

def mean_absolute(series):
    return series.mad()

def median_absolute(series):
    median = series.median()
    return (series - median).abs().median()

def average_range(series):
    return (series.max() - series.min()) / len(series)

def coefficient_variation(series):
    mean = series.mean()
    std_dev = series.std()
    return std_dev / mean

def quartile_deviation(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return (q3 - q1) / 2

def mean_deviation(data):
    mean = np.mean(data)
    deviation = data - mean
    md = np.mean(np.abs(deviation))
    
    return md

def root_square_deviation(data):
    mean = np.mean(data)
    deviation = data - mean
    rmsd = np.sqrt(np.mean(deviation**2))
    
    return rmsd

def median_deviation_from_the_mean(data):
    median = np.median(data)
    abs_deviation = np.abs(data - median)
    
    return np.median(abs_deviation)

def geometric_std(data):
    log_data = np.log(data)
    log_mean = np.mean(log_data)
    log_std_dev = np.std(log_data)
    
    return np.exp(log_std_dev)

def winsorized_variance(data, trim_percent=0.05):
    trimmed_data = scipy.stats.trimboth(data, trim_percent)
    mean = np.mean(trimmed_data)
    deviation = trimmed_data - mean
    winsorized_deviation = scipy.stats.mstats.winsorize(deviation, limits=trim_percent)
    return np.mean(np.square(winsorized_deviation))

def robust_coefficient_variation(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    rcv = mad / median
    
    return rcv



def scoring(rates_spread_ts,short_w,long_w,dispersion_function):
    data = rates_spread_ts
    short_window = short_w 
    long_window = long_w   

    data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_SHORT"] = data["SPREAD_COLOCAR_PLAZO_CERCANO"].rolling(window=short_window).apply(dispersion_function)
    data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_SHORT"] = data["SPREAD_COLOCAR_PLAZO_LEJANO"].rolling(window=short_window).apply(dispersion_function)
    data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_SHORT"] = data["SPREAD_TOMAR_PLAZO_CERCANO"].rolling(window=short_window).apply(dispersion_function)
    data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_SHORT"] = data["SPREAD_TOMAR_PLAZO_LEJANO"].rolling(window=short_window).apply(dispersion_function)

    data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_LONG"] = data["SPREAD_COLOCAR_PLAZO_CERCANO"].rolling(window=long_window).apply(dispersion_function)
    data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_LONG"] = data["SPREAD_COLOCAR_PLAZO_LEJANO"].rolling(window=long_window).apply(dispersion_function)
    data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_LONG"] = data["SPREAD_TOMAR_PLAZO_CERCANO"].rolling(window=long_window).apply(dispersion_function)
    data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_LONG"] = data["SPREAD_TOMAR_PLAZO_LEJANO"].rolling(window=long_window).apply(dispersion_function)

    data["SCORE"] = 0

    # Compare the short-term and long-term variances
    data.loc[data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_SHORT"] >= data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.50), "SCORE"] += 1
    data.loc[data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_SHORT"] >= data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.50), "SCORE"] += 1

    data.loc[data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_SHORT"] > data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.65), "SCORE"] += 1
    data.loc[data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_SHORT"] > data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.65), "SCORE"] += 1

    data.loc[data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_SHORT"] > data["SPREAD_COLOCAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.80), "SCORE"] += 1
    data.loc[data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_SHORT"] > data["SPREAD_COLOCAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.80), "SCORE"] += 1

    data.loc[data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_SHORT"] >= data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.50), "SCORE"] += 1
    data.loc[data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_SHORT"] >= data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.50), "SCORE"] += 1
    data.loc[data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_SHORT"] > data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.65), "SCORE"] += 1
    data.loc[data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_SHORT"] > data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.65), "SCORE"] += 1

    data.loc[data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_SHORT"] > data["SPREAD_TOMAR_PLAZO_CERCANO_VAR_LONG"].quantile(0.80), "SCORE"] += 1
    data.loc[data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_SHORT"] > data["SPREAD_TOMAR_PLAZO_LEJANO_VAR_LONG"].quantile(0.80), "SCORE"] += 1

    #data.dropna(inplace=True)
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
