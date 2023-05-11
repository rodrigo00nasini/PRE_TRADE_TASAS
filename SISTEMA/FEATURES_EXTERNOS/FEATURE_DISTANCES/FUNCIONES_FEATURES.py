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



# FEATURES DISTANCIAS:
def euclidean_distance(series1, series2):
    return np.sqrt(np.sum((series1 - series2) ** 2))

def manhattan_distance(series1, series2):
    return np.sum(np.abs(series1 - series2))

def minkowski_distance(series1, series2):
    return np.sum(np.abs(series1 - series2) ** 3) ** (1 / 3)

def chebyshev_distance(series1, series2):
    return np.max(np.abs(series1 - series2))

def cosine_distance(series1, series2):
    dot_product = np.dot(series1, series2)
    norm_series1 = np.linalg.norm(series1)
    norm_series2 = np.linalg.norm(series2)
    return 1 - (dot_product / (norm_series1 * norm_series2))
    return rcv

def canberra_distance(series1, series2):
    return np.sum(np.abs(series1 - series2) / (np.abs(series1) + np.abs(series2)))

def bray_curtis_dissimilarity(series1, series2):
    return np.sum(np.abs(series1 - series2)) / np.sum(np.abs(series1 + series2))

def jensen_shannon_divergence(series1, series2):
    series1 = np.array(series1)
    series2 = np.array(series2)
    average = (series1 + series2) / 2
    return (entropy(series1, average) + entropy(series2, average)) / 2

def chi_square_distance(series1, series2):
    return 0.5 * np.sum(((series1 - series2) ** 2) / (series1 + series2 + np.finfo(float).eps))

def hellinger_distance(series1, series2):
    return np.sqrt(0.5 * np.sum((np.sqrt(series1) - np.sqrt(series2)) ** 2))


def earth_movers_distance(series1, series2):
    return wasserstein_distance(series1, series2)



def scoring(data, short_w, long_w, distance_function):
    short_window = short_w 
    long_window = long_w 

    #SHORT WINDOW
    data["SPREAD_DISTANCIA_PLAZO_CERCANO_SHORT"] = data["SPREAD_COLOCAR_PLAZO_CERCANO"].rolling(window=short_window).apply(distance_function, args=(data["SPREAD_TOMAR_PLAZO_CERCANO"],))
    
    data["SPREAD_DISTANCIA_PLAZO_LEJANO_SHORT"] = data["SPREAD_COLOCAR_PLAZO_LEJANO"].rolling(window=short_window).apply(distance_function, args=(data["SPREAD_TOMAR_PLAZO_LEJANO"],))

    #LONG WINDOW 
    data["SPREAD_DISTANCIA_PLAZO_CERCANO_LONG"] = data["SPREAD_COLOCAR_PLAZO_CERCANO"].rolling(window=long_window).apply(distance_function, args=(data["SPREAD_TOMAR_PLAZO_CERCANO"],))
    data["SPREAD_DISTANCIA_PLAZO_LEJANO_LONG"] = data["SPREAD_COLOCAR_PLAZO_LEJANO"].rolling(window=long_window).apply(distance_function, args=(data["SPREAD_TOMAR_PLAZO_LEJANO"],))

    data["SCORE"] = 0

    # COMPARAR DISTANCIAS POR QUANTILES
    #QUANTIL 50
    data.loc[data["SPREAD_DISTANCIA_PLAZO_CERCANO_SHORT"]>
data["SPREAD_DISTANCIA_PLAZO_CERCANO_LONG"].quantile(0.50), "SCORE"] += 1
    
    data.loc[data["SPREAD_DISTANCIA_PLAZO_LEJANO_SHORT"] > data["SPREAD_DISTANCIA_PLAZO_LEJANO_LONG"].quantile(0.50), "SCORE"] += 1
    
    #QUANTIL 65

    data.loc[data["SPREAD_DISTANCIA_PLAZO_CERCANO_SHORT"]>
data["SPREAD_DISTANCIA_PLAZO_CERCANO_LONG"].quantile(0.65), "SCORE"] += 1
    
    data.loc[data["SPREAD_DISTANCIA_PLAZO_LEJANO_SHORT"] > data["SPREAD_DISTANCIA_PLAZO_LEJANO_LONG"].quantile(0.65), "SCORE"] += 1

    #QUANTIL 80
    data.loc[data["SPREAD_DISTANCIA_PLAZO_CERCANO_SHORT"]>
data["SPREAD_DISTANCIA_PLAZO_CERCANO_LONG"].quantile(0.80), "SCORE"] += 1
    
    data.loc[data["SPREAD_DISTANCIA_PLAZO_LEJANO_SHORT"] > data["SPREAD_DISTANCIA_PLAZO_LEJANO_LONG"].quantile(0.80), "SCORE"] += 1

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
