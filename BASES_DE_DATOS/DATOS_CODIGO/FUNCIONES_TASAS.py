#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis, skew,norm
from pandas.io.json import json_normalize
import numpy as np

def abrir_datos(url):
    with open(url, 'r') as f:
        json_data = json.load(f)
    dfs = []
    for data in json_data:
        tickers_data = data.pop('tickers')
        for ticker, ticker_data in tickers_data.items():
            df_ticker = json_normalize(ticker_data)
            df_ticker.insert(0, 'ticker', ticker)
            df_ticker.insert(1, 'horario_muestreo_datos', data['horario_muestreo_datos'])
            df_ticker.insert(2, 'caucion_a_un_dia_ultimo_trade_tasa', data['caucion_a_un_dia']['ultimo_trade'][0])
            df_ticker.insert(3, 'caucion_a_un_dia_ultimo_trade_monto', data['caucion_a_un_dia']['ultimo_trade'][1])
            dfs.append(df_ticker)
    result = pd.concat(dfs, ignore_index=True)
    return result


def limpiar_datos_tasas(df):
    columns_to_process = [
        'COLOCAR_TASA_ABRIENDO_PLAZO_CERCANO',
        'COLOCAR_TASA_ABRIENDO_PLAZO_LEJANO',
        'TOMAR_TASA_ABRIENDO_PLAZO_CERCANO',
        'TOMAR_TASA_ABRIENDO_PLAZO_LEJANO'
    ]

    for column in columns_to_process:
        tasa_column = column + "_TASA"
        df[tasa_column] = df[column].apply(lambda x: x[0][0] if len(x) > 0 else None)
    df.rename(columns={'horario_muestreo_datos': 'fecha'}, inplace=True)
    df.set_index("fecha",inplace=True)
    df.drop(columns=columns_to_process, inplace=True)
    return df



def limpiar_datos_volumen(df):
    columns_to_process = [
        'COLOCAR_TASA_ABRIENDO_PLAZO_CERCANO',
        'COLOCAR_TASA_ABRIENDO_PLAZO_LEJANO',
        'TOMAR_TASA_ABRIENDO_PLAZO_CERCANO',
        'TOMAR_TASA_ABRIENDO_PLAZO_LEJANO'
    ]

    for column in columns_to_process:
        volumen_column = column + "_VOLUMEN"
        df[volumen_column] = df[column].apply(lambda x: x[0][1] if len(x) > 0 else None)
    df.rename(columns={'horario_muestreo_datos': 'fecha'}, inplace=True)
    df.set_index("fecha",inplace=True)
    df.drop(columns=columns_to_process, inplace=True)
    return df