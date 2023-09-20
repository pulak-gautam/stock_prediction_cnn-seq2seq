import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tqdm import tqdm #for making cli


def read_data(filepath):
    df = pd.read_csv('data/GOOG-year.csv')
    df.head()

    #normalize data
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))

    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)
    df_log.head()

    return df, df_log, minmax

def plot_results(df, results, accuracies,test_size):
    plt.figure(figsize = (15, 5))
    for no, r in enumerate(results):
        plt.plot(r, label = 'forecast %d'%(no + 1))
    plt.plot(df['Close'].iloc[-test_size:].values, label = 'true trend', c = 'black')
    plt.legend()
    plt.title('average accuracy: %.4f'%(np.mean(accuracies)))
    plt.show()
