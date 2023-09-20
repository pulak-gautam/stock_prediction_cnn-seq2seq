import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import seaborn as sns



from tqdm import tqdm #for making cli

import sys
import warnings

from utils import read_data
from utils import plot_results
from model import forecast
from model import calculate_accuracy

if not sys.warnoptions:
    warnings.simplefilter('ignore')


if __name__ == '__main__':

    #data configuration    
    test_size = 30
    simulation_size = 10
    filepath = 'data/GOOG-year.csv'

    #initialization
    sns.set()
    tf.compat.v1.random.set_random_seed(1234)

    df, df_log, transform = read_data(filepath)
    df_train = df_log.iloc[:-test_size]
    df_test = df_log.iloc[-test_size:]
    print("Shape of Data : " + f"df : {df.shape}" + f"df_train : {df_train.shape}" + f"df _test : {df_test.shape}")


    #parameters
    num_layers = 1
    size_layer = 128
    timestamp = test_size
    epoch = 300
    dropout_rate = 0.7
    future_day = test_size
    learning_rate = 1e-3



    #running the model
    results = []
    for i in range(simulation_size):
        print('simulation %d'%(i + 1))
        
        results.append(forecast(num_layers=1, size_layer=128, timestamp=test_size, epoch=300, dropout_rate=0.7, learning_rate=1e-3, 
                                df=df, df_train=df_train, df_log=df_log, test_size=test_size, minmax=transform))

    accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

    plot_results(df, results, accuracies, test_size)
    