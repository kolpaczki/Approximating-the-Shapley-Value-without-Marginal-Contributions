import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_mse_history(data, step_size):
  experiment_data = data['experiments']
  number_of_runs = data['number_of_runs']
  methods = data['approx_methods']

  df_list = list()

  for method in methods:
    # averaging data
    df = pd.DataFrame(columns=['number_of_samples', 'mse', 'elapsed_time'])
    for run in range(number_of_runs):
      tmp_data = experiment_data[f'method: {method}'][f'run: {run}']
      tmp_df = pd.DataFrame(columns=['number_of_samples', 'mse', 'elapsed_time'])
      tmp_df['mse'] = tmp_data['mse_history']
      tmp_df['elapsed_time'] = np.cumsum(tmp_data['durations'])
      tmp_df['number_of_samples'] = list(range(0, len(tmp_df['mse']) * step_size, step_size))
      df = pd.concat([df, tmp_df])
    df = df.groupby(['number_of_samples']).mean()
    df_list.append((method, df))

    # plotting
    # plt.rcParams['figure.figsize'] = (12, 8)
    # fig, ax = plt.subplots(1, 3)
    # fig.tight_layout(pad=5.0)
    # fig.suptitle(f'Method Evaluation for {method}', fontsize=25)
    #
    # ax[0].set_title(f'MSE vs. SAMPLES')
    # ax[0].set_xlabel('number_of_samples')
    # ax[0].set_ylabel('mse')
    # ax[0].plot(df.index, df['mse'])
    # ax[0].grid()
    #
    # ax[1].set_title(f'MSE vs. ELAPSED_TIME')
    # ax[1].set_xlabel('time_elapsed')
    # ax[1].set_ylabel('mse')
    # ax[1].plot(df['elapsed_time'], df['mse'])
    # ax[1].grid()
    #
    # ax[2].set_title(f'SAMPLES vs. ELAPSED_TIME')
    # ax[2].set_xlabel('time_elapsed')
    # ax[2].set_ylabel('number_of_samples')
    # ax[2].plot(df['elapsed_time'], df.index)
    # ax[2].grid()
    #
    # plt.show()

  # plotting
  plt.rcParams['figure.figsize'] = (12, 8)
  fig, ax = plt.subplots(1, 3)
  fig.tight_layout(pad=5.0)
  fig.suptitle(f'Overall Method Evaluation', fontsize=25)

  for method, df in df_list:
    ax[0].set_title(f'MSE vs. SAMPLES')
    ax[0].set_xlabel('number_of_samples')
    ax[0].set_ylabel('mse')
    ax[0].plot(df.index, df['mse'], label=method)

    ax[1].set_title(f'MSE vs. ELAPSED_TIME')
    ax[1].set_xlabel('time_elapsed')
    ax[1].set_ylabel('mse')
    ax[1].plot(df['elapsed_time'], df['mse'])

    ax[2].set_title(f'SAMPLES vs. ELAPSED_TIME')
    ax[2].set_xlabel('time_elapsed')
    ax[2].set_ylabel('number_of_samples')
    ax[2].plot(df['elapsed_time'], df.index)

  ax[0].grid()
  ax[1].grid()
  ax[2].grid()
  ax[0].legend()
  plt.show()
