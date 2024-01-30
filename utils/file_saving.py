import os

import numpy as np
import pandas as pd



def save_mse_history_fo_file(data, filename, step_size):
  isExist = os.path.exists('Data')
  if not isExist:
    os.makedirs('Data')

  experiment_data = data['experiments']
  number_of_runs = data['number_of_runs']
  methods = data['approx_methods']

  df_list = list()

  for method in methods:
    # averaging data over all runs and putting them into a dataframe
    df = pd.DataFrame(columns=['number_of_samples', 'mse_mean', 'elapsed_time_mean', 'mse_var', 'elapsed_time_var'])
    for run in range(number_of_runs):
      tmp_data = experiment_data[f'method: {method}'][f'run: {run}']
      tmp_df = pd.DataFrame(columns=['number_of_samples', 'mse', 'elapsed_time'])
      tmp_df['mse'] = tmp_data['mse_history']
      tmp_df['elapsed_time'] = np.cumsum(tmp_data['durations'])
      tmp_df['number_of_samples'] = list(range(0, len(tmp_df['mse']) * step_size, step_size))
      df = pd.concat([df, tmp_df])
    df_mean = df.groupby(['number_of_samples']).mean()
    df_var = df.groupby(['number_of_samples']).var()
    df_std = np.sqrt((1.0 / (number_of_runs-1.0)) * df_var)
    df = pd.concat([df_mean, df_std], axis=1)
    df.columns = ['mse_mean', 'elapsed_time_mean', 'mse_var', 'elapsed_time_var']
    df_list.append((method, df))

  # save data to file
  for name, df in df_list:
    save_name = 'MseHistory__' + name + '__' + filename
    counter = 1
    tmp_name = save_name + f'__{counter}.txt'
    while os.path.exists(os.path.join('Data', tmp_name)):
      counter += 1
      tmp_name = save_name + f'__{counter}.txt'
    save_name += f'__{counter}.txt'
    df.insert(0, 'num_of_samples', df.index)
    header = 't \t mse \t time \t mseStdErr \t timeVar'
    np.savetxt(os.path.join('Data', save_name), df, delimiter='\t', header=header)
    with open(os.path.join('Data', save_name), mode='r') as file:
      data = file.readlines()
    if os.path.exists(os.path.join('Data', save_name)):
      os.remove(os.path.join('Data', save_name))

    with open(os.path.join('Data', save_name), mode='w') as file:
      data[0] = data[0].split('#')[1].strip()
      data[0] += '\n'
      file.truncate(16)
      file.writelines(data)



def save_shapley_history_to_file(data, filename, step_size, sample_run=False):
  isExist = os.path.exists('Data')
  if not isExist:
    os.makedirs('Data')

  experiment_data = data['experiments']
  methods = data['approx_methods']
  if not sample_run:
    number_of_runs = data['number_of_runs']
  else:
    number_of_runs = 1

  df_list = list()

  for method in methods:
    # averaging data over all runs and putting them into a dataframe
    df = pd.DataFrame()
    for run in range(number_of_runs):
      tmp_data = experiment_data[f'method: {method}'][f'run: {run}']
      tmp_df = pd.DataFrame()
      tmp_df['mean_shapley_value'] = tmp_data['mean_shapley_values']
      tmp_df['var_shapley_value'] = tmp_data['var_shapley_values']
      tmp_df['number_of_samples'] = list(range(0, len(tmp_df['mean_shapley_value']) * step_size, step_size))
      df = pd.concat([df, tmp_df])
    df = df.groupby(['number_of_samples']).mean()
    df_list.append((method, df))

  # save data to file
  for name, df in df_list:
    save_name = 'ShapleyHistory__' + name + '__' + filename
    counter = 1
    tmp_name = save_name + f'__{counter}.csv'
    while os.path.exists(os.path.join('Data', tmp_name)):
      counter += 1
      tmp_name = save_name + f'__{counter}.csv'
    save_name += f'__{counter}.csv'
    with open(os.path.join('Data', save_name), mode='w') as file:
      df.to_csv(file)
