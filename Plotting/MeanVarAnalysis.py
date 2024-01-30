import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_mean_var_history(data, step_size, display_type='mean'):
  experiment_data = data['experiments']
  number_of_runs = data['number_of_runs']
  methods = data['approx_methods']

  df_list = list()
  for method in methods:
    # averaging data
    df = pd.DataFrame(columns=['number_of_samples', 'mean', 'var', 'mse'])
    for run in range(number_of_runs):
      tmp_data = experiment_data[f'method: {method}'][f'run: {run}']
      tmp_df = pd.DataFrame(columns=['number_of_samples', 'mean', 'var', 'mse'])
      tmp_df['mse'] = tmp_data['mse_history']
      tmp_df['mean'] = tmp_data['mean_shapley_values']
      tmp_df['var'] = tmp_data['var_shapley_values']
      tmp_df['number_of_samples'] = list(range(0, len(tmp_df['mse']) * step_size, step_size))
      df = pd.concat([df, tmp_df])
    df = df.groupby(['number_of_samples']).mean()
    df_list.append((method, df))

  # plotting
  plt.rcParams['figure.figsize'] = (12, 8)

  for method, df in df_list:
    if display_type == 'mean':
      plt.title(f'Mean Visualisation - {method}')
    elif display_type == 'mean':
      plt.title(f'Variance Visualisation - {method}')
    ax = plt.axes(projection='3d')
    data_points = list()
    var_points = list()
    for element in df.iterrows():
      for i, el in enumerate(element[1]['mean']):
        data_points.append((element[0], i, el))
      for i, el in enumerate(element[1]['var']):
        var_points.append((element[0], i, el))

    x, y, z = zip(*data_points)
    u, v, w = zip(*var_points)
    if display_type == 'mean':
      ax.scatter(x, y, z, label=method)
    elif display_type == 'var':
      ax.scatter(u, v, w, label=method)

    ax.set_xlabel('time_steps t')
    ax.set_ylabel('player i')
    if display_type == 'mean':
      ax.set_zlabel('mean')
    elif display_type == 'var':
      ax.set_zlabel('var')
    ax.set_frame_on(False)
    plt.show()
