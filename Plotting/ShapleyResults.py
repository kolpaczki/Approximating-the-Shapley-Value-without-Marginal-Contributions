import math
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# # --------------------------------------------- #
# FILE = 'evaluation__10-24-2022_19:46:36.json'
# # --------------------------------------------- #
#
# with open(f'../Data/{FILE}', mode='r') as file:
#     data = json.load(file)


def plot_results(data):
    # general information
    #pprint.pprint(data)    # prints all data
    used_methods = data['approx_methods']
    true_shapley_value = data['experiments'][f'method: {used_methods[0]}']['run: 0']['GameInformation']['ground_truth_shapley_value']
    number_of_players = data['experiments'][f'method: {used_methods[0]}']['run: 0']['GameInformation']['number_of_players']
    number_of_runs = data['number_of_runs']
    budget = data['budget']
    v_N = data['experiments'][f'method: {used_methods[0]}']['run: 0']['v(N)']

    # runtimes
    runtimes = data['runtimes']
    mean_runtimes = np.mean(runtimes, axis=1)
    var_runtimes = np.var(runtimes, axis=1)

    # shapley value
    shapley_values = data['ShapleyValues']
    shapley_values = [[np.array(x) for x in y] for y in shapley_values]
    mean_shapley_values = np.mean(shapley_values, axis=1)
    var_shapley_values = np.var(shapley_values, axis=1)

    # mse
    mse = data['MeanSquaredError']
    algos = np.shape(mse)[0]
    runs = np.shape(mse)[1]
    mean_mse = np.mean(mse, axis=1)
    var_mse = np.var(mse, axis=1)
    std_mse = [math.sqrt((1.0 / (runs-1.0)) * i) for i in var_mse]

    #printing
    for i in range(0, algos):
        print(f'Algo {i+1}: {mean_mse[i]}\t{std_mse[i]}')

    # ------ plotting ----- #
    plt.rcParams['figure.figsize'] = (12, 8)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Experiment Analysis', fontsize=30)
    fig.tight_layout()

    # description
    text = f'Every sampling method was run {number_of_runs} times with a budget of {budget}. \n' \
           f'Game contains {number_of_players} players and v(N) = {v_N}'
    text = ax[0].text(-0.05, 0.03, text, fontsize=17)

    # first plot
    df = pd.DataFrame(data={'mean_mse': mean_mse, 'std_mse': std_mse,
                            'mean_time': mean_runtimes, 'var_time': var_runtimes})
    df = df.round(decimals=10)

    fig.patch.set_visible(False)
    ax[0].axis('off')
    ax[0].axis('tight')
    table = ax[0].table(cellText=df.values,
                        colLabels=df.columns,
                        rowLabels=used_methods,
                        rowLoc='right',
                        loc='center', fontsize=20)
    table.auto_set_font_size(False)
    table.scale(1, 1.5)

    # second plot
    x = [str(i) for i in range(number_of_players)]
    ax[1].set_title('Shapley Value Estimation Results')
    ax[1].set_xlabel('players')
    ax[1].set_ylabel('shapley_value')
    markers = ['*', '^', '>', '<', '1', '8', 'P', '*', '^', '>', '<', '1', '8', 'P']
    for i, label in enumerate(used_methods):
        ax[1].plot(x, mean_shapley_values[i], linestyle='None', marker=markers[i], label=label)
    ax[1].plot(x, true_shapley_value, linestyle='None', marker='.', label='TrueValues')
    ax[1].grid()
    ax[1].legend()

    plt.subplots_adjust(left=0.16)
    plt.show()
