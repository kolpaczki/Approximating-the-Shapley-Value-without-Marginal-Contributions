import matplotlib.pyplot as plt


# # --------------------------------------------- #
# METHOD = 'NaiveSampling'
# RUN = 1
# FILE = 'evaluation__10-24-2022_17:14:36.json'
# # --------------------------------------------- #
#
# with open(f'../Data/{FILE}', mode='r') as file:
#     data = json.load(file)


def plot_sample_experiment(data, RUN):
    experiment_data = data['experiments']
    game_data = data['GameInformation']
    number_of_players = game_data['number_of_players']
    methods = data['approx_methods']

    # setup plotting
    plt.rcParams['figure.figsize'] = (12, 8)
    if len(methods) == 1:
        number_of_axes = len(methods) + 1
    else:
        number_of_axes = len(methods)
    fig, ax = plt.subplots(number_of_axes, 1)
    fig.tight_layout(pad=5.0)
    fig.suptitle('Sample run for every methods', fontsize=25)

    for i, label in enumerate(methods):
        tmp_data = experiment_data[f'method: {label}'][f'run: {RUN}']['history']
        # first plot
        ax[i].set_title(f'Experiment: Shapley Value History (Method:{label}, Run: {RUN})')
        ax[i].set_xlabel('time_step')
        ax[i].set_ylabel('Shapley_value')

        for j in range(number_of_players):
            value_history = tmp_data[j]
            ax[i].plot(value_history)
            ax[i].grid()
    plt.show()


