import copy
import os

import numpy as np
import pandas as pd

from Experiment.GameEvaluator import GameEvaluator
from utils.file_saving import save_mse_history_fo_file


def run_experiment(game_list, approx_methods, approx_methods_extra, budget, number_of_runs,
                   step_size, use_exact_same_game=False, step_size_rerun_approximator=100, steps=None):

    first_game = game_list[0]

    game_evaluator = GameEvaluator(approx_methods=approx_methods,
                                   budget=budget,
                                   number_of_runs=number_of_runs,
                                   evaluation_penalty=0,
                                   step_size=step_size)
    # to run the exact same game over and over use this method
    if use_exact_same_game:
        evaluation = game_evaluator.evaluate_exact_same_game(first_game)
    # to run the same game class with different instantiations of a game, run this method
    else:
        evaluation = game_evaluator.evaluate_game(game_list)
    # ---------------------------------------------------- #

    if steps is not None:
        steps_iteration = steps
    else:
        steps_iteration = list(range(step_size_rerun_approximator, budget + step_size_rerun_approximator, step_size_rerun_approximator))

    data_for_approx_extra = {}
    for approx_method in approx_methods_extra:
        approximator_mse_history = []
        for budget_step in steps_iteration:
            budget_step = min(budget_step, budget)
            # run the approximator again
            game_evaluator_step = GameEvaluator(
                approx_methods=[approx_method], budget=budget_step,
                number_of_runs=number_of_runs, evaluation_penalty=0, step_size=1
            )
            evaluation_step = game_evaluator_step.evaluate_game(game_list)

            mse_values = copy.deepcopy(evaluation_step['MeanSquaredError'])[0]
            mse_mean = np.mean(mse_values)
            var = np.var(mse_values)
            try:
                mse_std = np.sqrt((1.0 / (number_of_runs - 1.0)) * var)
            except ZeroDivisionError:
                mse_std = np.zeros(mse_mean.shape)
            approximator_mse_history.append(
                {"t": budget_step, "mse": mse_mean, "mseStdErr": mse_std})
        data_for_approx_extra[approx_method.get_name()] = copy.deepcopy(approximator_mse_history)

    # ------------------- plotting ----------------------- #
    # plot_mse_history(evaluation, STEP_SIZE)
    # -----> only makes sense when using the same game
    # plot_results(evaluation)
    # ---------------------------------------------------- #

    # -------------------- toFile ------------------------ #
    # save_experiment_to_file(evaluation)
    # uncomment this line if you do not want to save data in file
    # this creates a directory Data where the evaluation results are stored
    filename = f'{first_game.get_name()}_N{first_game.n}_T{budget}_R{number_of_runs}'
    save_mse_history_fo_file(data=evaluation, filename=filename, step_size=step_size)
    # save_shapley_history_to_file(data=evaluation, filename=filename, step_size=STEP_SIZE)
    # ---------------------------------------------------- #

    for approximator_id, history in data_for_approx_extra.items():
        filename = f'MseHistory_{approximator_id}_{first_game.get_name()}_N{first_game.n}_T{budget}_R{number_of_runs}_{approximator_id}.csv'
        file_path = os.path.join('Data', filename)
        history_df = pd.DataFrame(history)
        history_df.to_csv(file_path, index=False)
