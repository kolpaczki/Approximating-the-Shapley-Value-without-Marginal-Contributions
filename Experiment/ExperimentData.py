from Games.AbstractGame import BaseGame



class ExperimentStorage:

  def __init__(self, game: BaseGame):
    self.game = game
    self.shapley_value = list()
    self.player_selected = list()
    self.duration_list = list()
    self.mse_history = list()
    self.shapley_mean_list = list()
    self.shapley_var_list = list()



  def add_shapley_values(self, shapley_value):
    self.shapley_value = shapley_value


  def add_time_evaluation(self, duration):
    self.duration_list.append(duration)


  def add_mse_value(self, mse):
    self.mse_history.append(mse)


  def add_mean_shapley_value(self, mean_value):
    self.shapley_mean_list.append(mean_value)


  def add_var_shapley_value(self, var_value):
      self.shapley_var_list.append(var_value)


  def to_json(self):
    result_dict = dict()
    result_dict['GameInformation'] = self.game.get_game_information()
    result_dict['v(N)'] = self.game.get_value(list(range(self.game.get_player_number())))
    result_dict['shapley_value'] = self.shapley_value
    result_dict['durations'] = self.duration_list
    result_dict['mse_history'] = self.mse_history
    result_dict['mean_shapley_values'] = self.shapley_mean_list
    result_dict['var_shapley_values'] = self.shapley_var_list
    return result_dict
