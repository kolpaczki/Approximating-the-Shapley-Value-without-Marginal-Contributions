from abc import abstractmethod


class BaseGame:
    @abstractmethod
    def get_game_information(self):
        pass

    @abstractmethod
    def get_value(self, S):
        pass

    @abstractmethod
    def get_player_number(self):
        pass

    @abstractmethod
    def get_shapley_values(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

