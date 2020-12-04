import gym


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        """
        Normalizes the actions to be in between action_space.high and action_space.low.
        If action_space.low == -action_space.high, this is equals to action_space.high*action.
        :param action:
        :return: normalized action
        """
        # take from [-a, a] to [-1, 1]
        action = action / self.action_space.high
        return action

    def reverse_action(self, action):
        """
        Reverts the normalization
        :param action:
        :return:
        """
        action = action * self.action_space.high
        return action
