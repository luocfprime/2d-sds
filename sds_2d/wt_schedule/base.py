from matplotlib import pyplot as plt

from ..utils.config import ValidateConfigMixin

# class fn_getitem_to_call:  # noqa
#     def __init__(self, fn):
#         self.fn = fn
#
#     def __getitem__(self, idx):
#         return self.fn(idx)


class BaseWTSchedule(ValidateConfigMixin):
    """
    Base class for all weight and timestep schedules
    """

    def t_schedule(self, optimization_step):
        """
        Get the timestep for the given optimization step according to the timestep schedule
        """
        raise NotImplementedError

    def w_schedule(self, optimization_step):
        """
        Get the weight for the given optimization step according to the weight schedule
        """
        raise NotImplementedError

    def show_plot(self, n_steps):
        """
        Plot the weight and timestep schedules
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        steps = list(range(n_steps))
        axes[0].plot(steps, [self.w_schedule(step) for step in steps])
        axes[0].set_title("Weight Schedule")
        axes[0].set_xlabel("Optimization Step")
        axes[0].set_ylabel("Weight")

        axes[1].plot(steps, [self.t_schedule(step) for step in steps])
        axes[1].set_title("Timestep Schedule")
        axes[1].set_xlabel("Optimization Step")
        axes[1].set_ylabel("Timestep")
        plt.show()
