import numpy as np

from innovization.constants import *

np.seterr(all='warn')


class DummyRepairAgent:
    def repair(self, x, **kwargs):
        return x


class PowerLawRepairAgent:
    def __init__(self, sig_multiplier=0):
        self.sig_multiplier = sig_multiplier

    def repair(self, x, var_pair, power_law):
        """Performs repair on a solution using the power laws associated with the specified variable pairs."""
        x_rep = np.copy(x)
        v_pair = np.copy(var_pair)
        if var_pair.ndim == 1:
            v_pair = v_pair.reshape([1, -1])
        for i, j in v_pair:
            # KLUGE: c can be inf
            if np.isinf(power_law.c[j, i]) or power_law.c[j, i] > 1000:
                continue
            c_rand = np.random.uniform(power_law.c[j, i] - self.sig_multiplier*power_law.sigma_c[j, i],
                                       power_law.c[j, i] + self.sig_multiplier*power_law.sigma_c[j, i])
            x_rep[j] = c_rand / x_rep[i] ** power_law.b[j, i]

        return x_rep


class InequalityRepairAgent:
    def __init__(self, siv_range=(0, 1)):
        # Default value of any SIV is [0, 1]. But user can change it if they want
        self.siv_range = siv_range

    def repair(self, x, var_pair, rule, xl, xu):
        """Performs repair on a solution using the power laws associated with the specified variable pairs."""
        x_rep = np.copy(x)
        v_pair = np.copy(var_pair)
        if var_pair.ndim == 1:
            v_pair = v_pair.reshape([1, -1])
        for i, j in v_pair:
            siv = self.siv_range[0] + np.random.rand()*(self.siv_range[1] - self.siv_range[0])
            # xi < xj
            if (rule == LESS_THAN or rule == LESS_THAN_OR_EQUAL) and x_rep[i] > x_rep[j]:
                x_rep[j] = x_rep[i] + siv*(xu[j] - x_rep[i])
            # xi > xj
            elif (rule == GREATER_THAN or rule == GREATER_THAN_OR_EQUAL) and x_rep[i] < x_rep[j]:
                x_rep[j] = xl[j] + siv*(x_rep[i] - xl[j])

        return x_rep


if __name__ == '__main__':
    pass
