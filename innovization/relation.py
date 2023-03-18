import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from innovization.constants import *


class InequalityRelation:
    """Represents the inequality relations learned from a training dataset."""
    # xi <= xj
    def __init__(self, n_var, eq_tol):
        self.n_var = n_var
        self.rule = -1 * np.ones([n_var, n_var])
        # self.ignore_vars = []
        # Tolerance below which variables are considered equal. For discrete variables with step size 1, eq_tol should
        # be set as 1.
        if type(eq_tol) == float or type(eq_tol) == int:
            self.eq_tol = eq_tol * np.ones(n_var)
        else:
            self.eq_tol = eq_tol
        self.min_eq_thresh_leq_geq = 0.3  # When counting <=, how much of it should be made up of = relations

        # Proportion of training data following a rule
        self.rule_score = np.zeros([n_var, n_var])

    def learn(self, training_data, ignore_vars=()):
        less_than = LESS_THAN
        greater_than = GREATER_THAN
        less_than_or_equal = LESS_THAN_OR_EQUAL
        greater_than_or_equal = GREATER_THAN_OR_EQUAL
        equal = EQUAL
        n_var = self.n_var
        # self.ignore_vars = ignore_vars
        n_sol = training_data.shape[0]

        for i in range(n_var - 1):
            if i in self.ignore_vars:
                # self.rule[i, :] = -1
                # self.rule[:, i] = -1
                continue

            for j in range(i + 1, n_var):
                if j in ignore_vars:
                    # self.rule[j, :] = -1
                    # self.rule[:, j] = -1
                    continue

                rule_count = {
                    less_than: 0,
                    greater_than: 0,
                    less_than_or_equal: 0,
                    greater_than_or_equal: 0,
                    equal: 0
                }
                for x in training_data:
                    if np.abs(x[i] - x[j]) <= self.eq_tol[i]:
                        rule_count[equal] += 1
                    elif x[i] < x[j]:
                        rule_count[less_than] += 1
                    elif x[i] > x[j]:
                        rule_count[greater_than] += 1
                rule_count[less_than_or_equal] = rule_count[less_than] + rule_count[equal]
                rule_count[greater_than_or_equal] = rule_count[greater_than] + rule_count[equal]

                majority_rule = np.argmax([rule_count[equal], rule_count[less_than], rule_count[greater_than]])
                if majority_rule == 0:
                    self.rule[i, j] = equal
                    self.rule_score[i, j] = rule_count[equal] / n_sol
                    self.rule[j, i] = self.rule[i, j]
                    self.rule_score[j, i] = self.rule_score[i, j]
                elif majority_rule == 1:
                    # Check if the number of equality instances are significant. If yes, mark the rule as <= .
                    # Else, mark the rule as < . Only relevant for discrete variables
                    if rule_count[equal] / (rule_count[less_than_or_equal]) >= self.min_eq_thresh_leq_geq:
                        self.rule[i, j] = less_than_or_equal
                        self.rule_score[i, j] = rule_count[less_than_or_equal] / n_sol
                        self.rule[j, i] = greater_than_or_equal
                    else:
                        self.rule[i, j] = less_than
                        self.rule_score[i, j] = rule_count[less_than] / n_sol
                        self.rule[j, i] = greater_than
                    self.rule_score[j, i] = self.rule_score[i, j]
                elif majority_rule == 2:
                    if rule_count[equal] / (rule_count[greater_than_or_equal]) >= self.min_eq_thresh_leq_geq:
                        self.rule[i, j] = greater_than_or_equal
                        self.rule_score[i, j] = rule_count[greater_than_or_equal] / n_sol
                        self.rule[j, i] = less_than_or_equal
                    else:
                        self.rule[i, j] = greater_than
                        self.rule_score[i, j] = rule_count[greater_than] / n_sol
                        self.rule[j, i] = less_than
                    self.rule_score[j, i] = self.rule_score[i, j]

        return np.copy(self.rule), np.copy(self.rule_score)

    def repair(self, x_rep, xl, xu, var_pair):
        # xj is always repaired based on xi. If xi is to be repaired then put [j, i] in var_pair instead of [i, j]
        x_backup = np.copy(x_rep)
        v_pair = np.copy(var_pair)
        if var_pair.ndim == 1:
            v_pair = v_pair.reshape([1, -1])
        for i, j in v_pair:
            if self.rule[i, j] == EQUAL:
                x_rep[j] = x_rep[i]
            elif self.rule[i, j] == LESS_THAN:
                x_rep[j] = x_rep[i] + self.eq_tol[i] + np.random.rand()*(xu[j] - x_rep[i])
            elif self.rule[i, j] == GREATER_THAN:
                x_rep[j] = x_rep[i] - self.eq_tol[i] - np.random.rand()*(x_rep[i] - xl[j])
            elif self.rule[i, j] == LESS_THAN_OR_EQUAL:
                x_rep[j] = x_rep[i] + np.random.rand()*(xu[j] - x_rep[i])
            elif self.rule[i, j] == GREATER_THAN_OR_EQUAL:
                x_rep[j] = x_rep[i] - np.random.rand()*(x_rep[i] - xl[j])

            if x_rep[j] < xl[j] or x_rep[j] > xu[j]:
                print("Limit violation during inequality repair.")
                x_rep[j] = x_backup[j]

        return x_rep


class ConstantRule:
    """Represents the constant rules learned from a training dataset."""
    # xi = c
    def __init__(self, n_var, const_tol, normalization_flag=False):
        self.n_var = n_var
        self.c = np.empty(n_var)
        self.c[:] = np.NaN
        self.const_var_flag = np.zeros(n_var)
        self.const_tol = const_tol
        # self.ignore_vars = []

        # Training error
        self.error = 1e100 * np.ones(n_var)
        self.max_error = np.ones(n_var)

        # Indicates whether learning and repair takes place on normalized data
        self.normalization_flag = normalization_flag

    def learn(self, training_data, ignore_vars=()):
        # ignore_vars are useful when we have variable subgroups
        # self.ignore_vars = ignore_vars
        n_var = self.n_var

        mean_x, std_x = np.mean(training_data, axis=0), np.std(training_data, axis=0)
        # TODO: Give a function for checking if a solution complies with an xi = c rule
        const_indx = np.where(std_x <= self.const_tol)[0]
        if len(const_indx) > 0:
            for i in const_indx:
                if i not in ignore_vars:
                    self.const_var_flag[i] = 1
                    self.c[i] = mean_x[i]
        else:
            print("No constant rule found.")

        return np.copy(self.c), np.copy(self.const_var_flag)

    def repair(self, x, var_list=None):
        x_rep = np.copy(x)
        if var_list is None:
            # v_list = np.arange(self.n_var)
            v_list = np.where(self.const_var_flag == 1)
        else:
            v_list = np.copy(var_list)

        for i in v_list:
            x_rep[i] = self.c[i]

        return x_rep

    def check_compliance(self, x_test, var_list, ignore_vars=()):
        compliance_arr = np.zeros(self.n_var)
        for i in var_list:
            if i in ignore_vars:
                continue
            c = self.c[i]
            xi = x_test[i]
            if np.abs(xi - c) <= self.const_tol:
                compliance_arr[i] = 1

        return compliance_arr


class PowerLaw:
    """Represents the power laws learned from a training dataset."""
    # xi * xj^b = c
    SUPPORTED_ERROR_METRICS = ['mse']

    def __init__(self, n_var, error_metric='mse', normalization_flag=False):
        self.n_var = n_var
        self.b = np.zeros([n_var, n_var])
        self.c = np.zeros([n_var, n_var])
        self.sigma_c = np.zeros([n_var, n_var])
        # self.ignore_vars = []

        # Training error
        self.error = np.zeros([n_var, n_var])
        self.max_error = np.ones([n_var, n_var])

        # Indicates whether learning and repair are performed on normalized data
        self.normalization_flag = normalization_flag

        # Sets the error metric to use for determining the goodness of fit
        if error_metric in PowerLaw.SUPPORTED_ERROR_METRICS:
            self.error_metric = error_metric
        else:
            warnings.warn("Unsupported error metric entered.")
            self.error_metric = error_metric

    def update_max_error(self):
        self.max_error = np.maximum(self.max_error, self.error)

    # def perform_linear_regression(self, training_data, log_x, pred_var, independent_var):
    #     """
    #     Perform linear regression where log(xi) is the predicted var and log(xj) is the independent var in order
    #     to learn the relation xi*xj^b = c.
    #     """
    #     xj_log_data = log_x[:, independent_var].reshape(-1, 1)  # Independent variable
    #     xi_log_data = log_x[:, pred_var]  # Predicted variable
    #     reg = LinearRegression().fit(X=xj_log_data, y=xi_log_data)
    #     b = -reg.coef_
    #     try:
    #         c = np.exp(reg.intercept_)
    #     except FloatingPointError:
    #         warnings.warn(f"Floating point error: {pred_var}, {independent_var}")
    #
    #     if self.error_metric == 'mse':
    #         # Predict log(xi) from log(xj) and calculate MSE in logspace
    #         xi_log_predicted = reg.predict(xj_log_data)
    #         mse_logspace = mean_squared_error(xi_log_data, xi_log_predicted)
    #
    #         # Convert predicted log(xi) to xi and re-calculate the MSE
    #         xi_predicted = np.exp(xi_log_predicted)
    #         mse_orig = mean_squared_error(y_true=training_data[:, pred_var], y_pred=xi_predicted)
    #
    #         self.error[pred_var, independent_var] = mse_orig
    #     else:
    #         warnings.warn("Error metric undefined.")
    #     self.b[pred_var, independent_var] = b
    #     self.c[pred_var, independent_var] = c

    def learn(self, training_data, ignore_vars=()):
        """
        Do not supply training data having variables with low standard deviation. Alternately, include such
        variables in the ignore_vars list. Training data should not have 0 or negative values.
        :param training_data: The list of solutions from which pairwise power laws need to be extracted.
        :type training_data: np.ndarray
        :param ignore_vars: Variables to ignore when learning power laws.
        :type ignore_vars: list
        """
        # self.ignore_vars = ignore_vars
        n_var = self.n_var
        log_x = np.log(training_data)
        for i in range(n_var - 1):
            if i in ignore_vars:
                # Set a very high error for vars in ignore list. This is to avoid having to use np.Inf
                # self.error[i, :] = 1e100
                # self.error[:, i] = 1e100
                # self.b[i, :], self.c[i, :] = 0, 0
                # self.b[:, i], self.c[:, i] = 0, 0
                continue

            for j in range(i + 1, n_var):
                if j in ignore_vars:
                    # self.error[j, :] = 1e100
                    # self.error[:, j] = 1e100
                    # self.b[j, :], self.c[j, :] = 0, 0
                    # self.b[:, j], self.c[:, j] = 0, 0
                    continue
                # # i as predicted var, j as independent var
                # self.perform_linear_regression(training_data=training_data, log_x=log_x, pred_var=i, independent_var=j)
                # # j as predicted var, i as independent var
                # self.perform_linear_regression(training_data=training_data, log_x=log_x, pred_var=j, independent_var=i)

                xj_log_data = log_x[:, j].reshape(-1, 1)
                y = log_x[:, i]
                reg = LinearRegression().fit(xj_log_data, y)
                b = -reg.coef_
                try:
                    c = np.exp(reg.intercept_)
                except FloatingPointError:
                    warnings.warn(f"Floating point error: {i}, {j}")
                    continue

                if self.error_metric == 'mse':
                    # Predict log(xi) from log(xj)
                    xi_log_predicted = reg.predict(xj_log_data)
                    mse_logspace = mean_squared_error(y, xi_log_predicted)

                    # Convert predicted log(xi) to xi
                    xi_predicted = np.exp(xi_log_predicted)
                    mse_orig = mean_squared_error(training_data[:, i], xi_predicted)

                    self.error[i, j] = mse_orig
                    self.error[j, i] = self.error[i, j]
                else:
                    warnings.warn("Error metric undefined.")
                self.b[i, j] = b[0]
                self.c[i, j] = c
                # Store the c values for each training data solution and calculate the std of c-values for a
                # particular power law
                c_arr = np.zeros(training_data.shape[0])
                for k, x_train in enumerate(training_data):
                    c_arr[k] = x_train[i] * x_train[j]**self.b[i, j]
                self.sigma_c[i, j] = np.std(c_arr)
                try:
                    self.b[j, i] = 1 / self.b[i, j]
                    self.c[j, i] = c ** self.b[j, i]
                    c_arr = np.zeros(training_data.shape[0])
                    for k, x_train in enumerate(training_data):
                        c_arr[k] = x_train[j] * x_train[i]**self.b[j, i]
                    self.sigma_c[j, i] = np.std(c_arr)
                except FloatingPointError:
                    warnings.warn(f"Floating point error encountered for i = {i}, j = {j}, b[i, j] = {self.b[i, j]}, "
                                  f"c[i, j] = {self.c[i, j]}.")
                    self.error[i, j] = 1e100
                    self.error[j, i] = self.error[i, j]

    def repair(self, x, var_pair):
        """Performs repair on a solution using the power laws associated with the specified variable pairs."""
        x_rep = np.copy(x)
        v_pair = np.copy(var_pair)
        if var_pair.ndim == 1:
            v_pair = v_pair.reshape([1, -1])
        for i, j in v_pair:
            x_rep[j] = self.c[j, i] / x_rep[i]**self.b[j, i]

        return x_rep

    def check_compliance(self, x_test, var_pair, error_threshold, ignore_vars=()):
        """
        Check if a particular solution is compliant with the power laws certain variable pairs.
        :param x_test:
        :type x_test:
        :param var_pair:
        :type var_pair:
        :param error_threshold:
        :type error_threshold:
        :param ignore_vars:
        :type ignore_vars:
        :return:
        :rtype:
        """
        v_pair = np.copy(var_pair)
        if var_pair.ndim == 1:
            v_pair = v_pair.reshape([1, -1])

        # compliance_matrix[i, j] = 1 denotes that power law between xi and xj satisfied by x_test
        compliance_matrix = np.zeros([self.n_var, self.n_var])
        for i, j in v_pair:
            if i in ignore_vars or j in ignore_vars:
                continue
            b, c = self.b[i, j], self.c[i, j]
            xi, xj = x_test[i], x_test[j]
            xi_predicted = c / xj**b

            error_ratio = (xi - xi_predicted)**2  # / self.max_error
            if error_ratio <= error_threshold:
                compliance_matrix[i, j] = 1

        return compliance_matrix
