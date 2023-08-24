import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np

from innovization.variable_relation_graph import VariableRelationGraph
from innovization.relation import InequalityRelation, ConstantRule, PowerLaw
from innovization.repair_agent import *
from innovization.constants import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from utils.general import get_all_repair_agents, get_repair_agent
from innovization.constants import *


class VRGInnovization:
    def __init__(self, n_var, vrg=None, rel_type=POWER_LAW_STR, xl=None, xu=None, **kwargs):
        self.n_var = n_var
        self.xl, self.xu = xl, xu

        # Check if variable groups are defined. If not, put all variables in one group.
        if 'groups' in kwargs:
            self.groups = kwargs['groups']
            # Arrange var groups in ascending order for convenience
            for i, g in enumerate(self.groups):
                self.groups[i] = np.sort(g)
        else:
            self.groups = [np.arange(n_var).reshape([1, -1])]

        # For each group, create a separate VRG
        if vrg is None:
            self.vrg = []
            for _ in self.groups:
                self.vrg.append(VariableRelationGraph())
        else:
            self.vrg = vrg

        self.rel_type = rel_type
        self.power_law_normalized = True
        if self.rel_type == POWER_LAW_STR or self.rel_type == POWER_LAW_CONSTANT_RULE_STR:
            if 'power_law_normalized' in kwargs:
                self.power_law_normalized = kwargs['power_law_normalized']
            # First learn constant rules, then learn power laws for the rest. This is necessary since for
            # xi = constant, power law xi * xj^b = c will have b = 0. This will cause problems during
            # the linear regression used for learning the power laws
            self.relation = [PowerLaw(n_var, evaluation_metric_name='r2',
                                      normalization_flag=self.power_law_normalized)]
            if self.rel_type == POWER_LAW_CONSTANT_RULE_STR:
                self.relation = [ConstantRule(n_var, kwargs['const_tol'],
                                              normalization_flag=self.power_law_normalized)] + self.relation

            # Add the repair agents. For power laws we use the ConstantRepairAgent and PowerLawRepairAgent.
            # PowerLawRepairAgent has sigma as a parameter.
            self.repair_agents = [None, []]
            self.repair_agent_names = ['constant_repair_generic', []]
            # all_power_law_rep_agents = get_all_repair_agents()
            if 'agent_names' in kwargs and len(kwargs['agent_names']) > 0:
                for agent_name in kwargs['agent_names']:
                    rep_class, rep_args = get_repair_agent(agent_name)
                    self.repair_agents[1].append(rep_class(**rep_args))
                    self.repair_agent_names[1].append(agent_name)
            else:
                # If no repair agent specified use a default repair agent
                rep_class, rep_args = get_repair_agent('power_law_rep')
                self.repair_agents[1].append(rep_class(**rep_args))

        elif self.rel_type == INEQUALITY_RULE_STR:
            self.relation = [InequalityRelation(self.n_var, kwargs['eq_tol'])]

            self.repair_agents = [[]]
            self.repair_agent_names = [[]]
            if 'agent_names' in kwargs and len(kwargs['agent_names']) > 0:
                for agent_name in kwargs['agent_names']:
                    rep_class, rep_args = get_repair_agent(agent_name)
                    self.repair_agents[0].append(rep_class(**rep_args))
                    self.repair_agent_names[0].append(agent_name)
            else:
                # If no repair agent specified use a default repair agent
                rep_class, rep_args = get_repair_agent('power_law_rep')
                self.repair_agents[0].append(rep_class(**rep_args))
        elif self.rel_type == CONSTANT_RULE_STR:
            self.relation = [ConstantRule(n_var, kwargs['const_tol'], normalization_flag=False)]
        else:
            warnings.warn(f"Unsupported relation type {rel_type} supplied.")

        self.repair_agent_probability = []
        self.repair_agent_min_probability = []
        for agent_ensemble in self.repair_agents:
            if agent_ensemble is None:
                self.repair_agent_probability.append(None)
                self.repair_agent_min_probability.append(None)
            else:
                # For each repair operator in the ensemble, have equal probability. Consider base optimization
                # as a repair operator. Thus for 4 repair operators, the initial probability for each
                # operator will be p(0) = 1 / (4 + 1) = 0.2.
                repair_prob = 1 / len(agent_ensemble)
                self.repair_agent_probability.append([repair_prob for _ in range(len(agent_ensemble))])
                min_probability_ref = 0.1  # Min. recommended selection probability of each repair agent
                if repair_prob <= min_probability_ref:  # Too many repair operators, should be avoided
                    min_probability_ref = repair_prob
                self.repair_agent_min_probability.append([min_probability_ref for _ in range(len(agent_ensemble))])

        self.correlation = np.zeros([n_var, n_var])
        self.min_correlation = 0
        if 'min_correlation' in kwargs:
            self.min_correlation = kwargs['min_correlation']

        self.training_data = None
        self.normalize_to_range = np.array([1, 2])
        self.normalized_NN_radius = 0.05

        self.rule_selection_criteria = {}
        # Select pairwise rules (xi, xj) only if their correlation is above a particular threshold
        if 'min_correlation' in kwargs:
            self.rule_selection_criteria['min_correlation'] = kwargs['min_correlation']
        # Select rule if the evaluation_metric is above a threshold
        if 'max_error' in kwargs:
            self.rule_selection_criteria['max_error'] = kwargs['max_error']
        # Select top n rules based on their evaluation_metric
        if 'top_n' in kwargs:
            self.rule_selection_criteria['top_n'] = kwargs['top_n']

    def normalize_data(self, data):
        x_min, x_max = self.normalize_to_range[0], self.normalize_to_range[1]
        data_normalized = x_min + (data - self.xl) / (self.xu - self.xl) * (x_max - x_min)

        return data_normalized

    def normalize_to_original_range(self, data_normalized):
        x_min, x_max = self.xl, self.xu
        data = x_min + (data_normalized - self.normalize_to_range[0]) / (self.normalize_to_range[1] - self.normalize_to_range[0]) * (x_max - x_min)

        return data

    def select_rule(self, vrg, edge_to_add):
        """
        A function to decide whether to include a particular edge (i, j) in the VRG. Selection criteria can include
        minimum correlation, maximum evaluation_metric, etc.

        :param vrg:
        :type vrg:
        :param edge_to_add:
        :type edge_to_add:
        :return:
        :rtype:
        """
        for edge in edge_to_add:
            i, j = edge
            edge_flag = True
            # If correlation between two variables is lower than minimum specified correlation (absolute),
            # do not add the edge
            if 'min_correlation' in self.rule_selection_criteria \
                    and np.abs(self.correlation[i, j]) < np.abs(self.min_correlation):
                edge_flag = False
            # If evaluation_metric of the fit between the two variables (i, j) under consideration is higher than the maximum
            # allowable evaluation_metric, do not add the edge
            if 'max_error' in self.rule_selection_criteria:
                if self.rel_type == POWER_LAW_STR:
                    error_mat = self.relation[1].evaluation_metric
                else:
                    error_mat = self.relation[0].evaluation_metric
                # TODO: Add simple_rel_type function argument. It denotes abstracted relations like
                #       proportional, inversely proportional, etc.
                if error_mat[i, j] > self.rule_selection_criteria['max_error']:
                    edge_flag = False
            # Select top n rules and use them to repair
            if 'top_n' in self.rule_selection_criteria:
                n_rules_to_select = self.rule_selection_criteria['top_n']
                if self.rel_type == POWER_LAW_STR:
                    error_mat = self.relation[1].evaluation_metric
                else:
                    error_mat = self.relation[0].evaluation_metric
                # Flatten the evaluation_metric array and sort in ascending order
                error_mat_1d = error_mat.flatten()
                error_mat_sorted_indx = np.argsort(error_mat_1d)
                # Map the 1D sorted indices to their corresponding 2D indices in the original evaluation_metric matrix.
                # Here row indx = floor(error_mat_sorted_indx / no. of columns)
                # col indx = error_mat_sorted_indx % no. of columns
                # For a 4x4 evaluation_metric matrix a 1D flatteneed index of 7 gets converted to [1, 3]
                error_mat_sorted_indx_2d = np.array([error_mat_sorted_indx // error_mat.shape[1],
                                                     error_mat_sorted_indx % error_mat.shape[1]]).transpose()
                # Select the top n rules with minimum evaluation_metric
                top_n_error_indx = error_mat_sorted_indx_2d[:n_rules_to_select, :]
                # If variable pair [i, j] not in the top n rules, do not create the edge
                if not [i, j] in top_n_error_indx.tolist():
                    edge_flag = False

            if edge_flag:
                vrg.add_edges([edge], rank=[1], rel_type=[POWER_LAW_CONSTANT_RULE])

    def evaluate_local_rule_performance(self, x_offspring, training_data, var_pair):
        """
        Evaluates the performance of a global rule in the neighborhood of one or more offspring solutions.
        :return:
        :rtype:
        """
        x_min, x_max = self.normalize_to_range[0], self.normalize_to_range[1]
        training_data_normalized = x_min + (training_data - self.xl) / (self.xu - self.xl) * (x_max - x_min)
        x_offspring_normalized = x_min + (x_offspring - self.xl) / (self.xu - self.xl) * (x_max - x_min)
        n_train = training_data.shape[0]

        if self.rel_type == POWER_LAW_STR:
            b, c = self.relation[1].b, self.relation[1].c
            if self.power_law_normalized:
                for v_pair in var_pair:
                    i, j = v_pair
                    neighbor_model = NearestNeighbors(radius=self.normalized_NN_radius)
                    neighbor_model.fit(training_data_normalized[:, v_pair])

                    neigh_dist, neigh_ind = neighbor_model.radius_neighbors(x_offspring_normalized[:, v_pair])
                    c_training = training_data_normalized[:, j] * training_data_normalized[:, i]**b[j, i]
                    c_training_mean = np.mean(c_training)
                    c_training_std = np.std(c_training)

                    # For every offspring, calculate the evaluation_metric of the associated global rules based on
                    # the local neighborhood
                    for offspring_nn_ind in neigh_ind:
                        n_neighbors = len(neigh_ind)
                        c_neighbor = []  # Evaluates xi*xj^b for neighborhood training data
                        for i, ind in enumerate(offspring_nn_ind):
                            training_rep_normalized = training_data_normalized[ind, :]
                            c_neighbor.append(training_rep_normalized[j]
                                              * training_rep_normalized[i]**b[j, i])
                        c_neighbor = np.array(c_neighbor)
                        c_neighbor_mean = np.mean(c_neighbor)
                        c_neighbor_std = np.std(c_neighbor)

                        # Find how many c values of the neighbors lie within 1 std of the c values of the total
                        # training dataset.
                        ind_c = np.where(np.abs(c_neighbor - c[j, i]) <= c_training_std)[0]
                        nn_within_1_std = len(ind_c)
                        local_confidence = nn_within_1_std / n_train
                        if self.relation[1].error_metric == 'mse':
                            i, j = v_pair
                            xi_training = training_data_normalized[offspring_nn_ind, i]
                            xj_training_predicted = c[j, i] / xi_training**b[j, i]
                            mse_local = mean_squared_error(y_true=training_data[:, j], y_pred=xj_training_predicted)
                        else:
                            warnings.warn("Undefined local performance metric.")
            else:
                warnings.warn("Power laws not set to be normalized. Nearest neighbor not defined in this case.")
        else:
            warnings.warn(f"Local rule performance metric not defined for relationship type {self.rel_type}.")

    def learn(self, training_data):
        self.training_data = training_data
        x_min, x_max = self.normalize_to_range[0], self.normalize_to_range[1]
        training_data_normalized = x_min + (self.training_data - self.xl) / (self.xu - self.xl) * (x_max - x_min)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.correlation = np.round(np.corrcoef(training_data_normalized, rowvar=False), decimals=2)
        if self.rel_type == POWER_LAW_STR or self.rel_type == POWER_LAW_CONSTANT_RULE_STR:
            for grp_indx, curr_grp in enumerate(self.groups):
                # For every group, ignore the variables not in the group
                ignore_vars = np.setdiff1d(np.arange(self.n_var), curr_grp)
                # Find constant relations first
                if self.relation == POWER_LAW_CONSTANT_RULE_STR:
                    if self.power_law_normalized:
                        c, const_var_flag = self.relation[0].learn(training_data=training_data_normalized,
                                                                   ignore_vars=ignore_vars)
                    else:
                        c, const_var_flag = self.relation[0].learn(training_data=training_data,
                                                                   ignore_vars=ignore_vars)
                    const_var_indx = np.where(const_var_flag == 1)[0]
                    power_law_rel_indx = 1
                else:
                    const_var_indx = []
                    power_law_rel_indx = 0

                # Learn power law afterwards ignoring the constant variables
                if self.power_law_normalized:
                    self.relation[power_law_rel_indx].learn(training_data=training_data_normalized,
                                                            ignore_vars=np.append(ignore_vars, const_var_indx))
                else:
                    self.relation[power_law_rel_indx].learn(training_data=training_data,
                                                            ignore_vars=np.append(ignore_vars, const_var_indx))

                # Construct the VRG graph
                self.vrg[grp_indx] = VariableRelationGraph(directed=False)
                self.vrg[grp_indx].add_nodes(curr_grp)
                for i in range(len(curr_grp) - 1):
                    if curr_grp[i] in const_var_indx:
                        continue
                    for j in range(i + 1, len(curr_grp)):
                        if curr_grp[j] in const_var_indx:
                            continue
                        edge_to_evaluate = [curr_grp[i], curr_grp[j]]
                        self.select_rule(vrg=self.vrg[grp_indx], edge_to_add=[edge_to_evaluate])

                removed_nodes = self.vrg[grp_indx].remove_nodes_with_degree(degree=0)

        elif self.rel_type == INEQUALITY_RULE_STR:
            for grp_indx, curr_grp in enumerate(self.groups):
                # For every group, ignore the variables not in the group
                ignore_vars = np.setdiff1d(np.arange(self.n_var), curr_grp)
                self.relation[0].learn(training_data=training_data_normalized, ignore_vars=ignore_vars)

                # Construct the VRG graph
                self.vrg[grp_indx] = VariableRelationGraph()
                self.vrg[grp_indx].add_nodes(curr_grp)
                for i in range(len(curr_grp) - 1):
                    for j in range(i + 1, len(curr_grp)):
                        if np.abs(self.correlation[curr_grp[i], curr_grp[j]]) >= np.abs(self.min_correlation):
                            # TODO: Add simple_rel_type function argument. It denotes abstracted relations like
                            #       proportional, inversely proportional, etc.
                            self.vrg[grp_indx].add_edges([[curr_grp[i], curr_grp[j]]], rank=[1],
                                                         rel_type=[self.relation[0].rule])

    def repair(self, x, debug=False, x_rep_operator=None):
        x_rep = np.copy(x)
        rep_indx = np.zeros_like(x)
        rep_operator_selected = -1 * np.ones(x.shape[0])
        const_repair_flag = False
        const_var_index = []
        if self.rel_type == POWER_LAW_STR or self.rel_type == CONSTANT_RULE_STR:
            const_var_index = np.where(self.relation[0].const_var_flag == 1)[0]

        exploration_history = []
        vrg_used = [[] for x_sol in x_rep]  # Store the VRGs used to repair each member
        for grp_indx, curr_grp in enumerate(self.groups):
            exploration_history.append([])
            node_list = self.vrg[grp_indx].get_nodes()
            edge_list = self.vrg[grp_indx].get_edges()
            # Length of node_list is 0 when there are no rules whose evaluation_metric is below threshold. Thus, the
            # VRG has 0 nodes and 0 edges
            if len(node_list) == 0:
                print(f"Group {grp_indx} does not have any low-evaluation_metric rules. Thus, number of VRG nodes is 0.")
                continue
            # This is just a check to see if an unexpected situation occurs where the graph has 1 or more nodes
            # but no edges
            if len(edge_list) == 0:
                warnings.warn(f"Number of VRG edges is 0.")
                continue
            for pop_indx, x_sol in enumerate(x_rep):
                if self.rel_type == POWER_LAW_STR or self.rel_type == CONSTANT_RULE_STR:
                    if self.relation[0].normalization_flag:
                        # Normalized current solution
                        x_sol_const_normalized = self.normalize_data(x_sol)
                        # Obtain repaired solution where some variables were set to a pre-specified value
                        x_const_repair_normalized = self.relation[0].repair(x_sol_const_normalized, const_var_index)
                        # Normalize solution to original range
                        x_rep[pop_indx, :] = self.normalize_to_original_range(x_const_repair_normalized)
                    else:
                        x_rep[pop_indx, :] = self.relation[0].repair(x_sol, const_var_index)
                # Create a random ordering D such that all directed edges (i, j) follow i <= j in D
                rand_ordering = np.random.permutation(node_list)
                vrg_directed = VariableRelationGraph(directed=True)
                vrg_directed.add_nodes(node_list)
                directed_edge_list = []
                for edge in edge_list:
                    if np.where(rand_ordering == edge[0])[0] <= np.where(rand_ordering == edge[1])[0]:
                        directed_edge_list.append([edge[0], edge[1]])
                        vrg_directed.add_edges([[edge[0], edge[1]]], rank=[1], rel_type=[POWER_LAW_CONSTANT_RULE])
                    else:
                        directed_edge_list.append([edge[1], edge[0]])
                        vrg_directed.add_edges([[edge[1], edge[0]]], rank=[1], rel_type=[POWER_LAW_CONSTANT_RULE])

                # print(f"Before reduction: VRG for group {grp_indx} has {len(vrg_directed.get_nodes())} nodes "
                #       f"and {len(vrg_directed.get_edges())} edges")
                vrg_directed.transitive_reduction()
                # print(f"After reduction: VRG for group {grp_indx} has {len(vrg_directed.get_nodes())} nodes "
                #       f"and {len(vrg_directed.get_edges())}edges ")
                # vrg_directed.draw()
                # plt.show()

                # node_list = list(vrg_directed.get_nodes())
                # edge_list_reduced = list(vrg_directed.get_edges())
                # print(node_list)
                # print(edge_list)
                nodes_fixed = []
                start_node = np.random.choice(node_list, 1, replace=False)[0]

                # Normalize solution, repair, and re-normalize to original range
                history = []
                repaired_vars = []
                if x_rep_operator is None:
                    repair_operator_selected = np.random.choice(a=np.arange(len(self.repair_agents[1])), size=1,
                                                                replace=False, p=self.repair_agent_probability[1])[0]
                else:
                    repair_operator_selected = x_rep_operator[pop_indx]
                # print(f"Repair operator {repair_operator_selected} selected for solution {pop_indx}")
                rep_operator_selected[pop_indx] = repair_operator_selected
                if self.power_law_normalized:
                    x_sol_normalized = self.normalize_data(x_sol)
                    self._dfs_repair(vrg=vrg_directed, x=x_sol_normalized,
                                     repair_operator_selected=repair_operator_selected, normalized=True,
                                     prev_node=start_node, curr_node=start_node,
                                     nodes_fixed=nodes_fixed, edge_type=OUT_EDGE,
                                     repaired_vars=repaired_vars, debug=debug,
                                     exploration_history=history)
                    x_rep[pop_indx, :] = self.normalize_to_original_range(x_sol_normalized)
                else:
                    self._dfs_repair(vrg=vrg_directed, x=x_sol,
                                     repair_operator_selected=repair_operator_selected, normalized=False,
                                     prev_node=start_node, curr_node=start_node,
                                     nodes_fixed=nodes_fixed, edge_type=OUT_EDGE,
                                     repaired_vars=repaired_vars, debug=debug,
                                     exploration_history=history)
                exploration_history[grp_indx].append(history)
                rep_indx[pop_indx, repaired_vars] = 1
                vrg_used[pop_indx].append(vrg_directed)

        print(np.unique(rep_operator_selected, return_counts=True))
        if debug:
            return x_rep, rep_indx, exploration_history, rep_operator_selected, vrg_used
        else:
            return x_rep, rep_indx

    def _dfs_repair(self, vrg, x, repair_operator_selected, normalized, prev_node, curr_node,
                    nodes_fixed, edge_type,
                    repaired_vars, debug,
                    exploration_history=None):
        """A DFS graph traversal algorithm to repair variables."""
        if debug:
            exploration_history.append(curr_node)
        if curr_node in nodes_fixed:
            return

        if curr_node != prev_node and curr_node not in nodes_fixed:
            if self.rel_type == POWER_LAW_STR:
                # x_rep = self.relation[1].repair(x, np.array([[prev_node, curr_node]]))
                x_rep = self.repair_agents[1][repair_operator_selected].repair(
                    x=x, var_pair=np.array([[prev_node, curr_node]]),
                    power_law=self.relation[1])
                # Check if repaired variable within limits
                if (not normalized) and (self.xl is not None) and (self.xu is not None) \
                        and (self.xl[curr_node] <= x_rep[curr_node] <= self.xu[curr_node]):
                    x[curr_node] = x_rep[curr_node]
                    repaired_vars.append(curr_node)
                elif normalized and (self.normalize_to_range[0] <= x_rep[curr_node] <= self.normalize_to_range[1]):
                    x[curr_node] = x_rep[curr_node]
                    repaired_vars.append(curr_node)
                else:
                    print(f"Either variable limits undefined or repair caused violations. Current node = {curr_node} "
                          f"and previous node = {prev_node}")
            elif self.rel_type == INEQUALITY_RULE_STR:
                x = self.relation[0].repair(x, self.xl, self.xu, np.array([[prev_node, curr_node]]))
                repaired_vars.append(curr_node)
            else:
                print(f"Repair for rel_type = {self.rel_type} undefined.")
        nodes_fixed.append(curr_node)

        in_edges = vrg.get_in_edges(curr_node)
        out_edges = vrg.get_out_edges(curr_node)
        # First follow the outgoing edges from the current node
        for edge in out_edges:
            next_node = edge[1]
            if next_node == prev_node:
                continue
            self._dfs_repair(vrg=vrg, x=x, repair_operator_selected=repair_operator_selected,
                             normalized=normalized, prev_node=curr_node, curr_node=next_node,
                             nodes_fixed=nodes_fixed, edge_type=OUT_EDGE, repaired_vars=repaired_vars,
                             debug=debug, exploration_history=exploration_history)
            if debug:
                exploration_history.append(curr_node)
        # Now, follow the incoming edges.
        for edge in in_edges:
            next_node = edge[0]
            if next_node == prev_node:
                continue
            self._dfs_repair(vrg=vrg, x=x, repair_operator_selected=repair_operator_selected,
                             normalized=normalized, prev_node=curr_node, curr_node=next_node,
                             nodes_fixed=nodes_fixed, edge_type=IN_EDGE, repaired_vars=repaired_vars,
                             debug=debug, exploration_history=exploration_history)
            if debug:
                exploration_history.append(curr_node)

        return x
