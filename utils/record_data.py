import logging
import os
import pickle
import warnings
from shutil import copyfile

import h5py
import numpy as np
import datetime

# from optimal_power_flow.run_mopf import results_dir
results_parent_dir = 'results'
# results_dir = os.path.join(results_parent_dir, f'{ts}')
time_now = datetime.datetime.now()
results_dir = os.path.join(results_parent_dir, f'{time_now.strftime("%Y%m%d_%H%M%S")}')


INNOVIZATION_DIR = 'innovization'
USER_INTERACT_DIR = 'user_interaction'
POWER_LAW_RANK_FILE_PREFIX = 'power_law_rank_gen'
CONSTANT_RULE_RANK_FILE_PREFIX = 'constant_rule_rank_gen'
INEQUALITY_RULE_RANK_FILE_PREFIX = 'inequality_rule_rank_gen'


def record_state(algorithm):
    global results_dir
    logging.getLogger().info(f"Gen {algorithm.n_gen}")
    with open(os.path.join(results_dir, 'current_gen'), 'w') as fp:
        fp.write(str(algorithm.n_gen))
    if os.path.exists(os.path.join(results_dir, '.pauserun')):
        print("Optimization paused")
        while True:
            if not os.path.exists(os.path.join(results_dir, '.pauserun')):
                print("Resuming optimization")
                break
    rank_pop = algorithm.pop.get('rank')
    x_pop = algorithm.pop.get('X')
    f_pop = algorithm.pop.get('F')
    rank_pop[rank_pop == None] = -1
    rep_time_pop = algorithm.pop.get('rep_time')
    rep_indx_pop = algorithm.pop.get('rep_indx')
    rep_operator_survivors = algorithm.pop.get('rep_operator_selected')
    rep_operator_selected_off = algorithm.problem.rep_operator_selected

    x_pf = x_pop[rank_pop == 0]
    pf = f_pop[rank_pop == 0]
    x_points_of_interest = np.copy(x_pf)
    # print(len(pf))
    # if len(pf) > 10:
    #     knee = get_knee(pf, epsilon=0.125)
    #     min_x_knee = np.min(x_pf[knee, 0])
    #     max_x_knee = np.max(x_pf[knee, 0])
    #     if min_x_knee != max_x_knee:
    #         mean_x_knee = max_x_knee - min_x_knee
    #     else:
    #         mean_x_knee = min_x_knee
    #     x_points_of_interest = x_pf[(x_pf[:, 0] >= (mean_x_knee - 0.15)) & (x_pf[:, 0] <= (mean_x_knee + 0.15)), :]
    #     if len(x_points_of_interest) <= 1:
    #         x_points_of_interest = np.copy(x_pf)

    g_pop = np.array([])
    cv_pop = np.array([])
    if algorithm.problem.n_constr > 0:
        g_pop = algorithm.pop.get('G')
        cv_pop = algorithm.pop.get('CV')

    # Percentage of pop in ND set serves as a general confidence of rules learned
    algorithm.problem.percent_pf = pf.shape[0] / algorithm.pop_size
    # if algorithm.problem.percent_pf >= 0.5:
    innov_info_available = False  # Indicates whether at least one learning phase has taken place
    innov = None
    # If atleast 10 sols are in pf and 5 gens have passed before learning occurs
    if pf.shape[0] > 10:
        if (algorithm.n_gen >= 5
                and (algorithm.n_gen % algorithm.problem.learning_interval == 0
                     or algorithm.n_gen == algorithm.termination.n_max_gen)
                and x_points_of_interest.shape[0] >= 1 and algorithm.problem.innov is not None):
            innov_info_available = True
            algorithm.problem.innov_info_available = innov_info_available
            innov = algorithm.problem.innov
            rep_agent_names = innov.repair_agent_names
            print(f"Learning at gen {algorithm.n_gen}")
            innov.learn(x_points_of_interest)
            decision_maker = algorithm.problem.decision_maker
            # If a decision maker is involved perform an interaction and re-learn the rules
            if decision_maker is not None:
                decision_maker.interact(innov)
                innov.learn(x_points_of_interest)

            vrg_used_for_repair = algorithm.pop.get('vrg_used_for_repair')
            # DEBUG
            for vrg in vrg_used_for_repair:
                if vrg is not None:
                    print(vrg)
            gen_str = f'innov_gen{algorithm.n_gen}.pkl'
            with open(os.path.join(results_dir, INNOVIZATION_DIR, gen_str), 'wb') as f:
                pickle.dump(innov, f)

            # Currently knowledge base refers to a VRG. Will be generalized in the future
            knowledge_base_str = f'knowledge_base_gen{algorithm.n_gen}.pkl'
            with open(os.path.join(results_dir, INNOVIZATION_DIR, knowledge_base_str), 'wb') as f:
                pickle.dump(vrg_used_for_repair, f)

    # Just after one repair, track which solutions repaired have survived
    if algorithm.problem.repair_power and algorithm.problem.innov_info_available \
            and algorithm.n_gen >= 5 and ((algorithm.n_gen - 1) % algorithm.problem.repair_interval == 0):
        print(f"Updating repair probabilities at gen {algorithm.n_gen}")
        rep_agent_ensemble = algorithm.problem.innov.repair_agents[1]
        n_rep_agents = len(rep_agent_ensemble)
        n_survivors = np.zeros(n_rep_agents)
        n_off_repaired = np.zeros(n_rep_agents)
        off_survival_rate = np.nan * np.ones(n_rep_agents)
        for r_indx in range(n_rep_agents):
            n_survivors[r_indx] = np.sum(rep_operator_survivors == r_indx)
            n_off_repaired[r_indx] = np.sum(rep_operator_selected_off == r_indx)
            if n_off_repaired[r_indx] > 0:
                off_survival_rate[r_indx] = n_survivors[r_indx] / n_off_repaired[r_indx]

        alpha = 0.6
        # If off_survival_rate has no nan elements
        if not np.any(np.isnan(off_survival_rate)):
            for agent_no in range(len(algorithm.problem.innov.repair_agents)):
                min_probability = algorithm.problem.innov.repair_agent_min_probability[agent_no]
                if min_probability is None:
                    continue
                rep_probability_old = np.array(algorithm.problem.innov.repair_agent_probability[agent_no])
                # KLUGE: divide by 0
                if np.sum(off_survival_rate) == 0:
                    rep_probability_new = rep_probability_old
                else:
                    rep_probability_new = np.maximum(min_probability,
                                                     (alpha * off_survival_rate / np.sum(off_survival_rate)
                                                      + (1 - alpha) * rep_probability_old))
                prob_sum = np.sum(rep_probability_new)
                diff = prob_sum - 1
                # Which repair agents were set to min. probability
                min_prob_indx = np.where(rep_probability_new == 0.1)[0]
                not_min_prob_indx = np.where(rep_probability_new > 0.1)[0]
                n_min_prob = len(min_prob_indx)
                n_not_min_prob = len(not_min_prob_indx)
                rep_probability_new_scaled = np.copy(rep_probability_new)
                rep_probability_new_scaled[not_min_prob_indx] -= diff / n_not_min_prob
                if not np.isclose(np.sum(rep_probability_new_scaled), 1):
                    print(rep_probability_new_scaled)
                    warnings.warn("Sum of repair probabilities is not 1.")
                # rep_probability_new_scaled = rep_probability_new / np.sum(rep_probability_new)

                # rep_probability_new = rep_probability_new / np.sum(rep_probability_new)
                algorithm.problem.innov.repair_agent_probability[agent_no] = rep_probability_new_scaled
                print("Old repair agent probabilities: ", rep_probability_old)
                print("New repair agent probabilities: ", rep_probability_new_scaled)
        # print(rep_operator_survivors)

    if algorithm.n_gen % 100 == 0:
        # print("Save state disabled.")
        print("Saving state")
        with open(os.path.join(results_dir, 'state.pkl'), 'wb') as f:
            pickle.dump(algorithm, f)

    optim_state_hdf_file = os.path.join(results_dir, 'optim_state.hdf5')
    with h5py.File(optim_state_hdf_file, 'a') as hf:
        hf.attrs['obj_label'] = algorithm.problem.obj_label
        hf.attrs['current_gen'] = algorithm.n_gen
        hf.attrs['xl'] = algorithm.problem.xl
        hf.attrs['xu'] = algorithm.problem.xu
        hf.attrs['n_obj'] = algorithm.problem.n_obj
        hf.attrs['n_constr'] = algorithm.problem.n_constr
        if 'innov_info_latest_gen' not in hf.attrs.keys():
            hf.attrs['innov_info_latest_gen'] = -1
        if 'total_rep_time' not in hf.attrs:
            hf.attrs['total_rep_time'] = 0
        if rep_time_pop[0] is not None:
            hf.attrs['total_rep_time'] += rep_time_pop[0]
            logging.info(f"Total repair time = {hf.attrs['total_rep_time']}")
        hf.attrs['ignore_vars'] = algorithm.problem.ignore_vars

        g1 = hf.create_group(f'gen{algorithm.n_gen}')

        # Basic population data
        g1.create_dataset('X', data=x_pop)
        g1.create_dataset('F', data=f_pop)
        g1.create_dataset('rank', data=rank_pop.astype(int))
        g1.create_dataset('G', data=g_pop)
        g1.create_dataset('CV', data=cv_pop)
        g1.create_dataset('rep_time', data=rep_time_pop[0])
        g1.create_dataset('rep_indx', data=rep_indx_pop)

        # Innovization information
        if algorithm.problem.innov_info_available:
            if algorithm.n_gen % algorithm.problem.learning_interval == 0:
                hf.attrs['innov_info_latest_gen'] = algorithm.n_gen
            hf.attrs['n_groups'] = len(algorithm.problem.var_groups)
            g1.create_dataset('percent_pf', data=algorithm.problem.percent_pf)
            for grp_indx, g in enumerate(algorithm.problem.var_groups):
                g1.create_dataset(f'var_groups_{grp_indx}', data=g)
            if algorithm.problem.repair_power:
                # Constant rule
                g1.create_dataset('c_const', data=algorithm.problem.innov.relation[0].c)
                g1.create_dataset('const_var_flag', data=algorithm.problem.innov.relation[0].const_var_flag)
                g1.create_dataset('const_tol', data=algorithm.problem.innov.relation[0].const_tol)
                g1.create_dataset('error_const', data=algorithm.problem.innov.relation[0].error)
                g1.create_dataset('max_error_const', data=algorithm.problem.innov.relation[0].max_error)

                # Power law
                g1.create_dataset('b_pow', data=algorithm.problem.innov.relation[1].b)
                g1.create_dataset('c_pow', data=algorithm.problem.innov.relation[1].c)
                g1.create_dataset('error_pow', data=algorithm.problem.innov.relation[1].error)
                g1.create_dataset('max_error_pow', data=algorithm.problem.innov.relation[1].max_error)
                g1.create_dataset('error_metric', data=algorithm.problem.innov.relation[1].error_metric)
            elif algorithm.problem.repair_inequality:
                g1.create_dataset('rule', data=algorithm.problem.innov.relation[0].rule)
                g1.create_dataset('eq_tol', data=algorithm.problem.innov.relation[0].eq_tol)
                g1.create_dataset('min_eq_thresh_leq_geq', data=algorithm.problem.innov.relation[0].min_eq_thresh_leq_geq)
                g1.create_dataset('rule_score', data=algorithm.problem.innov.relation[0].rule_score)

    if os.path.exists(optim_state_hdf_file):
        copyfile(optim_state_hdf_file, optim_state_hdf_file + ".bak")


def get_repair_out_suffix(cmd_args, results_parent_dir, problem_out_str, time_now):
    """
    Get a standardized path for the output folder of an optimization run. The path has 3 components:
    parent directory + problem-specific string + current time.
    :param cmd_args:
    :type cmd_args:
    :param results_parent_dir:
    :type results_parent_dir:
    :param problem_out_str:
    :type problem_out_str:
    :param time_now:
    :type time_now:
    :return:
    :rtype:
    """
    repair_str = 'repair'
    if cmd_args.repair_power:
        repair_str += f'_power_freq{cmd_args.repair_interval}'
    elif cmd_args.repair_inequality:
        repair_str += f'_inequality_freq{cmd_args.repair_interval}'
    else:
        repair_str = 'baseline'

    if cmd_args.save is not None:
        exp_dir = os.path.join(results_parent_dir, cmd_args.save)
    else:
        exp_dir = os.path.join(results_parent_dir,
                               f'{problem_out_str}_{cmd_args.popsize}pop_'
                               f'{cmd_args.ngen}gen_{repair_str}'
                               f'_{time_now.strftime("%Y%m%d_%H%M%S")}')

    return exp_dir, repair_str


def create_run_output_dir(exp_dir, run, seed, additional_folders=()):
    results_dir = os.path.join(exp_dir, f"run{run}_seed{seed}")
    os.mkdir(results_dir)
    innov_sub_dir = os.path.join(results_dir, INNOVIZATION_DIR)  # Used to store gen-wise innovization results
    os.mkdir(innov_sub_dir)
    interact_sub_dir = os.path.join(results_dir, USER_INTERACT_DIR)  # Used to store gen-wise innovization results
    os.mkdir(interact_sub_dir)
    for folder in additional_folders:
        os.mkdir(folder)

    return results_dir, innov_sub_dir, interact_sub_dir
