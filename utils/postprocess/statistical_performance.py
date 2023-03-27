import copy

import h5py
import os
import datetime
import numpy as np
import pickle

# from pymoo.factory import get_performance_indicator
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
from scipy.stats import ranksums

plt.rcParams["figure.autolayout"] = True


def calc_igd(f_normalized, ref_pf):
    igd = IGD(ref_pf)
    # print("IGD", igd.calc(f_normalized))
    return igd(f_normalized)


def calc_igd_plus(f_normalized, ref_pf):
    igd_plus = IGDPlus(ref_pf)
    # print("IGD+", igd_plus.calc(f_normalized))
    return igd_plus(f_normalized)


def calc_hv(f_normalized, ref_point=np.array([1.1, 1.1])):
    hv = HV(ref_point=ref_point)
    # print("HV", hv.calc(f_normalized))
    return hv(f_normalized)


def find_pf(front):
    rank = np.ones(front.shape[0])

    for i in range(front.shape[0]):
        for j in range(front.shape[0]):
            if i == j:
                continue
            if dominates(front[i, :], front[j, :]):
                rank[j] = 0

    pf = front[rank == 1, :]

    return pf


def dominates(x1, x2):

    if np.sum(x1 < x2) == len(x1):
        d = 1
    else:
        d = 0
    return d


if __name__ == '__main__':
    max_gen_fixed = None
    feval_multiplier = None
    max_nruns = None
    custom_target_HV = None
    feval_flag = True  # If true, print fevals on HV, IGD x-axis instead of generations
    range_band_flag = False
    median_feval_hv_target_flag = False
    percentage_target_HV = 0.8
    base_exp_no = 0  # Set which element of result_dir represents the base algorithm
    time_now = datetime.datetime.now()
    cmp_results_dir = os.path.join('stat_output',
                                   f'comparison_{time_now.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(cmp_results_dir)

    results_dir = [
        '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/moea_kdo/optimal_power_flow/output/case118_fe_60pop_100gen_baseline_20220423_180457',
        '/home/abhiroop/Insync/ghoshab1@msu.edu/Google Drive/Abhiroop/Data/MSU/Research/DARPA/Code/CP3/moea_kdo/optimal_power_flow/output/case118_fe_60pop_100gen_repair_power_freq10_20220423_190936',
                   ]

    name_list = [
        'Base NSGA-II',
        'NSGA-II/VRG-ensemble',
        # 'NSGA-II/VRG-s0',
        # 'NSGA-II/VRG-s1',
        # 'NSGA-II/VRG-s2',
    ]

    print(cmp_results_dir)
    with open(os.path.join(cmp_results_dir, 'compare_list'), 'w') as fp:
        for r_dir in results_dir:
            fp.writelines(r_dir + "\n")

    # Find a pareto front for all runs and experiments combined to calculate the normalization values
    pf_list = []
    for exp_no, exp_dir in enumerate(results_dir):
        for x in os.listdir(exp_dir):
            r_dir = os.path.join(exp_dir, x)
            res_file_path = os.path.join(r_dir, 'optim_state.hdf5')
            if os.path.exists(res_file_path):
                try:
                    # For handling errors where hdf file is incomplete
                    with h5py.File(res_file_path, 'r') as hf:
                        if max_gen_fixed is None:
                            max_gen = hf.attrs['current_gen']
                        else:
                            max_gen = max_gen_fixed
                        pf = np.array(hf[f'gen{max_gen}']['F'])
                        rank = np.array(hf[f'gen{max_gen}']['rank'])
                        final_pop = np.array(hf[f'gen{max_gen}']['X'])
                        pf_list.append(pf[rank == 0, :])
                except OSError:
                    pass
            else:
                print("File read evaluation_metric")

    pop_size = final_pop.shape[0]
    all_pf = find_pf(np.concatenate(pf_list, axis=0))
    ideal_point = np.min(all_pf, axis=0)
    nadir_point = np.max(all_pf, axis=0)

    min_val = ideal_point
    max_val = nadir_point

    all_pf_normalized = (all_pf - min_val) / (max_val - min_val)
    best_hv = calc_hv(all_pf_normalized)
    if custom_target_HV is None:
        target_hv = np.round(percentage_target_HV * best_hv, decimals=2)
    else:
        target_hv = custom_target_HV

    with open(os.path.join(cmp_results_dir, 'normalization'), 'w') as fp:
        for val in min_val:
            fp.write(f"{val} ")
        fp.write("\n")
        for val in max_val:
            fp.write(f"{val} ")
        fp.write("\n")

    # These lists will contain all the gen-wise HV and IGD for every experiment and every run_no
    hv_hist = []
    igd_hist = []
    igd_plus_hist = []
    median_hv = []
    median_igd = []
    median_igd_plus = []
    min_hv, min_igd, min_igd_plus = [], [], []
    max_hv, max_igd, max_igd_plus = [], [], []
    for exp_no, exp_dir in enumerate(results_dir):
        n_run_dirs = len(os.listdir(exp_dir))
        hv_hist.append([[] for _ in range(n_run_dirs)])
        igd_hist.append([[] for _ in range(n_run_dirs)])
        igd_plus_hist.append([[] for _ in range(n_run_dirs)])
        median_hv.append([])
        median_igd.append([])
        median_igd_plus.append([])
        min_hv.append([])
        min_igd.append([])
        min_igd_plus.append([])
        max_hv.append([])
        max_igd.append([])
        max_igd_plus.append([])

    fig_pf = plt.figure()
    ax_pf = fig_pf.add_subplot(111)

    fig_hv = plt.figure()
    ax_hv = fig_hv.add_subplot(111)
    ax_hv_plot_arr = []

    fig_igd = plt.figure()
    ax_igd = fig_igd.add_subplot(111)
    # Repeat for every experiment
    for exp_no, exp_dir in enumerate(results_dir):
        n_runs = len(hv_hist[exp_no])
        if n_runs > 0:
            # For every run_no collect the pareto fronts and hv values
            run_no = 0
            for r_indx, x in enumerate(os.listdir(exp_dir)):
                r_dir = os.path.join(exp_dir, x)
                print(r_dir)
                res_file_path = os.path.join(exp_dir, r_dir, 'optim_state.hdf5')
                try:
                    # For handling errors where hdf file is incomplete
                    with h5py.File(res_file_path, 'r') as hf:
                        if max_gen_fixed is None:
                            max_gen = hf.attrs['current_gen']
                        else:
                            max_gen = max_gen_fixed
                        for gen in range(max_gen):
                            if gen % 100 == 0:
                                print(gen)
                            f_gen = np.array(hf[f'gen{gen + 1}']['F'])
                            rank_gen = np.array(hf[f'gen{gen + 1}']['rank'])
                            pf_gen = f_gen[rank_gen == 0]
                            if pf_gen.shape[0] > 0:
                                pf_gen_normalized = (pf_gen - min_val) / (max_val - min_val)
                                hv_gen = calc_hv(pf_gen_normalized, ref_point=np.array([1.1, 1.1]))
                                igd_gen = calc_igd(pf_gen_normalized, all_pf_normalized)
                                # igd_plus_gen = calc_igd_plus(pf_gen_normalized, all_pf_normalized)

                                hv_hist[exp_no][run_no].append([gen + 1, hv_gen])
                                igd_hist[exp_no][run_no].append([gen + 1, igd_gen])
                                # igd_plus_hist.append([gen + 1, igd_plus_gen])
                            else:
                                hv_hist[exp_no][run_no].append([gen + 1, 0.0])
                                igd_hist[exp_no][run_no].append([gen + 1, np.Inf])
                except OSError:
                    pass

                run_no += 1
                if max_nruns is not None and run_no == max_nruns:
                    break

            gen_run_arr = []
            hv_run_arr = []
            igd_run_arr = []
            igd_plus_run_arr = []
            # Collect the HV and IGD of every run
            for run_no in range(len(hv_hist[exp_no])):
                h_arr = np.array(hv_hist[exp_no][run_no])
                i_arr = np.array(igd_hist[exp_no][run_no])
                # i_plus_arr = np.array(igd_plus_hist[exp_no][run_no])

                gen_run_arr.append(h_arr[:, 0].reshape(-1, 1))
                hv_run_arr.append(h_arr[:, 1].reshape(-1, 1))
                igd_run_arr.append(i_arr[:, 1].reshape(-1, 1))
                # igd_plus_run_arr.append(i_plus_arr[:, 1].reshape(-1, 1))

            hv_run_arr = np.concatenate(hv_run_arr, axis=1)
            median_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                          np.median(hv_run_arr, axis=1).reshape(-1, 1), axis=1)
            min_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                       np.min(hv_run_arr, axis=1).reshape(-1, 1), axis=1)
            max_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                       np.max(hv_run_arr, axis=1).reshape(-1, 1), axis=1)

            # KLUGE
            # if exp_no == 0:
            #     # median_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
            #     #                               np.mean(hv_run_arr, axis=1).reshape(-1, 1), axis=1)
            #     median_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
            #                                   np.minimum(np.mean(hv_run_arr, axis=1).reshape(-1, 1), np.median(hv_run_arr, axis=1).reshape(-1, 1)), axis=1)

            if exp_no == 3:
                # median_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
                #                               np.mean(hv_run_arr, axis=1).reshape(-1, 1), axis=1)
                median_hv[exp_no] = np.append(np.array(hv_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                              np.minimum(np.min(hv_run_arr, axis=1).reshape(-1, 1), np.min(hv_run_arr, axis=1).reshape(-1, 1)), axis=1)

            igd_run_arr = np.concatenate(igd_run_arr, axis=1)
            median_igd[exp_no] = np.append(np.array(igd_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                           np.median(igd_run_arr, axis=1).reshape(-1, 1), axis=1)
            min_igd[exp_no] = np.append(np.array(igd_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                        np.min(igd_run_arr, axis=1).reshape(-1, 1), axis=1)
            max_igd[exp_no] = np.append(np.array(igd_hist[exp_no])[0][:, 0].reshape(-1, 1),
                                        np.max(igd_run_arr, axis=1).reshape(-1, 1), axis=1)

            ax_pf.scatter(pf_gen[:, 0], pf_gen[:, 1], label=name_list[exp_no], alpha=0.5)
            if feval_flag:
                xval = median_hv[exp_no][:, 0] * pop_size
                if feval_multiplier is not None:
                    xval *= feval_multiplier
            else:
                xval = median_hv[exp_no][:, 0]
            ax_hv.plot(xval, median_hv[exp_no][:, 1], label=name_list[exp_no])
            ax_igd.plot(xval, median_igd[exp_no][:, 1], label=name_list[exp_no])
            print(f"Experiment {exp_no + 1}, Final median HV = {median_hv[exp_no][:, 1][-1]}, "
                  f"median IGD = {median_igd[exp_no][:, 1][-1]}")
            if n_runs > 1 and range_band_flag:
                ax_hv.fill_between(xval, min_hv[exp_no][:, 1], max_hv[exp_no][:, 1],
                                   label=name_list[exp_no] + ' range', alpha=0.3)
                ax_igd.fill_between(xval, min_igd[exp_no][:, 1], max_igd[exp_no][:, 1],
                                    label=name_list[exp_no] + ' range', alpha=0.3)

            # ax_igd_plus.plot(median_igd_plus[exp_no][:, 0], median_igd_plus[exp_no][:, 1], label=name_list[exp_no])

    feval_targetHV_arr = []
    gen_targetHV_arr = []
    if median_feval_hv_target_flag:
        ax_hv.axhline(target_hv, color='k', linestyle='--', label='Target HV')
    for exp_no, exp_dir in enumerate(results_dir):
        # At index idx+1, median_hv crosses target_hv
        idx = np.argwhere(np.diff(np.sign(target_hv - median_hv[exp_no][:, 1]))).flatten()
        if feval_flag:
            if idx == (len(median_hv[exp_no]) - 1) or np.isclose(median_hv[exp_no][:, 1][idx], target_hv):
                median_feval_to_target_HV = median_hv[exp_no][:, 0][idx] * pop_size
            else:
                # idx only points out the index beyond which the median_hv crosses target_hv. We have to
                # estimate the exact point between gen[idx] and gen[idx+1] when median_hv touches target_hv
                # hv_diff = median_hv[exp_no][:, 1][idx + 1] - median_hv[exp_no][:, 1][idx]
                # feval_diff = (median_hv[exp_no][:, 0][idx + 1] - median_hv[exp_no][:, 0][idx]) * pop_size
                # slope = hv_diff / feval_diff
                median_feval_to_target_HV = median_hv[exp_no][:, 0][idx + 1] * pop_size
            # if exp_no == 0:
            #     median_feval_to_target_HV = np.array([40000])
            # if exp_no == 0:
            #     median_feval_to_target_HV = np.array([30000])
            if feval_multiplier is not None:
                median_feval_to_target_HV *= feval_multiplier
            # feval_targetHV_arr.append(median_feval_to_target_HV[0])
            print(f"Experiment {exp_no + 1}, target HV = {target_hv}, "
                  f"median fevals to reach target = {median_feval_to_target_HV}")
            # ax_hv.axvline(median_feval_to_target_HV, color='g', linestyle='--')
            # if len(median_feval_to_target_HV) == 0:
            #     median_feval_to_target_HV = median_hv[exp_no][-1, 0][idx] * pop_size
            # KLUGE
            line_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][exp_no]

            if len(median_feval_to_target_HV) != 0 and median_feval_hv_target_flag:
                ax_hv.axvline(median_feval_to_target_HV, color=line_color, linestyle='--')
                ax_hv.text(0.96*median_feval_to_target_HV, 0.2, s=f'{np.round(median_feval_to_target_HV[0]).astype(int)}',
                           rotation='vertical',
                           color=line_color)

            # if len(median_feval_to_target_HV) > 0:
            #     if exp_no == 0:
            #         ax_hv.axvline(median_feval_to_target_HV, color=line_color, linestyle='--', label='Base feval')
            #     else:
            #         line_color = 'orange'
            #         ax_hv.axvline(median_feval_to_target_HV, color=line_color, linestyle='--', label='Repair feval')
            #     ax_hv.text(0.95*median_feval_to_target_HV, 0.2, s=f'{np.round(median_feval_to_target_HV[0]).astype(int)}',
            #                rotation='vertical',
            #                color=line_color)
            # else:
            #     print(f"Exp {exp_no} did not reach target HV")
        else:
            if idx == (len(median_hv[exp_no]) - 1) or np.isclose(median_hv[exp_no][:, 1][idx], target_hv):
                median_gen_to_target_HV = median_hv[exp_no][:, 0][idx]
            else:
                # idx only points out the index beyond which the median_hv crosses target_hv. We have to
                # estimate the exact point between gen[idx] and gen[idx+1] when median_hv touches target_hv
                # hv_diff = median_hv[exp_no][:, 1][idx + 1] - median_hv[exp_no][:, 1][idx]
                # feval_diff = (median_hv[exp_no][:, 0][idx + 1] - median_hv[exp_no][:, 0][idx]) * pop_size
                # slope = hv_diff / feval_diff
                median_gen_to_target_HV = median_hv[exp_no][:, 0][idx + 1]
            # median_gen_to_target_HV = median_hv[exp_no][:, 0][idx]
            # gen_targetHV_arr.append(median_gen_to_target_HV[0])
            print(f"Experiment {exp_no + 1}, target HV = {target_hv}, "
                  f"median gens to reach target = {median_gen_to_target_HV}")
            ax_hv.axvline(median_gen_to_target_HV, color='#4F4C1B', linestyle='--')

        # feval_targetHV_arr.append([])
        # gen_targetHV_arr.append([])
        f_arr = []
        g_arr = []
        for run_no in range(len(hv_hist[exp_no])):
            hv_exp_run = np.array(hv_hist[exp_no][run_no])
            idx = np.argwhere(np.diff(np.sign(target_hv - hv_exp_run[:, 1]))).flatten()
            if len(idx) == 0:
                f_arr.append(1e30)
                g_arr.append(1e30)
            else:
                if feval_flag:
                    f_arr.append(hv_exp_run[idx[0], 0] * pop_size)
                else:
                    g_arr.append(hv_exp_run[idx[0], 0])
        feval_targetHV_arr.append(copy.copy(f_arr))
        gen_targetHV_arr.append(copy.copy(g_arr))

    # feval_targetHV_arr = np.array(feval_targetHV_arr)
    # gen_targetHV_arr = np.array(gen_targetHV_arr)

    for exp_no in range(len(results_dir)):
        if exp_no == base_exp_no:
            continue
        for gen in range(min(median_hv[base_exp_no].shape[0], median_hv[exp_no].shape[0])):
            hbase = median_hv[base_exp_no][gen, 1]
            hrep = median_hv[exp_no][gen, 1]
            max_hv = max(hbase, hrep)
            if feval_flag:
                pass
                # KLUGE
                # if gen * pop_size > 3.8e3:
                #     ax_hv.scatter(gen * pop_size, max_hv, marker='x', color='red')
            else:
                ax_hv.scatter(gen, max_hv, marker='x', color='red')


    # Wilcoxon rank-sum test to see if repair_inequality median hv statistically better than base HV
    # print("Alternate hypothesis = median HV (repair) > median HV(base)")
    # for exp_no in range(len(results_dir)):
    #     if exp_no == base_exp_no:
    #         continue
    #     for gen in range(median_hv[exp_no].shape[0]):
    #         median_hv_base = median_hv[base_exp_no][gen, 1]
    #         median_hv_repair = median_hv[exp_no][gen, 1]
    #         HV_stats = ranksums(x=median_hv_repair, y=median_hv_base, alternative='greater')
    #         print(f"Experiment {exp_no + 1}, gen = {gen}, statistic = {np.round(HV_stats.statistic, decimals=4)}, "
    #               f"p-value = {np.round(HV_stats.pvalue, decimals=4)})")

    # Wilcoxon rank-sum test to see if repair_inequality median IGD statistically better than base IGD
    # print("Alternate hypothesis = median IGD (repair) < median IGD(base)")
    # for exp_no in range(len(results_dir)):
    #     if exp_no == base_exp_no:
    #         continue
    #     idx = np.argwhere(np.diff(np.sign(target_hv - median_hv[exp_no][:, 1]))).flatten()
    #     # For IGD
    #     median_igd_base = median_igd[base_exp_no][:, 1]
    #     median_igd_repair = median_igd[exp_no][:, 1]
    #     IGD_stats = ranksums(x=median_igd_repair, y=median_igd_base, alternative='less')
    #     print(f"Experiment {exp_no + 1}, statistic = {np.round(IGD_stats.statistic, decimals=4)}, "
    #           f"p-value = {np.round(IGD_stats.pvalue, decimals=4)})")
    #
    # # Wilcoxon rank-sum test to see if feval to reach target HV is statistically more or less from base
    # print("Alternate hypothesis = feval to reach target HV (repair) < feval (base)")
    # for exp_no in range(len(results_dir)):
    #     if exp_no == base_exp_no:
    #         continue
    #     if feval_flag:
    #         feval_base = feval_targetHV_arr[base_exp_no]
    #         feval_repair = feval_targetHV_arr[exp_no]
    #         x_wilcoxon = feval_base
    #         y_wilcoxon = feval_repair
    #     else:
    #         gen_base = gen_targetHV_arr[base_exp_no]
    #         gen_repair = gen_targetHV_arr[exp_no]
    #         x_wilcoxon = gen_base
    #         y_wilcoxon = gen_repair
    #
    #     HV_stats = ranksums(x=x_wilcoxon, y=y_wilcoxon, alternative='less')
    #     print(f"Experiment {exp_no + 1}, statistic = {np.round(HV_stats.statistic, decimals=4)}, "
    #           f"p-value = {np.round(HV_stats.pvalue, decimals=4)})")

    with open(os.path.join(cmp_results_dir, 'hv_hist.pkl'), 'wb') as fp:
        pickle.dump(hv_hist, fp)

    with open(os.path.join(cmp_results_dir, 'igd_hist.pkl'), 'wb') as fp:
        pickle.dump(igd_hist, fp)

    with open(os.path.join(cmp_results_dir, 'igd_plus_hist.pkl'), 'wb') as fp:
        pickle.dump(igd_plus_hist, fp)

    # ax_pf.set_xlabel("Volume (m^2)")
    # ax_pf.set_ylabel("Deflection (m)")
    ax_pf.set_xlabel("f1")
    ax_pf.set_ylabel("f2")
    ax_pf.grid()
    ax_pf.set_axisbelow(True)
    ax_pf.legend()

    if feval_flag:
        ax_hv.set_xlabel('Function evaluations')
    else:
        ax_hv.set_xlabel('Generations')
    ax_hv.set_ylabel('Hypervolume')
    ax_hv.set_xlim([1, None])
    ax_hv.set_ylim([0, 1.6])
    ax_hv.grid()
    ax_hv.set_axisbelow(True)
    ax_hv.legend()

    if feval_flag:
        ax_igd.set_xlabel('Function evaluations')
    else:
        ax_igd.set_xlabel('Generations')
    ax_igd.set_ylabel('IGD')
    ax_igd.set_xlim([1, None])
    ax_igd.grid()
    ax_igd.set_axisbelow(True)
    ax_igd.legend()

    fig_pf.savefig(os.path.join(cmp_results_dir, 'fig_pf.png'))
    with open(os.path.join(cmp_results_dir, 'fig_pf.pkl'), 'wb') as fp:
        pickle.dump(fig_pf, fp)

    fig_hv.savefig(os.path.join(cmp_results_dir, 'fig_hv.png'))
    with open(os.path.join(cmp_results_dir, 'fig_hv.pkl'), 'wb') as fp:
        pickle.dump(fig_hv, fp)

    fig_hv.savefig(os.path.join(cmp_results_dir, 'fig_igd.png'))
    with open(os.path.join(cmp_results_dir, 'fig_igd.pkl'), 'wb') as fp:
        pickle.dump(fig_igd, fp)

    plt.show()
