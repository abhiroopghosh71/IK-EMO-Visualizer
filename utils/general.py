import numpy as np
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints


def get_all_decision_makers():
    from innovization.interactive.human_decision_maker import HumanDM
    from innovization.interactive.artificial_decision_maker import ArtificialDM

    decision_makers = {
        'human_dm': (HumanDM, {'async_mode': False}),
        'human_dm_async': (HumanDM, {'async_mode': True}),
        'artificial_dm': (ArtificialDM, {}),
    }

    return decision_makers


def get_decision_maker(dm_name):
    return get_all_decision_makers()[dm_name]


def get_all_repair_agents():
    from innovization.repair_agent import PowerLawRepairAgent, DummyRepairAgent

    repair_agents = {
        'power_law_rep': (PowerLawRepairAgent, {'sig_multiplier': 0}),
        'power_law_rep_sig_0': (PowerLawRepairAgent, {'sig_multiplier': 0}),
        'power_law_rep_sig_1': (PowerLawRepairAgent, {'sig_multiplier': 1}),
        'power_law_rep_sig_2': (PowerLawRepairAgent, {'sig_multiplier': 2}),

        'ineq_rep': (PowerLawRepairAgent, {'sig_multiplier': 0}),

        'dummy_repair': (DummyRepairAgent, {}),
        'base': (DummyRepairAgent, {}),
    }

    return repair_agents


def get_repair_agent(agent_name):
    return get_all_repair_agents()[agent_name]


def get_agent_names_from_cmd_args(cmd_args):
    kwargs = {'agent_names': []}
    all_power_law_rep_agents = get_all_repair_agents()
    cmd_args_dict = vars(cmd_args)
    for key in cmd_args_dict:
        if key in all_power_law_rep_agents and cmd_args_dict[key]:
            kwargs['agent_names'].append(key)

    return kwargs


def get_knee(f, epsilon=0.125):
    # KLUGE: Pymoo changed from original. Line 23 in mcdm.high_tradeoff.py
    # neighbors = neighbors_finder.find(i) replaced by the next line
    # neighbors = [j for j in neighbors_finder.find(i) if j != i]

    # dm = HighTradeoffPoints(epsilon=epsilon)
    dm = HighTradeoffPoints(zero_to_one=True, ideal=np.array([0, 0]), nadir=np.array([3, 0.012]))
    knee_indx = dm.do(f)

    return knee_indx
