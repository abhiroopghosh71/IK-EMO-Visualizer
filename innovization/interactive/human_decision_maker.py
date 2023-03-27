import os

import numpy as np

from innovization.constants import *
from innovization.interactive.decision_making import DecisionMaker
from utils.record_data import USER_INTERACT_DIR, POWER_LAW_RANK_FILE_PREFIX, \
    INEQUALITY_RULE_RANK_FILE_PREFIX, CONSTANT_RULE_RANK_FILE_PREFIX


class HumanDM(DecisionMaker):
    def __init__(self, async_mode=False):
        self.async_mode = async_mode
        self.last_power_law_interaction = 0
        self.last_const_rule_interaction = 0
        self.last_ineq_rule_interaction = 0
        super().__init__(dm_type=HUMAN_DM)

    def interact(self, result_path):
        if not self.async_mode:
            input("Paused for decision making. Press any key to continue.")
        else:
            print("Ready for user interaction (asynchronous).")
        print("Analyzing user input.")

        user_input_dir = os.path.join(result_path, USER_INTERACT_DIR)
        # Get the latest generation at which user provided a ranking & read the corresponding input file
        max_gen = HumanDM.get_max_gen_user_interaction(POWER_LAW_RANK_FILE_PREFIX, user_input_dir)
        rank_list = {}
        # If gen 100 is the latest gen when user gave a rule ranking, then read a file
        # "<POWER_LAW_RANK_FILE_PREFIX>100"
        latest_power_law_rank_file = os.path.join(user_input_dir, f'{POWER_LAW_RANK_FILE_PREFIX}{max_gen}')
        if max_gen is None or not os.path.exists(latest_power_law_rank_file):
            print("No power law ranking file found.")
        elif max_gen > self.last_power_law_interaction:
            power_law_rank_list = np.loadtxt(latest_power_law_rank_file, delimiter=',')
            self.last_power_law_interaction = max_gen
            rank_list['power_law_rank_list'] = power_law_rank_list
        else:
            print(f"No new power law ranking was given since generation {max_gen}")

        # Repeat the above for constant rule
        max_gen = HumanDM.get_max_gen_user_interaction(CONSTANT_RULE_RANK_FILE_PREFIX, user_input_dir)
        latest_const_rule_rank_file = os.path.join(user_input_dir, f'{CONSTANT_RULE_RANK_FILE_PREFIX}{max_gen}')
        if max_gen is None or not os.path.exists(latest_const_rule_rank_file):
            print("No constant rule ranking file found.")
        elif max_gen > self.last_const_rule_interaction:
            const_rule_rank_list = np.loadtxt(latest_const_rule_rank_file, delimiter=',')
            self.last_const_rule_interaction = max_gen
            rank_list['const_rule_rank_list'] = const_rule_rank_list
        else:
            print(f"No new constant rule ranking was given since generation {max_gen}")

        # Repeat the above for inequality rule
        max_gen = HumanDM.get_max_gen_user_interaction(INEQUALITY_RULE_RANK_FILE_PREFIX, user_input_dir)
        latest_ineq_rule_rank_file = os.path.join(user_input_dir, f'{INEQUALITY_RULE_RANK_FILE_PREFIX}{max_gen}')
        if max_gen is None or not os.path.exists(latest_ineq_rule_rank_file):
            print("No inequality rule ranking file found.")
        elif max_gen > self.last_ineq_rule_interaction:
            ineq_rule_rank_list = np.loadtxt(latest_ineq_rule_rank_file, delimiter=',')
            self.last_ineq_rule_interaction = max_gen
            rank_list['ineq_rule_rank_list'] = ineq_rule_rank_list
        else:
            print(f"No new inequality rule ranking was given since generation {max_gen}")
        print("Resuming optimization")

        return rank_list

    @staticmethod
    def get_max_gen_user_interaction(file_prefix, user_input_dir):
        gen_list = [int(fname[len(file_prefix):])
                    for fname in os.listdir(user_input_dir)
                    if fname[len(file_prefix):].isdigit()]
        if len(gen_list) == 0:
            return None
        max_gen = np.max(gen_list)

        return max_gen
