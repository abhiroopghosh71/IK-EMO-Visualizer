from innovization.interactive.decision_making import DecisionMaker
from innovization.constants import *


class ArtificialDM(DecisionMaker):
    def __init__(self, rule_selection_criteria):
        super().__init__(dm_type=ARTIFICIAL_DM)
        self.rule_selection_criteria = rule_selection_criteria

    def interact(self, innov):
        for key in self.rule_selection_criteria.keys():
            # The 'del' key consists of the keys we need to remove from the rule_selection_criteria dict.
            if key == 'delete':
                keys_to_delete = self.rule_selection_criteria[key]
                for del_key in keys_to_delete:
                    innov.rule_selection_criteria.pop(del_key)
            # Add or modify a rule selection criteria
            else:
                innov.rule_selection_criteria[key] = self.rule_selection_criteria[key]
