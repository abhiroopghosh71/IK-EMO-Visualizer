import pytest
from utils.sample_data import problem_name_arg_valid


@pytest.mark.parametrize("prob_name_input,expected_val",
                         [([''], False),
                          (['prob_1'], True), (['prob_2'], True), (['prob_3'], True),
                          (['prob_1', 'prob_3'], True),
                          (['prob_1', 'prob_2', 'prob_3'], True),
                          (['prob_1', 'prob_', 'prob_3'], False)])
def test_problem_name_valid(prob_name_input: list, expected_val: bool):
    """
    Test problem name argument validation
    :param prob_name_input:
    :param expected_val:
    """
    valid_problem_names = ['prob_1', 'prob_2', 'prob_3']
    assert problem_name_arg_valid(prob_name_input, valid_problem_names) == expected_val
