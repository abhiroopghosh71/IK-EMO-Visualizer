import json
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.termination import get_termination


def generate_data(problem_name: str, output_path: str, verbose: bool = False):
    """
    Generate sample data for IK-EMO-Viz using PyMOO
    :param verbose: Enables verbose messaging
    :param problem_name: The name of the problem
    :param output_path: The output data file
    """
    problem = get_problem(problem_name)
    algorithm = NSGA2(
        pop_size=40,
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.title("Objective Space")

    np.savetxt(os.path.join(output_path, 'X.DAT'), res.X, delimiter=',')
    np.savetxt(os.path.join(output_path, 'F.DAT'), res.F, delimiter=',')

    params = {'problem': problem_name,
              'n_obj': problem.n_obj, 'n_var': problem.n_var,
              'xl': problem.xl.tolist(), 'xu': problem.xu.tolist(),
              'obj_label': ['f1', 'f2'], 'ignore_vars': []
              }
    with open(os.path.join(output_path, 'params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)


def problem_name_arg_valid(prob_name_input, valid_problem_names):
    for p in prob_name_input:
        if p not in valid_problem_names:
            return False

    return True


if __name__ == '__main__':
    # Generate data for sample problems
    VALID_PROBLEM_LIST = ('truss2d', 'welded_beam')

    parser = argparse.ArgumentParser(
        prog='sample_data.py',
        description='Generates sample DAT file(s) compatible with IK-EMO-Viz')

    parser.add_argument('--seed', default='0',
                        help="Random number generator seed. Defaults to 0. Changing this may result in different "
                             "data than the default.")
    parser.add_argument('--problem-name', default='all',
                        help="Problem name whose data is to be generated. Generates values for all problems by "
                             "default.")
    parser.add_argument('--problem-list', action='store_true',
                        help="Problem name whose data is to be generated. Generates values for all problems by "
                             "default.")
    parser.add_argument('--out-path', default=None,
                        help="Output path of the sample data. Defaults to ./data/<problem name>")
    parser.add_argument('--plot-data', default=False, action='store_true',
                        help="If present, make visual plots of the sample data")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help="Print detailed messages")

    args = parser.parse_args()

    if args.problem_list:
        print("Valid problem names:")
        print(VALID_PROBLEM_LIST)

    # If problem name not provided, generate data for all problems. Otherwise, check validity of problem arguments
    if args.problem_name == 'all':
        problem_name_list = VALID_PROBLEM_LIST
    else:
        problem_name_list = [p.lower().strip() for p in args.problem_name.split(',')]
        if not problem_name_arg_valid(problem_name_list, VALID_PROBLEM_LIST):
            parser.print_help()
            print("Valid problem names:")
            print(VALID_PROBLEM_LIST)

    if args.seed.isdigit():
        np.random.seed(int(args.seed))
    else:
        parser.print_help()

    if args.out_path is not None:
        out_folder_path = args.out_path
    else:
        out_folder_path = 'data'

    if args.verbose:
        print("Valid problem names:")
        print(VALID_PROBLEM_LIST)
        print("Problem name input:")
        print(args.problem_name)
        print("Problem name input formatted to:")
        print(problem_name_list)
        print("Output path:")
        print(out_folder_path)

    for sample_problem_name in problem_name_list:
        out_folder_problem_data = os.path.join(out_folder_path, sample_problem_name)
        print(f"Generating data for problem {sample_problem_name}")

        if not os.path.exists(out_folder_problem_data):
            print(f"out_folder_problem_data {out_folder_problem_data}. Creating it now.")
            os.makedirs(out_folder_problem_data)

        generate_data(problem_name=sample_problem_name, output_path=out_folder_problem_data, verbose=args.verbose)
        print(f"Problem {sample_problem_name} sample data generation complete")

    if args.plot_data:
        plt.show()
