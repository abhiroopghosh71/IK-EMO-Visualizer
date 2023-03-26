import json

import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import os


def generate_data(problem_name, output_path):
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
    plt.title("Design Space")

    np.savetxt(os.path.join(output_path, 'X.DAT'), res.X, delimiter=',')
    np.savetxt(os.path.join(output_path, 'F.DAT'), res.F, delimiter=',')

    params = {'problem': problem_name,
              'n_obj': problem.n_obj, 'n_var': problem.n_var,
              'xl': problem.xl.tolist(), 'xu': problem.xu.tolist(),
              'obj_label': ['f1', 'f2'], 'ignore_vars': []
              }
    with open(os.path.join(output_path, 'params.DAT'), 'w') as fp:
        json.dump(params, fp, indent=4)


if __name__ == '__main__':
    np.random.seed(713)
    problem_name_default = 'truss2d'
    out_folder_path = os.path.join('data', problem_name_default)
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    generate_data(problem_name=problem_name_default, output_path=out_folder_path)
    plt.show()
