import argparse
import multiprocessing as mp
from utils.general import get_all_repair_agents, get_all_decision_makers


def get_argparser(description='Optimization problem', n_var=None):
    """Intended to replace the need for defining standard arguments."""
    # Command line args accepted by the program
    parser = argparse.ArgumentParser("IK-EMO Visualizer")
    parser.add_argument("--result-path", type=str, help="Path to the results to be visualized")
    parser.add_argument("--port", type=int, default=8050, help="Port to host the dash server")
    parser.add_argument("--special-flag", type=str, default=None,
                        help="Any special flag to be passed to the viz portal")
    parser.add_argument("--app-mode", type=str, default='single-input',
                        help="Choose whether to start IK-EMO Visualizer in single-input mode or interactive mode")

    parser.add_argument("--X-file", type=str, default=None,
                        help="Input a CSV file containing the decision variable values of an EMO population")
    parser.add_argument("--F-file", type=str, default=None,
                        help="Input a CSV file containing the objective values of an EMO population")
    parser.add_argument("--params-file", type=str, default=None,
                        help="Input a CSV file containing the EMO parameters (for example, no. of objectives)")

    return parser
