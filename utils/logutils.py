import calendar
import datetime
import logging
import os
import sys
import time

gmt = time.gmtime()
ts = calendar.timegm(gmt)
results_parent_dir = 'results'
# results_dir = os.path.join(results_parent_dir, f'{ts}')
time_now = datetime.datetime.now()
results_dir = os.path.join(results_parent_dir, f'{time_now.strftime("%Y%m%d_%H%M%S")}')
# To convert timestamp to human-readable form
ts_simple = time.ctime(ts)


class Logger(object):
    def __init__(self, filepath=None):
        self.terminal = sys.stdout
        if filepath is None:
            self.log = open('logfile.log', 'a')
        else:
            self.log = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def setup_logging(res_dir, print_log_to_console=False):
    # logging.getLogger().addHandler(logging.StreamHandler())
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{res_dir}/optim_log_{ts}.log")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    if print_log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    sys.stdout = Logger(filepath=f"{res_dir}/terminal_output_{ts}.log")