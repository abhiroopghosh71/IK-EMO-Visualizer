import os
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, askopenfilenames


def open_file_selection_dialog(multi_file=False, title="Select file", initialdir=os.getcwd(),
                               file_types=(("all files", "*.*"), ("CSV files", "*.csv"), ("HDF5 files", "*.hdf5"))):
    """Opens a file selection dialog for the user to select the csv file to open. Returns the full path of the
    selected file.

        Returns:
            file_path (str): The path of the file_path selected by the user
    """
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if multi_file:
        file_path = askopenfilenames(initialdir=initialdir,
                                     title=title,
                                     filetypes=file_types)
    else:
        file_path = askopenfilename(initialdir=initialdir,
                                    title=title,
                                    filetypes=file_types)

    return file_path


def open_dir_selection_dialog(title="Select folder", initialdir=os.getcwd()):
    """Opens a file selection dialog for the user to select the csv file to open. Returns the full path of the
        selected file.

        Returns:
            folder_path (str): The path of the file_path selected by the user
    """
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    folder_selected = filedialog.askdirectory()

    return folder_selected


if __name__ == "__main__":
    # print(open_file_selection_dialog(multi_file=True))
    # print(open_file_selection_dialog(multi_file=False))
    print(open_dir_selection_dialog())
