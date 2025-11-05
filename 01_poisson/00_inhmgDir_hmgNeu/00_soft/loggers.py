"""
********************************************************************************
loggers, utils for logging
********************************************************************************
"""

import pathlib
import datetime


def make_logger(path):
    """
    make a logger

    args:
        path: path to save
    return:
        logger: logger
    """

    # if the file already exists, ask whether to overwrite
    if pathlib.Path(path).exists():
        while True:
            answer = input(f"The file '{path}' already exists. Do you want to overwrite? (y/n) ")
            if answer == "y":
                break
            elif answer == "n":
                path = input("Please enter a new file name: ")
                if pathlib.Path(path).exists():
                    continue
                else:
                    break
            else:
                continue

    # write nothing and return the path
    with open(path, mode="w") as f:
        pass
    return path


def write_logger(path, log):
    """
    write a log
    """

    with open(path, mode="a") as f:
        print(log, file=f)


