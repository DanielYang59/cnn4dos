"""Utility script to recording system/Python env information."""

import json
import logging
import platform
import socket
import subprocess

import psutil


def record_python_package_ver(logfile="python_package_ver.log") -> None:
    """Record Python package versions to file.

    Args:
        logfile (str, optional): logfile name.
            Defaults to "python_package_ver.log".

    Notes:
        ref: https://stackoverflow.com/questions/50964992/
        save-all-currently-installed-packages-in-anaconda-to-a-file

    """
    # Get all installed Python packages
    result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )

    # Write results to log file
    with open(logfile, mode="w") as f:
        for line in result:
            f.write(line)


def record_system_info(logfile="sys_info.log"):
    """Record system related information to file.

    Args:
        logfile (str, optional): name of logfile. Defaults to "sys_info.log".

    Notes:
        ref: https://stackoverflow.com/questions/3103178/
        how-to-get-the-system-info-with-python

    """
    # Get system info
    try:
        info = {}
        info["platform"] = platform.system()
        info["platform-release"] = platform.release()
        info["platform-version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["processor"] = platform.processor()
        info["ram"] = f"{str(
            round(psutil.virtual_memory().total / 1024.0**3)
        )} GB"
    except Exception as e:
        logging.exception(e)

    # Write info to file
    log_content = json.loads(json.dumps(info))
    with open(logfile, mode="w") as f:
        for key, value in log_content.items():
            f.write(f"{key}: {value}\n")
