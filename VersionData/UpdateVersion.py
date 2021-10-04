import os
import sys


def read_config(filename):
    config_dict = {}
    with open(filename) as f:
        for lines in f:
            if len(lines.strip()) == 0:
                continue
            items = lines.replace(" ", "").strip("\n").split('=')
            config_dict[items[0]] = eval(items[1])

    return config_dict


versionTemplae = \
"""
major = {major}
minor = {minor}
build = {build}
""".strip("\n")

versionFileName = "version.py"
config = os.path.join(os.path.dirname(sys.argv[0]), versionFileName)
version = read_config(config)

versionData = versionTemplae.format(major=version["major"], minor=version["minor"], build=version["build"]+1)

with open(config, "w") as versionFile:
    versionFile.write(versionData)
