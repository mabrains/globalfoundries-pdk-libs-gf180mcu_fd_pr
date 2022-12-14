# Copyright 2022 GlobalFoundries PDK Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:
  model_reg.py [--num_cores=<num>]

  -h, --help             Show help text.
  -v, --version          Show version.
  --num_cores=<num>      Number of cores to be used by simulator
"""

from docopt import docopt
import pandas as pd
import os
from jinja2 import Template
import concurrent.futures
import shutil
import multiprocessing as mp

import subprocess
import glob

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

DEFAULT_TEMP = 25.0
PASS_THRESH = 2.0


def find_diode(filename):
    """
    Find diode in log
    """
    cmd = 'grep "0  " {} | head -n 1'.format(filename)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    return process.communicate()


def call_simulator(file_name):
    """Call simulation commands to perform simulation.
    Args:
        file_name (str): Netlist file name.
    """
    return os.system(f"ngspice -b -a {file_name} -o {file_name}.log > {file_name}.log")


def ext_cv_measured(dev_data_path, device, corners, dev_path):
    # Read Data
    df = pd.read_excel(dev_data_path)
    dim_df = df[["L (um)", "W (um)"]].copy()
    all_dfs = []

    loops = dim_df["L (um)"].count()
    for i in range(0, loops):
        width = dim_df["W (um)"].iloc[int(i)]
        length = dim_df["L (um)"].iloc[int(i)]

        if i % 4 == 0:
            temp = -40
        elif i % 4 == 1:
            temp = 25
        elif i % 4 == 2:
            temp = 125
        else:
            temp = 175

        for corner in corners:
            idf = df[["Vj", f"diode_{corner}"]].copy()
            idf.rename(
                columns={f"diode_{corner}": "diode_measured", "Vj": "measured_volt"},
                inplace=True,
            )

            idf["corner"] = corner
            idf["length"] = length
            idf["width"] = width
            idf["temp"] = temp
            # all_dfs.append(idf)
            os.makedirs(f"{dev_path}/measured_cv", exist_ok=True)
            idf.to_csv(
                f"{dev_path}/measured_cv/measured_A{width}\
                    _P{length}_t{temp}_{corner}.csv"
            )

    sf = glob.glob(f"{dev_path}/measured_cv/*")  # stored_files
    for i in range(len(sf)):
        sdf = pd.read_csv(sf[i])
        all_dfs.append(sdf)

    df = pd.concat(all_dfs)
    # df["temp"] = DEFAULT_TEMP
    df["device"] = device
    df.dropna(axis=0, inplace=True)
    df = df[
        [
            "device",
            "corner",
            "length",
            "width",
            "temp",
            "measured_volt",
            "diode_measured",
        ]
    ]

    return df


def run_sim(dirpath, device, length, width, corner, temp):
    """ Run simulation at specific information and corner """
    netlist_tmp = "./device_netlists/cv.spice"

    info = {}
    info["device"] = device
    info["corner"] = corner
    info["temp"] = temp
    info["width"] = width
    info["length"] = length

    width_str = "{:.1f}".format(width)
    length_str = "{:.1f}".format(length)
    temp_str = "{:.1f}".format(temp)

    netlist_path = f"{dirpath}/{device}_netlists_cv/netlist_A{width_str}_P{length_str}_t{temp_str}_{corner}.spice"

    with open(netlist_tmp) as f:
        tmpl = Template(f.read())
        os.makedirs(f"{dirpath}/{device}_netlists_cv", exist_ok=True)
        os.makedirs(f"{dirpath}/simulated_cv", exist_ok=True)

        with open(netlist_path, "w") as netlist:
            netlist.write(
                tmpl.render(
                    device=device,
                    area=width_str,
                    pj=length_str,
                    corner=corner,
                    temp=temp_str,
                )
            )

    # Running ngspice for each netlist

    call_simulator(netlist_path)

    return info


def run_sims(df, dirpath, num_workers=mp.cpu_count()):

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_list = []
        for j, row in df.iterrows():
            futures_list.append(
                executor.submit(
                    run_sim,
                    dirpath,
                    row["device"],
                    row["length"],
                    row["width"],
                    row["corner"],
                    row["temp"],
                )
            )

        for future in concurrent.futures.as_completed(futures_list):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print("Test case generated an exception: %s" % (exc))

    all_dfs = []
    sf = glob.glob(f"{dirpath}/simulated_cv/*.csv")  # stored simulated data files
    for i in range(len(sf)):
        sdf = pd.read_csv(sf[i], header=None, delimiter=r"\s+",)
        sdf.rename(
            columns={1: "diode_simulated", 0: "simulated_volt"}, inplace=True,
        )
        sdf.to_csv(sf[i])
        all_dfs.append(sdf)

    df_res = pd.concat(all_dfs)
    df_res.dropna(axis=0, inplace=True)
    df_res.to_csv("try.csv")

    df = pd.DataFrame(results)

    df = df[["device", "corner", "length", "width", "temp"]]
    print(type(df_res["diode_simulated"]))

    df_res["device"] = df["device"]
    print(df_res)

    return df_res


def main():

    # pandas setup
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("display.width", 1000)

    main_regr_dir = "diode_regr"

    # diode var.
    corners = ["typical", "ff", "ss"]

    devices = [
        "diode_dw2ps",
        # "diode_pw2dw",
        # "diode_nd2ps_03v3",
        # "diode_nd2ps_06v0",
        # "diode_nw2ps_03v3",
        # "diode_nw2ps_06v0",
        # "diode_pd2nw_03v3",
        # "diode_pd2nw_06v0",
        # "sc_diode",
    ]

    for i, dev in enumerate(devices):
        dev_path = f"{main_regr_dir}/{dev}"

        if os.path.exists(dev_path) and os.path.isdir(dev_path):
            shutil.rmtree(dev_path)

        os.makedirs(f"{dev_path}", exist_ok=False)

        print("######" * 10)
        print(f"# Checking Device {dev}")

        diode_cv_data_files = glob.glob(f"./0_measured_data/{dev}_cv.nl_out.xlsx")
        if len(diode_cv_data_files) < 1:
            print("# Can't find mimcap file for device: {}".format(dev))
            diode_cv_file = ""
        else:
            diode_cv_file = diode_cv_data_files[0]
        print("# diode_cv data points file : ", diode_cv_file)

        if diode_cv_file == "":
            print(f"# No datapoints available for validation for device {dev}")
            continue

        if diode_cv_file != "":
            meas_cv_df = ext_cv_measured(diode_cv_file, dev, corners, dev_path)
        else:
            meas_cv_df = []

        print(
            "# Device {} number of measured_datapoints : ".format(dev), len(meas_cv_df)
        )

        sim_df = run_sims(meas_cv_df, dev_path, 3)
        print("# Device {} number of simulated datapoints : ".format(dev), len(sim_df))


# # ================================================================
# -------------------------- MAIN --------------------------------
# ================================================================

if __name__ == "__main__":

    # Args
    arguments = docopt(__doc__, version="comparator: 0.1")
    workers_count = (
        os.cpu_count() * 2
        if arguments["--num_cores"] is None
        else int(arguments["--num_cores"])
    )

    # Calling main function
    main()
