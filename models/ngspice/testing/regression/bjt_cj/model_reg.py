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
  models_regression.py [--num_cores=<num>]

  -h, --help             Show help text.
  -v, --version          Show version.
  --num_cores=<num>      Number of cores to be used by simulator
"""

from docopt import docopt
import pandas as pd
import numpy as np
import os
from jinja2 import Template
import concurrent.futures
import shutil
import multiprocessing as mp

import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
PASS_THRESH = 2.0  # threshold value for passing devices
no_rows_npn = 54  # no.of combinations extracted from npn sheet
no_rows_pnp = 24  # no.of combinations extracted from pnp sheet


def call_simulator(file_name):
    """Call simulation commands to perform simulation.
    Args:
        file_name(str): Netlist file name.
    """
    return os.system(
        f"ngspice -b -a {file_name} -o {file_name}.log > {file_name}.log"
    )


def ext_measured(
    cj_file: str, dev: str, devices: list, dev_path: str, no_rows=int
) -> pd.DataFrame:
    """Extracting the measured data of npn devices from excel sheet

    Args:
         cj_file(str): path to the data sheet
         dev(str): device name whether npn or pnp
         devices(list): list for undertest devices
         dev_path(str): A path where extracted data is stored
         no_rows(int): no.of combinations extracted from npn, pnp sheet

    Returns:
         df_measured(pd.DataFrame): A data frame contains all extracted data

    """

    # Reading excel sheet and creating data frame
    df = pd.read_excel(cj_file)

    # temp_range is threshold for switching between 25, -40, 125
    temp_range = int(no_rows / 3)
    # initiating empty list for appendindurationg dataframes
    all_dfs = list()

    # Extracting measured values for each Device
    for i in range(no_rows):

        # building up temperature
        if i in range(0, temp_range):
            temp = 25
        elif i in range(temp_range, 2 * temp_range):
            temp = -40
        else:
            temp = 175

        # extracted columns from sheet
        temp_value = list()
        dev_names = list()
        cj_measured = list()
        cap_names = list()

        # reading third column for getting device name, cap_name
        data = df["Unnamed: 2"][i]  # data is a string now
        cap_name = data[4:7]
        end_location = data.find("(")
        read_dev_name = data[8:end_location]
        space = read_dev_name.find(" ")
        read_dev_name = read_dev_name[0:space]

        # renaming the device like the standard
        # for npn devices:
        if dev == "npn":
            if read_dev_name == "10x10":
                dev_name = devices[0]
            elif read_dev_name == "5x5":
                dev_name = devices[1]
            elif read_dev_name == "0p54x16":
                dev_name = devices[2]
            elif read_dev_name == "0p54x8":
                dev_name = devices[3]
            elif read_dev_name == "0p54x4":
                dev_name = devices[4]
            elif read_dev_name == "0p54x2":
                dev_name = devices[5]

        # for pnp devices:
        elif dev == "pnp":
            if read_dev_name == "0p42x10":
                dev_name = devices[0]
            elif read_dev_name == "0p42x5":
                dev_name = devices[1]
            elif read_dev_name == "10x10":
                dev_name = devices[2]
            elif read_dev_name == "5x5":
                dev_name = devices[3]

        # extracting C-V measured data
        # Special case for 1st measured values
        if i == 0:
            cj_values = df[
                [
                    "Vj",
                    "bjt_typical",
                    "bjt_ff",
                    "bjt_ss",
                ]
            ].copy()

            cj_values.rename(
                columns={
                    "Vj": "measured_volt",
                    "bjt_typical": "measured_bjt_typical",
                    "bjt_ff": "measured_bjt_ff",
                    "bjt_ss": "measured_bjt_ss",
                },
                inplace=True,
            )

        else:
            cj_values = df[
                [
                    "Vj",
                    f"bjt_typical.{i}",
                    f"bjt_ff.{i}",
                    f"bjt_ss.{i}",
                ]
            ].copy()

            cj_values.rename(
                columns={
                    "Vj": "measured_volt",
                    f"bjt_typical.{i}": "measured_bjt_typical",
                    f"bjt_ff.{i}": "measured_bjt_ff",
                    f"bjt_ss.{i}": "measured_bjt_ss",
                },
                inplace=True,
            )

        os.makedirs(f"{dev_path}/cj_measured", exist_ok=True)
        cj_values.to_csv(
            f"{dev_path}/cj_measured/measured_{dev_name}_t{temp}.csv"
        )

        dev_names.append(dev_name)
        cap_names.append(cap_name)
        temp_value.append(temp)
        cj_measured.append(
            f"{dev_path}/cj_measured/measured_{dev_name}_t{temp}.csv"
        )

        sdf = {
            "device": dev_names,
            "temp": temp_value,
            "cap": cap_names,
            "cj_measured": cj_measured,
        }
        sdf = pd.DataFrame(sdf)
        all_dfs.append(sdf)

    df = pd.concat(all_dfs)
    df.dropna(axis=0, inplace=True)
    return df


def run_sim(dirpath: str, cap: str, device: str, temp: float) -> dict:
    """Run simulation at specific information and corner
    Args:
        dirpath(str): path to the file where we write data
        cap(str): under-test cap to select proper netlist
        device(str): the device instance will be simulated
        temp(float): a specific temp for simulation

    Returns:
        info(dict): results are stored in,
        and passed to the run_sims function to extract data
    """

    corners = ["typical", "ff", "ss"]
    for corner in corners:
        info = dict()
        info["device"] = device
        info["temp"] = temp
        dev = device.split("_")[0]

        netlist_tmp = f"./device_netlists/{dev}_{cap}.spice"
        temp_str = "{:.1f}".format(temp)

        netlist_path = (
            f"{dirpath}/{dev}_netlists/"
            + f"netlist_{device}_t{temp_str}_c{corner}.spice"
        )

        result_path = (
            f"{dirpath}/cj_simulated/"
            + f"simulated_{device}_t{temp_str}_c{corner}.csv"
        )

        # initiating the directory in which results will be stored
        os.makedirs(f"{dirpath}/cj_simulated", exist_ok=True)

        with open(netlist_tmp) as f:
            tmpl = Template(f.read())
            os.makedirs(f"{dirpath}/{dev}_netlists", exist_ok=True)

            with open(netlist_path, "w") as netlist:
                netlist.write(
                    tmpl.render(
                        device=device,
                        temp=temp_str,
                        corner=corner,
                    )
                )

        # Running ngspice for each netlist
        try:
            call_simulator(netlist_path)

            # check if results stored in csv file or not!
            if os.path.exists(result_path):
                bjt_simu_cj = result_path
            else:
                bjt_simu_cj = "None"

        except Exception:
            bjt_simu_cj = "None"

    info["cj_simulated"]= bjt_simu_cj

    return info


def run_sims(
    df: pd.DataFrame, dirpath: str, num_workers=mp.cpu_count()
) -> None:
    """passing netlists to run_sim function
        and storing the results csv files into dataframes

    Args:
        df(pd.DataFrame): dataframe passed from the ext_measured function
        dirpath(str): the path to the file where we write data
        num_workers=mp.cpu_count() (int): num of cpu used

    Returns:
        df(pd.DataFrame): dataframe contains simulated results
    """

    results = list()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures_list = list()
        for j, row in df.iterrows():
            futures_list.append(
                executor.submit(
                    run_sim, dirpath, row["cap"], row["device"], row["temp"]
                )
            )

        for future in concurrent.futures.as_completed(futures_list):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print("Test case generated an exception: %s" % (exc))

    return None


def main():
    """Main function applies all regression steps"""

    # pandas setup
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("display.width", 1000)

    main_regr_dir = "bjt_cj_regr"
    devices = ["npn", "pnp"]
    caps_npn = ["EBJ", "CBJ", "CSJ"]
    caps_pnp = ["EBJ", "CBJ"]
    npn_devices = [
        "npn_10p00x10p00",
        "npn_05p00x05p00",
        "npn_00p54x16p00",
        "npn_00p54x08p00",
        "npn_00p54x04p00",
        "npn_00p54x02p00",
    ]

    pnp_devices = [
        "pnp_10p00x00p42",
        "pnp_05p00x00p42",
        "pnp_10p00x10p00",
        "pnp_05p00x05p00",
    ]

    for i, dev in enumerate(devices):
        dev_path = f"{main_regr_dir}/{dev}"

        if os.path.exists(dev_path) and os.path.isdir(dev_path):
            shutil.rmtree(dev_path)

        os.makedirs(f"{dev_path}", exist_ok=False)

        print("######" * 10)
        print(f"# Checking Device {dev}")

        cj_data_files = glob.glob(
            f"../../180MCU_SPICE_DATA/BJT/bjt_cv_{dev}.nl_out.xlsx"
        )
        if len(cj_data_files) < 1:
            print("# Can't find data file for device: {}".format(dev))
            cj_file = ""
        else:
            cj_file = cj_data_files[0]
        print("# bjt_cj data points file : ", cj_file)

        if cj_file == "":
            print(f"# No datapoints available for validation for device {dev}")
            continue

        if dev == "npn":
            list_dev = npn_devices
            caps = caps_npn
            no_rows = no_rows_npn
        elif dev == "pnp":
            list_dev = pnp_devices
            caps = caps_pnp
            no_rows = no_rows_pnp

        if cj_file != "":
            meas_df = ext_measured(cj_file, dev, list_dev, dev_path, no_rows)
        else:
            meas_df = list()

        meas_len = len(
            pd.read_csv(glob.glob(f"{dev_path}/cj_measured/*.csv")[1])
        )

        print(
            f"# Device {dev} number of measured_datapoints : ",
            len(meas_df) * meas_len,
        )

        # assuming number of used cores is 3
        # calling run_sims function for simulating devices
        run_sims(meas_df, dev_path, 3)


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
