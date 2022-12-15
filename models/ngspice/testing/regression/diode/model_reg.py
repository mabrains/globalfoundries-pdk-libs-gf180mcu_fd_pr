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

from unittest.mock import DEFAULT
from docopt import docopt
import pandas as pd
import numpy as np
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
MAX_VOLTAGE = 3.3

def find_diode(filepath):
    """
    Find diode in csv files
    """
    return os.path.exists(filepath)
    


def call_simulator(file_name):
    """Call simulation commands to perform simulation.
    Args:
        file_name (str): Netlist file name.
    """
    return os.system(f"ngspice -b -a {file_name} -o {file_name}.log > {file_name}.log")

def ext_cv_measured(dev_data_path, device, corners,dev_path):
    # Read Data
    df = pd.read_excel(dev_data_path)

    dim_df = df[["L (um)", "W (um)"]].copy()
    dim_df.rename(
        columns={
            "L (um)": "length",
            "W (um)" : "width"
        },
        inplace=True,
    )
    
    all_dfs = []
    
    loops = dim_df["length"].count()
    for corner in corners:

        for i in range(0, loops):
            width = dim_df["width"].iloc[int(i)]
            length = dim_df["length"].iloc[int(i)]

            if i % 4 == 0:
                temp = -40
            elif i % 4 == 1:
                temp = 25
            elif i % 4 == 2:
                temp = 125
            else:
                temp = 175
            
            leng = []
            wid = []
            tempr = []
            cor = []
            meas = []

            if i == 0 : 
                idf = df[["Vj", f"diode_{corner}"]].copy()
                idf.rename(
                    columns={
                        f"diode_{corner}": "diode_measured",
                        "Vj" : "measured_volt"
                    },
                    inplace=True,
                )
                
        
            else :

                idf = df[["Vj", f"diode_{corner}.{i}"]].copy()
                idf.rename(
                    columns={
                        f"diode_{corner}.{i}": "diode_measured",
                        "Vj" : "measured_volt"
                    },
                    inplace=True,
                ) 

            meas_volt = []
            meas_diode = []
            for j in range(idf["measured_volt"].count()) : 
                if abs(idf["measured_volt"][j]) < MAX_VOLTAGE :
                    meas_volt.append(idf["measured_volt"][j])
                    meas_diode.append(idf["diode_measured"][j])
                else :
                    break  
            meas_data = pd.DataFrame({"measured_volt":meas_volt,"diode_measured":meas_diode})

            os.makedirs(f"{dev_path}/measured_cv", exist_ok=True)
            meas_data.to_csv(f"{dev_path}/measured_cv/measured_A{width}_P{length}_t{temp}_{corner}.csv")
            
            leng.append(length)
            wid.append(width)
            tempr.append(temp)
            cor.append(corner)
            meas.append(f"{dev_path}/measured_cv/measured_A{width}_P{length}_t{temp}_{corner}.csv")

        
            sdf = {"length":leng,"width":wid,"temp":tempr,"corner":cor,"diode_measured":meas}
            sdf = pd.DataFrame(sdf)
            all_dfs.append(sdf)

    
    df = pd.concat(all_dfs)       
    df.dropna(axis=0, inplace=True)
    df["device"] = device
    df = df[["device","length","width","temp","corner","diode_measured"]]

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
    result_path = f"{dirpath}/simulated_cv/simulated_A{width_str}_P{length_str}_t{temp_str}_{corner}.csv"
    
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
    try:
        call_simulator(netlist_path)
        # Find diode in csv
        if (find_diode(result_path)):
            diode_simu = result_path
        else :
            diode_simu = "None"
    except Exception as e:
        diode_simu = "None"

    info["diode_sim_unscaled"] = diode_simu
    
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


    sf = glob.glob(f"{dirpath}/simulated_cv/*.csv") # stored simulated data files 
    for i in range(len(sf)):
        sdf = pd.read_csv(
                sf[i],
                header=None,
                delimiter=r"\s+",
            )
        sdf.rename(
                columns={
                    1 : "diode_simulated",
                    0 : "simulated_volt"
                },
                inplace=True,
            )
        sdf.to_csv(sf[i])
        
    df = pd.DataFrame(results)
    
    df = df[
        ["device","length", "width", "temp", "corner",  "diode_sim_unscaled"]
    ]
    df["diode_sim"] = df["diode_sim_unscaled"] 

    return df

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
        "diode_pw2dw",
        "diode_nd2ps_03v3",
        "diode_nd2ps_06v0",
        "diode_nw2ps_03v3",
        "diode_nw2ps_06v0",
        "diode_pd2nw_03v3",
        "diode_pd2nw_06v0",
        "sc_diode",
    ]

    for i, dev in enumerate(devices):
        dev_path = f"{main_regr_dir}/{dev}"

        if os.path.exists(dev_path) and os.path.isdir(dev_path):
            shutil.rmtree(dev_path)

        os.makedirs(f"{dev_path}", exist_ok=False)

        print("######" * 10)
        print(f"# Checking Device {dev}")

        diode_cv_data_files = glob.glob(
            f"./0_measured_data/{dev}_cv.nl_out.xlsx"
        )
        if len(diode_cv_data_files) < 1:
            print("# Can't find mimcap file for device: {}".format(dev))
            diode_cv_file = ""
        else:
            diode_cv_file = diode_cv_data_files[0]
        print("# diode_cv data points file : ", diode_cv_file)

        if diode_cv_file == "" :
            print(f"# No datapoints available for validation for device {dev}")
            continue
            
        if diode_cv_file != "":
            meas_cv_df = ext_cv_measured(diode_cv_file, dev, corners,dev_path)
        else:
            meas_cv_df = []
        
        meas_len = len(pd.read_csv(glob.glob(f"{dev_path}/measured_cv/*.csv")[1]))


        print("# Device {} number of measured_datapoints : ".format(dev), len(meas_cv_df)*meas_len)
        

        sim_cv_df = run_sims(meas_cv_df, dev_path, 3)
        sim_len = len(pd.read_csv(glob.glob(f"{dev_path}/simulated_cv/*.csv")[1]))
        print("# Device {} number of simulated datapoints : ".format(dev), len(sim_cv_df)*sim_len)


        merged_cv_df = meas_cv_df.merge(
            sim_cv_df, on=["device", "corner", "length", "width", "temp"], how="left"
        )
        
        merged_cv_dfs = []
        for i in range(len(merged_cv_df)):
            measured_cv_data = pd.read_csv(merged_cv_df["diode_measured"][i])
            simulated_cv_data = pd.read_csv(merged_cv_df["diode_sim"][i])
            result_cv_data = simulated_cv_data.merge(
            measured_cv_data, how="left"
            )
            result_cv_data["corner"] = merged_cv_df["diode_measured"][i].split("/")[-1].split("_")[-1].split(".")[0]
            result_cv_data["device"] = merged_cv_df["diode_measured"][i].split("/")[1]
            result_cv_data["length"] = merged_cv_df["diode_measured"][i].split("/")[-1].split("_")[1].split("A")[1]
            result_cv_data["width"] = merged_cv_df["diode_measured"][i].split("/")[-1].split("_")[2].split("P")[1]
            result_cv_data["temp"] = merged_cv_df["diode_measured"][i].split("/")[-1].split("_")[3].split("t")[1]
            
            
            result_cv_data["error"] = (
                np.abs(result_cv_data["diode_simulated"] - result_cv_data["diode_measured"])
                 * 100.0
                / result_cv_data["diode_measured"]
            )

            result_cv_data = result_cv_data [
                ["device","length","width","temp","corner","measured_volt","diode_measured","diode_simulated","error"]
            ]

            merged_cv_dfs.append(result_cv_data)
        
        merged_cv_out = pd.concat(merged_cv_dfs)


        merged_cv_out.to_csv(f"{dev_path}/error_analysis_cv.csv", index=False)

        print(
            "# Device {} min error: {:.2f} , max error: {:.2f}, mean error {:.2f}".format(
                dev,
                merged_cv_out["error"].min(),
                merged_cv_out["error"].max(),
                merged_cv_out["error"].mean(),
            )
        )

        if result_cv_data["error"].max() < PASS_THRESH:
            print("# Device {} has passed regression.".format(dev))
        else:
            print("# Device {} has failed regression. Needs more analysis.".format(dev))

        print("\n\n")
        


# # ================================================================
# -------------------------- MAIN --------------------------------
# ================================================================

if __name__ == "__main__":

    # Args
    arguments = docopt(__doc__, version="comparator: 0.1")
    workers_count = (
        os.cpu_count() * 2
        if arguments["--num_cores"] == None
        else int(arguments["--num_cores"])
    )

    # Calling main function
    main()

