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

def find_bjt(filepath):
    """
    Find bjt in csv files
    """
    return os.path.exists(filepath)
    


def call_simulator(file_name):
    """Call simulation commands to perform simulation.
    Args:
        file_name (str): Netlist file name.
    """
    return os.system(f"ngspice -b -a {file_name} -o {file_name}.log > {file_name}.log")

def ext_npn_measured(dev_data_path, device,devices,dev_path):
    # Read Data
    df = pd.read_excel(dev_data_path)

    all_dfs = []
    
    loops = df["corners"].count()

    for i in range(loops):

        temp_range = int(loops / 4)
        if i in range(0, temp_range):
            temp = 25
        elif i in range(temp_range, 2 * temp_range):
            temp = -40
        elif i in range(2 * temp_range, 3 * temp_range):
            temp = 125
        else:
            temp = 175

        tempr = []
        dev = []
        ic_meas = []
        ib_meas = []


        k = i
        if i >= len(devices):
            while k >= len(devices):
                k = k - len(devices)

        # Special case for 1st measured values

        if i == 0:

            idf_ic = df[["vbp ","vcp =1","vcp =2","vcp =3"]].copy()
            idf_ic.rename(
                    columns={
                        "vbp " : "measured_base_volt",
                        "vcp =1" : "measured_Ic_vcp_step1",
                        "vcp =2" : "measured_Ic_vcp_step2",
                        "vcp =3" : "measured_Ic_vcp_step3",
                    },
                    inplace=True,
                )
            
            
        else :

            idf_ic = df[["vbp ",f"vcp =1.{2*i}",f"vcp =2.{2*i}",f"vcp =3.{2*i}"]].copy()
            idf_ic.rename(
                    columns={
                        "vbp " : "measured_base_volt",
                        f"vcp =1.{2*i}" : "measured_Ic_vcp_step1",
                        f"vcp =2.{2*i}" : "measured_Ic_vcp_step2",
                        f"vcp =3.{2*i}" : "measured_Ic_vcp_step3",
                    },
                    inplace=True,
                )

        os.makedirs(f"{dev_path}/ic_measured", exist_ok=True)
        idf_ic.to_csv(f"{dev_path}/ic_measured/measured_{devices[k]}_t{temp}.csv")
            
        idf_ib = df[["vbp ",f"vcp =1.{2*i+1}",f"vcp =2.{2*i+1}",f"vcp =3.{2*i+1}"]].copy()
        idf_ib.rename(
                columns={
                    "vbp " : "measured_base_volt",
                    f"vcp =1.{2*i+1}" : "measured_Ib_vcp_step1",
                    f"vcp =2.{2*i+1}" : "measured_Ib_vcp_step2",
                    f"vcp =3.{2*i+1}" : "measured_Ib_vcp_step3",
                },
                inplace=True,
            )
        
        os.makedirs(f"{dev_path}/ib_measured", exist_ok=True)
        idf_ib.to_csv(f"{dev_path}/ib_measured/measured_{devices[k]}_t{temp}.csv")

        dev.append(devices[k])
        tempr.append(temp)
        ic_meas.append(f"{dev_path}/ic_measured/measured_{devices[k]}_t{temp}.csv")
        ib_meas.append(f"{dev_path}/ic_measured/measured_{devices[k]}_t{temp}.csv")

        sdf = {"device":dev,"temp":tempr,"ic_measured":ic_meas,"ib_measured":ib_meas}
        sdf = pd.DataFrame(sdf)
        all_dfs.append(sdf)
    

    df = pd.concat(all_dfs)       
    df.dropna(axis=0, inplace=True)
    df["corner"] = "typical"
    df = df[["device","temp","corner","ic_measured","ib_measured"]]
    
    return df

def run_sim(char,dirpath, device, temp):
    """ Run simulation at specific information and corner """
    
    info = {}
    info["device"] = device
    info["temp"] = temp
    dev = device.split("_")[0]

    netlist_tmp = f"./device_netlists/{dev}.spice"
    
    temp_str = "{:.1f}".format(temp)
    
    
    netlist_path = f"{dirpath}/{dev}_netlists/netlist_{device}_t{temp_str}.spice"

    for c in char:
        result_path = f"{dirpath}/{c}_simulated/simulated_{device}_t{temp_str}.csv"
        os.makedirs(f"{dirpath}/{c}_simulated", exist_ok=True)
    
    with open(netlist_tmp) as f:
        tmpl = Template(f.read())
        os.makedirs(f"{dirpath}/{dev}_netlists", exist_ok=True)
        

        with open(netlist_path, "w") as netlist:
            netlist.write(
                tmpl.render(
                    device=device,
                    temp=temp_str,
                )
            )
        
    # Running ngspice for each netlist
    try:
        call_simulator(netlist_path)
        # Find bjt in csv
        if (find_bjt(result_path)):
            bjt_simu = result_path
        else :
            bjt_simu = "None"
    except Exception as e:
        bjt_simu = "None"

    info["beta_sim_unscaled"] = bjt_simu
    
    return info


def run_sims(char,df, dirpath, num_workers=mp.cpu_count()):


    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_list = []
        for j, row in df.iterrows():
            futures_list.append(
                executor.submit(
                    run_sim,
                    char,
                    dirpath,
                    row["device"],
                    row["temp"],
                )
            )

        for future in concurrent.futures.as_completed(futures_list):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print("Test case generated an exception: %s" % (exc))

    #for c in char :
    # sf = glob.glob(f"{dirpath}/simulated_{char}/*.csv") # stored simulated data files 
    # for i in range(len(sf)):
    #     sdf = pd.read_csv(
    #             sf[i],
    #             header=None,
    #             delimiter=r"\s+",
    #         )
    #     sdf.rename(
    #             columns={
    #                 1 : "diode_simulated",
    #                 0 : "simulated_volt"
    #             },
    #             inplace=True,
    #         )
    #     sdf.to_csv(sf[i])
        
    df = pd.DataFrame(results)
    
    df = df[
        ["device", "temp",  "beta_sim_unscaled"]
    ]
    df["beta_sim"] = df["beta_sim_unscaled"] 

    return df

def main():

    # pandas setup
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("display.width", 1000)

    main_regr_dir = "bjt_beta_regr"

    # bjt var.

    devices = [ 
        "npn",
        #"pnp"
    ]

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
        "pnp_05p00x05p00"
    ]

    char = ["ib","ic"]

    for i, dev in enumerate(devices):
        dev_path = f"{main_regr_dir}/{dev}"

        if os.path.exists(dev_path) and os.path.isdir(dev_path):
            shutil.rmtree(dev_path)

        os.makedirs(f"{dev_path}", exist_ok=False)

        print("######" * 10)
        print(f"# Checking Device {dev}")

        print("\n")

        #for c in char :

        beta_data_files = glob.glob(
            f"../../180MCU_SPICE_DATA/BJT/bjt_{dev}_beta_f.nl_out.xlsx"
        )
        if len(beta_data_files) < 1:
            print("# Can't find diode file for device: {}".format(dev))
            beta_file = ""
        else:
            beta_file = beta_data_files[0]
        print(f"# bjt_beta data points file : ", beta_file)


        if beta_file == "" :
            print(f"# No datapoints available for validation for device {dev}")
            continue
            
        if dev == "npn" :
            f = ext_npn_measured
            list_dev = npn_devices

        if beta_file != "":
            meas_df = ext_npn_measured(beta_file, dev,list_dev,dev_path)
        else:
            meas_df = []

            
        meas_len = len(pd.read_csv(glob.glob(f"{dev_path}/ic_measured/*.csv")[1]))


        print(f"# Device {dev} number of measured_datapoints : ", len(meas_df)*meas_len)
            

        sim_df = run_sims(char,meas_df, dev_path, 3)
            #sim_len = len(pd.read_csv(glob.glob(f"{dev_path}/{c}_simulated/*.csv")[1]))
            #print(f"# Device {dev} number of {c}_simulated datapoints : ", len(sim_df)*sim_len)

            # # compare section 

         

            # merged_df = meas_df.merge(
            #     sim_df, on=["device", "corner", "length", "width", "temp"], how="left"
            # )
            
            # merged_dfs = []
            # for i in range(len(merged_df)):
            #     measured_data = pd.read_csv(merged_df["diode_measured"][i])
            #     simulated_data = pd.read_csv(merged_df["diode_sim"][i])
            #     result_data = simulated_data.merge(
            #     measured_data, how="left"
            #     )
            #     result_data["corner"] = merged_df["diode_measured"][i].split("/")[-1].split("_")[-1].split(".")[0]
            #     result_data["device"] = merged_df["diode_measured"][i].split("/")[1]
            #     result_data["length"] = merged_df["diode_measured"][i].split("/")[-1].split("_")[1].split("A")[1]
            #     result_data["width"] = merged_df["diode_measured"][i].split("/")[-1].split("_")[2].split("P")[1]
            #     result_data["temp"] = merged_df["diode_measured"][i].split("/")[-1].split("_")[3].split("t")[1]
                
                
            #     result_data["error"] = (
            #         np.abs(result_data["diode_simulated"] - result_data["diode_measured"])
            #         * 100.0
            #         / result_data["diode_measured"]
            #     )

            #     result_data = result_data [
            #         ["device","length","width","temp","corner","measured_volt","diode_measured","diode_simulated","error"]
            #     ]

            #     merged_dfs.append(result_data)
            
            # merged_out = pd.concat(merged_dfs)


            # merged_out.to_csv(f"{dev_path}/error_analysis_{c}.csv", index=False)

            # print(
            #     "# Device {} min error: {:.2f} , max error: {:.2f}, mean error {:.2f}".format(
            #         dev,
            #         merged_out["error"].min(),
            #         merged_out["error"].max(),
            #         merged_out["error"].mean(),
            #     )
            # )

            # if result_data["error"].max() < PASS_THRESH:
            #     print("# Device {} has passed regression.".format(dev))
            # else:
            #     print("# Device {} has failed regression. Needs more analysis.".format(dev))

            # print("\n\n")

            


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

