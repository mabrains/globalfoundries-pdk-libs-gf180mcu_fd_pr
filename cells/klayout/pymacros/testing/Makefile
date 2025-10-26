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


#=========================================================================
# ---------------------------------- DRC ---------------------------------
#=========================================================================

SHELL        := /bin/bash
Testing_DIR  ?= $(shell pwd)
run_folder   := $(shell date +'pcells_run_%Y_%m_%d_%H_%M')


.DEFAULT_GOAL := all

all : test-pcell

test-pcell: test-diode  test-MIM  test-cap_mos  test-FET  test-RES

#=================================
# ---------- test-BJT --------
#=================================

.ONESHELL:
test-bjt:
	@cd $(Testing_DIR)
	@mkdir -p $(run_folder)/bjt
	@echo "===== test BJT pcells ====="
	@pytest --device=bjt pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/bjt

#=================================
# -------- test-diode -------
#=================================

.ONESHELL:
test-diode:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/diode
	@echo "===== test diode pcells ====="
	@pytest --device=diodes pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/diode

#=================================
# --------- test-MIM ---------
#=================================

.ONESHELL:
test-MIM:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/cap_mim
	@echo "===== test MIM pcells ====="
	@pytest --device=mim_caps pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/cap_mim

#=================================
# --------- test-MOS ---------
#=================================

.ONESHELL:
test-FET : test-nfet_03v3 test-nfet_05v0 test-nfet_06v0  test-pfet_03v3 test-pfet_05v0 test-pfet_06v0 

test-nfet_03v3:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/nfet_03v3
	@echo "===== test nfet_03v3 pcells ====="
	@pytest --device=nfet_03v3 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/nfet_03v3

test-nfet_05v0:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/nfet_05v0
	@echo "===== test nfet_05v0 pcells ====="
	@pytest --device=nfet_05v0 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/nfet_05v0

test-nfet_06v0:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/nfet_06v0
	@echo "===== test nfet_06v0 pcells ====="
	@pytest --device=nfet_06v0 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/nfet_06v0
	
test-pfet_03v3:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/pfet_03v3
	@echo "===== test pfet_03v3 pcells ====="
	@pytest --device=pfet_03v3 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/pfet_03v3

test-pfet_05v0:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/pfet_05v0
	@echo "===== test pfet_05v0 pcells ====="
	@pytest --device=pfet_05v0 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/pfet_05v0

test-pfet_06v0:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/pfet_06v0
	@echo "===== test pfet_06v0 pcells ====="
	@pytest --device=pfet_06v0 pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/pfet_06v0


#=================================
# ------- test-cap_mos --------
#=================================

.ONESHELL:
test-cap_mos:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/cap_mos
	@echo "===== test cap_mos pcells ====="
	@pytest --device=mos_caps pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/cap_mos

#=================================
# --------- test-RES ---------
#=================================

.ONESHELL:
test-RES:
	@cd $(Testing_DIR)
	@mkdir -p  $(run_folder)/res
	@echo "===== test res pcells ====="
	@pytest --device=res pcell_reg_Pytest.py
	@rm -rf $(Testing_DIR)/../../../../globalfoundries-pdk-libs-gf180mcu_fd_pv/
	@mv -f drc_run_* lvs_run_* testcases $(run_folder)/res

#==========================
# --------- HELP ----------
#==========================

# Help Target
help:
	@echo "\n ==== The following are some of the valid targets for this Makefile ====\n"
	@echo "... all                  (Default if no target is provided )"
	@echo "... tes-pcell            (To test DRC/LVS for all pcells   )"
	@echo "... test-bjt             (To test DRC/LVS for bjt pcells   )"
	@echo "... test-diode           (To run DRC/LVS for diode pcells  )"
	@echo "... test-MIM             (To run DRC/LVS for MIM pcells    )"
	@echo "... test-FET             (To run DRC/LVS for FET pcells    )"
	@echo "... test-cap_mos         (To run DRC/LVS for cap_mos pcells)"
	@echo "... test-RES             (To run DRC/LVS for RES pcells    )"
