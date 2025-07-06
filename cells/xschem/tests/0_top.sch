v {xschem version=3.4.5 file_version=1.2

* Copyright 2022 GlobalFoundries PDK Authors
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

}
G {}
K {}
V {}
S {}
E {}
L 7 360 -990 360 -60 {}
L 7 700 -990 700 -60 {}
L 7 1040 -990 1040 -60 {}
L 7 1380 -990 1380 -60 {}
L 7 1720 -990 1720 -60 {}
L 7 20 -990 20 -60 {}
T {MOSFETS} 40 -970 0 0 0.8 0.8 {}
T {CAPACITORS} 380 -970 0 0 0.8 0.8 {}
T {BJTs} 730 -970 0 0 0.8 0.8 {}
T {DIODES} 1060 -970 0 0 0.8 0.8 {}
T {RESISTORS} 1410 -970 0 0 0.8 0.8 {}
T {Deprecated symbol:
cap_mim_2p0fF.sym
equivalent to
cap_mim_2f0fF.sym} 430 -890 0 0 0.4 0.4 {}
C {test_nfet_03v3.sym} 190 -460 0 0 {name=x1}
C {test_nfet_03v3_dss.sym} 190 -410 0 0 {name=x2}
C {test_pfet_03v3.sym} 190 -360 0 0 {name=x3}
C {test_pfet_03v3_dss.sym} 190 -310 0 0 {name=x4}
C {devices/title.sym} 160 -30 0 0 {name=l5 author="GlobalFoundries PDK Authors"}
C {devices/launcher.sym} 90 -1100 0 0 {name=h1
descr="List of devices (Google docs)"
url="https://docs.google.com/spreadsheets/d/1xxxg_VzZWJ1NNysSMcOVzyUin7M2dvtb_E7X-5WQKJ0/edit#gid=1056368354"}
C {test_nfet_06v0.sym} 190 -260 0 0 {name=x5}
C {test_pfet_06v0.sym} 190 -160 0 0 {name=x6}
C {test_nfet_06v0_nvt.sym} 190 -110 0 0 {name=x7}
C {test_nplus_u.sym} 1550 -660 0 0 {name=x8}
C {devices/launcher.sym} 90 -1050 0 0 {name=h2
descr="Gitlab"
url="https://gitlab.com"}
C {test_cap_nmos_03v3.sym} 530 -260 0 0 {name=x9}
C {test_cap_pmos_03v3.sym} 530 -210 0 0 {name=x10}
C {test_pplus_u.sym} 1550 -610 0 0 {name=x11}
C {test_npn_10p00x10p00.sym} 870 -110 0 0 {name=x12}
C {test_pnp_10p00x10p00.sym} 870 -410 0 0 {name=x13}
C {test_diode_nd2ps_03v3.sym} 1210 -110 0 0 {name=x14}
C {test_nwell.sym} 1550 -110 0 0 {name=x15}
C {test_npolyf_u.sym} 1550 -210 0 0 {name=x16}
C {test_ppolyf_u.sym} 1550 -160 0 0 {name=x17}
C {test_cap_nmos_06v0.sym} 530 -110 0 0 {name=x18}
C {test_cap_pmos_06v0.sym} 530 -160 0 0 {name=x20}
C {test_diode_pw2dw.sym} 1210 -160 0 0 {name=x21}
C {test_rm1.sym} 1550 -710 0 0 {name=x22}
C {test_cap_mim_2f0fF.sym} 530 -510 0 0 {name=x23}
C {test_nfet_05v0.sym} 190 -210 0 0 {name=x19}
C {test_rm2.sym} 1550 -760 0 0 {name=x24}
C {test_rm3.sym} 1550 -810 0 0 {name=x25}
C {test_cap_nmos_03v3_b.sym} 530 -310 0 0 {name=x26}
C {test_cap_pmos_03v3_b.sym} 530 -360 0 0 {name=x27}
C {test_cap_pmos_06v0_b.sym} 530 -410 0 0 {name=x28}
C {test_cap_nmos_06v0_b.sym} 530 -460 0 0 {name=x29}
C {test_ppolyf_s.sym} 1550 -560 0 0 {name=x30}
C {test_npolyf_s.sym} 1550 -510 0 0 {name=x31}
C {test_ppolyf_u_1k_6p0.sym} 1550 -460 0 0 {name=x33}
C {test_ppolyf_u_2k_6p0.sym} 1550 -410 0 0 {name=x34}
C {test_ppolyf_u_1k.sym} 1550 -360 0 0 {name=x35}
C {test_ppolyf_u_2k.sym} 1550 -310 0 0 {name=x36}
C {test_ppolyf_u_3k.sym} 1550 -260 0 0 {name=x37}
C {test_cap_mim_1f5fF.sym} 530 -560 0 0 {name=x32}
C {test_cap_mim_1f0fF.sym} 530 -610 0 0 {name=x38}
C {test_cap_mim_analog.sym} 530 -660 0 0 {name=x39}
C {test_npn_05p00x05p00.sym} 870 -160 0 0 {name=x40}
C {test_npn_00p54x16p00.sym} 870 -210 0 0 {name=x41}
C {test_npn_00p54x08p00.sym} 870 -260 0 0 {name=x42}
C {test_npn_00p54x04p00.sym} 870 -310 0 0 {name=x43}
C {test_npn_00p54x02p00.sym} 870 -360 0 0 {name=x44}
C {test_pnp_05p00x05p00.sym} 870 -460 0 0 {name=x45}
C {test_pnp_05p00x00p42.sym} 870 -510 0 0 {name=x46}
C {test_pnp_10p00x00p42.sym} 870 -560 0 0 {name=x47}
C {test_diode_pd2nw_03v3.sym} 1210 -210 0 0 {name=x48}
C {test_diode_nd2ps_06v0.sym} 1210 -260 0 0 {name=x49}
C {test_diode_pd2nw_06v0.sym} 1210 -310 0 0 {name=x50}
C {test_diode_nw2ps_03v3.sym} 1210 -360 0 0 {name=x51}
C {test_diode_nw2ps_06v0.sym} 1210 -410 0 0 {name=x52}
C {test_diode_dw2ps.sym} 1210 -460 0 0 {name=x53}
C {test_sc_diode.sym} 1210 -510 0 0 {name=x54}
C {test_nfet_06v0_dss.sym} 190 -510 0 0 {name=x55}
C {test_pfet_06v0_dss.sym} 190 -560 0 0 {name=x56}
C {test_nfet_10v0_asym.sym} 190 -610 0 0 {name=x57}
C {test_pfet_10v0_asym.sym} 190 -660 0 0 {name=x58}
C {symbols/cap_mim_2p0fF.sym} 520 -740 0 0 {name=C1
W=1e-6
L=1e-6
model=cap_mim_2f0fF
spiceprefix=X
m=1}
