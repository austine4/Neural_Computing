# Neural_Computing

This repository is meant include all files and code necessary for building microelectrode arrays (MEAs), recording neural signals, and analyzing the data.


Design of MEA includes:

  CAD files for photolithography (.gds) (KLayout)
  
  CAD files for design of topological layer (.stl) (?)


Design of recording device includes:

  CAD files for PCB design (.PcbDoc) (Altium)
  
  Neural Recording Chips (Cadence) *not included yet but may be in future*
  
    For now, we use Intran chips
    
  Aquisition FPGA Board
  
    For now, we use Open Ethys


Software for spike sorting includes:

  Conversion from open_ethys file to readable file in MATLAB
  
  MATLAB code for analyzing spikes can be found here: https://github.com/csn-le/wave_clus
