clc
clear
cdl = nrCDLChannel;
   cdl.DelayProfile = 'CDL-A';
   i = info(cdl);
   disp(i.AveragePathGains)  % powers in dB
   disp(i.AnglesAoD)         % AoD in degrees