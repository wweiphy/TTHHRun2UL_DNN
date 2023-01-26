#!/bin/bash 

# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230119_evaluation_old/plots/output_limit.root'
export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230119_evaluation_new/plots/output_limit.root'
export SEPARATOR='__'
export NOMHISTKEY='$CHANNEL__$PROCESS'
export ORIGNAME='ttlf'
export SYSTHISTKEY='$CHANNEL__$PROCESS__$SYSTEMATIC'


python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSyst.py