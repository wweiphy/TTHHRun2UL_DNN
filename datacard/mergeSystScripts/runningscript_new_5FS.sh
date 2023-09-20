#!/bin/bash 

# post
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230523_evaluation_new_2/plots/output_limit.root'
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230523_evaluation_new_6j4b_2/plots/output_limit.root'
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230523_evaluation_new_5j4b_2/plots/output_limit.root'

#  pre
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230515_evaluation_new_2/plots/output_limit.root'
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230515_evaluation_new_6j4b_2/plots/output_limit.root'
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230515_evaluation_new_5j4b_2/plots/output_limit.root'

/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSystScripts/mergeSysts_ttH.sh
/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSystScripts/mergeSysts_ttcc.sh
/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSystScripts/mergeSysts_ttlf.sh
# Todo - comment out if applying on 5FS
# /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSystScripts/mergeSysts_ttmb.sh
