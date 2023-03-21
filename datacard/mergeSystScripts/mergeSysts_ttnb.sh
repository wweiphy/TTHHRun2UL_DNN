#!/bin/bash 
# export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
# source $VO_CMS_SW_DIR/cmsset_default.sh
# export SCRAM_ARCH=slc7_amd64_gcc820
# cd /uscms_data/d3/wwei/SM_TTHH/CMSSW_11_1_0_pre4/src
# eval `scram runtime -sh`
# cd - 
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230220_evaluation_new/plots/ttnb_discriminator.root'
export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230220_evaluation_new_5j4b/plots/ttnb_discriminator.root'
# export INFILE='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230220_evaluation_new_6j5b/plots/ttnb_discriminator.root'
export SEPARATOR='__'
export NOMHISTKEY='$CHANNEL__$PROCESS'
export ORIGNAME='ttnb'
export SYSTHISTKEY='$CHANNEL__$PROCESS__$SYSTEMATIC'


python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/mergeSyst.py