#!/bin/bash

echo "Setting Up Environment"
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.sh

xrdcp -r root://cmseos.fnal.gov//store/user/wwei/Eval/CMSSW_11_1_2.tgz .

tar -xf CMSSW_11_1_2.tgz
rm CMSSW_11_1_2.tgz
#tar -xf CMSSW_11_1_0_pre4.tar.gz
echo "Attempting setenv command"
export SCRAM_ARCH=slc7_amd64_gcc820
cd CMSSW_11_1_0_pre4/src/
scramv1 b ProjectRename
eval `scramv1 runtime -sh`
echo $CMSSW_BASE

# cd ${_CONDOR_SCRATCH_DIR}/CMSSW_11_1_0_pre4/src/DRACO-MLfoy/
pwd

echo "transferring input file from EOS"

cd workdir
env -i X509_USER_PROXY=${X509_USER_PROXY} xrdcp -r root://cmseos.fnal.gov//store/user/wwei/Eval/Eval_0119_UL_nominal/ .

echo "done with transferring"

export KERAS_BACKEND=tensorflow

cd ..
cd runscript
echo "Starting Executable"

python eval_template_new.py -o 230119_evaluation_new_2 -i 221130_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0119_UL_nominal

echo "transferring output files into EOS"

cd ..
cd workdir

env -i X509_USER_PROXY=${X509_USER_PROXY} xrdcp -r 230119_evaluation_new_2/ root://cmseos.fnal.gov//store/user/wwei/Eval/.

echo "files have been transfered to EOS"

rm -r 230119_evaluation_new_2/
rm -r Eval_0119_UL_nominal/

echo "deleted the input and output files"
