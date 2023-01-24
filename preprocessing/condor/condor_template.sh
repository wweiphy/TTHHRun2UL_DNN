#!/bin/bash

echo "Setting Up Environment"
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "Attempting setenv command"


xrdcp -s root://cmseos.fnal.gov//store/user/wwei/Eval/CMSSW_12_1_1.tgz .

tar -xf CMSSW_12_1_1.tgz
rm CMSSW_12_1_1.tgz
cd CMSSW_12_1_1/src/
scramv1 b ProjectRename 


eval `scramv1 runtime -sh`

echo "set up"

{{SETUP}}



echo "Starting Executable"

{{RUNCOMMAND}}




echo "Transfer output file(s) to EOS"



{{TRANSFEROUTFILE}}


cd ..

{{DELETE}}

echo "clear space"



