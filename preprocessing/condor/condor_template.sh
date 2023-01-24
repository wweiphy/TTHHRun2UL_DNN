#!/bin/bash

echo "Setting Up Environment"
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "Attempting setenv command"
# export SCRAM_ARCH={{SCRAM_ARCH}}
# export CMSSW_VERSION={{CMSSW_VERSION}}
# cmsrel $CMSSW_VERSION

xrdcp -s root://cmseos.fnal.gov//store/user/wwei/Eval/CMSSW_12_1_1.tgz .

tar -xf CMSSW_12_1_1.tgz
rm CMSSW_12_1_1.tgz
cd CMSSW_12_1_1/src/
scramv1 b ProjectRename 

# cd $CMSSW_VERSION/src/
# cmsenv
eval `scramv1 runtime -sh`

echo "set up"

{{SETUP}}

# echo "Transfer input file(s) from EOS"

# xrdcp -f {{INFILE}} .
# {{INFILE}}

echo "Starting Executable"

{{RUNCOMMAND}}


# OUTDIR = {{OUTDIR}}

# for FILE in *{{OUTFILE_FORMAT}}
# do
#   echo "transfer ${FILE} to ${OUTDIR}"
#  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1
#   XRDEXIT=$?
#   if [[ $XRDEXIT -ne 0 ]]; then
#     rm *.{{OUTFILE_FORMAT}}
#     echo "exit code $XRDEXIT, failure in xrdcp"
#     exit $XRDEXIT
#   fi
#   rm ${FILE}

echo "Transfer output file(s) to EOS"

# DB_AWS_ZONE=('us-east-2a' 'us-west-1a' 'eu-central-1a')


{{TRANSFEROUTFILE}}
# xrdcp -f {{EVEN_OUTFILE}} {{OUTDIR}}/{{EVEN_OUTFILE}} 
# xrdcp -f {{ODD_OUTFILE}} {{OUTDIR}}/{{ODD_OUTFILE}} 


# root://cms-xrd-global.cern.ch/
# root://cmseos.fnal.gov/

# cd $CMSSW_VERSION/src
cd ..

{{DELETE}}
# rm -rf genproductions

echo "clear space"



