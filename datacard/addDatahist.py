import sys
import os
import subprocess
import datetime
import stat
import ROOT
import glob
import imp 
import types


nodeNames = ['ttHH','ttH','ttZbb','ttZH','ttZZ','ttlf','ttcc','ttb','tt2b','ttbb','ttbbb','tt4bb']
# nodeNames = ['ttHH']
classNames = ['ttHH','ttH','ttZbb','ttZH','ttZZ','ttlf','ttcc','ttb','tt2b','ttbb','ttbbb','tt4bb']

# nodeNames = ['ttHH','ttH','ttZbb','ttZH','ttZZ','ttlf','ttcc','ttmb','ttnb']
# classNames = ['ttHH','ttH','ttZbb','ttZH','ttZZ','ttlf','ttcc','ttmb','ttnb']


# hadd -f output_limit.root 1 2 3

filepath = '/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/221003_test_evaluation_old/plots/output_limit.root'


print(classNames)
rootFile = ROOT.TFile(filepath, "UPDATE")
"""
get histograms and add them to data obs
"""

for node in nodeNames:
    print("doing node {}".format(node))
#        histNameTemplate = self.nominalHistoKey.replace("$CHANNEL", label)
    # histName = histNameTemplate.replace("$PROCESS", str(sampleNicks[0]))

    newHist = None
    for label in classNames:
#            sampleName = histNameTemplate.replace("$PROCESS",str(nick))
        sampleName = "ljets_ge4j_ge3t_"+node+"_node__"+label
        print("doing process: "+label)
        bufferHist = rootFile.Get(sampleName)
        print("bufferHist: "+str(bufferHist))
        if newHist is None:
            dataname = "ljets_ge4j_ge3t_"+node+"_node__data_obs"
            newHist = bufferHist.Clone(dataname)
        else:
            newHist.Add(bufferHist)
    newHist.Write()

rootFile.Close()



# hadd -f output_limit.root ttHH_discriminator.root ttH_discriminator.root ttZZ_discriminator.root ttZH_discriminator.root ttZbb_discriminator.root ttlf_discriminator.root ttcc_discriminator.root ttmb_discriminator.root ttnb_discriminator.root 

# hadd -f output_limit.root ttHH_discriminator.root ttH_discriminator.root ttZZ_discriminator.root ttZH_discriminator.root ttZbb_discriminator.root ttlf_discriminator.root ttcc_discriminator.root ttb_discriminator.root tt2b_discriminator.root ttbb_discriminator.root ttbbb_discriminator.root tt4b_discriminator.root 



# python /uscms_data/d3/wwei/SM_TTHH/CMSSW_11_1_0_pre4/src/pyroot-plotscripts/util/DatacardScript.py --categoryname=ljets_ge4j_ge3t_ttmb_node --rootfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/220920_JABDT_2e5_ge4j_ge3t_final_evaluation_553/plots/output_limit.root --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/220920_JABDT_2e5_ge4j_ge3t_final_evaluation_553/datacards/ljets_ge4j_ge3t_ttmb_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/CMSSW_11_1_0_pre4/src/datacardMaker --signaltag=ttHH --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/datacard.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'

