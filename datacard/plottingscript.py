import os
import optparse

usage = "usage=%prog [options] \n"
usage += "USE: python plottingscript.py -n new_plotting "

# evaluation - discriminators
# 2017
# python plottingscript.py -n new -f 230119_evaluation_new_5j4b -c new_230119_5j4b_sys 
# python plottingscript.py -n new -f 230119_evaluation_new_6j4b -c new_230119_6j4b_sys

# 2016pre
# python plottingscript.py -n new -f 230515_evaluation_new_5j4b -c new_230515_5j4b_sys 
# python plottingscript.py -n new -f 230515_evaluation_new_6j4b -c new_230515_6j4b_sys
# python plottingscript.py -n new -f 230515_evaluation_new -c new_230515_sys
# python plottingscript.py -n new -f 230515_evaluation_old -c old_230515_sys

# 2016
# python plottingscript.py -n new -f 230523_evaluation_new_5j4b -c new_230523_5j4b_sys
# python plottingscript.py -n new -f 230523_evaluation_new_6j4b -c new_230523_6j4b_sys
# python plottingscript.py -n new -f 230523_evaluation_new -c new_230523_sys
# python plottingscript.py -n new -f 230523_evaluation_old -c old_230523_sys


# kinematics
# python plottingscript.py -n new_plotting -f 230119_evaluation_new_5j4b -c new_230119_5j4b_sys

parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--new", dest="new", default="new",
        help="making datacard for new categorizations, total 9", metavar="new")

parser.add_option("-f", "--filefolder", dest="filefolder", default="230220_evaluation_new",
                  help="file folder name", metavar="filefolder")

parser.add_option("-c", "--cardfolder", dest="cardfolder", default="new_230119_new_sys",
                  help="file folder name", metavar="filefolder")
parser.add_option("-j", "--njets", dest="njets", default=4,
                  help="number of jets selection", metavar="bjets")
parser.add_option("-b", "--nbjets", dest="nbjets", default=3,
                  help="number of bjets selection", metavar="nbjets")

(options, args) = parser.parse_args()

process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttb', 'ttbb', 'tt2b', 'ttbbb', 'tt4b']
# variables = ['N_BTagsM', 'Electron_E[0]', 'Jet_CSV[5]']
variables = [
    'N_BTagsL',
    'N_BTagsM',
    'N_BTagsT',
    'N_Jets',
    'N_LooseElectrons',
    'N_LooseJets',
    'N_LooseMuons',
    'N_PrimaryVertices',
    'N_TightElectrons',
    'N_TightMuons',
    'CSV[0]',
    'CSV[1]',
    'CSV[2]',
    'CSV[3]',
    'CSV[4]',
    'CSV[5]',
    'CSV[6]',
    'CSV[7]',
    'Electron_E[0]',
    'Electron_M[0]',
    'Electron_Pt[0]',
    'Jet_CSV[0]',
    'Jet_CSV[1]',
    'Jet_CSV[2]',
    'Jet_CSV[3]',
    'Jet_CSV[4]',
    'Jet_CSV[5]',
    'Jet_CSV[6]',
    'Jet_CSV[7]',
    'Jet_E[0]',
    'Jet_E[1]',
    'Jet_E[2]',
    'Jet_E[3]',
    'Jet_E[4]',
    'Jet_E[5]',
    'Jet_E[6]',
    'Jet_E[7]',
    'Jet_M[0]',
    'Jet_M[1]',
    'Jet_M[2]',
    'Jet_M[3]',
    'Jet_M[4]',
    'Jet_M[5]',
    'Jet_M[6]',
    'Jet_M[7]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    'Jet_Pt[2]',
    'Jet_Pt[3]',
    'Jet_Pt[4]',
    'Jet_Pt[5]',
    'Jet_Pt[6]',
    'Jet_Pt[7]',
    'LooseElectron_E[0]',
    'LooseElectron_M[0]',
    'LooseElectron_Pt[0]',
    'LooseJet_CSV[0]',
    'LooseJet_CSV[1]',
    'LooseJet_CSV[2]',
    'LooseJet_CSV[3]',
    'LooseJet_E[0]',
    'LooseJet_E[1]',
    'LooseJet_E[2]',
    'LooseJet_E[3]',
    'LooseJet_M[0]',
    'LooseJet_M[1]',
    'LooseJet_M[2]',
    'LooseJet_M[3]',
    'LooseJet_Pt[0]',
    'LooseJet_Pt[1]',
    'LooseJet_Pt[2]',
    'LooseJet_Pt[3]',
    'LooseLepton_E[0]',
    'LooseLepton_M[0]',
    'LooseLepton_Pt[0]',
    'LooseMuon_E[0]',
    'LooseMuon_M[0]',
    'LooseMuon_Pt[0]',
    'Muon_E[0]',
    'Muon_M[0]',
    'Muon_Pt[0]',
    'TaggedJet_CSV[0]',
    'TaggedJet_CSV[1]',
    'TaggedJet_E[0]',
    'TaggedJet_E[1]',
    'TaggedJet_M[0]',
    'TaggedJet_M[1]',
    'TaggedJet_Pt[0]',
    'TaggedJet_Pt[1]',
    'TightLepton_E[0]',
    'TightLepton_M[0]',
    'TightLepton_Pt[0]',
]


if options.new == "new":

    evaluation = True

    for node in process_new:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.filefolder)
        workdir = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/{}".format(options.cardfolder)
        # selectionlabel = "\geq {} jets, \geq {} b-tags".format(options.njets, options.nbjets)
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq {} jets, \geq {} b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir={} --evaluation={}'.format(
            "new", node, options.njets,options.nbjets,rootfile, workdir, evaluation)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

elif options.new == "old":

    evaluation = True

    for node in process_old:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.filefolder)
        workdir = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/{}".format(
            options.cardfolder)
        # selectionlabel = "\geq {} jets, \geq {} b-tags".format(
            # options.njets, options.nbjets)
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq {} jets, \geq {} b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir={} --evaluation={}'.format(
            "old", node, options.njets,options.nbjets,rootfile, workdir, evaluation)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

else:

    evaluation = False
    for var in variables:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/output_limit.root"
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={}'.format(
            "new", var, rootfile,evaluation)

        os.system(runcommand)

        print("finish plotting variable: "+var)


