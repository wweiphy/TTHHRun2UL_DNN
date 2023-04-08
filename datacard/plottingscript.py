import os
import optparse

usage = "usage=%prog [options] \n"
usage += "USE: python plottingscript.py -n new "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--new", dest="new", default="new",
        help="making datacard for new categorizations, total 9", metavar="new")

(options, args) = parser.parse_args()

process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttb', 'ttbb', 'tt2b', 'ttbbb', 'tt4b']
variables = ['Evt_CSV_avg', 'Evt_CSV_avg_tagged', 'Evt_CSV_dev']


if options.new == "new":

    for node in process_new:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230220_evaluation_new/plots/output_limit.root"
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq 4 jets, \geq 3 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new"'.format("new", node, rootfile)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

elif options.new == "old":

    for node in process_old:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/230220_evaluation_old/plots/output_limit.root"
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq 4 jets, \geq 3 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old"'.format("old", node, rootfile)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

else:

    for var in variables:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/output_limit.root"
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="2 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new"'.format(
            "new", var, rootfile)

        os.system(runcommand)

        print("finish plotting variable: "+var)


