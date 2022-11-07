import os
import optparse

usage = "usage=%prog [options] \n"
usage += "USE: python plottingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--new", dest="new", default=True,
        help="making datacard for new categorizations, total 9", metavar="new")

(options, args) = parser.parse_args()

process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttb', 'ttbb', 'tt2b', 'ttbbb', 'tt4b']



if options.new:

    for node in process_new:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/221107_test_evaluation_new/plots/output_limit.root"
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq 4 jets, \geq 3 b-tags" --rootfile=rootfile  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new"'.format("new", node, rootfile)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

else:

    for node in process_old:

        rootfile = ""
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq 4 jets, \geq 3 b-tags" --rootfile=rootfile  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old"'.format("old", node, rootfile)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))


