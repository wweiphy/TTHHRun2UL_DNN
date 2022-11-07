import os
import optparse

usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

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

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/221107_test_evaluation_new_2/plots/output_limit.root"
        outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
        signaltag = node

        runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile={} --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/{}.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile,  outputfile, signaltag, "datacard_new")


        os.system(runcommand)

        print("finish making datacard for process {}".format(node))

else:

    for node in process_old:

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
        rootfile = ""
        outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
        signaltag = node

        runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile={} --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/{}.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile,  outputfile, signaltag, "datacard_old")


        os.system(runcommand)

        print("finish making datacard for process {}".format(node))


