import os
import optparse


# python cardmakingscript.py -n old -f 230220_evaluation_old
# python cardmakingscript.py -n new -f 230220_evaluation_new_4j5b


usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--new", dest="new", default="new",
        help="making datacard for new categorizations, total 9", metavar="new")

parser.add_option("-f", "--folder", dest="folder", default="221204_test_evaluation_new",
                  help="folder name", metavar="folder")

(options, args) = parser.parse_args()

process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttb','ttbb','tt2b','ttbbb','tt4b']


if options.new == "new":

    for node in process_new:

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.folder)
        # outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new_{}/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(str(node), options.folder)
        # signaltag = "ttHH"

        # runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag=ttHH --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_new_sys_test.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        #     categoryname, options.folder, node)

        runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/new/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag=ttHH --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_new_sys_test.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
            categoryname, rootfile, node)


        os.system(runcommand)

        print("finish making datacard for process {}".format(node))

elif options.new == "old":

    for node in process_old:

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.folder)
        # outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old_{}/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(str(node),options.folder)
        signaltag = "ttHH"

        runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_old_sys_test.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile, node, signaltag)


        os.system(runcommand)

        print("finish making datacard for process {}".format(node))


