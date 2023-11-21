import os
import optparse
import sys

# 17&18
# python cardmakingscript.py -n new -f TwoYear-4FS5j4b
# python cardmakingscript.py -n new -f TwoYear-4FS6j4b
# python cardmakingscript.py -n new -f TwoYear-5j4b
# python cardmakingscript.py -n new -f TwoYear-5j4b

# 2017
# python cardmakingscript.py -n old -f 230119_evaluation_old_2
# python cardmakingscript.py -n new -f 230119_evaluation_new_2
# python cardmakingscript.py -n new -f 230119_evaluation_new_5j4b
# python cardmakingscript.py -n new -f 230119_evaluation_new_5j4b_4FS
# python cardmakingscript.py -n new -f 230119_evaluation_new_5j4b_5FS
# python cardmakingscript.py -n new -f 230119_evaluation_new_6j4b
# python cardmakingscript.py -n new -f 230119_evaluation_new_6j4b_4FS
# python cardmakingscript.py -n new -f 230119_evaluation_new_6j4b_5FS

# 2018
# python cardmakingscript.py -n old -f 230220_evaluation_old
# python cardmakingscript.py -n new -f 230220_evaluation_new
# python cardmakingscript.py -n new -f 230220_evaluation_new_5j4b
# python cardmakingscript.py -n new -f 230220_evaluation_new_5j4b_5FS
# python cardmakingscript.py -n new -f 230220_evaluation_new_5j4b_4FS
# python cardmakingscript.py -n new -f 230220_evaluation_new_6j4b
# python cardmakingscript.py -n new -f 230220_evaluation_new_6j4b_5FS
# python cardmakingscript.py -n new -f 230220_evaluation_new_6j4b_4FS
# python cardmakingscript.py -n new -f 231011_evaluation_new_6j4b_4FS

# python cardmakingscript.py -n old -f 230515_evaluation_old_2
# python cardmakingscript.py -n new -f 230515_evaluation_new_2
# python cardmakingscript.py -n new -f 230515_evaluation_new_5j4b_2
# python cardmakingscript.py -n new -f 230515_evaluation_new_6j4b_2

# python cardmakingscript.py -n old -f 230523_evaluation_old_2
# python cardmakingscript.py -n new -f 230523_evaluation_new_2
# python cardmakingscript.py -n new -f 230523_evaluation_new_5j4b_2
# python cardmakingscript.py -n new -f 230523_evaluation_new_6j4b_2


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

filedir = os.path.dirname(os.path.realpath(__file__))
# datacarddir = os.path.dirname(filedir)
basedir = os.path.dirname(filedir)
sys.path.append(basedir)


if options.new == "new":

    if not os.path.exists("new"):
        os.mkdir("new")
    if not os.path.exists("new_nosys"):
        os.mkdir("new_nosys")

    if "4FS" in options.folder:
        systfile = "datacard_new_sys_reduce_4FS.csv"
    else:
        systfile = "datacard_new_sys_reduce.csv"

    for node in process_new:

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)

        if "TwoYear" in options.folder or "ThreeYear" in options.folder:
            rootfile = filedir + "/combineRun2/"+options.folder+"/output_limit.root"
            print(rootfile)

        else:
            rootfile = basedir + "/workdir/{}/plots/output_limit.root".format(options.folder)

        scriptfile = filedir+"/DatacardScript.py"
        outfile = filedir+"/new_nosys/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
        cardmaker = filedir+"/datacardMaker"
        csvfile_nosys = filedir+"/datacard_new.csv"
        
        runcommand1 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
            scriptfile, categoryname, rootfile, outfile, cardmaker, csvfile_nosys)

        csvfile_sys = filedir+"/"+systfile
        runcommand2 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
            scriptfile, categoryname, rootfile, outfile, cardmaker, csvfile_sys)


        os.system(runcommand1)
        os.system(runcommand2)

        print("finish making datacard for process {}".format(node))

elif options.new == "old":

    for node in process_old:

        categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.folder)
        # outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old_{}/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(str(node),options.folder)
        signaltag = "ttHH"

        runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_old.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile, node, signaltag)
        # runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_old_sys_test.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile, node, signaltag)


        os.system(runcommand)

        print("finish making datacard for process {}".format(node))


