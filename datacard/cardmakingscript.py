import os
import optparse
import sys

# 17&18
# python cardmakingscript.py -n new -f TwoYear-4FS5j4b
# python cardmakingscript.py -n new -f TwoYear-4FS6j4b
# python cardmakingscript.py -n new -f TwoYear5j4b
# python cardmakingscript.py -n new -f TwoYear6j4b
# python cardmakingscript.py -n new -f 20166j4b
# python cardmakingscript.py -n new -f 2016-4FS6j4b

# 16pre&16post

# python cardmakingscript.py -y 2016 -f 20166j4b_oldtthh

# Run2

# python cardmakingscript.py -y run2 -f ThreeYear6j4b_oldtthh



# 2017
# python cardmakingscript.py -n old -f 230119_evaluation_old_2
# python cardmakingscript.py -n new -f 230119_evaluation_new_2
# python cardmakingscript.py -n new -f 230119_evaluation_new_5j4b
# python cardmakingscript.py -y 2017 -f 230119_evaluation_new_6j4b_oldtthh_3
# python cardmakingscript.py -y 2017 -f 230119_evaluation_new_5j4b_oldtthh_3


# 2018
# python cardmakingscript.py -n old -f 230220_evaluation_old
# python cardmakingscript.py -n new -f 230220_evaluation_new
# python cardmakingscript.py -y 2018 -f 230220_evaluation_new_5j4b
# python cardmakingscript.py -n new -f 230220_evaluation_new_5j4b_5FS
# python cardmakingscript.py -n new -f 230220_evaluation_new_5j4b_4FS
# python cardmakingscript.py -y 2018 -f 230220_evaluation_new_6j4b_oldtthh
# python cardmakingscript.py -n new -f 230220_evaluation_new_6j4b_5FS
# python cardmakingscript.py -n new -f 230220_evaluation_new_6j4b_4FS
# python cardmakingscript.py -n new -f 231011_evaluation_new_6j4b_4FS

# 2016

# python cardmakingscript.py -n old -f 230515_evaluation_old_2
# python cardmakingscript.py -n new -f 230515_evaluation_new_2
# python cardmakingscript.py -n new -f 230515_evaluation_new_5j4b_2
# python cardmakingscript.py -y 2016 -f 230515_evaluation_new_6j4b_oldtthh_2
# python cardmakingscript.py -y 2016 -f 230515_evaluation_new_5j4b

# python cardmakingscript.py -n old -f 230523_evaluation_old_2
# python cardmakingscript.py -n new -f 230523_evaluation_new_2
# python cardmakingscript.py -y 2016 -f 230523_evaluation_new_5j4b
# python cardmakingscript.py -y 2016 -f 230523_evaluation_new_6j4b_oldtthh_2


usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-y", "--year", dest="year", default="2017",
        help="year", metavar="year")

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


# if options.new == "new":

if "4FS" in options.folder:
    systfile = "datacard_new_sys_reduce_4FS.csv"
else:
    if options.year == "2016" and "2016" in options.folder:
        fullsystfile = "datacard_new_sys_2016.csv" 
        fullsystfilerate = "datacard_new_sys_2016_rate.csv" 
        systfile = "datacard_new_sys_reduce_2016.csv"
        systfilerate = "datacard_new_sys_reduce_2016_rate.csv"
    # elif options.year == "2016postVFP":
        # systfile = "datacard_new_sys_reduce_2016postVFP.csv"
    elif options.year == "2016" and "230515" in options.folder:
        systfile = "datacard_new_sys_reduce_2016-TwoEra.csv"
        fullsystfile = "datacard_new_sys_2016-TwoEra.csv" 
        
        systfilerate = "datacard_new_sys_reduce_2016-TwoEra_rate.csv"
        fullsystfilerate = "datacard_new_sys_2016-TwoEra_rate.csv" 

    elif options.year == "2016" and "230523" in options.folder:
        systfile = "datacard_new_sys_reduce_2016-TwoEra.csv"
        fullsystfile = "datacard_new_sys_2016-TwoEra.csv" 
        
        if "5j4b" in options.folder: 
            systfilerate = "datacard_new_sys_reduce_2016-TwoEra_rate_2016post_5j4b.csv"
            fullsystfilerate = "datacard_new_sys_2016-TwoEra_rate_2016post_5j4b.csv" 
        if "6j4b" in options.folder: 
            systfilerate = "datacard_new_sys_reduce_2016-TwoEra_rate_2016post_6j4b.csv"
            fullsystfilerate = "datacard_new_sys_2016-TwoEra_rate_2016post_6j4b.csv"  
        
    elif options.year == "2017":
        systfile = "datacard_new_sys_reduce_2017.csv"
        systfilerate = "datacard_new_sys_reduce_2017_rate.csv"
        fullsystfile = "datacard_new_sys_2017.csv"
        fullsystfilerate = "datacard_new_sys_2017_rate.csv"
    elif options.year == "2018":
        systfile = "datacard_new_sys_reduce_2018.csv"
        systfilerate = "datacard_new_sys_reduce_2018_rate.csv"
        fullsystfile = "datacard_new_sys_2018.csv"
        fullsystfilerate = "datacard_new_sys_2018_rate.csv"
    elif options.year == "run2":
        systfile = "datacard_new_sys_reduce_run2.csv"
        fullsystfile = "datacard_new_sys_run2.csv"
        systfilerate = "datacard_new_sys_reduce_run2_rate.csv"
        fullsystfilerate = "datacard_new_sys_run2_rate.csv"

for node in process_new:

    categoryname = "ljets_ge4j_ge3t_{}_node".format(node)

    if "TwoYear" in options.folder or "ThreeYear" in options.folder or "2016" in options.folder:
        rootfile = filedir + "/combineRun2/"+options.folder+"/output_limit.root"
        print(rootfile)
    else:
        rootfile = basedir + "/workdir/{}/plots/output_limit.root".format(options.folder)

    scriptfile = filedir+"/DatacardScript.py"
    outfile1 = filedir+"/new_nosys/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
    cardmaker = filedir+"/datacardMaker"
    csvfile_nosys = filedir+"/datacard_new.csv"

    runcommand1 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        scriptfile, categoryname, rootfile, outfile1, cardmaker, csvfile_nosys)

    csvfile_sys = filedir+"/"+systfile
    outfile2 = filedir + "/new/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
    runcommand2 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        scriptfile, categoryname, rootfile, outfile2, cardmaker, csvfile_sys)
    

    csvfile_sys_rate = filedir+"/"+systfilerate
    outfile3 = filedir + "/new_rate/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
    runcommand3 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        scriptfile, categoryname, rootfile, outfile3, cardmaker, csvfile_sys_rate)
    
    csvfile_sys_full = filedir+"/"+fullsystfile
    outfile4 = filedir + "/new_full/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
    runcommand4 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        scriptfile, categoryname, rootfile, outfile4, cardmaker, csvfile_sys_full)
    
    csvfile_sys_fullrate = filedir+"/"+fullsystfilerate
    outfile5 = filedir + "/new_fullrate/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(node)
    runcommand5 = "python {} --categoryname={} --rootfile={} --outputfile={} --directory={} --signaltag=ttHH --csvfile={} --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(
        scriptfile, categoryname, rootfile, outfile5, cardmaker, csvfile_sys_fullrate)


    if not os.path.exists("new_nosys"):
        os.mkdir("new_nosys")
    os.system(runcommand1)

    # if not os.path.exists("new"):
    #         os.mkdir("new")
    # if not os.path.exists("new_rate"):
    #     os.mkdir("new_rate")
    # if not os.path.exists("new_fullrate"):
    #     os.mkdir("new_fullrate")
    # os.system(runcommand2)
    # os.system(runcommand3)
    if not os.path.exists("new_full"):
        os.mkdir("new_full")
    os.system(runcommand4)
    # os.system(runcommand5)

    
    



    print("finish making datacard for process {}".format(node))

# elif options.new == "old":

#     for node in process_old:

#         categoryname = "ljets_ge4j_ge3t_{}_node".format(node)
#         rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.folder)
#         # outputfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old_{}/ljets_ge4j_ge3t_{}_node_hdecay.txt".format(str(node),options.folder)
#         signaltag = "ttHH"

#         runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_old.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile, node, signaltag)
#         # runcommand = "python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/DatacardScript.py --categoryname={} --rootfile={} --outputfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/old/ljets_ge4j_ge3t_{}_node_hdecay.txt --directory=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacardMaker --signaltag={} --csvfile=/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/datacard_old_sys_test.csv --nominal_key='$CHANNEL__$PROCESS' --syst_key='$CHANNEL__$PROCESS__$SYSTEMATIC'".format(categoryname, rootfile, node, signaltag)


#         os.system(runcommand)

#         print("finish making datacard for process {}".format(node))


