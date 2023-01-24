import os
import sys
import optparse
from jinja2 import Environment, FileSystemLoader
import glob


usage="usage=%prog [options] \n"
usage+="USE: python createCondors.py"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-e", "--dataEra", dest="dataEra", default=2017,
        help="data year of the JABDT training", metavar="dataEra")

# parser.add_option("-o", "--outPath", dest="outPath", default="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/JABDT/condor",
        # help="Path of Output Folder containing condor files, default: 'condor'", metavar="outPath")

# parser.add_option("-p", "--process", dest="process", default="ttHH",
#                   help="Process of the card and Name of the Output Folder", metavar="process")

parser.add_option("-c", "--cpus", dest="cpus", default=4,
                  help="CPU request", metavar="cpus")

parser.add_option("-m", "--memory", dest="memory", default=10000,
                  help="Memory request", metavar="memory")

parser.add_option("-n", "--new", dest="new", default="new",
                  help="new or old categorization", metavar="new")

(options, args) = parser.parse_args()


SETUP = "cd TTHHRun2UL_DNN/preprocessing"


DELETE = "rm -rf TTHHRun2UL_DNN"

MEMORY = options.memory
# DISK = options.disk
CPU = options.cpus



OUTDIR = "root://cmseos.fnal.gov//store/user/wwei/Eval/{}/".format(
   options.dataEra)
outPath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/preprocessing/condor/{}/".format(options.dataEra)



environment = Environment(loader=FileSystemLoader("."))
scriptTemplate = environment.get_template("condor_template.sh")
jdlTemplate = environment.get_template("condor_template.jdl")

eospath = "root://cmseos.fnal.gov/"


systs = [
  'JESup',
  'JESdown',
  'JERup',
  'JERdown',
  'JESFlavorQCDup',
  'JESRelativeBalup',
  'JESHFup',
  'JESBBEC1up', 
  'JESEC2up',
  'JESAbsoluteup',
  'JESBBEC1yearup',
  'JESRelativeSampleyearup',
  'JESEC2yearup',
  'JESHFyearup',
  'JESAbsoluteyearup',
  'JESFlavorQCDdown',
  'JESRelativeBaldown',
  'JESHFdown',
  'JESBBEC1down',
  'JESEC2down',
  'JESAbsolutedown',
  'JESBBEC1yeardown',
  'JESRelativeSampleyeardown',
  'JESEC2yeardown',
  'JESHFyeardown',
  'JESAbsoluteyeardown',
]


for i, syst in enumerate(systs):
    
    if not os.path.exists(outPath):
        os.system("mkdir -p " + outPath)

    if options.new == "new":

        RUNCOMMAND = "python3 template_UL_Eval_Syst.py --outputdirectory=Eval_0119_UL --variableselection=dnn_variables --maxentries=20000 --cores={}  --syst={}".format(options.cpus, syst)


        TRANSFEROUTFILE = "cd .. " + "\n" + "env -i X509_USER_PROXY=${X509_USER_PROXY} xrdcp -r /workdir/Eval_0119_UL_{}".format(syst) + " " + eospath + "/store/user/wwei/Eval/230119/."

    else:

        RUNCOMMAND = "python3 template_UL_Eval_Syst_old.py  --outputdirectory=Eval_0119_UL_old --variableselection=dnn_variables --maxentries=20000 --cores={}  --syst={}".format(options.cpus, syst)

        TRANSFEROUTFILE = "cd .. " + "\n" + "env -i X509_USER_PROXY=${X509_USER_PROXY} xrdcp -r /workdir/Eval_0119_UL_old_{}".format(syst) + " " + eospath + "/store/user/wwei/Eval/230119/."

    scriptFileName = outPath + "/" + options.new + "_{}.sh".format(syst)
    scriptcontent = scriptTemplate.render(
        SETUP = SETUP,
        RUNCOMMAND = RUNCOMMAND,
        # OUTDIR = OUTDI
        TRANSFEROUTFILE=TRANSFEROUTFILE,
        DELETE = DELETE
    )
    with open(scriptFileName, mode="w") as scriptFile:
        scriptFile.write(scriptcontent)
        print("... wrote {}".format(scriptFileName))

    # for jdl file

    EXECUTABLE = scriptFileName
    FILES = scriptFileName
    OUTFILES = scriptFileName.split(".")[0]

    jdlFileName = outPath + "/" + options.new + "_{}.jdl".format(syst)
    jdlcontent = jdlTemplate.render(
        MEMORY = MEMORY,
        # DISK = DISK,
        CPU = CPU,
        EXECUTABLE = EXECUTABLE,
        FILES = FILES,
        OUTFILES = OUTFILES
    )
    with open(jdlFileName, mode="w") as jdlFile:
        jdlFile.write(jdlcontent)
        print("... wrote {}".format(jdlFileName))


