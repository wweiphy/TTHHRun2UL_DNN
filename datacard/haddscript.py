import os
import sys
import optparse
import glob

# python haddscript.py -f 221204_evaluation_new

filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
# basedir = os.path.dirname(DRACOdir)

usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

# parser.add_option("-n", "--new", dest="new", default="new",
#         help="making datacard for new categorizations, total 9", metavar="new")

parser.add_option("-f", "--folder", dest="folder", default="221204_test_evaluation_new",
                  help="folder name", metavar="folder")

(options, args) = parser.parse_args()

# process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
#                'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
# process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
#                'ttZZ', 'ttlf', 'ttcc', 'ttb','ttbb','tt2b','ttbbb','tt4b']


syst = [
  'JESup',
  'JESdown',
  'JERup',
  'JERdown',
#   'JESFlavorQCDup',
#   'JESRelativeBalup',
#   'JESHFup',
#   'JESBBEC1up',
#   'JESEC2up',
#   'JESAbsoluteup',
#   'JESBBEC1yearup',
#   'JESRelativeSampleyearup',
#   'JESEC2yearup',
#   'JESHFyearup',
#   'JESAbsoluteyearup',
#   'JESFlavorQCDdown',
#   'JESRelativeBaldown',
#   'JESHFdown',
#   'JESBBEC1down',
#   'JESEC2down',
#   'JESAbsolutedown',
#   'JESBBEC1yeardown',
#   'JESRelativeSampleyeardown',
#   'JESEC2yeardown',
#   'JESHFyeardown',
#   'JESAbsoluteyeardown',
]

filepath = basedir+"/workdir/{}/plots/*discriminator.root".format(options.folder)
allFiles = sorted(glob.glob(filepath))
print(filepath)



files = ""

print ("doing nominal files")
for file in allFiles:
    files += " " + file


for sys in syst:

    print ("doing {} files".format(sys))

    sys_filepath = basedir+"/workdir/{}/plots/*.root".format(options.folder + "_" + sys)
    
    allFiles = sorted(glob.glob(sys_filepath))

    for file in allFiles:
        files += " " + file

print(files)
command = "hadd "+basedir+"workdir/{}/plots/output_limit.root ".format(options.folder) + files

# command = "hadd /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root ".format(options.folder) + files

print ("hadd files: ")
# print (command)

os.system(command)

