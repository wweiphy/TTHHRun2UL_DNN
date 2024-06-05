import os
import sys
import optparse
import pandas as pd
import ROOT


# python mergeNode.py -c 6j4b_5 


usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

# parser.add_option("--twoyear", dest="twoyear", action = "store_true", default=False,
#         help="combine 18 with 17", metavar="twoyear")

# parser.add_option("-y", "--year", dest="year", default="2017",
#         help="year", metavar="year")


parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="selection cateogiry", metavar="category")

(options, args) = parser.parse_args()

# processlist = ['ttlf', 'ttcc', 'ttmb', 'ttHH','ttH','ttZ','ttZH','ttZZ','ttnb']
processlist = ['ttbar', 'ttHH','ttH','ttZ','ttZH','ttZZ']

# decorrelated_systs = ['effTrigger_mu','effTrigger_e','eff_mu','eff_e','btag_hfstats1','btag_hfstats2','btag_lfstats1','btag_lfstats2','JER']

filedir = os.path.dirname(os.path.realpath(__file__))
combdir = os.path.dirname(filedir)
basedir = os.path.dirname(combdir)
sys.path.append(basedir)


folder_path = basedir + "/workdir/"

files = ['230220','230119','230515','230523']
# files = ['230220']
# decorrelated_systs = ['effTrigger_mu','effTrigger_e','eff_mu','eff_e','btag_hfstats1','btag_hfstats2','btag_lfstats1','btag_lfstats2','JER']


for file in files:

    if file == "230220":
        year = "2018"
    if file == "230119":
        year = "2017"
    if file == "230515":
        year = "2016preVFP"
    if file == "230523":
        year = "2016postVFP"

    filepath = file+"_evaluation_new_"+options.category+"/plots/output_limit.root"

    file = ROOT.TFile.Open(folder_path+filepath, "UPDATE")
    print('file: '+filepath)

    for node in processlist:

        for process in processlist:

            histonameup = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__JER_"+year+"Up"
            histonamedown = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__JER_"+year+"Down"

            new_histonameup = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__JER_"+process+"_"+year+"Up"
            new_histonamedown = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__JER_"+process+"_"+year+"Down"


            new_histup = histonameup.Clone()
            new_histdown = histonamedown.Clone()


            new_histup.Write(histoname)
            new_histdown.Write(histoname)

        

    print("done with renaming JER histograms")













