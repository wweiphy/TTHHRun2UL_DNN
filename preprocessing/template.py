import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
# basedir = os.path.dirname(os.path.dirname(filedir))
basedir = filedir
sys.path.append(basedir)
import preprocessing
# import additional_variables as add_var
# import selections


"""
USE: python template.py --outputdirectory=DIR --variableselection=variables --maxentries=20000 
"""

usage="usage=%prog [options] \n"
usage+="USE: python template.py --outputdirectory=DIR --variableselection=FILE --maxentries=INT --name=STR\n"
usage+="OR: python template.py -o DIR -v FILE -e INT -n STR"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="InputFeatures",
        help="DIR for output", metavar="outputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-e", "--maxentries", dest="maxEntries", default=50000,
        help="INT used for maximal number of entries for each batch (to restrict memory usage)", metavar="maxEntries")

parser.add_option("-n", "--name", dest="Name", default="dnn",
        help="STR of the output file name", metavar="Name")

parser.add_option("-c", "--cores", dest="numCores", default=1,
                  help="number of cores to run the preprocessing", metavar="NumCores")

# parser.add_option("-l", "--islocal", dest="islocal", default=False,
#                   help="True if the ntuple files are stored in the eos space, False if the ntuple files are in local space", metavar="islocal")

(options, args) = parser.parse_args()

if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

if not os.path.isabs(options.outputDir):
    outputdir = basedir+"/workdir/"+options.outputDir
elif os.path.exists(options.outputDir) or os.path.exists(os.path.dirname(options.outputDir)):
    outputdir=options.outputDir
else:
    sys.exit("ERROR: Output Directory does not exist!")

# define a base event selection which is applied for all Samples
# select only events with GEN weight > 0 because training with negative weights is weird
# base = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_MET > 20. and Weight_GEN_nom > 0.)"
base = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_MET > 20.)"

# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1)"
single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1))"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

ttHH_selection = "(Evt_Odd == 1)"  # Should I do this on ttHH

# define output classes
ttHH_categories = preprocessing.EventCategories()
ttHH_categories.addCategory("ttHH", selection = None)

# ttHH_even_categories = preprocessing.EventCategories()
# ttHH_even_categories.addCategory("ttHH_even", selection = None)
# ttHH_even_selection = "(Evt_Odd == 0)"

ttZH_categories = preprocessing.EventCategories()
ttZH_categories.addCategory("ttZH", selection = None)

ttZZ_categories = preprocessing.EventCategories()
ttZZ_categories.addCategory("ttZZ", selection = None)

ttZ_categories = preprocessing.EventCategories()
ttZ_categories.addCategory("ttZ", selection = None)

ttH_categories = preprocessing.EventCategories()
ttH_categories.addCategory("ttH", selection = None)

ttZbb_categories = preprocessing.EventCategories()
ttZbb_categories.addCategory("ttZbb", selection = None)


# ttbar_categories = preprocessing.EventCategories()
# ttbar_categories.addCategory("ttbb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt2b", selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttb",  selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
#ttbar_categories.addCategory("ttlf", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
#ttbar_categories.addCategory("ttcc", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
# ttbar_categories.addCategory("ttbbb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt4b", selection = "(GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ttmb_categories = preprocessing.EventCategories()
ttmb_categories.addCategory("ttmb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
ttmb_categories.addCategory("ttnb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")


# initialize dataset class
dataset = preprocessing.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    maxEntries  = options.maxEntries,
    ncores      = options.numCores,
    )

# add base event selection
dataset.addBaseSelection(base_selection)

ntuplesPath = "/uscms/home/wwei/nobackup/SM_TTHH/CMSSW_10_2_18/src/BoostedTTH/crab/lpcgroup/crab_ntuple"

dataset.addSample(
    sampleName  = "TT4b",
    ntuples     = ntuplesPath+"/crab_TT4b_TuneCP5_13TeV_madgraph_pythia8_2017_ntuple_0_0_5/results/*nominal*.root",
    categories  = ttmb_categories,
#    lumiWeight  = 41.5,
    selections  = None,#ttbar_selection,
#    selections  = ttbar_selection,
    islocal     = True
      ) # not finished

      
dataset.addSample(
    sampleName  = "TTbbToSL",
    ntuples     = ntuplesPath+"/crab_TTbb_Powheg_Openloops_SL_2017_ntuple_0_0_no_sys/results/*nominal*.root",
    categories  = ttmb_categories,
#    lumiWeight  = 41.5,
    selections  = None,#ttbar_selection,
#    selections  = ttbar_selection,
    islocal     = True
      )
      
      
# initialize variable list
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
# additional_variables = [
#     "Evt_Odd",
#     #    "N_Jets",
#     #    "N_BTagsL",
#     #    "N_BTagsM",
#     #    "N_BTagsT",
#     "Weight_XS",
#     "Weight_CSV",
#     "Weight_GEN_nom",
#     "Evt_ID",
#     "Evt_Run",
#     "Evt_Lumi"]
sys.path.append(basedir+"/variable_sets/")
import additional_variables as add_var
import sf_variables as sf_var
# add these variables to the variable list
dataset.addVariables(add_var.additional_variables)
dataset.addVariables(sf_var.scalefactor_variables['nominal'])
# dataset.addVariables(add_var.all_additional_variables)

# run the preprocessing
dataset.runPreprocessing()
