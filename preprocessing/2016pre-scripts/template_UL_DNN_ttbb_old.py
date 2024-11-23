import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
# basedir = os.path.dirname(os.path.dirname(filedir))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import preprocessing
# import additional_variables as add_var
# import selections


"""
USE: python3 /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/preprocessing/template_UL_DNN_ttbb_old.py --outputdirectory=DNN_0509_UL_2016pre --variableselection=variables --maxentries=20000 --cores=8  --dataEra=2016preVFP
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

parser.add_option("-y", "--dataEra", dest="dataEra", default=2017,
                  help="dataera", metavar="dataEra")
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


ttbar_categories = preprocessing.EventCategories()
ttbar_categories.addCategory("ttbb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("tt2b", selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttb",  selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttlf", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttcc", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")

ttbar2_categories = preprocessing.EventCategories()
ttbar2_categories.addCategory("ttbbb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0)")
ttbar2_categories.addCategory("tt4b", selection = "(GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")


ttmb_categories = preprocessing.EventCategories()
ttmb_categories.addCategory("ttmb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
ttmb_categories.addCategory("ttnb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ntuplesPath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_10_6_29/src/BoostedTTH/crab/2017UL/ntuple/crab_ntuple"
ntuplesPath2 = "/eos/uscms/store/group/lpctthrun2/wwei/UL"


# initialize dataset class
dataset = preprocessing.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    maxEntries  = options.maxEntries,
    ncores      = options.numCores,
    dataEra=options.dataEra,
    do_EvalSFs=False,
    )

# add base event selection
dataset.addBaseSelection(base_selection)


# dataset.addSample(
#     sampleName="TTHHTo4b",
#     ntuples=ntuplesPath +
#     "/crab_TTHHTo4b_5f_LO_TuneCP5_13TeV_madgraph_pythia8_2017_ntuple_0_0_11/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttHH_4b.root",
#     categories=ttHH_categories,
#     #    lumiWeight  = 41.5,
#     selections  = None,
#     # selections=ttHH_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName="TTH",
#     ntuples=ntuplesPath +
#     "/crab_ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_2017_ntuple_0_0_7/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttH_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttH_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName="TTZZ",
#     ntuples=ntuplesPath+"/crab_TTZZTo4b_5f_LO_TuneCP5_13TeV_madgraph_pythia8_2017_ntuple_0_0_no_sys/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttZZ_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=True
# )


# dataset.addSample(
#     sampleName="TTZZ2",
#     ntuples=ntuplesPath+"/crab_TTZZTo4b_5f_LO_TuneCP5_13TeV_madgraph_pythia8_Ext_2017_ntuple_0_0_no_sys/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttZZ_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName="TTZH",
#     ntuples=ntuplesPath +
#     "/crab_TTZH_TuneCP5_13TeV-madgraph-pythia8_2017_ntuple_0_0_no_sys/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttZH_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName="TTZH2",
#     ntuples=ntuplesPath +
#     "/crab_TTZH_TuneCP5_13TeV-madgraph-pythia8_pmx_2017_ntuple_0_0_no_sys/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttZH_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName="TTZbb",
#     ntuples=ntuplesPath +
#     "/crab_TTZToBB_TuneCP5_13TeV-amcatnlo-pythia8_2017_ntuple_0_0_no_sys_2/results/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttZbb_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=True
# )

# dataset.addSample(
#     sampleName  = "TT4b",
#     ntuples=ntuplesPath2 +
#     "/2018/ntuple/TT4b_TuneCP5_13TeV_madgraph_pythia8/sl_LEG_ntuple_2018/230212_022936/*/*nominal*.root",
#     categories  = ttbar2_categories,
#     process = "tt4b",
# #    lumiWeight  = 41.5,
#     selections=ttHH_selection,  # ttbar_selection,
# #    selections  = ttbar_selection,
#     islocal     = False
#       ) 

      
dataset.addSample(
    sampleName  = "TTbbSL",
    ntuples=ntuplesPath2 +
    "/2016pre/ntuple/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2016preVFP/230501_213507/*/*nominal*.root",
    categories  = ttbar_categories,
    process = "ttbbSL",
#    lumiWeight  = 41.5,
    selections=ttHH_selection,  # ttbar_selection,
#    selections  = ttbar_selection,
    islocal     = False
      )

# dataset.addSample(
#     sampleName="TTbbToDL",
#     ntuples=ntuplesPath2 +
#     "/TTbb_Powheg_Openloops/sl_LEG_ntuple_2017_DL/220322_040450/0000/*nominal*.root",
#     categories=ttbar_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=False
# )

# dataset.addSample(
#     sampleName="TTbbToDL2",
#     ntuples=ntuplesPath2 +
#     "/TTbb_Powheg_Openloops/sl_LEG_ntuple_2017_DL_Ext/220322_041515/0000/*nominal*.root",
#     categories=ttbar_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=False
# )
      

# initialize dataset class
# dataset2 = preprocessing.Dataset(
#     outputdir=outputdir,
#     naming=options.Name,
#     maxEntries=options.maxEntries,
#     ncores=options.numCores,
# )

# add base event selection
# dataset2.addBaseSelection(base_selection)

# dataset2.addSample(
#     sampleName="TTToSL",
#     ntuples=ntuplesPath2+"/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/220315_154526/0000/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttSL_220210.root",
#     categories=ttbar_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection
#     islocal=False
# )

# dataset2.addSample(
#     sampleName="TTToDL",
#     ntuples=ntuplesPath2+"/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/sl_LEG_ntuple_2017_2/220721_041939/*/*nominal*.root",
#     #    ntuples     = ntuplesPath+"/ttSL_220210.root",
#     categories=ttbar_categories,
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection
#     islocal=False
# )
# initialize variable list
dataset.addVariables(variable_set.all_variables)
# dataset2.addVariables(variable_set.all_variables)

sys.path.append(basedir+"/variable_sets/")

# print (basedir)
import additional_variables as add_var
# import sf_variables as sf_var
import sf_variables as sf_var
# add these variables to the variable list
dataset.addVariables(add_var.additional_variables)
# dataset2.addVariables(add_var.additional_variables)
# dataset.addVariables(sf_var.scalefactor_variables)
# dataset2.addVariables(sf_var.scalefactor_variables)
# dataset2.addVariables(sf_var.ttbar_variables)

# run the preprocessing
dataset.runPreprocessing()
# dataset2.runPreprocessing()
