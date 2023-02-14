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
USE: python3 /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/preprocessing/template_UL_Eval_nominal_test.py --outputdirectory=Eval_0103_UL_test --variableselection=dnn_variables --maxentries=20000 --cores=4
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


ttbar_categories = preprocessing.EventCategories()
# ttbar_categories.addCategory("ttbb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt2b", selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttb",  selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttlf", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttcc", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
# ttbar_categories.addCategory("ttbbb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt4b", selection = "(GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ttmb_categories = preprocessing.EventCategories()
ttmb_categories.addCategory("ttmb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")

ttnb_categories = preprocessing.EventCategories()
ttnb_categories.addCategory("ttnb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ntuplesPath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_10_6_29/src/BoostedTTH/crab/2017UL/ntuple/crab_ntuple"
ntuplesPath2 = "/eos/uscms/store/group/lpctthrun2/wwei/UL"


# initialize dataset class
dataset = preprocessing.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    maxEntries  = options.maxEntries,
    ncores      = options.numCores,
    do_EvalSFs=True,
    )

# add base event selection
dataset.addBaseSelection(base_selection)


dataset.addSample(
    sampleName="TTHH",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTHHTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2017/221126_052019/*/ntuples_nominal_Tree_10.root",
    #    ntuples     = ntuplesPath+"/ttHH_4b.root",
    categories=ttHH_categories,
    process = "ttHH",
    # lumiWeight  = 1.105,
    selections  = None,
    # selections=ttHH_selection,
    islocal=False
)



dataset.addSample(
    sampleName="TTZZ",
        ntuples=ntuplesPath2 +
        "/2017/ntuple/TTZZTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2017/221126_052936/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZZ_categories,
    process="ttZZ",
    # lumiWeight  = 6.75E-02,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)


dataset.addSample(
    sampleName="TTZZ2",
    ntuples=ntuplesPath2 +
        "/2017/ntuple/TTZZTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2017_Ext/221126_053245/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZZ_categories,
    process="ttZZ",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=True
)

dataset.addSample(
    sampleName="TTZH",
    ntuples=ntuplesPath2 +
        "/2017/ntuple/TTZHTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2017/221126_052419/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZH_categories,
    process="ttZH",
    # lumiWeight  = 1.1035,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)

dataset.addSample(
    sampleName="TTZH2",
    ntuples=ntuplesPath2 +
        "/2017/ntuple/TTZHTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2017_Ext/221126_052726/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZH_categories,
    process="ttZH",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=True
)

dataset.addSample(
    sampleName="TTZ",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTZToBB_TuneCP5_13TeV-amcatnlo-pythia8/sl_LEG_ntuple_2017/221126_052843/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZ_categories,
    process="ttZ",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)

      


dataset.addSample(
    sampleName="TT4b",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TT4b_TuneCP5_13TeV_madgraph_pythia8/sl_LEG_ntuple_2017/221126_051927/*/ntuples_nominal_Tree_1.root",
    categories=ttnb_categories,
    process="tt4b",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)  # not finished





    
# initialize dataset class
dataset2 = preprocessing.Dataset(
    outputdir=outputdir,
    naming=options.Name,
    maxEntries=options.maxEntries,
    ncores=options.numCores,
    do_EvalSFs=True,
)

# add base event selection
dataset2.addBaseSelection(base_selection)

# dataset2.addSample(
#     sampleName="TTH",
#     ntuples=ntuplesPath2 +
#     "/2017/ntuple/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/221126_054209/*/ntuples_nominal_Tree_1.root",
#     #    ntuples     = ntuplesPath+"/ttH_220208.root",
#     categories=ttH_categories,
#     process="ttH",
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttH_selection,
#     islocal=False
# )

dataset2.addSample(
    sampleName="TTSL",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/230115_045248/*/ntuples_nominal_Tree_9.root",
    #    ntuples     = ntuplesPath+"/ttSL_220210.root",
    categories=ttbar_categories,
    process="ttSL",
    # lumiWeight  = 5.0,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection
    islocal=False
)

dataset2.addSample(
    sampleName="TTDL",
    # ntuples="/eos/uscms/store/user/wwei/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/230115_045248/*/ntuples_nominal_Tree_1.root",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/230115_045026/*/ntuples_nominal_Tree_9.root",
    categories=ttbar_categories,
    process="ttDL",
    # lumiWeight  = 5.0,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection
    islocal=False
)


# initialize dataset class
dataset3 = preprocessing.Dataset(
    outputdir=outputdir,
    naming=options.Name,
    maxEntries=options.maxEntries,
    ncores=options.numCores,
    do_EvalSFs=True,
)
dataset3.addBaseSelection(base_selection)


dataset3.addSample(
    sampleName="TTbbSL",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2017/230115_045551/*/ntuples_nominal_Tree_9.root",
    categories=ttmb_categories,
    process="ttbbSL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)

# dataset3.addSample(
#     sampleName="TTbbSL2",
#     ntuples=ntuplesPath2 +
#     "/2017/ntuple/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2017/221126_053456/*/ntuples_nominal_Tree_2.root",
#     categories=ttmb_categories,
#     process="ttbbSL",
#     #    lumiWeight  = 41.5,
#     selections=None,  # ttbar_selection,
#     #    selections  = ttbar_selection,
#     islocal=False
# )

dataset3.addSample(
    sampleName="TTbbDL",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/TTbb_4f_TTTo2L2Nu_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2017/230115_050027/*/ntuples_nominal_Tree_1.root",
    categories=ttmb_categories,
    process="ttbbDL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)

# initialize dataset class
dataset4 = preprocessing.Dataset(
    outputdir=outputdir,
    naming=options.Name,
    maxEntries=options.maxEntries,
    ncores=options.numCores,
    do_EvalSFs=True,
)

# add base event selection
dataset4.addBaseSelection(base_selection)

dataset4.addSample(
    sampleName="TTH",
    ntuples=ntuplesPath2 +
    "/2017/ntuple/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2017/221126_054916/*/ntuples_nominal_Tree_1.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttH_categories,
    process="ttH",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttH_selection,
    islocal=False
)

# initialize variable list
dataset.addVariables(variable_set.all_variables)
dataset2.addVariables(variable_set.all_variables)
dataset3.addVariables(variable_set.all_variables)
dataset4.addVariables(variable_set.all_variables)

sys.path.append(basedir+"/variable_sets/")

# print (basedir)
import additional_variables as add_var
# import sf_variables as sf_var
import sf_variables as sf_var
# add these variables to the variable list
dataset.addVariables(add_var.additional_variables)
dataset2.addVariables(add_var.additional_variables)
dataset3.addVariables(add_var.additional_variables)
dataset4.addVariables(add_var.additional_variables)
dataset.addVariables(sf_var.scalefactor_variables)
dataset2.addVariables(sf_var.scalefactor_variables)
dataset3.addVariables(sf_var.scalefactor_variables)
dataset4.addVariables(sf_var.scalefactor_variables)
dataset2.addVariables(sf_var.ttbar_variables)
dataset3.addVariables(sf_var.ttbar_variables)
dataset4.addVariables(sf_var.ttH_variables)
dataset2.addVariables(sf_var.PDF_tt)
dataset3.addVariables(sf_var.PDF_ttbb)
dataset4.addVariables(sf_var.PDF_ttH)

# run the preprocessing
dataset.runPreprocessing()
dataset2.runPreprocessing()
dataset3.runPreprocessing()
dataset4.runPreprocessing()