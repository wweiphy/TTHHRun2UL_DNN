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

sys.path.append(basedir+"/variable_sets/")
# print (basedir)
import additional_variables as add_var
# import sf_variables as sf_var
import sf_variables as sf_var


"""
USE: python3 /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/preprocessing/template_UL_bTag_Syst.py --outputdirectory=BTag_0308_UL --variableselection=variables_bTagCorrection --maxentries=20000 --cores=8  --syst=JESup --dataEra=2018


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

parser.add_option("-s", "--syst", dest="syst", default="JERup",
        help="Systematics", metavar="syst")

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
base_selection = "(N_Jets >= 5)"

# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1)"
single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1))"

# base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

ttHH_selection = "(Evt_Odd == 0)"  # Should I do this on ttHH

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

ttHSL_categories = preprocessing.EventCategories()
ttHSL_categories.addCategory("ttHSL", selection = None)

ttHDL_categories = preprocessing.EventCategories()
ttHDL_categories.addCategory("ttHDL", selection = None)


ttSL_categories = preprocessing.EventCategories()
ttSL_categories.addCategory("ttSL", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")

ttDL_categories = preprocessing.EventCategories()
ttDL_categories.addCategory("ttDL", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
# ttbar_categories.addCategory("ttbb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt2b", selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttb",  selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttlf", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("ttcc", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
# ttbar_categories.addCategory("ttbbb", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0)")
# ttbar_categories.addCategory("tt4b", selection = "(GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ttmbSL_categories = preprocessing.EventCategories()
ttmbSL_categories.addCategory("ttbbSL", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")

ttmbDL_categories = preprocessing.EventCategories()
ttmbDL_categories.addCategory("ttbbDL", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")

ttnb_categories = preprocessing.EventCategories()
ttnb_categories.addCategory("tt4b", selection = "(GenEvt_I_TTPlusBB == 4 and GenEvt_I_TTPlusCC == 0) or (GenEvt_I_TTPlusBB == 5 and GenEvt_I_TTPlusCC == 0)")

ntuplesPath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_10_6_29/src/BoostedTTH/crab/2017UL/ntuple/crab_ntuple"
ntuplesPath2 = "/eos/uscms/store/group/lpctthrun2/wwei/UL"

# syst = [
# #   'JESup',
# #   'JESdown',
# #   'JERup',
#   'JERdown',
# #   'JESFlavorQCDup',
# #   'JESRelativeBalup',
# #   'JESHFup',
# #   'JESBBEC1up', 
# #   'JESEC2up',
# #   'JESAbsoluteup',
# #   'JESBBEC1yearup',
# #   'JESRelativeSampleyearup',
# #   'JESEC2yearup',
# #   'JESHFyearup',
# #   'JESAbsoluteyearup',
# #   'JESFlavorQCDdown',
# #   'JESRelativeBaldown',
# #   'JESHFdown',
# #   'JESBBEC1down',
# #   'JESEC2down',
# #   'JESAbsolutedown',
# #   'JESBBEC1yeardown',
# #   'JESRelativeSampleyeardown',
# #   'JESEC2yeardown',
# #   'JESHFyeardown',
# #   'JESAbsoluteyeardown',
# ]


# for sys in syst:

# initialize dataset class
dataset = preprocessing.Dataset(
    outputdir   = outputdir+"_"+options.syst,
    naming      = options.Name,
    maxEntries  = options.maxEntries,
    ncores      = options.numCores,
    dataEra=options.dataEra,
    do_BTagCorrection=True,
    # do_EvalSFs=True,

    )

# add base event selection
dataset.addBaseSelection(base_selection)


dataset.addSample(
    sampleName="TTHHTo4b",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTHHTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2018/240329_160626/*/*" +
    options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttHH_4b.root",
    categories=ttHH_categories,
    process = "ttHH",
    #    lumiWeight  = 41.5,
    # selections  = None,
    selections=None,
    islocal=False
)


dataset.addSample(
    sampleName="TTZZ",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTZZTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2018/240329_205536/*/*" + options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZZ_categories,
    process = "ttZZ",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)


dataset.addSample(
    sampleName="TTZH",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTZHTo4b_TuneCP5_13TeV-madgraph-pythia8/sl_LEG_ntuple_2018/240329_200040/*/*" + options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZH_categories,
    process = "ttZH",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)




dataset.addSample(
    sampleName="TTZ",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTZToBB_TuneCP5_13TeV-amcatnlo-pythia8/sl_LEG_ntuple_2018/240329_205344/*/*" + options.syst+"*.root",
    # ntuples="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_10_6_29/src/BoostedTTH/BoostedAnalyzer/test/ntuples_nominal_Tree.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttZ_categories,
    process = "ttZ",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)  # almost

      
dataset.addSample(
    sampleName  = "TTbbSL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2018/240330_004830/*/*"+options.syst+"*.root",
    categories  = ttmbSL_categories,
    process = "ttbbSL",
#    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
#    selections  = ttbar_selection,
    islocal     = False
      )

dataset.addSample(
    sampleName="TTbbDL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTbb_4f_TTTo2L2Nu_TuneCP5-Powheg-Openloops-Pythia8/sl_LEG_ntuple_2018/240330_005217/*/*"+options.syst+"*.root",
    # 221118_234955
    categories=ttmbDL_categories,
    process="ttbbDL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)

dataset.addSample(
    sampleName="TT4b",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TT4b_TuneCP5_13TeV_madgraph_pythia8/sl_LEG_ntuple_2018/240329_194937/*/*" +
    options.syst+"*.root",
    categories=ttnb_categories,
    process="tt4b",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection,
    islocal=False
)  # not finished


dataset.addSample(
    sampleName="TTHSL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2018/240330_010702/*/*"+options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttHSL_categories,
    process = "ttHSL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttH_selection,
    islocal=False
)

dataset.addSample(
    
    sampleName="TTHDL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/ttHTobb_ttTo2L2Nu_M125_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2018/240330_005509/*/*"+options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttH_220208.root",
    categories=ttHDL_categories,
    process = "ttHDL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttH_selection,
    islocal=False
)

dataset.addSample(
    sampleName="TTSL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2018/240329_161542/*/*" + options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttSL_220210.root",
    categories=ttSL_categories,
    process = "ttSL",
    #    lumiWeight  = 41.5,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection
    islocal=False
)

dataset.addSample(
    sampleName="TTDL",
    ntuples=ntuplesPath2 +
    "/2018/ntuple/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/sl_LEG_ntuple_2018/240329_162107/*/*" + options.syst+"*.root",
    #    ntuples     = ntuplesPath+"/ttSL_220210.root",
    categories=ttDL_categories,
    process="ttDL",
    # lumiWeight  = 1.0,
    selections=None,  # ttbar_selection,
    #    selections  = ttbar_selection
    islocal=False
)

# initialize variable list
dataset.addVariables(variable_set.all_variables)
# dataset2.addVariables(variable_set.all_variables)

# add these variables to the variable list
dataset.addVariables(add_var.additional_variables)
# dataset2.addVariables(add_var.additional_variables)
dataset.addVariables(sf_var.scalefactor_variables)
# dataset2.addVariables(sf_var.scalefactor_variables)
# dataset2.addVariables(sf_var.ttbar_variables)

# run the preprocessing
dataset.runPreprocessing()
# dataset2.runPreprocessing()
