# global imports
# import ROOT
# ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse
import json
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DNN_framework.DNN as DNN
import DNN_framework.data_frame as df


# python eval_template_new_sys.py -o 221204_test_evaluation_new -i 221114_test_new_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_1204_UL_test --syst=JESup


"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-d", "--dataset", dest="dataset", default="Eval_1204_UL_test",
                  help="folder of h5 files", metavar="dataset")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

parser.add_option("--signalclass", dest="signal_class", default=None,
        help="STR of signal class for plots", metavar="signal_class")

parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        help="activate to print ROC value for confusion matrix", metavar="printROC")

parser.add_option("--binary", dest="binary", action = "store_true", default=False,
        help="activate to perform binary classification instead of multiclassification. Takes the classes passed to 'signal_class' as signals, all others as backgrounds.")

parser.add_option("-t", "--binaryBkgTarget", dest="binary_bkg_target", default = 0.,
        help="target value for training of background samples (default is 0, signal is always 1)")

# parser.add_option("--total-weight-expr", dest="total_weight_expr",default="x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom",
#         help="string containing expression of total event weight (use letter \"x\" for event-object; example: \"x.weight\")", metavar="total_weight_expr")

parser.add_option("-f", "--test_percentage", dest="test_percentage", default=0.2, type=float, help="set fraction of events used for testing, rest is used for training", metavar="test_percentage")

parser.add_option("-s", "--syst", dest="syst", default="JESup",
                  help="JER JES uncertainties", metavar="syst")

# parser.add_option("-c", "--category", dest="category",default="4j_ge3t",
#                 help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="CATEGORY")
                
parser.add_option("--evaluationEpoch", dest="evaluation_epoch_model", default = None,
                  help="model saved in this epoch used for evaluation", metavar="evaluation_epoch_model")
    
(options, args) = parser.parse_args()

#get input directory path
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

if not options.outDir:
    outPath = inPath
elif not os.path.isabs(options.outDir):
    outPath = basedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

if options.signal_class:
    signal=options.signal_class.split(",")
else:
    signal=None

if options.binary:
    if not signal:
        sys.exit("ERROR: need to specify signal class if binary classification is activated")

configFile = inPath+"/checkpoints/net_config.json"
  # TODO - modify this
dfDirectory = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/"+options.dataset
   
if not os.path.exists(configFile):
        sys.exit(
        "config needed to load trained DNN not found\n{}".format(configFile))

with open(configFile) as f:
        config = f.read()
        config = json.loads(config)


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


# TODO - remove the addSample part because future DNN will save the data df
# TODO - add the dealing with data

for sys in syst:

    # load samples
        input_samples = df.InputSamples(input_path=dfDirectory+"_"+sys+"/",
                                        addSampleSuffix=config["addSampleSuffix"], test_percentage=options.test_percentage)

        for sample in config["eventClasses"]:
                total_weight_expr = "x.Weight_XS * x.Weight_CSV_UL * x.Weight_GEN_nom * x.lumiWeight"
                # normalization_weight = 1
                if sample["sampleLabel"] == "ttHH":
                        # sample_train_weight = 0.5
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttHH_dnn.h5"
                elif sample["sampleLabel"] == "ttZH":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttZH_dnn.h5"
                elif sample["sampleLabel"] == "ttZZ":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttZZ_dnn.h5"
                elif sample["sampleLabel"] == "ttZ":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # '/ (0.001571054/0.00016654)'
                        # sample_path = dfDirectory+"ttZ_dnn.h5"
                elif sample["sampleLabel"] == "ttmb":
                #     sample_train_weight = 1
                        normalization_weight = 26.3
                #     sample_path = dfDirectory+"ttmb_dnn.h5"
                elif sample["sampleLabel"] == "ttnb":
                #     sample_train_weight = 1
                        normalization_weight = 1.1
                #     sample_path = dfDirectory+"ttnb_dnn.h5"
                elif sample["sampleLabel"] == "ttcc":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttcc_dnn.h5"
                elif sample["sampleLabel"] == "ttlf":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttlf_dnn.h5"
                elif sample["sampleLabel"] == "ttH":
                        # sample_train_weight = 1
                        normalization_weight = 1.
                        # sample_path = dfDirectory+"ttH_dnn.h5"
                # normalization_weight = 1
                input_samples.addSample(sample_path=dfDirectory+"_"+sys+"/"+sample["sampleLabel"]+"_dnn.h5", label=sample["sampleLabel"], train_weight=1,
                                        normalization_weight=normalization_weight, total_weight_expr=total_weight_expr)
                # sample_train_weight = 1
                # input_samples.addSample(sample["samplePath"], sample["sampleLabel"],
                #                         normalization_weight=normalization_weight, train_weight=sample_train_weight, total_weight_expr=total_weight_expr)


        print("shuffle seed: {}".format(config["shuffleSeed"]))

        # init DNN class
        dnn = DNN.DNN(
        save_path=outPath+"_"+sys,
        # sample_save_path=sample_save_path,
        input_samples=input_samples,
        category_name=config["JetTagCategory"],
        train_variables=config["trainVariables"],
        Do_Evaluation = True,
        shuffle_seed=config["shuffleSeed"],
        addSampleSuffix=config["addSampleSuffix"],
        )

        # load the trained model
        dnn.load_trained_model(inPath, options.evaluation_epoch_model)


        # plot the output discriminators
        dnn.save_JESJERdiscriminators(log = options.log, privateWork = options.privateWork, printROC = options.printROC, syst = sys)

