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

# 2018 


# python eval_template_new_data.py -o 230220_evaluation_new_6j4b_2_data -i 230220_50_2_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_3_data --notequalbin --lumi=1 -c 6j4b --year=2018 

# python eval_template_new_data.py -o 230220_evaluation_new_5j4b_2_data -i 230220_50_2_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_3_data --notequalbin --lumi=1 -c 5j4b --year=2018



"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-d", "--dataset", dest="dataset", default="Eval_1204_UL_test_nominal",
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

parser.add_option("-f", "--test_percentage", dest="test_percentage", default=0.2, type=float, help="set fraction of events used for testing, rest is used for training", metavar="test_percentage")

parser.add_option("--ttmb", dest="ttmb", default=1.0, type=float,
                  help="factor for ttmb events", metavar="ttmb")

parser.add_option("--ttnb", dest="ttnb", default=1.0, type=float,
                  help="factor for ttnb events", metavar="ttnb")

parser.add_option("--lumi", dest="lumi", default=83, type=float,
                  help="luminosity", metavar="lumi")

parser.add_option("--notequalbin", dest="notequalbin", action="store_false",
                  default=True, help="set up equal bin or not", metavar="notequalbin")
parser.add_option("--year", dest="year", default = "2017",
                  help="year", metavar="year")
parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="category", metavar="category")
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
dfDirectory = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/"+options.dataset+"/"
   
if not os.path.exists(configFile):
        sys.exit(
        "config needed to load trained DNN not found\n{}".format(configFile))

with open(configFile) as f:
        config = f.read()
        config = json.loads(config)

# load samples
# input_samples = data_frame.InputSamples(
#     config["inputData"], addSampleSuffix=config["addSampleSuffix"], test_percentage = options.test_percentage)
input_samples = df.InputSamples(input_path=dfDirectory, addSampleSuffix=config["addSampleSuffix"],dataEra=options.year, test_percentage = options.test_percentage)

total_weight_expr = "x.Weight_XS * x.lumiWeight"
# "x.Weight_XS * x.Weight_CSV_UL * x.Weight_GEN_nom * x.lumiWeight"

input_samples.addSample(sample_path=dfDirectory+"eledata_dnn.h5", label="data",normalization_weight=1., train_weight=1, total_weight_expr=total_weight_expr)

input_samples.addSample(sample_path=dfDirectory+"singlemuon_dnn.h5", label="data",normalization_weight=1., train_weight=1, total_weight_expr=total_weight_expr)


print("shuffle seed: {}".format(config["shuffleSeed"]))


sample_save_path = basedir+"/workdir/"
# init DNN class
dnn = DNN.DNN(
is_Data=True,
save_path=outPath,
input_samples=input_samples,
input_path = inPath,
lumi=options.lumi,
category_name=config["JetTagCategory"],
train_variables=config["trainVariables"],
Do_Evaluation = True,
shuffle_seed=config["shuffleSeed"],
addSampleSuffix=config["addSampleSuffix"],
Do_Control = False,
)


dnn.load_trained_model(inPath, options.evaluation_epoch_model)

dnn.saveData_discriminators(event_classes = ['ttHH', 'ttmb', 'ttcc', 'ttlf', 'ttnb','ttH', 'ttZH', 'ttZZ','ttZ'], log=options.log, privateWork=options.privateWork, printROC=options.printROC, category=options.category, year=options.year,
                        lumi=options.lumi, equalbin=options.notequalbin)  #