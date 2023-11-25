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
# python eval_template_new.py -o 230220_evaluation_new_5FS -i 230220_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal

# python eval_template_new.py -o 230220_evaluation_new_4FS_test -i 230220_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal

# python eval_template_new.py -o 230220_evaluation_new_5j4b -i 230220_50_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=5.654803199 --ttnb=1.0 --notequalbin

# --ttnb=1.240415029
# python eval_template_new.py -o 230220_evaluation_new_5j4b_5FS_test -i 230220_50_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=1.0 --ttnb=1.0

# python eval_template_new.py -o 230220_evaluation_new_5j4b_4FS -i 230220_50_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=5.654803199 --ttnb=3.611169031 --notequalbin

# python eval_template_new.py -o 230220_evaluation_new_6j4b -i 230220_50_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=5.654803199 --ttnb=1.0 --notequalbin
# --ttnb=1.212174627

# python eval_template_new.py -o 230220_evaluation_new_6j4b_5FS_test -i 230220_50_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=1.0 --ttnb=1.0

# python eval_template_new.py -o 230220_evaluation_new_6j4b_4FS -i 230220_50_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal --ttmb=5.654803199 --ttnb=3.611169031 --notequalbin

# python eval_template_new.py -o 230220_evaluation_new_6j4b_4FS_2 -i 231011_50_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_nominal_4FS

# 2017
# python eval_template_new.py -o 230119_evaluation_new_4FS -i 221130_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0119_UL_nominal

# python eval_template_new.py -o 230119_evaluation_new_6j4b -i 230119_50_ge6j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0119_UL_nominal
# python eval_template_new.py -o 230119_evaluation_new_5j4b -i 230119_50_ge5j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0119_UL_nominal

# 2016post
# python eval_template_new.py -o 230523_evaluation_new_2 -i 230523_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0523_UL_nominal

# python eval_template_new.py -o 230523_evaluation_new_6j4b_2 -i 230523_50_ge6j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0523_UL_nominal
# python eval_template_new.py -o 230523_evaluation_new_5j4b_2 -i 230523_50_ge5j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0523_UL_nominal


# 2016pre
# python eval_template_new.py -o 230515_evaluation_new_2 -i 230515_50_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0515_UL_nominal

# python eval_template_new.py -o 230515_evaluation_new_6j4b_2 -i 230515_50_ge6j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0515_UL_nominal

# python eval_template_new.py -o 230515_evaluation_new_5j4b_2 -i 230515_50_ge5j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0515_UL_nominal




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

# parser.add_option("--total-weight-expr", dest="total_weight_expr",default="x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom",
#         help="string containing expression of total event weight (use letter \"x\" for event-object; example: \"x.weight\")", metavar="total_weight_expr")

parser.add_option("-f", "--test_percentage", dest="test_percentage", default=0.2, type=float, help="set fraction of events used for testing, rest is used for training", metavar="test_percentage")

parser.add_option("--ttmb", dest="ttmb", default=1.0, type=float,
                  help="factor for ttmb events", metavar="ttmb")

parser.add_option("--ttnb", dest="ttnb", default=1.0, type=float,
                  help="factor for ttnb events", metavar="ttnb")

parser.add_option("--lumi", dest="lumi", default=83, type=float,
                  help="luminosity", metavar="lumi")

parser.add_option("--notequalbin", dest="notequalbin", action="store_false",
                  default=True, help="set up equal bin or not", metavar="notequalbin")

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
input_samples = df.InputSamples(input_path=dfDirectory, addSampleSuffix=config["addSampleSuffix"], test_percentage = options.test_percentage)

# TODO - remove the addSample part because future DNN will save the data df
# TODO - add the dealing with data

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
                normalization_weight = options.ttmb
                # normalization_weight = 61.  # for 2017
                # normalization_weight = 5.505191209  # for 2018 ttbb 5j4b
                # normalization_weight = 5.467833742  # for 2018 ttbb 6j4b
                # normalization_weight = 1.  # for 2018
        #     sample_path = dfDirectory+"ttmb_dnn.h5"
        elif sample["sampleLabel"] == "ttnb":
        #     sample_train_weight = 1
                # normalization_weight = 1.
                normalization_weight = options.ttnb
                # normalization_weight = 1.35 # for 2018 tt4b
                # normalization_weight = 3.538023785  # for 2018 ttbb 5j4b
                # normalization_weight = 3.363282228  # for 2018 ttbb 6j4b
                # normalization_weight = 1.240415029  # for 2018 tt4b 5j4b
                # normalization_weight = 1.212174627  # for 2018 tt4b 6j4b
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

        # if sample["sampleLabel"] == "ttH":
        input_samples.addSample(sample_path=dfDirectory+sample["sampleLabel"]+"_dnn.h5", label=sample["sampleLabel"],
                                normalization_weight=normalization_weight, train_weight=1, total_weight_expr=total_weight_expr)
        # sample_train_weight = 1
        # input_samples.addSample(sample["samplePath"], sample["sampleLabel"],
        #                         normalization_weight=normalization_weight, train_weight=sample_train_weight, total_weight_expr=total_weight_expr)


print("shuffle seed: {}".format(config["shuffleSeed"]))

#TODO-modify this
sample_save_path = basedir+"/workdir/"
# init DNN class
dnn = DNN.DNN(
save_path=outPath,
# sample_save_path=sample_save_path,
input_samples=input_samples,
# lumi = 119.66,
lumi=options.lumi,
# lumi = 67.24, # 2016post
# lumi = 78.08, # 2016pre
# lumi = 83,
category_name=config["JetTagCategory"],
train_variables=config["trainVariables"],
Do_Evaluation = True,
shuffle_seed=config["shuffleSeed"],
addSampleSuffix=config["addSampleSuffix"],
Do_Control = False,
)

#    dnn._load_datasets(shuffle_seed=config["shuffleSeed"],balanceSamples=True)
# load the trained model
dnn.load_trained_model(inPath, options.evaluation_epoch_model)
# dnn.predict_event_query()


# dnn = DNN.loadDNN(inPath, outPath, sample_save_path, binary=options.binary, signal=signal,
#                   binary_target=options.binary_bkg_target, total_weight_expr=options.total_weight_expr, model_epoch=options.evaluation_epoch_model)

# plotting
# if options.plot:
    # if options.binary:
        # plot output node
        # bin_range = [options.binary_bkg_target, 1.]
        # dnn.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC, bin_range = bin_range)

    # else:
#        # plot the confusion matrix
#        dnn.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)

        # plot the output discriminators
# dnn.save_discriminators(log = options.log, privateWork = options.privateWork, printROC = options.printROC, lumi=41.5)
# dnn.save_discriminators(log = options.log, privateWork = options.privateWork, printROC = options.printROC, lumi=67.24) # 59.7 * 2 , because select only Evt_Odd = 0 
# dnn.save_discriminators(log = options.log, privateWork = options.privateWork, printROC = options.printROC, lumi=78.08) # 59.7 * 2 , because select only Evt_Odd = 0 
# dnn.save_discriminators(log = options.log, privateWork = options.privateWork, printROC = options.printROC, lumi=83) # 59.7 * 2 , because select only Evt_Odd = 0 
dnn.save_discriminators(log=options.log, privateWork=options.privateWork, printROC=options.printROC,
                        lumi=119.66, equalbin=options.notequalbin)  # 59.7 * 2 , because select only Evt_Odd = 0
#
#        # plot the output nodes
#        dnn.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC, sigScale = -1)
#
#        # plot closure test
#        dnn.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)
