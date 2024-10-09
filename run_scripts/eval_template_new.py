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

# python eval_template_new.py -o 230220_evaluation_new_6j4b_8 -i 230220_50_2_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_3_nominal --ttmb=5.696962364 --ttnb=1.0 --notequalbin --lumi=119.66 --year=2018 --ttHH=1.0 -c 6j4b

# python eval_template_new.py -o 230220_evaluation_new_5j4b_5 -i 230220_50_2_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_4_nominal --ttHH=2. --ttmb=11.23889254 --ttcc=2.0 --ttlf=2.0 --ttnb=2. --ttH=2. --ttZH=2. --ttZZ=2. --ttZ=2. --notequalbin --lumi=59.8 --year=2018 -c 5j4b
# -ttmb=2.0*5.61944626999348

# python eval_template_new.py -o 230220_evaluation_new_4j3b_test -i 230220_50_2_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0308_UL_3_nominal --ttmb=5.704889739467897  --ttnb=1.0 --notequalbin --lumi=119.66 --year=2018 --ttHH=1.0 -c 4j3b



# 2017

# python eval_template_new.py -o 230119_evaluation_new_6j4b_test -i 230119_50_ge6j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0119_UL_3_nominal --ttmb=62.58449531 --ttnb=1.0 --notequalbin --lumi=83.0 --year=2017 --ttHH=1.0 -c 6j4b


# python eval_template_new.py -o 230119_evaluation_new_5j4b_7 -i 230119_50_ge5j_ge4t --signalclass=ttHH --plot --printroc -d Eval_0119_UL_4_nominal --ttHH=2. --ttmb=123.5667701 --ttcc=2.0 --ttlf=2.0 --ttnb=2. --ttH=2. --ttZH=2. --ttZZ=2. --ttZ=2. --notequalbin --lumi=41.5 --year=2017 -c 5j4b 


# 2016post

# python eval_template_new.py -o 230523_evaluation_new_6j4b_test -i 230523_50_2_ge6j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0523_UL_3_nominal --ttmb=6.705125691 --ttnb=1.0 --lumi=33.62 -c 6j4b --notequalbin --year=2016postVFP --ttHH=1.0

# python eval_template_new.py -o 230523_evaluation_new_5j4b_5 -i 230523_50_2_ge5j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0523_UL_4_nominal --ttHH=2. --ttmb=13.45424116 --ttcc=2.0 --ttlf=2.0 --ttnb=2. --ttH=2. --ttZH=2. --ttZZ=2. --ttZ=2. --notequalbin --lumi=16.81 --year=2016postVFP -c 5j4b 



# python eval_template_new.py -o 230523_evaluation_new_4j3b -i 230523_50_2_ge4j_ge3t  --signalclass=ttHH --plot --printroc -d Eval_0523_UL_3_nominal --ttmb=6.861631179 --ttnb=1.0 --lumi=33.62 -c 4j3b --notequalbin --year=2016postVFP --ttHH=1.0


# 2016pre
# python eval_template_new.py -o 230515_evaluation_new_4j3b -i 230515_50_2_ge4j_ge3t --signalclass=ttHH --plot --printroc -d Eval_0515_UL_3_nominal -c 4j3b --ttmb=6.629180384 --ttnb=1.0 --lumi=39.04 --notequalbin --year=2016preVFP --ttHH=1.0

# python eval_template_new.py -o 230515_evaluation_new_6j4b_test -i 230515_50_2_ge6j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0515_UL_3_nominal --ttmb=6.414488975 --ttnb=1.0 --lumi=39.04 -c 6j4b --notequalbin --year=2016preVFP --ttHH=1.0

# python eval_template_new.py -o 230515_evaluation_new_5j4b_5 -i 230515_50_2_ge5j_ge4t  --signalclass=ttHH --plot --printroc -d Eval_0515_UL_4_nominal --ttHH=2. --ttmb=12.69219617 --ttcc=2.0 --ttlf=2.0 --ttnb=2. --ttH=2. --ttZH=2. --ttZZ=2. --ttZ=2. --notequalbin --lumi=19.52 --year=2016preVFP -c 5j4b 








"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="category", metavar="category")

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

parser.add_option("--ttHH", dest="ttHH", default=1.0, type=float,
                  help="factor for ttHH events", metavar="ttHH")

parser.add_option("--ttH", dest="ttH", default=1.0, type=float,
                  help="factor for ttH events", metavar="ttH")

parser.add_option("--ttZH", dest="ttZH", default=1.0, type=float,
                  help="factor for ttZH events", metavar="ttZH")

parser.add_option("--ttZZ", dest="ttZZ", default=1.0, type=float,
                  help="factor for ttZZ events", metavar="ttZZ")

parser.add_option("--ttZ", dest="ttZ", default=1.0, type=float,
                  help="factor for ttZ events", metavar="ttZ")

parser.add_option("--ttlf", dest="ttlf", default=1.0, type=float,
                  help="factor for ttHH events", metavar="ttlf")

parser.add_option("--ttcc", dest="ttcc", default=1.0, type=float,
                  help="factor for ttHH events", metavar="ttcc")

parser.add_option("--lumi", dest="lumi", default=83, type=float,
                  help="luminosity", metavar="lumi")
parser.add_option("--year", dest="year", default = "2017",
                  help="year", metavar="year")

parser.add_option("--notequalbin", dest="notequalbin", action="store_false",
                  default=True, help="set up equal bin or not", metavar="notequalbin")

parser.add_option("--evaluationEpoch", dest="evaluation_epoch_model", default = None,
                  help="model saved in this epoch used for evaluation", metavar="evaluation_epoch_model")
    
(options, args) = parser.parse_args()

basedir = '/work/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN'
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
dfDirectory = basedir+"/workdir/"+options.dataset+"/"
   
if not os.path.exists(configFile):
        sys.exit(
        "config needed to load trained DNN not found\n{}".format(configFile))

with open(configFile) as f:
        config = f.read()
        config = json.loads(config)

# load samples
# input_samples = data_frame.InputSamples(
#     config["inputData"], addSampleSuffix=config["addSampleSuffix"], test_percentage = options.test_percentage)
input_samples = df.InputSamples(input_path=dfDirectory, dataEra = options.year, addSampleSuffix=config["addSampleSuffix"], test_percentage = options.test_percentage)

# TODO - remove the addSample part because future DNN will save the data df
# TODO - add the dealing with data
# sampleList = ['ttHH','ttZ','ttZZ','ttZH','ttH','ttH2','ttmb','ttmb2','ttnb','ttlf','ttlf2','ttcc','ttcc2']
for sample in config["eventClasses"]:
        total_weight_expr = "x.Weight_XS * x.Weight_CSV_UL * x.Weight_GEN_nom * x.lumiWeight"
        # normalization_weight = 1
        if sample["sampleLabel"] == "ttHH":
                # sample_train_weight = 0.5
                normalization_weight = options.ttHH 
                # normalization_weight = 0.861419355
                # ratio = new cross section (0.6676)/old cross section (0.775)
                # sample_path = dfDirectory+"ttHH_dnn.h5"
        elif sample["sampleLabel"] == "ttZH":
                # sample_train_weight = 1
                normalization_weight = options.ttZH
                # sample_path = dfDirectory+"ttZH_dnn.h5"
        elif sample["sampleLabel"] == "ttZZ":
                # sample_train_weight = 1
                normalization_weight = options.ttZZ
                # sample_path = dfDirectory+"ttZZ_dnn.h5"
        elif sample["sampleLabel"] == "ttZ":
                # sample_train_weight = 1
                normalization_weight = options.ttZ
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
                normalization_weight = options.ttcc
                # sample_path = dfDirectory+"ttcc_dnn.h5"
        elif sample["sampleLabel"] == "ttlf":
                # sample_train_weight = 1
                normalization_weight = options.ttlf
                # sample_path = dfDirectory+"ttlf_dnn.h5"
        elif sample["sampleLabel"] == "ttH":
                # sample_train_weight = 1
                normalization_weight = options.ttH
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
input_path = inPath,
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
                        lumi=options.lumi, year = options.year, category=options.category, equalbin=options.notequalbin)  # 59.7 * 2 , because select only Evt_Odd = 0
#
#        # plot the output nodes
#        dnn.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC, sigScale = -1)
#
#        # plot closure test
#        dnn.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)
