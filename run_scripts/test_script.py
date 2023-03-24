
# 2017

# python test_script.py -i Eval_0119_UL_nominal -o Control_test -c ge4j_ge3t --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc


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

import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DNN_framework.DNN as DNN
import DNN_framework.data_frame as df

options.initArguments()


input_samples = df.InputSamples(options.getInputDirectory(), options.getTestPercentage())

weight_expr = "x.Weight_XS * x.Weight_CSV_UL * x.Weight_GEN_nom * x.lumiWeight"
input_samples.addSample(options.getDefaultName("ttHH"),  label = "ttHH",  normalization_weight = options.getNomWeight(), train_weight = 1, total_weight_expr = weight_expr)



print("shuffle seed: {}".format(config["shuffleSeed"]))

#TODO-modify this
sample_save_path = basedir+"/workdir/"
# init DNN class
dnn = DNN.DNN(
save_path=outPath,
# sample_save_path=sample_save_path,
input_samples=input_samples,
lumi = 119.4,
# lumi = 41.5,
category_name=config["JetTagCategory"],
train_variables=config["trainVariables"],
Do_Evaluation = True,
shuffle_seed=config["shuffleSeed"],
addSampleSuffix=config["addSampleSuffix"],
)



dnn.save_DNNInput(log = options.log, privateWork = options.privateWork, printROC = options.printROC) # 59.7 * 2 , because select only Evt_Odd = 0 
