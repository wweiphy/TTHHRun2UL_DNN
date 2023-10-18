
# 2017

# python plotting_ttbb.py -i Eval_0119_UL_nominal -o ttbb_2017 -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2017 5FS ttmb
# python plotting_ttbb.py -i Eval_0119_UL_nominal_5FS -o ttbb_2017_5FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2018
# python plotting_ttbb.py -i Eval_0308_UL_nominal -o ttbb_2018_test -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2018 5FS ttmb
# python plotting_ttbb.py -i Eval_0308_UL_nominal_5FS -o ttbb_2018_5FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# python plotting_ttbb.py -i Eval_0523_UL_nominal -o ttbb_2016post -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc
# python plotting_ttbb.py -i Eval_0515_UL_nominal -o ttbb_2016pre -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc


# python plotting_ttbb.py -i Control_0409  -o ttbb_4_rescale -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_ttbb.py -i Control_0822_2017  -o ttbb_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc
# python plotting_ttbb.py -i Control_1718  -o ttbb_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_ttbb.py -i Control_1718  -o ttbb_2018 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_ttbb.py -i Control_1718  -o ttbb_1718 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc


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
input_samples.addSample(options.getDefaultName("ttmb_5FS"),  label = "ttmb",  normalization_weight = 1, train_weight = 1, total_weight_expr = weight_expr) # 2018
# input_samples.addSample(options.getDefaultName("ttmb"),  label = "ttmb",  normalization_weight = 5.65, train_weight = 1, total_weight_expr = weight_expr) # 2018
# input_samples.addSample(options.getDefaultName("ttmb"),  label = "ttmb",  normalization_weight = 61.34, train_weight = 1, total_weight_expr = weight_expr) # 2017

# input_samples.addSample(options.getDefaultName("ttmb_2017"),  label = "ttmb",  normalization_weight = 4.26, train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttmb_2017"),  label = "ttmb",  normalization_weight = 4.26 * 82.96, train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttmb_2018"),  label = "ttmb",  normalization_weight = 5.65, train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttmb_2018"),  label = "ttmb",  normalization_weight = 5.65 * 119.66, train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttHH"),  label = "ttHH",  normalization_weight = options.getNomWeight(), train_weight = 1, total_weight_expr = weight_expr)

# init DNN class
# dnn = DNN.DNN(
# save_path=outPath,
# # sample_save_path=sample_save_path,
# input_samples=input_samples,
# lumi = 119.4,
# # lumi = 41.5,
# category_name=config["JetTagCategory"],
# train_variables=config["trainVariables"],
# Do_Evaluation = True,
# shuffle_seed=config["shuffleSeed"],
# addSampleSuffix=config["addSampleSuffix"],
# )

dnn = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    lumi = 119.66,
    # lumi=67.24,  # 2016post
    # lumi = 78.08, # 2016pre
    # lumi=82.96,
    # lumi = 1.,
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    Do_Control=True,
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    # balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection()
    )



dnn.save_DNNInput(node_cls="ttbb_5FS", isData=False) 
