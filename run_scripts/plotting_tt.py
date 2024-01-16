
# 2017

# python plotting_tt.py -i Eval_0119_UL_nominal -o tt_2017 -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# python plotting_tt.py -i Eval_0308_UL_nominal -o tt_2018 -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc --lumi=119.66

# python plotting_tt.py -i Eval_0523_UL_nominal -o tt_2016post -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# python plotting_tt.py -i Eval_0515_UL_nominal -o tt_2016pre -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc


# python plotting_tt.py -i Control_0409 -o tt -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt.py -i Control_0822_2017 -o tt_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt.py -i Control_1718 -o tt_2018 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt.py -i Control_1718 -o tt_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt.py -i Control_1718 -o tt_1718 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc


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
# input_samples.addSample(options.getDefaultName("ttlf_2017"),  label="ttlf",
#                         normalization_weight=1., train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttcc_2017"),  label="ttcc",  normalization_weight=1., train_weight=1, total_weight_expr=weight_expr)

# input_samples.addSample(options.getDefaultName("ttlf_2018"),  label="ttlf",
                        # normalization_weight=1, train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttcc_2018"),  label="ttcc",  normalization_weight=1., train_weight=1, total_weight_expr=weight_expr)

# input_samples.addSample(options.getDefaultName("ttlf_2017"),  label="ttlf",
# normalization_weight=82.96, train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttcc_2017"),  label="ttcc",  normalization_weight=82.96, train_weight=1, total_weight_expr=weight_expr)

# input_samples.addSample(options.getDefaultName("ttlf_2018"),  label="ttlf",
#                         normalization_weight=119.66, train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttcc_2018"),  label="ttcc",  normalization_weight=119.66, train_weight=1, total_weight_expr=weight_expr)


input_samples.addSample(options.getDefaultName("ttlf"),  label="ttlf",
                        normalization_weight=1., train_weight=1, total_weight_expr=weight_expr)
input_samples.addSample(options.getDefaultName("ttcc"),  label="ttcc",
                        normalization_weight=1., train_weight=1, total_weight_expr=weight_expr)

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
    # lumi = 119.66,
    # lumi=67.24,  # 2016post
    # lumi = 78.08, # 2016pre
    lumi = options.lumi,
    # lumi = 1.,
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # Do_Evaluation=False,
    # Do_plotting=True,
    Do_Control=True,
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    # balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection()
    )



dnn.save_DNNInput(node_cls="tt", isData=False) 
