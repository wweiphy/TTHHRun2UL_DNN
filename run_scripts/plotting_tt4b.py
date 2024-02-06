
# 2017

# python plotting_tt4b.py -i Eval_0119_UL_nominal -o tt4b_2017 -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2017 4FS tt4b
# python plotting_tt4b.py -i Eval_0119_UL_nominal_4FS -o tt4b_2017_4FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2017 tt4b from ttbar
# python plotting_tt4b.py -i Eval_0119_UL_nominal_5FS -o tt4b_2017_5FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2018
# python plotting_tt4b.py -i Eval_0308_UL_nominal -o tt4b_2018 -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc --lumi=119.66 --ttnb=1.351130313 

# 2018 4FS tt4b
# python plotting_tt4b.py -i Eval_0308_UL_nominal_4FS -o tt4b_2018_4FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# 2018 tt4b from ttbar
# python plotting_tt4b.py -i Eval_0308_UL_nominal_5FS -o tt4b_2018_5FS -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# python plotting_tt4b.py -i Eval_0523_UL_nominal -o tt4b_2016post -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc
# python plotting_tt4b.py -i Eval_0515_UL_nominal -o tt4b_2016pre -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables -n ge4j_ge3t_ttH --plot --printroc

# python plotting_tt4b.py -i Control_0409  -o tt4b -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_2018_4FS  -o tt4b_4FS -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_0822_2017  -o tt4b_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_2017_4FS  -o tt4b_4FS_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_1718  -o tt4b_2017 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_1718_4FS  -o tt4b_1718_4FS -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_1718  -o tt4b_2018_new -c ge4j_ge3t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc

# python plotting_tt4b.py -i Control_1718  -o tt4b_1718 -c ge4j_2t -v variables -n ge4j_ge3t_ttH --epochs=500 --signalclass=ttHH -f 0.2 -v variables --plot --printroc


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
input_samples.addSample(options.getDefaultName("ttnb"),  label = "ttnb",  normalization_weight = options.getNormttnb(), train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttnb_2017"),  label="ttnb",
                        # normalization_weight=82.96*0.92, train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttnb_2018"),  label = "ttnb",  normalization_weight = 1, train_weight = 1, total_weight_expr = weight_expr)
# input_samples.addSample(options.getDefaultName("ttnb_2018"),  label = "ttnb",  normalization_weight = 119.66*1.35, train_weight = 1, total_weight_expr = weight_expr)

# 2018 4FS
# input_samples.addSample(options.getDefaultName("ttnb"),  label="ttnb",
                        # normalization_weight=3.538023785, train_weight=1, total_weight_expr=weight_expr)  # 4FS 2018
# 2017 4FS
# input_samples.addSample(options.getDefaultName("ttnb"),  label="ttnb",
#                         normalization_weight=38.92963248, train_weight=1, total_weight_expr=weight_expr)  # 4FS 2017

# 17&18 4FS
# input_samples.addSample(options.getDefaultName("ttnb_2018"),  label="ttnb",
                        # normalization_weight=3.538023785 * 119.66, train_weight=1, total_weight_expr=weight_expr)
# input_samples.addSample(options.getDefaultName("ttnb_2017"),  label="ttnb",
                        # normalization_weight=82.96*38.92963248, train_weight=1, total_weight_expr=weight_expr)

# input_samples.addSample(options.getDefaultName("ttnb"),  label = "ttnb",  normalization_weight = 3.61, train_weight = 1, total_weight_expr = weight_expr) # 4FS 2018
# input_samples.addSample(options.getDefaultName("ttnb"),  label = "ttnb",  normalization_weight = 1.35, train_weight = 1, total_weight_expr = weight_expr) # tt4b 2018

# input_samples.addSample(options.getDefaultName("ttnb"),  label = "ttnb",  normalization_weight = 39.43, train_weight = 1, total_weight_expr = weight_expr) # 4FS 2017
# input_samples.addSample(options.getDefaultName("ttnb"),  label = "ttnb",  normalization_weight = 1.32, train_weight = 1, total_weight_expr = weight_expr) # tt4b 2017


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
    # lumi = 82.96,
    lumi=options.getLumi(),
    # lumi = 1.,
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    Do_Control=True,
    # Do_plotting=True,
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    # balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection()
    )



dnn.save_DNNInput(node_cls="tt4b", isData=False) 
