
# 4j3b 50
# python train_template.py -i 0515_DIR_4b -o 2201028_test  --plot --printroc -c ge4j_ge3t --epochs=5 --signalclass=ttHH -f 0.2 -v dnn_variables -n ge4j_ge3t_ttH 


# global imports
#import ROOT
#ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys

#os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4
#os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=4
#os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=4

# option handler
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

# print ("test percentage")
# print (options.getTestPercentage())


# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getTestPercentage())

weight_expr = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight'

weight_expr_ttZZ_ttZH = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight / 2'

weight_expr_ttZ = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight / 0.001571054 * 0.00016654'

weight_expr_ttmb = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight *6.89'

weight_expr_ttnb = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight *1.09'


weight_expr_ttHH = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom * x.lumiWeight * 2'

#weight_expr = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'
#weight_expr = '1'

# define all samples
# only ttH sample needs even/odd splitting for 2017 MC, not sure if ttHH needs it??

#input_samples.addSample(options.getDefaultName("ttHH"),  label = "ttHH",  normalization_weight = options.getNomWeight(), train_weight = 2, total_weight_expr = weight_expr_ttHH)
input_samples.addSample(options.getDefaultName("ttHH"),  label = "ttHH",  normalization_weight = options.getNomWeight(), train_weight = 1, total_weight_expr = weight_expr_ttHH)

#input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
#input_samples.addSample(options.getDefaultName("tt2b"), label = "tt2b", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
#input_samples.addSample(options.getDefaultName("ttb"),  label = "ttb",  normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttmb"),  label = "ttmb",  normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttnb"), label = "ttnb", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
#input_samples.addSample(options.getDefaultName("ttbbb"), label = "ttbbb", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
#input_samples.addSample(options.getDefaultName("tt4b"), label = "tt4b", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttZH"), label = "ttZH", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttZZ"), label = "ttZZ", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
#input_samples.addSample(options.getDefaultName("ttZ"), label = "ttZ", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("ttZbb"), label = "ttZbb", normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# initializing DNN training class
dnn = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    Do_Evaluation   = False,
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    # balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection()
    )

# build DNN model
dnn.build_model(options.getNetConfig())

# perform the training
dnn.train_model(signal_class = options.getSignal())

# evalute the trained model
dnn.eval_model()

# save information
dnn.save_model(sys.argv, filedir, options.getNetConfigName(), get_gradients = options.doGradients())

# save and print variable ranking
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
#dnn.get_weights()

#dnn.get_gradients(
#    is_binary = options.isBinary()
#)

# plotting 
if options.doPlots():
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        dnn.plot_binaryOutput(
            log         = options.doLogPlots(), 
            privateWork = options.isPrivateWork(), 
            printROC    = options.doPrintROC(), 
            bin_range   = bin_range, 
            name        = options.getName())
    else:
        # plot the confusion matrix
        dnn.plot_confusionMatrix(
            privateWork = options.isPrivateWork(), 
            printROC    = options.doPrintROC())

        # plot the output discriminators
        dnn.plot_discriminators(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot the output nodes
#        dnn.plot_outputNodes(
#            log                 = options.doLogPlots(),
#            signal_class        = options.getSignal(),
#            privateWork         = options.isPrivateWork(),
#            printROC            = options.doPrintROC(),
#            sigScale            = options.getSignalScale())
#
#        # plot event yields
#        dnn.plot_eventYields(
#            log                 = options.doLogPlots(),
#            signal_class        = options.getSignal(),
#            privateWork         = options.isPrivateWork(),
#            sigScale            = options.getSignalScale())
#
#        # plot closure test
#        dnn.plot_closureTest(
#            log                 = options.doLogPlots(),
#            signal_class        = options.getSignal(),
#            privateWork         = options.isPrivateWork())
#
#
