import os
import sys
import numpy as np
import pandas as pd
# import ROOT
import array
import math
import generateJTcut as JTcut
import data_frame
import csv
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Derivatives import Inputs, Outputs, Derivatives
import tensorflow as tf
# from tensorflow import keras
# from keras import optimizers
# from keras import models
# from keras import backend
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir = os.path.dirname(DRACOdir)
sys.path.append(basedir)

from evaluationScripts import plottingScripts
# Limit gpu usage
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config = tf.compat.v1.ConfigProto(device_count={
                                  'GPU': 1, 'CPU': 4}, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# TODO WARNING: tensorflow: From / uscms_data/d3/wwei/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/DNN_framework/DNN.py: 38: The name tf.keras.backend.set_session is deprecated. Please use tf.compat.v1.keras.backend.set_session instead.

# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.config.threading.set_inter_op_parallelism_threads(4)

# config = tf.ConfigProto(device_count={
#                                   'GPU': 1, 'CPU': 4}, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
# config.gpu_options.allow_growth = True
# tf.Session(config=config)

class EarlyStopping(keras.callbacks.Callback):
    ''' custom implementation of early stopping
        with options for
            - stopping when val/train loss difference exceeds a percentage threshold
            - stopping when val loss hasnt increased for a set number of epochs '''

    def __init__(self, monitor="loss", value=None, min_epochs=20, stopping_epochs=None, patience=10, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.val_monitor = "val_"+monitor
        self.train_monitor = monitor
        self.patience = patience
        self.n_failed = 0

        self.stopping_epochs = stopping_epochs
        self.best_epoch = 0
        self.best_validation = 999.
        self.min_epochs = min_epochs
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_val = logs.get(self.val_monitor)
        if epoch == 0:
            self.best_validation = current_val
        current_train = logs.get(self.train_monitor)

        if current_val is None or current_train is None:
            print("Early stopping requires {} and {} available".format(
                self.val_monitor, self.train_monitor), RuntimeWarning)

        if current_val < self.best_validation:
            self.best_validation = current_val
            self.best_epoch = epoch

        # check loss by percentage difference
        if self.value:
            if (current_val-current_train)/(current_train) > self.value and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nEpoch {}: early stopping threshold reached".format(epoch))
                self.n_failed += 1
                if self.n_failed > self.patience:
                    self.model.stop_training = True

        # check loss by validation performance increase
        if self.stopping_epochs:
            if self.best_epoch + self.stopping_epochs < epoch and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nValidation loss has not decreased for {} epochs".format(
                        epoch - self.best_epoch))
                self.model.stop_training = True


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, cp_path=None):
        self.cp_path = cp_path

    def on_epoch_end(self, epoch, logs={}):
        if epoch >= 299:  # or save after some epoch, each k-th epoch etc.
            out_file = self.cp_path + "/trained_model_{}.h5py".format(epoch)
            self.model.save(out_file, save_format='h5')


class MonitorSignalToBackground(keras.callbacks.Callback):

    def __init__(self, test_label=None, test_data=None, classes=None, cp_path=None, signal_class=None, test_weights=None):
        #        super(MonitorSignalToBackground, self).__init__()
        super(keras.callbacks.Callback, self).__init__()
        self.test_label = test_label
        self.test_data = test_data
        self.classes = classes
        self.cp_path = cp_path
        self.signal_class = signal_class
        self.test_weights = test_weights

    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch %05d: " % (epoch))
        # get output values of ttHH node

        self.model_prediction_vector = self.model.predict(self.test_data)
        self.predicted_classes = np.argmax(
            self.model_prediction_vector, axis=1)

        for i, node_cls in enumerate(self.classes):
            if self.signal_class[0] == node_cls:
                out_values = self.model_prediction_vector[:, i]
                signal_label = i
                continue

        asimovsig_max = 0.
        unc = 0.1
        cut_max = 0
        for x in range(2, 20):
            cut = x * 0.05
            signal_test_weights = [self.test_weights[k] for k in range(len(out_values))
                                   if self.true_label[k] == signal_label and out_values[k] >= cut]

            background_test_weights = [self.test_weights[k] for k in range(len(out_values))
                                       if self.true_label[k] != signal_label and out_values[k] >= cut]

            s = np.sum(signal_test_weights)
            b = np.sum(background_test_weights)

            print("s and b before optimization: {} and {}".format(len(signal_test_weights), len(background_test_weights)))
            print("s and b: {} and {}".format(s, b))

            if b < 0.1:
                continue
            varb = b*unc*b*unc

#            varb = b
            tot = s + b
            sign_check = tot*math.log(float((tot*(varb+b)))/float(((b*b)+tot*varb)))-(
                1/unc/unc)*math.log(1.+float((varb*s))/float((b*(b+varb))))
            if sign_check <= 0:
                continue
            asimovsig = math.sqrt(2*(tot*math.log(float((tot*(varb+b)))/float(
                ((b*b)+tot*varb)))-(1/unc/unc)*math.log(1.+float((varb*s))/float((b*(b+varb))))))
            print("Z: {}".format(asimovsig))
            print(cut)
            if asimovsig > asimovsig_max:
               asimovsig_max = asimovsig
               cut_max = cut
               print("Z max: {}".format(asimovsig_max))

        monitor = [epoch, asimovsig_max, cut_max]
        with open(self.cp_path+"/monitor.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(monitor)
            f.close()


class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(backend.get_value(
            self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(lr)
        # Set the value back to the optimizer before this epoch starts
        backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


def lr_schedule(previous_lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
#    decaylr = keras.callbacks.LearningRateScheduler(lambda x: 1e-2 * 0.99312 ** x )
#    lr = 1e-03 * 0.99312 ** epoch # 1e-03 - 1e-06
#    lr = 1e-04 * 0.99312 ** epoch # 1e-04 - 1e-07
#    lr = 5e-05 * 0.99312 ** epoch # 5e-05 - 5e-08
    next_lr = previous_lr * 0.99312 # lr = 2e-05 * 0.99312 ** epoch  2e-05 - 3e-08
#    lr = 1e-05 * 0.99312 ** epoch # 1e-05 - 1e-08
    return next_lr


class DNN():
    def __init__(self,
                 save_path,
                 input_samples,
                #  sample_save_path,
                 category_name,
                 train_variables,
                 Do_Evaluation = False,
                 is_Data = False,
                 category_cutString=None,
                 category_label=None,
                 train_epochs=500,
                 test_percentage=0.2,
                 eval_metrics=None,
                 shuffle_seed=None,
                 evenSel=None,
                 addSampleSuffix=""):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples
        # self.sample_save_path = sample_save_path

        # suffix of additional (ttbb) sample
        self.addSampleSuffix = addSampleSuffix
        self.Do_Evaluation = Do_Evaluation

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # name of event category (usually nJet/nTag category)
        self.category_name = category_name

        # string containing event selection requirements;
        # if not specified (default), deduced via JTcut
        self.category_cutString = (
            category_cutString if category_cutString is not None else JTcut.getJTstring(category_name))
        # category label (string);
        # if not specified (default), deduced via JTcut
        self.category_label = (
            category_label if category_label is not None else JTcut.getJTlabel(category_name))

        # selection
        self.evenSel = ""
        self.oddSel = "1."
        if not evenSel == None:
            if evenSel == True:
                self.evenSel = "(Evt_Odd==0)"
                self.oddSel = "(Evt_Odd==1)"
            elif evenSel == False:
                self.evenSel = "(Evt_Odd==1)"
                self.oddSel = "(Evt_Odd==0)"

        # list of input variables
        self.train_variables = train_variables

        # percentage of events saved for testing
        self.test_percentage = test_percentage

        # number of train epochs
        self.train_epochs = train_epochs

        # additional metrics for evaluation of the training process
        self.eval_metrics = eval_metrics

        # load data set
        # if not Do_Evaluation:
        self.cp_path = self.save_path+"/checkpoints/"
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)

        self.data = self._load_datasets(shuffle_seed)
        self.event_classes = self.data.output_classes
        # # else:
        #     if not is_Data:
        #         df = pd.read_hdf(self.sample_save_path+"/df.h5",'df')
        #         self.data = df

        #         # get previously saved event classes and classes translation object
        #         self.event_classes = []
        #         with open(self.sample_save_path+"/output_classes.txt") as f:
        #             for line in f:
        #                 self.event_classes.append(line)
        #         self.data.n_output_neurons = len(self.event_classes)

        #         self.class_translation = json.load(
        #             open(self.sample_save_path+"/class_translation.txt"))
        #         print("class_translation:")
        #         print(self.class_translation)
                

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        # layer names for in and output (needed for c++ implementation)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"

    def _load_datasets(self, shuffle_seed):
        ''' load data set '''
        data = data_frame.DataFrame(
            input_samples=self.input_samples,
            save_path=self.save_path,
            event_category=self.category_cutString,
            train_variables=self.train_variables,
            test_percentage=self.test_percentage,
            shuffleSeed=shuffle_seed,
            evenSel=self.evenSel,
            addSampleSuffix=self.addSampleSuffix,
            Do_Evaluation = self.Do_Evaluation)
        
        return data.loadDatasets()

    def _load_architecture(self, config):
        ''' load the architecture configs '''

        # define default network configuration
        self.architecture = {
            "layers":                   [512,256,128,64],
            "loss_function":            "categorical_crossentropy",
            "Dropout":                  0.3,
            "L1_Norm":                  0.,
            "L2_Norm":                  0,
            "batch_size":               512,
            "optimizer":                optimizers.Adam(learning_rate=2e-5),
            "activation_function":      "leakyrelu",
            "output_activation":        "Softmax",
            "earlystopping_percentage": None,
            "earlystopping_epochs":     None,
            "saveEpoch":                False
        }

        for key in config:
            self.architecture[key] = config[key]

    def build_default_model(self):
        ''' build default straight forward DNN from architecture dictionary '''

        # infer number of input neurons from number of train variables
        number_of_input_neurons     = self.data.n_input_neurons

        # get all the architecture settings needed to build model
        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        l2_regularization_beta      = self.architecture["L2_Norm"]
        l1_regularization_beta      = self.architecture["L1_Norm"]
        output_activation           = self.architecture["output_activation"]

        # define input layer
        Inputs = keras.layers.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)
        X = Inputs

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            if self.architecture["activation_function"] != "leakyrelu":
                X = keras.layers.Dense(
                    units               = nNeurons,
                    activation          = activation_function,
                    # kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
                    kernel_regularizer  = keras.regularizers.l1_l2(l1 = l1_regularization_beta, l2 = l2_regularization_beta), 
                    name                = "DenseLayer_"+str(iLayer)
                    )(X)

            elif self.architecture["activation_function"] == "leakyrelu":
                X = keras.layers.Dense(
                    units=nNeurons,
                    # activation=activation_function,
                    # kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
                    kernel_regularizer=keras.regularizers.l1_l2(
                        l1=l1_regularization_beta, l2=l2_regularization_beta),
                    name="DenseLayer_"+str(iLayer)
                )(X)
                X = keras.layers.LeakyReLU(alpha=0.3)(X)

            # add dropout percentage to layer if activated
            if not dropout == 0:
                X = keras.layers.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)

        # generate output layer
        X = keras.layers.Dense(
            units               = self.data.n_output_neurons,
            activation          = output_activation.lower(),
            # kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
            kernel_regularizer  = keras.regularizers.l1_l2(l1 = l1_regularization_beta, l2 = l2_regularization_beta),
            name                = self.outputName
            )(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    def build_model(self, config=None, model=None, penalty=None):
        ''' build a DNN model
            use options defined in 'config' dictionary '''

        if config:
            self._load_architecture(config)
            print("loading non default net configs")

        if model == None:
            print("building model from config")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss=self.architecture["loss_function"],
            optimizer=self.architecture["optimizer"],
            metrics=self.eval_metrics)

        # save the model
        self.model = model

        # save net information
        out_file = self.save_path+"/model_summary.yml"
        yml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)

    def train_model(self, signal_class= None):
        ''' train the model '''
        # save monitor_z
        header = ["epoch", "Z", "cut"]
        out_file = self.cp_path+"/monitor.csv"
        f = open(out_file, 'w')
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()
        print("create monitor csv file {}".format(out_file))

        callbacks = [CustomLearningRateScheduler(lr_schedule)]

        # add early stopping if activated
        if self.architecture["earlystopping_percentage"] or self.architecture["earlystopping_epochs"]:
            callbacks.append(EarlyStopping(
                                            monitor="loss",
                                            value=self.architecture["earlystopping_percentage"],
                                            min_epochs=500,
                                            stopping_epochs=self.architecture["earlystopping_epochs"],
                                            verbose=1))
        
        if self.architecture["saveEpoch"]:
            callbacks.append(CustomSaver(cp_path=self.cp_path))
            callbacks.append(MonitorSignalToBackground(
                test_label=self.data.get_test_labels(as_categorical=False),
                                                        test_data=self.data.get_test_data(as_matrix=True),
                                                        classes=self.event_classes,
                                                        cp_path=self.cp_path,
                                                        signal_class=signal_class,
                                                        test_weights=self.data.get_lumi_weights()))

        # train main net
        self.trained_model = self.model.fit(
            x=self.data.get_train_data(as_matrix=True),
            y=self.data.get_train_labels(),
            batch_size=self.architecture["batch_size"],
            epochs=self.train_epochs,
            shuffle=True,
            callbacks=callbacks,
            validation_split=0.25,
            sample_weight=self.data.get_train_weights())

    def eval_model(self):
        ''' evaluate trained model '''

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix=True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix=True))
        self.model_train_prediction = self.model.predict(
            self.data.get_train_data(as_matrix=True))

        #figure out ranges
        # self.get_ranges()

        # save predicted classes with argmax
        self.predicted_classes = np.argmax(
            self.model_prediction_vector, axis=1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical=False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(
            self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(
                    metric, self.model_eval[im+1]))

    def save_model(self, argv, execute_dir, netConfigName, get_gradients=False):
        ''' save the trained model '''

        # save executed command
        argv[0] = execute_dir+"/"+argv[0].split("/")[-1]
        execute_string = "python "+" ".join(argv)
        out_file = self.cp_path+"/command.sh"
        with open(out_file, "w") as f:
            f.write(execute_string)
        print("saved executed command to {}".format(out_file))

        # save model as h5py file
        out_file = self.cp_path + "/trained_model.h5py"
        self.model.save(out_file, save_format='h5')
        print("saved trained model at "+str(out_file))

        # save config of model
        model_config = self.model.get_config()
        out_file = self.cp_path + "/trained_model_config"
        with open(out_file, "w") as f:
            f.write(str(model_config))
        print("saved model config at "+str(out_file))

        # save weights of network
        out_file = self.cp_path + "/trained_model_weights.h5"
        self.model.save_weights(out_file)
        print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

        self.netConfig = netConfigName

        # produce json file with configs
        configs = self.architecture
        configs["inputName"] = self.inputName
        configs["outputName"] = self.outputName + \
            "/"+configs["output_activation"]
        configs = {key: configs[key]
                   for key in configs if not "optimizer" in key}

        # more information saving
        configs["inputData"] = self.input_samples.input_path
        configs["eventClasses"] = self.input_samples.getClassConfig()
        configs["JetTagCategory"] = self.category_name
        configs["categoryLabel"] = self.category_label
        configs["Selection"] = self.category_cutString
        configs["trainEpochs"] = self.train_epochs
        configs["trainVariables"] = self.train_variables
        configs["shuffleSeed"] = self.data.shuffleSeed
        configs["trainSelection"] = self.evenSel
        configs["evalSelection"] = self.oddSel
        configs["addSampleSuffix"] = self.addSampleSuffix
        configs["netConfig"] = self.netConfig
        configs["optimizer"] = str(self.architecture["optimizer"])

        # save information for binary DNN
        if self.data.binary_classification:
            configs["binaryConfig"] = {
                "minValue": self.input_samples.bkg_target,
                "maxValue": 1.,
            }

        json_file = self.cp_path + "/net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent=2, separators=(",", ": "))
        print("wrote net configs to "+str(json_file))

        '''  save configurations of variables for plotscript '''
        plot_file = self.cp_path+"/plot_config.csv"
        variable_configs = pd.read_csv(
            basedir+"/TTHHRun2UL_DNN/DNN_framework/plot_configs/variableConfig.csv").set_index("variablename", drop=True)
        variables = variable_configs.loc[self.train_variables]
        variables.to_csv(plot_file, sep=",")
        print("wrote config of input variables to {}".format(plot_file))

        # Serialize the test inputs for the analysis of the gradients
        if get_gradients:
            pickle.dump(self.data.get_test_data(), open(
                self.cp_path+"/inputvariables.pickle", "wb"))

    def get_input_weights(self):
        ''' get the weights of the input layer and sort input variables by weight sum '''

        # get weights
        first_layer = self.model.layers[1]
        weights = first_layer.get_weights()[0]

        self.weight_dict = {}
        for out_weights, variable in zip(weights, self.train_variables):
            w_sum = np.sum(np.abs(out_weights))
            self.weight_dict[variable] = w_sum

        # sort weight dict
        rank_path = self.save_path + "/first_layer_weight_sums.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_sum\n")
            # for key, val in sorted(self.weight_dict.iteritems(), key=lambda k, v: (v, k)): # python3
            for key, val in sorted(self.weight_dict.iteritems(), key=lambda (k, v): (v, k)): # for python2
                #print("{:50s}: {}".format(key, val))
                f.write("{},{}\n".format(key, val))
        print("wrote weight ranking to "+str(rank_path))

    def get_gradients(self, is_binary):

        # Load keras model
        checkpoint_path = self.cp_path+"/trained_model.h5py"
        # get the keras model
        model_keras = keras.models.load_model(checkpoint_path, compile=False)

        # Get TensorFlow graph
        inputs = Inputs(self.train_variables)
        try:
            import net_configs_tensorflow
        except:
            print("Failed to import Tensorflow models.")
            quit()

        try:
            name_keras_model = self.netConfig
            model_tensorflow_impl = getattr(
                net_configs_tensorflow, self.netConfig + "_tensorflow")
        except:
            print(
                "Failed to load TensorFlow version of Keras model {}.".format(
                    name_keras_model))
            quit()

        #  Get weights as numpy arrays, load weights in tensorflow variables and build tensorflow graph with weights from keras model
        model_tensorflow = model_tensorflow_impl(
            inputs.placeholders, model_keras)

        # Load test data
        x_in = pickle.load(open(self.cp_path+"/inputvariables.pickle", "rb"))

        if is_binary:
            outputs = Outputs(model_tensorflow, ['sig'])
            event_classes = ['sig']
        else:
            outputs = Outputs(model_tensorflow, self.event_classes)
            event_classes = self.event_classes

#        sess = tf.Session()
        sess = tf.compat.v1.Session()

#        sess.run(tf.global_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Get operations for first-order derivatives
        deriv_ops = {}
        derivatives = Derivatives(inputs, outputs)
        for class_ in event_classes:
            deriv_ops[class_] = []
            for variable in self.train_variables:
                deriv_ops[class_].append(derivatives.get(class_, [variable]))

        mean_abs_deriv = {}

        for class_ in event_classes:

            # weight = array("f", [-999])
            # deriv_class = np.zeros(
            #     (len(self.data.get_test_data(as_matrix=True)), len(self.train_variables)))
            # weights = np.zeros(len(self.data.get_test_data()))

            # Calculate first-order derivatives
            deriv_values = sess.run(
                deriv_ops[class_],
                feed_dict={
                    inputs.placeholders: x_in
                })  # calculate y accourding to function deriv_ops (= dy/dx) when x = x_in
            # Remove axes of length one from a.
            deriv_values = np.squeeze(deriv_values)

            mean_abs_deriv[class_] = np.average(np.abs(deriv_values), axis=1)

        # Normalize rows
        matrix = np.vstack([mean_abs_deriv[class_]
                           for class_ in event_classes])
        for i_class, class_ in enumerate(event_classes):
            matrix[i_class, :] = matrix[i_class, :] / \
                np.sum(matrix[i_class, :])

        # Make plot
        variables = self.train_variables
        plt.figure(0, figsize=(len(variables), len(event_classes)))
        axis = plt.gca()

        print(matrix.shape[0])
        print(matrix.shape[1])

        csvtext = "variable,"+",".join(event_classes)
        for j in range(matrix.shape[1]):
            csvtext += "\n"+variables[j]
            for i in range(matrix.shape[0]):
                csvtext += ",{:.3f}".format(matrix[i, j])
                axis.text(
                    j + 0.5,
                    i + 0.5,
                    '{:.3f}'.format(matrix[i, j]),
                    ha='center',
                    va='center')

        q = plt.pcolormesh(matrix, cmap='Oranges')
        #cbar = plt.colorbar(q)
        #cbar.set_label("mean(abs(Taylor coefficients))", rotation=270, labelpad=20)
        variables_label = [v.replace("_", "\_") for v in variables]
        event_classes_label = [v.replace("_", "\_") for v in event_classes]
        plt.xticks(
            np.array(range(len(variables))) + 0.5, variables_label, rotation='vertical')
        plt.yticks(
            np.array(range(len(event_classes))) + 0.5, event_classes_label, rotation='horizontal')
        plt.xlim(0, len(variables))
        plt.ylim(0, len(event_classes_label))
        output_path = os.path.join(self.cp_path,
                                   "keras_taylor_1D")

        plt.savefig(output_path+".png", bbox_inches='tight')
        print("Save plot to {}.png".format(output_path))
        plt.savefig(output_path+".pdf", bbox_inches='tight')
        print("Save plot to {}.pdf".format(output_path))

        with open(output_path+".csv", "w") as f:
            f.write(csvtext)
        print("wrote coefficient information to {}.csv".format(output_path))

    def load_trained_model(self, inputDirectory, ModelEpoch = None):
        ''' load an already trained model '''
        if not ModelEpoch:
            checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"
        else:
            checkpoint_path = inputDirectory+"/checkpoints/trained_model_{}.h5py".format(ModelEpoch)

        print("import model from directory: {}".format(checkpoint_path))
        # get the keras model
        self.model = keras.models.load_model(checkpoint_path)
        # self.model = models.load_model(checkpoint_path)
        self.model.summary()

        # print(str(configs["optimizer"]))
        # self.model.compile(
        #     loss=configs["loss_function"],
        #     optimizer="adam",
        #     metrics=self.eval_metrics)

        # evaluate whole dataset with keras model
        self.model_eval = self.model.evaluate(
            x=self.data.get_full_data_after_preprocessing(as_matrix=True), 
            y=self.data.get_full_labels_after_preprocessing(),
            sample_weight=self.data.get_full_train_weights())

        # save predictions with keras model
        self.model_prediction_vector = self.model.predict(
            self.data.get_full_data_after_preprocessing(as_matrix=True))

        # save predicted classes with argmax with keras model
        self.predicted_classes = np.argmax(
            self.model_prediction_vector, axis=1)
#        np.argmax: Returns the indices of the maximum values along an axis.

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_full_labels_after_preprocessing(as_categorical=False), self.predicted_classes)

        # print evaluations  with keras model
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(
            self.data.get_full_labels_after_preprocessing(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

    def save_discriminators(self, log=False, printROC=False, privateWork=False,
                            signal_class=None, nbins=None, bin_range=None, lumi=41.5, sigScale=-1):
        ''' plot all events classified as one category '''
        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons, 2), 1.]
        if not nbins:
            nbins = int(25*(1.-bin_range[0]))

        saveDiscrs = plottingScripts.savenominalDiscriminators(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            predicted_classes=self.predicted_classes,
            event_classes=self.event_classes,
            nbins=nbins,
            bin_range=bin_range,
            event_category=self.category_label,
            savedir=self.plot_path,
            lumi=lumi,
            logscale=log)

        saveDiscrs.save()

    def save_JESJERdiscriminators(self, log=False, printROC=False, syst = "JESup", privateWork=False,
                            signal_class=None, nbins=None, bin_range=None,
                            sigScale=-1):
        ''' plot all events classified as one category '''
        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons, 2), 1.]
        if not nbins:
            nbins = int(25*(1.-bin_range[0]))

        saveDiscrs = plottingScripts.saveJESJERDiscriminators(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            predicted_classes=self.predicted_classes,
            event_classes=self.event_classes,
            nbins=nbins,
            bin_range=bin_range,
            event_category=self.category_label,
            savedir=self.plot_path,
            syst = syst,
            logscale=log)

        saveDiscrs.save()

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------

    def plot_metrics(self, privateWork=False):
        plt.rc('text', usetex=True)

        ''' plot history of loss function and evaluation metrics '''
        metrics = ["loss"]
        if self.eval_metrics:
            metrics += self.eval_metrics

        # loop over metrics and generate matplotlib plot
        for metric in metrics:
            plt.clf()
            plt.figure(figsize=(10, 8))  # added by Wei
            # get history of train and validation scores
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1, n_epochs+1, 1)

            # plot histories
            plt.plot(epochs, train_history, "b-", label="train", lw=2)
            plt.plot(epochs, val_history, "r-", label="validation", lw=2)
            if privateWork:
                plt.title("CMS private work", loc="left", fontsize=16)

            # add title
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc="right", fontsize=16)

            # make it nicer
            plt.grid()
            plt.xlabel("epoch", fontsize=16)
            plt.ylabel(metric.replace("_", " "), fontsize=16)
            # plt.ylim(ymin=0.)

            # add legend
            plt.legend()

            # save
            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

    def plot_outputNodes(self, log=False, printROC=False, signal_class=None,
                         privateWork=False, nbins=30, bin_range=[0., 1.],
                         sigScale=-1):
        ''' plot distribution in outputNodes '''
        plotNodes = plottingScripts.plotOutputNodes(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            event_classes=self.event_classes,
            nbins=nbins,
            bin_range=bin_range,
            signal_class=signal_class,
            event_category=self.category_label,
            plotdir=self.plot_path,
            logscale=log,
            sigScale=sigScale)

        plotNodes.plot(ratio=False, printROC=printROC, privateWork=privateWork)

    def plot_discriminators(self, log=False, printROC=False, privateWork=False,
                            signal_class=None, nbins=None, bin_range=None,
                            sigScale=-1):
        ''' plot all events classified as one category '''
        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons, 2), 1.]
        if not nbins:
            nbins = int(25*(1.-bin_range[0]))

        plotDiscrs = plottingScripts.plotDiscriminators(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            event_classes=self.event_classes,
            nbins=nbins,
            bin_range=bin_range,
            signal_class=signal_class,
            event_category=self.category_label,
            plotdir=self.plot_path,
            logscale=log,
            sigScale=sigScale)

        bkg_hist, sig_hist = plotDiscrs.plot(
            ratio=False, printROC=printROC, privateWork=privateWork)
        #print("ASIMOV: mu=0: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 0))
        #print("ASIMOV: mu=1: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 1))

    def plot_confusionMatrix(self, norm_matrix=True, privateWork=False, printROC=False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            event_classes=self.event_classes,
            event_category=self.category_label,
            plotdir=self.save_path)

        plotCM.plot(norm_matrix=norm_matrix,
                    privateWork=privateWork, printROC=printROC)

    def plot_closureTest(self, log=False, privateWork=False,
                         signal_class=None, nbins=None, bin_range=None):
        ''' plot comparison between train and test samples '''

        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons, 2), 1.]
        if not nbins:
            nbins = int(20*(1.-bin_range[0]))

        closureTest = plottingScripts.plotClosureTest(
            data=self.data,
            test_prediction=self.model_prediction_vector,
            train_prediction=self.model_train_prediction,
            event_classes=self.event_classes,
            nbins=nbins,
            bin_range=bin_range,
            signal_class=signal_class,
            event_category=self.category_label,
            plotdir=self.plot_path,
            logscale=log)

        closureTest.plot(ratio=False, privateWork=privateWork)

    def plot_eventYields(self, log= False, privateWork = False, signal_class = None, sigScale = -1):
        eventYields = plottingScripts.plotEventYields(
            data=self.data,
            prediction_vector=self.model_prediction_vector,
            event_classes=self.event_classes,
            event_category=self.category_label,
            signal_class=signal_class,
            plotdir=self.save_path,
            logscale=log)

        eventYields.plot(privateWork=privateWork)
