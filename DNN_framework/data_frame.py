import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler


class Sample:
    def __init__(self, path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom', addSampleSuffix=""):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight
        self.isSignal = None
        self.train_weight = train_weight
        self.min = 0.0
        self.max = 1.0
        self.total_weight_expr = total_weight_expr
        self.addSampleSuffix = addSampleSuffix

    def load_dataframe(self, event_category, lumi, evenSel=""):
        # loading samples from one .h5 file or mix it with one uncertainty variation (default is without mixing)
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore(self.path, mode="r") as store:
            df = store.select("data")
            samp = int(df.shape[0]*1.0)
#                df = df.astype('float64') # added by Wei
            df = df.head(samp)
            print("number of events before selections: "+str(df.shape[0]))

        # apply event category cut
        query = event_category

        if not evenSel == "":
            query += " and "+evenSel
        df.query(query, inplace=True)
        print("number of events after selections:  "+str(df.shape[0]))
        self.nevents = df.shape[0]

        # add event weight
        df = df.assign(total_weight=lambda x: eval(self.total_weight_expr))
        print("total weight: {}".format(df["total_weight"].values))
        # assign train weight
        weight_sum = sum(df["total_weight"].values)
        print("weight sum: {}".format(weight_sum))
        print("self train weight: {}".format(self.train_weight))
        df = df.assign(train_weight=self.train_weight)
        print("sum of train weights: {}".format(
            sum(df["train_weight"].values)))

        if self.addSampleSuffix in self.label:
            df["class_label"] = pd.Series(
                [c + self.addSampleSuffix for c in df["class_label"].values], index=df.index)

        # add lumi weight

        df = df.assign(lumi_weight=lambda x: x.total_weight *
                       lumi * self.normalization_weight / self.train_weight)
        print("sum of lumi weights: {}".format(sum(df["lumi_weight"].values)))
        self.data = df
        print("-"*50)


    def getConfig(self):
        config = {}
        config["sampleLabel"] = self.label
        config["samplePath"] = self.path
        config["sampleWeight"] = self.normalization_weight
        config["sampleEvents"] = self.nevents
        config["min"] = self.min
        config["max"] = self.max
        return config


class InputSamples:
    def __init__(self, input_path, activateSamples=None, test_percentage=0.2, addSampleSuffix=""):
        self.binary_classification = False
        self.input_path = input_path
        self.samples = []
        self.addSampleSuffix = addSampleSuffix
        self.test_percentage = float(test_percentage)
        if self.test_percentage <= 0. or self.test_percentage >= 1.:
            sys.exit("fraction of events to be used for testing (test_percentage) set to {}. this is not valid. choose something in range (0.,1.)")

    def addSample(self, sample_path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path

        self.samples.append(Sample(sample_path, label, normalization_weight, train_weight, total_weight_expr=total_weight_expr, addSampleSuffix=self.addSampleSuffix))

    def getClassConfig(self):
        configs = []
        for sample in self.samples:
            configs.append(sample.getConfig())
        return configs

    def addBinaryLabel(self, signals, bkg_target):
        self.binary_classification = True
        self.signal_classes = signals
        self.bkg_target = float(bkg_target)
        for sample in self.samples:
            if sample.label in signals:
                sample.isSignal = True
            else:
                sample.isSignal = False


class DataFrame(object):
    ''' takes a path to a folder where one h5 per class is located
        the events are cut according to the event_category
        variables in train_variables are used as input variables
        the dataset is shuffled and split into a test and train sample according to test_percentage
        for better training '''

    def __init__(self,
                 input_samples,
                 save_path,
                 event_category,
                 train_variables,
                 test_percentage=0.2,
                 lumi=41.5,
                 shuffleSeed=None,
                 evenSel="",
                 addSampleSuffix=""):

        self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel
        self.save_path = save_path
        self.input_samples = input_samples
        self.train_variables = train_variables
        self.test_percentage = test_percentage
        self.shuffleSeed = shuffleSeed
        self.addSampleSuffix = addSampleSuffix

        self.binary_classification = input_samples.binary_classification
        if self.binary_classification:
            self.bkg_target = input_samples.bkg_target

    def loadDatasets(self):

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in self.input_samples.samples:
            sample.load_dataframe(self.event_category,
                                  self.lumi, self.evenSel)
            train_samples.append(sample.data)

        # concatenating all dataframes
        df = pd.concat(train_samples, sort=True)
        del train_samples

        # multiclassification labelling
        if not self.binary_classification:
            # add class_label translation
            index = 0
            self.class_translation = {}
            self.classes = []

            for sample in self.input_samples.samples:
                self.class_translation[sample.label] = index
                self.classes.append(sample.label)
                index += 1
            self.index_classes = [self.class_translation[c]
                                  for c in self.classes]
                                  
            print("class translation: "+self.class_translation)
            with open(self.save_path+"class_translation.txt", "w") as txt_file:
                for line in self.class_translation:
                    txt_file.write(" ".join(line) + "\n")

            df["index_label"] = pd.Series(
                [self.class_translation[c] for c in df["class_label"].values], index=df.index)

            # save some meta data about network
            self.n_input_neurons = len(self.train_variables)
            self.n_output_neurons = len(
                self.classes)

        # binary classification labelling
        else:

            # class translations
            self.class_translation = {}
            self.class_translation["sig"] = 1
            self.class_translation["bkg"] = float(self.bkg_target)

            self.classes = ["sig", "bkg"]
            self.index_classes = [self.class_translation[c]
                                  for c in self.classes]

            df["index_label"] = pd.Series(
                [1 if c in self.input_samples.signal_classes else 0 for c in df["class_label"].values], index=df.index)

            # add_bkg_df = None
            bkg_df = df.query("index_label == 0")
            sig_df = df.query("index_label == 1")

            signal_weight = sum(sig_df["train_weight"].values)
            bkg_weight = sum(bkg_df["train_weight"].values)
            sig_df["train_weight"] = sig_df["train_weight"] / \
                (2*signal_weight)*df.shape[0]
            bkg_df["train_weight"] = bkg_df["train_weight"] / \
                (2*bkg_weight)*df.shape[0]

            sig_df["binaryTarget"] = 1.
            bkg_df["binaryTarget"] = float(self.bkg_target)

            df = pd.concat([sig_df, bkg_df])
            print("True")

            self.n_input_neurons = len(self.train_variables)
            self.n_output_neurons = 1

        # shuffle dataframe
        if not self.shuffleSeed:
           self.shuffleSeed = np.random.randint(low=0, high=2**16)

        print("using shuffle seed {} to shuffle input data".format(self.shuffleSeed))

        df = shuffle(df, random_state=self.shuffleSeed)

        self.unsplit_df = df.copy()
        
        # normal splitting
        # X = df[train_variables].values
        # Y = df["index_label"].values
        # X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        #     X, Y, test_size=test_percentage, random_state=None)
        
        # customized splitting
        n_test_samples = int(df.shape[0]*self.test_percentage)
        df_test = df.head(n_test_samples)
        df_train = df.tail(df.shape[0] - n_test_samples)

        print("start preprocessing")

        QTScaler = QuantileTransformer(
            n_quantiles=2000, output_distribution='uniform', random_state=0)
        MScaler = MinMaxScaler(feature_range=(0, 1))

        df_final_train = df_train.copy(deep=True)
        df_final_test = df_test.copy(deep=True)

        df_final_train[self.train_variables] = MScaler.fit_transform(
            QTScaler.fit_transform(df_train[self.train_variables]))
        df_final_test[self.train_variables] = MScaler.transform(
            QTScaler.transform(df_test[self.train_variables]))

        print("end preprocessing")

        self.df_unsplit_preprocessing = pd.concat(
            [df_final_test, df_final_train])

        # adjust weights via 1/test_percentage for test and 1/(1 - test_percentage) for train samples such that yields in plots correspond to complete dataset

        df_final_train["lumi_weight"] = df_train["lumi_weight"] / \
            (1 - self.test_percentage)
        df_final_test["lumi_weight"] = df_test["lumi_weight"] / self.test_percentage

        self.df_test = df_final_test
        self.df_train = df_final_train

        # save variable lists
        self.output_classes = self.classes

        with open(self.save_path+"output_classes.txt", "w") as txt_file:
            for line in self.output_classes:
                txt_file.write(" ".join(line) + "\n")
        
        
        print("total events after cuts:  "+str(df.shape[0]))
        print("events used for training: "+str(self.df_train.shape[0]))
        print("events used for testing:  "+str(self.df_test.shape[0]))
        del df

        # save dataframe after preprocessing
        outFile_df = self.save_path+"/"+"df.h5" 
        outFile_df_train = self.save_path+"/"+"df_train.h5" 
        outFile_df_test = self.save_path+"/"+"df_test.h5" 

        self.saveDatasets(df,outFile_df)
        self.saveDatasets(df_train, outFile_df_train)
        self.saveDatasets(df_test, outFile_df_test)

    def saveDatasets(self, df, outFile):
            print("creating dataset after preprocessing")
            with pd.HDFStore(outFile, "a") as store:
                store.append("data", df, index=False)

    # train data -----------------------------------

    def get_train_data(self, as_matrix=True):
        if as_matrix:
            return self.df_train[self.train_variables].values
        else:
            return self.df_train[self.train_variables]

    def get_train_weights(self):
        return self.df_train["train_weight"].values

    def get_train_labels(self, as_categorical=True):
        if self.binary_classification:
            return self.df_train["binaryTarget"].values
        else:
            return self.df_train["index_label"].values

    def get_train_lumi_weights(self):
        return self.df_train["lumi_weight"].values

    # test data ------------------------------------
    def get_test_data(self, as_matrix=True):
        if as_matrix:
            return self.df_test[self.train_variables].values
        else:
            return self.df_test[self.train_variables]

    def get_all_test_data(self):
        return self.df_test

    def get_test_weights(self):
        return self.df_test["total_weight"].values

    def get_lumi_weights(self):
        return self.df_test["lumi_weight"].values

    def get_test_labels(self, as_categorical=True):
        if self.binary_classification:
            return self.df_test["binaryTarget"].values
        else:
            return self.df_test["index_label"].values

    # full sample after preprocessing ----------------------------------
    def get_full_data_after_preprocessing(self, as_matrix=True):
        if as_matrix:
            return self.df_unsplit_preprocessing[self.train_variables].values
        else:
            return self.df_unsplit_preprocessing[self.train_variables]

    def get_full_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight"].values

    def get_full_lumi_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["lumi_weight"].values

    def get_full_labels_after_preprocessing(self, as_categorical=True):
        return self.df_unsplit_preprocessing["index_label"].values
