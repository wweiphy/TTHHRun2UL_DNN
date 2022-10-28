import os
import sys
import ROOT
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
# import ROOT

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
pyrootdir = os.path.dirname(filedir)
basedir = os.path.dirname(pyrootdir)
sys.path.append(pyrootdir)
sys.path.append(basedir)
from plot_configs import setupPlots

# save DNN outputs for evaluation 
class saveDiscriminators:
    def __init__(self, data, prediction_vector, predicted_classes, event_classes, nbins, bin_range, event_category, savedir, logscale=False):
        self.data = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = predicted_classes
        self.event_classes = event_classes
        self.n_classes = len(self.event_classes) 
        self.nbins = nbins
        self.bin_range = bin_range
        self.event_category = event_category
        self.savedir = savedir
        self.logscale = logscale

    def save(self):

        # allBKGhists = []
        # allSIGhists = []

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):

            f = ROOT.TFile(self.savedir + "/" + node_cls +
                           "_discriminator" + ".root", "RECREATE")
            print("name of the root file: ")
            print(self.savedir + "/" + node_cls + "_discriminator" + ".root")

            if i >= self.n_classes:
                continue
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"

# TODO - add the loading of class translation in DNN.py for evaluation instead of here
            # get index of node
            # self.class_translation = []
            # with open(self.sample_save_path+"class_translation.txt") as f:
            #     for line in f:
            #         self.class_translation.append(line)
            # print("class_translation: " + self.class_translation)

            # nodeIndex = self.class_translation[node_cls]

            nodeIndex = self.data.class_translation[node_cls]

            # get output values of this node
            out_values = self.prediction_vector[:, i]
#            out_values2 =

            # fill lists according to class
            bkgHists = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                if j >= self.n_classes:
                    continue
                # classIndex = self.class_translation[truth_cls]
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [out_values[k] for k in range(len(out_values))
                                   if self.data.get_full_labels_after_preprocessing(as_categorical=False)[k] == classIndex
                                   and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [self.data.get_full_lumi_weights_after_preprocessing()[k] for k in range(len(out_values))
                                    if self.data.get_full_labels_after_preprocessing(as_categorical=False)[k] == classIndex
                                    and self.predicted_classes[k] == nodeIndex]

                print("{} events in discriminator: {}\t(Integral: {})".format(
                    truth_cls, len(filtered_values), sum(filtered_weights)))

                weightIntegral += sum(filtered_weights)

                histogram = setupPlots.setupHistogram(
                    values=filtered_values,
                    weights=filtered_weights,
                    nbins=self.nbins,
                    bin_range=self.bin_range,
                    #                        color     = setup.GetPlotColor(truth_cls),
                    xtitle="ljets_ge4j_ge3t_" + \
                    str(node_cls)+"_node__"+str(truth_cls),
                    ytitle=setupPlots.GetyTitle(),
                    filled=True)

                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)
#            allBKGhists.append( bkgHists )

            f.cd()
            f.Write()
            f.Close()


class plotConfusionMatrix:
    def __init__(self, data, prediction_vector, event_classes, event_category, plotdir):
        self.data = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis=1)

        self.event_classes = event_classes
        self.n_classes = len(self.event_classes) 

        self.event_category = event_category
        self.plotdir = plotdir

        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical=False), self.predicted_classes)
        

        # default settings
        self.ROCScore = None

    def plot(self, norm_matrix=True, privateWork=False, printROC=False):
        if printROC:
            self.ROCScore = roc_auc_score(
                self.data.get_test_labels(), self.prediction_vector)

        # norm confusion matrix if activated
        if norm_matrix:
            new_matrix = np.empty(
                (self.n_classes, self.n_classes), dtype=np.float64)
            for yit in range(self.n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit, :]))
                for xit in range(self.n_classes):
                    new_matrix[yit, xit] = 1e-9
                    if self.confusion_matrix[yit, xit] != 0:
                        new_matrix[yit, xit] = self.confusion_matrix[yit,
                                                                     xit]/(evt_sum+1e-9)
                    # print("y: ", yit, "x: ", xit,"matrix: ", new_matrix[yit, xit])

            self.confusion_matrix = new_matrix

        # initialize Histogram
        cm = setupPlots.setupConfusionMatrix(
            matrix=self.confusion_matrix.T,
            ncls=self.n_classes,
            xtitle="predicted class",
            ytitle="true class",
            binlabel=self.event_classes)

        canvas = setupPlots.drawConfusionMatrixOnCanvas(
            cm, "confusion matrix", self.event_category, self.ROCScore, privateWork=privateWork)
        setupPlots.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")


class plotDiscriminators:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale=False, sigScale=-1):
        self.data = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis=1)

        self.event_classes = event_classes
        self.n_classes = len(self.event_classes) - \
            self.data.input_samples.additional_samples
        self.nbins = nbins
        self.bin_range = bin_range
        self.signal_class = signal_class
        self.event_category = event_category
        self.plotdir = plotdir
        self.logscale = logscale
        self.sigScale = sigScale
        self.signalIndex = []
        self.signalFlag = []

        if self.signal_class:
            #            for signal in signal_class:

            self.signalIndex.append(self.data.class_translation[signal_class])
            self.signalFlag.append(self.data.get_class_flag(signal_class))

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio=False, printROC=False, privateWork=False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        allBKGhists = []
        allSIGhists = []

        for i in range(50):
            out_values_i = self.prediction_vector[i, :]
            print("\nout_values: '"+str(i))+"'"
#            print ("out_values: ")
            print(out_values_i)
            print(sum(out_values_i))
            print(self.predicted_classes[i])

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            if i >= self.n_classes:
                continue
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"

            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag = self.signalFlag
            else:
                signalIndex = [nodeIndex]
                signalFlag = [self.data.get_class_flag(node_cls)]

            # get output values of this node
            out_values = self.prediction_vector[:, i]

            if self.printROCScore and len(signalIndex) == 1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

            # fill lists according to class
            bkgHists = []
            bkgLabels = []
            weightIntegral = 0

            sig_values = []
            sig_labels = []
            sig_weights = []

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                if j >= self.n_classes:
                    continue
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [out_values[k] for k in range(len(out_values))
                                   if self.data.get_test_labels(as_categorical=False)[k] == classIndex
                                   and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [self.data.get_lumi_weights()[k] for k in range(len(out_values))
                                    if self.data.get_test_labels(as_categorical=False)[k] == classIndex
                                    and self.predicted_classes[k] == nodeIndex]

                print("{} events in discriminator: {}\t(Integral: {})".format(
                    truth_cls, len(filtered_values), sum(filtered_weights)))

                if j in signalIndex:
                    # signal histogram
                    sig_values.append(filtered_values)
                    sig_labels.append(str(truth_cls))
                    sig_weights.append(filtered_weights)
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)

                    histogram = setup.setupHistogram(
                        values=filtered_values,
                        weights=filtered_weights,
                        nbins=self.nbins,
                        bin_range=self.bin_range,
                        color=setup.GetPlotColor(truth_cls),
                        xtitle=str(truth_cls)+" at "+str(node_cls)+" node",
                        ytitle=setup.GetyTitle(self.privateWork),
                        filled=True)

                    bkgHists.append(histogram)

                    bkgLabels.append(truth_cls)
            allBKGhists.append(bkgHists)
            sigHists = []
            scaleFactors = []
            for iSig in range(len(sig_labels)):
                # setup signal histogram
                sigHist = setup.setupHistogram(
                    values=sig_values[iSig],
                    weights=sig_weights[iSig],
                    nbins=self.nbins,
                    bin_range=self.bin_range,
                    color=setup.GetPlotColor(sig_labels[iSig]),
                    xtitle=str(sig_labels[iSig])+" at "+str(node_cls)+" node",
                    ytitle=setup.GetyTitle(self.privateWork),
                    filled=False)

                # set signal histogram linewidth
                sigHist.SetLineWidth(3)

                # set scalefactor
                if self.sigScale == -1:
                    scaleFactor = weightIntegral/(sum(sig_weights[iSig])+1e-9)
                else:
                    scaleFactor = float(self.sigScale)
                allSIGhists.append(sigHist.Clone())
                sigHist.Scale(scaleFactor)
                sigHists.append(sigHist)
                scaleFactors.append(scaleFactor)

            # rescale histograms if privateWork is enabled
            if privateWork:
                for sHist in sigHists:
                    sHist.Scale(1./sHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
            canvas = setup.drawHistsOnCanvas(
                sigHists, bkgHists, plotOptions,
                canvasName=node_cls+" final discriminator")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            for i, h in enumerate(sigHists):
                legend.AddEntry(
                    h, sig_labels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore and len(signalIndex) == 1:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(
                    canvas, plotOptions["ratio"], nodePlot=True)
            else:
                setup.printLumi(canvas, ratio=plotOptions["ratio"])

            # add category label
            setup.printCategoryLabel(
                canvas, self.event_category, ratio=plotOptions["ratio"])

            out_path = self.plotdir + "/finaldiscr_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/finaldiscr_*.pdf " + \
            str(workdir)+"/discriminators.pdf"
        print(cmd)
        os.system(cmd)

        # create combined histos for max Likelihood fit
        h_bkg = np.array([])
        h_sig = np.array([])
        for l_h in allBKGhists:
            h_tmp = l_h[0].Clone()
            h_tmp.Reset()
            for h in l_h:
                h_tmp.Add(h)
            h_bkg = np.concatenate((h_bkg, hist2array(h_tmp)), axis=None)

        for h in allSIGhists:
            h_sig = np.concatenate((h_sig, hist2array(h)), axis=None)
        return h_bkg, h_sig


class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale=False, sigScale=-1):
        self.data = data
        self.prediction_vector = prediction_vector
        self.event_classes = event_classes
        self.n_classes = len(self.event_classes) - \
            self.data.input_samples.additional_samples
        self.nbins = nbins
        self.bin_range = bin_range
        self.signal_class = signal_class
        self.event_category = event_category
        self.plotdir = plotdir
        self.logscale = logscale
        self.sigScale = sigScale
        self.signalIndex = []
        self.signalFlag = []

        if self.signal_class:
            #            for signal in signal_class:
            self.signalIndex.append(self.data.class_translation[signal_class])
            self.signalFlag.append(self.data.get_class_flag(signal_class))

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio=False, printROC=False, privateWork=False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            if i >= self.n_classes:
                continue
            # get output values of this node
            out_values = self.prediction_vector[:, i]

            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag = self.signalFlag
            else:
                signalIndex = [nodeIndex]
                signalFlag = [self.data.get_class_flag(node_cls)]

            if self.printROCScore and len(signalIndex) == 1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

            # fill lists according to class
            bkgHists = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                if j >= self.n_classes:
                    continue
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [out_values[k] for k in range(len(out_values))
                                   if self.data.get_test_labels(as_categorical=False)[k] == classIndex]

                filtered_weights = [self.data.get_lumi_weights()[k] for k in range(len(out_values))
                                    if self.data.get_test_labels(as_categorical=False)[k] == classIndex]

                if j in signalIndex:
                    # signal histogram
                    sig_values = filtered_values
                    sig_label = str(truth_cls)
                    sig_weights = filtered_weights
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)

                    histogram = setup.setupHistogram(
                        values=filtered_values,
                        weights=filtered_weights,
                        nbins=self.nbins,
                        bin_range=self.bin_range,
                        color=setup.GetPlotColor(truth_cls),
                        xtitle=str(truth_cls)+" at "+str(node_cls)+" node",
                        ytitle=setup.GetyTitle(self.privateWork),
                        filled=True)

                    bkgHists.append(histogram)
                    bkgLabels.append(truth_cls)

            # setup signal histogram
            sigHist = setup.setupHistogram(
                values=sig_values,
                weights=sig_weights,
                nbins=self.nbins,
                bin_range=self.bin_range,
                color=setup.GetPlotColor(sig_label),
                xtitle=str(sig_label)+" at "+str(node_cls)+" node",
                ytitle=setup.GetyTitle(self.privateWork),
                filled=False)

            # set signal histogram linewidth
            sigHist.SetLineWidth(3)

            # set scalefactor
            if self.sigScale == -1:
                scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            else:
                scaleFactor = float(self.sigScale)
            sigHist.Scale(scaleFactor)

            # rescale histograms if privateWork enabled
            if privateWork:
                sigHist.Scale(1./sigHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions,
                canvasName=node_cls+" node")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            legend.AddEntry(sigHist, sig_label +
                            " x {:4.0f}".format(scaleFactor), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore and len(signalIndex) == 1:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(
                    canvas, plotOptions["ratio"], nodePlot=True)
            else:
                setup.printLumi(canvas, ratio=plotOptions["ratio"])

            # add category label
            setup.printCategoryLabel(
                canvas, self.event_category, ratio=plotOptions["ratio"])

            out_path = self.plotdir + "/outputNode_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/outputNode_*.pdf " + \
            str(workdir)+"/outputNodes.pdf"
        print(cmd)
        os.system(cmd)


class plotClosureTest:
    def __init__(self, data, test_prediction, train_prediction, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale=False):
        self.data = data
        self.test_prediction = test_prediction
        self.train_prediction = train_prediction

        self.pred_classes_test = np.argmax(self.test_prediction, axis=1)
        self.pred_classes_train = np.argmax(self.train_prediction, axis=1)

        self.event_classes = event_classes
        self.nbins = nbins
        self.bin_range = bin_range
        self.signal_class = signal_class
        self.event_category = event_category
        self.plotdir = plotdir
        self.logscale = logscale
        self.signalIndex = []

        if self.signal_class:
            #            for signal in signal_class:
            self.signalIndex.append(self.data.class_translation[signal_class])

        # generate sub directory
        self.plotdir += "/ClosurePlots/"
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

        # default settings
        self.privateWork = False

    def plot(self, ratio=False, privateWork=False):
        self.privateWork = privateWork

        ksvalues = ""
        for i, node_cls in enumerate(self.event_classes):
            nodeIndex = self.data.class_translation[node_cls]

            test_values = self.test_prediction[:, i]
            train_values = self.train_prediction[:, i]

            for j, process in enumerate(self.event_classes):
                procIndex = self.data.class_translation[process]

                test_values_proc = [test_values[k] for k in range(len(test_values))
                                    if self.data.get_test_labels(as_categorical=False)[k] == procIndex
                                    and self.pred_classes_test[k] == nodeIndex]

                train_values_proc = [train_values[k] for k in range(len(train_values))
                                     if self.data.get_train_labels(as_categorical=False)[k] == procIndex
                                     and self.pred_classes_train[k] == nodeIndex]

                test_weights_proc = [self.data.get_lumi_weights()[k] for k in range(len(test_values))
                                     if self.data.get_test_labels(as_categorical=False)[k] == procIndex
                                     and self.pred_classes_test[k] == nodeIndex]

                train_weights_proc = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values))
                                      if self.data.get_train_labels(as_categorical=False)[k] == procIndex
                                      and self.pred_classes_train[k] == nodeIndex]

                train = setup.setupHistogram(
                    values=train_values_proc,
                    weights=train_weights_proc,
                    nbins=self.nbins*3,
                    bin_range=self.bin_range,
                    color=ROOT.kBlue,
                    xtitle="train {} at {} node".format(process, node_cls),
                    ytitle=setup.GetyTitle(privateWork=True),
                    filled=True)
                if train.Integral() != 0:
                   train.Scale(1./train.Integral())
                train.SetLineWidth(1)
                train.SetFillColorAlpha(ROOT.kBlue, 0.5)

                test = setup.setupHistogram(
                    values=test_values_proc,
                    weights=test_weights_proc,
                    nbins=self.nbins*3,
                    bin_range=self.bin_range,
                    color=ROOT.kRed,
                    xtitle="test {} at {} node".format(process, node_cls),
                    ytitle=setup.GetyTitle(privateWork=True),
                    filled=False)
                if test.Integral() != 0:
                   test.Scale(1./test.Integral())
                test.SetMarkerStyle(20)
                test.SetLineWidth(1)
                test.SetMarkerSize(2)

                plotOptions = {"logscale": self.logscale}

                # init canvas
                canvas = setup.drawClosureTestOnCanvas(
                    train, None, test, None, plotOptions,
                    canvasName="{} at {} node".format(process, node_cls))

                # setup legend
                legend = setup.getLegend()

                legend.SetTextSize(0.02)
                ks = train.KolmogorovTest(test)
                ksvalues += "{}_AT_{}_NODE".format(process,
                                                   node_cls)+", "+str(ks)+"\n"
                # add entries
                legend.AddEntry(train, "train {}".format(process), "F")
                legend.AddEntry(
                    test,  "test {} (KS = {:.3f})".format(process, ks), "L")

                # draw legend
                legend.Draw("same")

                # prit private work label if activated
                if self.privateWork:
                    setup.printPrivateWork(canvas)
                # add category label
                setup.printCategoryLabel(canvas, self.event_category)

                # add private work label if activated
                if self.privateWork:
                    setup.printPrivateWork(
                        canvas, plotOptions["ratio"], nodePlot=True)

                out_path = self.plotdir + \
                    "/closureTest_for_{}_at_{}_node.pdf".format(
                        process, node_cls)
                setup.saveCanvas(canvas, out_path)

        # loop over output nodes
        for i, node_cls in enumerate(self.event_classes):
            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalClass = self.signal_class
            else:
                signalIndex = [nodeIndex]
                signalClass = node_cls

            # get output values of this node
            test_values = self.test_prediction[:, i]
            train_values = self.train_prediction[:, i]

            sig_test_values = [test_values[k] for k in range(len(test_values))
                               if self.data.get_test_labels(as_categorical=False)[k] in signalIndex
                               and self.pred_classes_test[k] == nodeIndex]
            bkg_test_values = [test_values[k] for k in range(len(test_values))
                               if not self.data.get_test_labels(as_categorical=False)[k] in signalIndex
                               and self.pred_classes_test[k] == nodeIndex]

            sig_train_values = [train_values[k] for k in range(len(train_values))
                                if self.data.get_train_labels(as_categorical=False)[k] in signalIndex
                                and self.pred_classes_train[k] == nodeIndex]
            bkg_train_values = [train_values[k] for k in range(len(train_values))
                                if not self.data.get_train_labels(as_categorical=False)[k] in signalIndex
                                and self.pred_classes_train[k] == nodeIndex]

            sig_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values))
                                if self.data.get_test_labels(as_categorical=False)[k] in signalIndex
                                and self.pred_classes_test[k] == nodeIndex]
            bkg_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values))
                                if not self.data.get_test_labels(as_categorical=False)[k] in signalIndex
                                and self.pred_classes_test[k] == nodeIndex]

            sig_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values))
                                 if self.data.get_train_labels(as_categorical=False)[k] in signalIndex
                                 and self.pred_classes_train[k] == nodeIndex]
            bkg_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values))
                                 if self.data.get_train_labels(as_categorical=False)[k] in signalIndex
                                 and self.pred_classes_train[k] == nodeIndex]

            # setup train histograms
            sig_train = setup.setupHistogram(
                values=sig_train_values,
                weights=sig_train_weights,
                nbins=self.nbins,
                bin_range=self.bin_range,
                color=ROOT.kBlue,
                xtitle="signal train at "+str(node_cls)+" node",
                ytitle=setup.GetyTitle(privateWork=True),
                filled=True)
            sig_train.Scale(1./sig_train.Integral())
            sig_train.SetLineWidth(1)
            sig_train.SetFillColorAlpha(ROOT.kBlue, 0.5)

            bkg_train = setup.setupHistogram(
                values=bkg_train_values,
                weights=bkg_train_weights,
                nbins=self.nbins,
                bin_range=self.bin_range,
                color=ROOT.kRed,
                xtitle="bkg train at "+str(node_cls)+" node",
                ytitle=setup.GetyTitle(privateWork=True),
                filled=True)
            bkg_train.Scale(1./bkg_train.Integral())
            bkg_train.SetLineWidth(1)
            bkg_train.SetFillColorAlpha(ROOT.kRed, 0.5)

            # setup test histograms
            sig_test = setup.setupHistogram(
                values=sig_test_values,
                weights=sig_test_weights,
                nbins=self.nbins,
                bin_range=self.bin_range,
                color=ROOT.kBlue,
                xtitle="signal test at "+str(node_cls)+" node",
                ytitle=setup.GetyTitle(privateWork=True),
                filled=False)
            sig_test.Scale(1./sig_test.Integral())
            sig_test.SetLineWidth(1)
            sig_test.SetMarkerStyle(20)
            sig_test.SetMarkerSize(2)

            bkg_test = setup.setupHistogram(
                values=bkg_test_values,
                weights=bkg_test_weights,
                nbins=self.nbins,
                bin_range=self.bin_range,
                color=ROOT.kRed,
                xtitle="bkg test at "+str(node_cls)+" node",
                ytitle=setup.GetyTitle(privateWork=True),
                filled=False)
            bkg_test.Scale(1./bkg_test.Integral())
            bkg_test.SetLineWidth(1)
            bkg_test.SetMarkerStyle(20)
            bkg_test.SetMarkerSize(2)

            plotOptions = {"logscale": self.logscale}

            # init canvas
            canvas = setup.drawClosureTestOnCanvas(
                sig_train, bkg_train, sig_test, bkg_test, plotOptions,
                canvasName="closure test at {} node".format(node_cls))

            # setup legend
            legend = setup.getLegend()

            legend.SetTextSize(0.02)
            ksSig = sig_train.KolmogorovTest(sig_test)
            ksBkg = bkg_train.KolmogorovTest(bkg_test)
            # add entries
            legend.AddEntry(sig_train, "train {}".format(
                "+".join(signalClass)), "F")
            legend.AddEntry(bkg_train, "train bkg", "F")
            legend.AddEntry(sig_test,  "test {} (KS = {:.3f})".format(
                "+".join(signalClass), ksSig), "L")
            legend.AddEntry(
                bkg_test,  "test bkg (KS = {:.3f})".format(ksBkg), "L")

            # draw legend
            legend.Draw("same")

            # prit private work label if activated
            if self.privateWork:
                setup.printPrivateWork(canvas)
            # add category label
            setup.printCategoryLabel(canvas, self.event_category)

            # add private work label if activated
            if self.privateWork:
                setup.printPrivateWork(
                    canvas, plotOptions["ratio"], nodePlot=True)

            out_path = self.plotdir + \
                "/closureTest_at_{}_node.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(os.path.dirname(self.plotdir[:-1]))
        cmd = "pdfunite "+str(self.plotdir)+"/closureTest_*.pdf " + \
            str(workdir)+"/closureTest.pdf"
        print(cmd)
        os.system(cmd)

        outfile = workdir+"/KSValues.txt"
        with open(outfile, "w") as f:
            f.write(ksvalues)
        print("wrote ks values to {}".format(outfile))
        print(ksvalues)


class plotEventYields:
    def __init__(self, data, prediction_vector, event_classes, event_category, signal_class, plotdir, logscale, sigScale=-1):
        self.data = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis=1)

        self.event_classes = event_classes
        self.n_classes = len(self.event_classes) - \
            self.data.input_samples.additional_samples
        self.signal_class = signal_class
        self.signalIndex = []

        if self.signal_class:
            #            for signal in signal_class:
            self.signalIndex.append(self.data.class_translation[signal_class])
        else:
            self.signalIndex = [self.data.class_translation["ttHH"]]

        self.event_category = event_category
        self.plotdir = plotdir

        self.logscale = logscale
        self.sigScale = sigScale

        self.privateWork = False

    def plot(self, privateWork=False, ratio=False):
        self.privateWork = privateWork

        # loop over processes
        sigHists = []
        sigLabels = []
        bkgHists = []
        bkgLabels = []

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}
        yTitle = "event Yield"
        if privateWork:
            yTitle = setup.GetyTitle(privateWork)

        totalBkgYield = 0

        # generate one plot per output node
        for i, truth_cls in enumerate(self.event_classes):
            if i >= self.data.n_output_neurons:
                continue
            classIndex = self.data.class_translation[truth_cls]

            class_yields = []

            # loop over output nodes
            for j, node_cls in enumerate(self.event_classes):
                if j >= self.data.n_output_neurons:
                    continue

                # get output values of this node
                out_values = self.prediction_vector[:, i]

                nodeIndex = self.data.class_translation[node_cls]

                # get yields
                class_yield = sum([self.data.get_lumi_weights()[k] for k in range(len(out_values))
                                   if self.data.get_test_labels(as_categorical=False)[k] == classIndex
                                   and self.predicted_classes[k] == nodeIndex])
                class_yields.append(class_yield)

            if i in self.signalIndex:
                histogram = setup.setupYieldHistogram(
                    yields=class_yields,
                    classes=self.event_classes,
                    n_classes=self.n_classes,
                    xtitle=str(truth_cls)+" event yield",
                    ytitle=yTitle,
                    color=setup.GetPlotColor(truth_cls),
                    filled=False)

                # set signal histogram linewidth
                histogram.SetLineWidth(2)
                sigHists.append(histogram)
                sigLabels.append(truth_cls)

            else:
                histogram = setup.setupYieldHistogram(
                    yields=class_yields,
                    classes=self.event_classes,
                    n_classes=self.n_classes,
                    xtitle=str(truth_cls)+" event yield",
                    ytitle=yTitle,
                    color=setup.GetPlotColor(truth_cls),
                    filled=True)
                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)

                totalBkgYield += sum(class_yields)

        # scale histograms according to options
        scaleFactors = []
        for sig in sigHists:
            if self.sigScale == -1:
                scaleFactors.append(totalBkgYield/sig.Integral())
            else:
                scaleFactors.append(float(self.sigScale))
        if privateWork:
            for sig in sigHists:
                sig.Scale(1./sig.Integral())
            for h in bkgHists:
                h.Scale(1./totalBkgYield)
        else:
            for i, sig in enumerate(sigHists):
                sig.Scale(scaleFactors[i])

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, plotOptions,
            canvasName="event yields per node")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        for i, sig in enumerate(sigHists):
            legend.AddEntry(
                sig, sigLabels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

        # add background entries
        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")

        # draw legend
        legend.Draw("same")

        # add lumi
        setup.printLumi(canvas, ratio=plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(
            canvas, self.event_category, ratio=plotOptions["ratio"])

        out_path = self.plotdir + "/event_yields.pdf"
        setup.saveCanvas(canvas, out_path)
