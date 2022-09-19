import os
import sys
import uproot
filedir = os.path.dirname(os.path.realpath(__file__))
pyrootdir = os.path.dirname(filedir)
basedir = os.path.dirname(pyrootdir)
sys.path.append(pyrootdir)
sys.path.append(basedir)
from plot_configs import setupPlots

class saveDiscriminators:
    def __init__(self, data, prediction_vector, predicted_classes, event_classes, nbins, bin_range, event_category, savedir, logscale=False, sigScale=-1):
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
        self.sigScale = sigScale

    def save(self):

        # allBKGhists = []
        # allSIGhists = []

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):

            f = uproot.open(self.savedir + "/" + node_cls +
                           "_discriminator" + ".root", "RECREATE")
            print("name of the root file: ")
            print(self.savedir + "/" + node_cls + "_discriminator" + ".root")

            if i >= self.n_classes:
                continue
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"

            # get index of node
            self.class_translation = []
            with open(self.sample_save_path+"class_translation.txt") as f:
                for line in f:
                    self.class_translation.append(line)
            print("class_translation: " + self.class_translation)

            nodeIndex = self.class_translation[node_cls]

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
                classIndex = self.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [out_values[k] for k in range(len(out_values))
                                   if self.get_full_labels_after_preprocessing()[k] == classIndex
                                   and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [self.data.get_full_lumi_weights_after_preprocessing()[k] for k in range(len(out_values))
                                    if self.data.get_full_labels_after_preprocessing()[k] == classIndex
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
                    ytitle=setupPlots.GetyTitle(self.privateWork),
                    filled=True)

                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)
#            allBKGhists.append( bkgHists )

            f.cd()
            f.Write()
            f.Close()
