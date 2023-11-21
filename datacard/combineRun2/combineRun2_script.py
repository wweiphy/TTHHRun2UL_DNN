import os
import sys
import optparse
import pandas as pd
import ROOT


# python combineRun2_script.py --flavor -c 5j4b

usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

# parser.add_option("-twoyear", "--twoyear", dest="twoyear", action = "store_true", default=False,
#         help="combine 18 with 17", metavar="twoyear")

parser.add_option("--threeyear", dest="threeyear", action = "store_true", default=False,
        help="combine 18 with 17 and 16", metavar="threeyear")

parser.add_option("-f", "--flavor", dest="flavor", action = "store_true", default=False, help="use 4FS on ttnb or not", metavar="flavor")


parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="selection cateogiry", metavar="category")

(options, args) = parser.parse_args()

processlist = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']


filedir = os.path.dirname(os.path.realpath(__file__))
combdir = os.path.dirname(filedir)
basedir = os.path.dirname(combdir)
sys.path.append(basedir)


folder_path = basedir + "/workdir/"


if options.flavor:

    histofile = "histo-name-4FS.csv"

    file1path = "230220_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
    file2path = "230119_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
    outFolder = 'TwoYear-4FS'+options.category


else:

    histofile = "histo-name.csv"

    file1path = "230220_evaluation_new_"+options.category+"/plots/output_limit.root"
    file2path = "230119_evaluation_new_"+options.category+"/plots/output_limit.root"
    outFolder = 'TwoYear'+options.category


file1 = ROOT.TFile.Open(folder_path+file1path)
file2 = ROOT.TFile.Open(folder_path+file2path)

if options.threeyear:

    if options.flavor:

        file3path = "230515_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        file4path = "230523_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        outFolder = 'ThreeYear-4FS'+options.category


    else:

        file3path = "230515_evaluation_new_"+options.category+"/plots/output_limit.root"
        file4path = "230523_evaluation_new_"+options.category+"/plots/output_limit.root"
        outFolder = 'ThreeYear'+options.category

    file3 = ROOT.TFile.Open(folder_path+file3path)
    file4 = ROOT.TFile.Open(folder_path+file4path)



if not os.path.exists(outFolder):
    os.mkdir(outFolder)


df = pd.read_csv(histofile, index_col=None)

systlist = df['Uncertainty'].tolist()

output_file = ROOT.TFile(outFolder+"/output_limit.root", "RECREATE")

for node in processlist:

    print("start combining node "+node)

    for process in processlist:

        histoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process

        hist1 = file1.Get(histoname)
        hist2 = file2.Get(histoname)

        combined_hist = hist1.Clone()
        combined_hist.Add(hist2)

        if not hist1 or not hist2:
            print("Error: Unable to load histograms for 17 or 18.")
            exit(1)

        if options.threeyear:

            hist3 = file3.Get(histoname)
            hist4 = file4.Get(histoname)

            if not hist3 or not hist4:
                print("Error: Unable to load histograms for 16APV or 16.")
                exit(1)
            combined_hist.Add(hist3)
            combined_hist.Add(hist4)

        combined_hist.Write(histoname)

        
        for sys in systlist:

            if df[(df['Uncertainty']==sys)][process].item() == '1':

                uphistoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+sys+"Up"
                downhistoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+sys+"Down"

                uphist1 = file1.Get(uphistoname)
                uphist2 = file2.Get(uphistoname)
                downhist1 = file1.Get(downhistoname)
                downhist2 = file2.Get(downhistoname)

                upcombined_hist = uphist1.Clone()
                upcombined_hist.Add(uphist2)

                downcombined_hist = downhist1.Clone()
                downcombined_hist.Add(downhist2)

                if not uphist1 or not uphist2 or not downhist1 or not downhist2:
                    print("Error: Unable to load histograms for 17 or 18 systematic "+sys)
                    exit(1)

                if options.threeyear:

                    uphist3 = file3.Get(uphistoname)
                    uphist4 = file4.Get(uphistoname)
                    downhist3 = file3.Get(downhistoname)
                    downhist4 = file4.Get(downhistoname)

                    if not uphist3 or not uphist4 or not downhist3 or not downhist4:
                        print("Error: Unable to load histograms for 16APV or 16 systematic "+sys)
                        exit(1)

                    upcombined_hist.Add(uphist3)
                    upcombined_hist.Add(uphist4)
                    downcombined_hist.Add(downhist3)
                    downcombined_hist.Add(downhist4)

                upcombined_hist.Write(uphistoname)
                downcombined_hist.Write(downhistoname)

output_file.Close()

print("done combining histograms")











