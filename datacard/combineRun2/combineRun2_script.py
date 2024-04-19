import os
import sys
import optparse
import pandas as pd
import ROOT


# python combineRun2_script.py --flavor -c 6j4b_oldtthh --threeyear 

usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

parser.add_option("--twoyear", dest="twoyear", action = "store_true", default=False,
        help="combine 18 with 17", metavar="twoyear")

parser.add_option("--threeyear", dest="threeyear", action = "store_true", default=False,
        help="combine 18 with 17 and 16", metavar="threeyear")

parser.add_option("-f", "--flavor", dest="flavor", action = "store_true", default=False, help="use 4FS on ttnb or not", metavar="flavor")


parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="selection cateogiry", metavar="category")

(options, args) = parser.parse_args()

processlist = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']

decorrelated_systs = ['effTrigger_mu','effTrigger_e','eff_mu','eff_e','btag_hfstats1','btag_hfstats2','btag_lfstats1','btag_lfstats2']

filedir = os.path.dirname(os.path.realpath(__file__))
combdir = os.path.dirname(filedir)
basedir = os.path.dirname(combdir)
sys.path.append(basedir)


folder_path = basedir + "/workdir/"

if options.twoyear:

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

else:
    
    if options.flavor:

        histofile = "histo-name-4FS.csv"

        file1path = "230523_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        file2path = "230515_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        outFolder = '2016-4FS'+options.category


    else:

        histofile = "histo-name.csv"

        file1path = "230523_evaluation_new_"+options.category+"/plots/output_limit.root"
        file2path = "230515_evaluation_new_"+options.category+"/plots/output_limit.root"
        outFolder = '2016'+options.category 

    file1 = ROOT.TFile.Open(folder_path+file1path)
    file2 = ROOT.TFile.Open(folder_path+file2path)

if options.threeyear:


    if options.flavor:

        file1path = "230220_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        file2path = "230119_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"

        file3path = "230515_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        file4path = "230523_evaluation_new_"+options.category+"_4FS/plots/output_limit.root"
        outFolder = 'ThreeYear-4FS'+options.category


    else:

        file1path = "230220_evaluation_new_"+options.category+"/plots/output_limit.root"
        file2path = "230119_evaluation_new_"+options.category+"/plots/output_limit.root"

        file3path = "230515_evaluation_new_"+options.category+"/plots/output_limit.root"
        file4path = "230523_evaluation_new_"+options.category+"/plots/output_limit.root"
        outFolder = 'ThreeYear'+options.category

    file1 = ROOT.TFile.Open(folder_path+file1path)
    file2 = ROOT.TFile.Open(folder_path+file2path)
    file3 = ROOT.TFile.Open(folder_path+file3path)
    file4 = ROOT.TFile.Open(folder_path+file4path)
    
print('file1: '+file1path)
print('file2: '+file2path)



if not os.path.exists(outFolder):
    os.mkdir(outFolder)


df = pd.read_csv(histofile, index_col=None)

systlist = df['Uncertainty'].tolist()

output_file = ROOT.TFile(outFolder+"/output_limit.root", "RECREATE")

for node in processlist:

    print("start combining node "+node)

    datahistname = 'ljets_ge4j_ge3t_'+node+"_node__data_obs"

    datahist1 = file1.Get(datahistname)
    datahist2 = file2.Get(datahistname)

    datacombined_hist = datahist1.Clone()
    datacombined_hist.Add(datahist2)

    if not datahist1 or not datahist2:
        print("Error: Unable to load data histograms for 17 or 18.")
        exit(1)
        
    print(datahistname)
    if options.threeyear:

        datahist3 = file3.Get(datahistname)
        datahist4 = file4.Get(datahistname)

        if not datahist3 or not datahist4:
            print("Error: Unable to load histograms for 16APV or 16.")
            exit(1)
        datacombined_hist.Add(datahist3)
        datacombined_hist.Add(datahist4)

    datacombined_hist.Write(datahistname) 

    for process in processlist:

        histoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process

        hist1 = file1.Get(histoname)
        hist2 = file2.Get(histoname)

        # for decorrelate_syst in decorrelated_systs:

        #     uphistoname1 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2018"+"Up"
        #     downhistoname1 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2018"+"Down"
        #     # print(uphistoname1)
        #     # testhist1 = file1.Get('ljets_ge4j_ge3t_ttHH_node__ttHH__effTrigger_mu_2018Up')
        #     # testhist2 = file2.Get('ljets_ge4j_ge3t_ttHH_node__ttHH__effTrigger_mu_2018Up')
        #     # print(testhist1)
        #     # print(testhist2)
        #     # print('done')
        #     syst_histup1 = file1.Get(uphistoname1)
        #     syst_histdown1 = file1.Get(downhistoname1)
        #     systup1 = syst_histup1.Clone()
        #     systdown1 = syst_histdown1.Clone()

        #     systup1.Write(uphistoname1) 
        #     systdown1.Write(downhistoname1) 

        #     uphistoname2 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2017"+"Up"
        #     downhistoname2 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2017"+"Down"
        #     syst_histup2 = file2.Get(uphistoname2)
        #     syst_histdown2 = file2.Get(downhistoname2)
            
        #     systup2 = syst_histup2.Clone()
        #     systdown2 = syst_histdown2.Clone()

        #     systup2.Write(uphistoname2) 
        #     systdown2.Write(downhistoname2) 


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

        if options.threeyear:

            for decorrelate_syst in decorrelated_systs:

                uphistoname1 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2018"+"Up"
                downhistoname1 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2018"+"Down"
                syst_histup1 = file1.Get(uphistoname1)
                syst_histdown1 = file1.Get(downhistoname1)
                # systup1 = syst_histup1.Clone()
                # systdown1 = syst_histdown1.Clone()

                syst_histup1.Write(uphistoname1) 
                syst_histdown1.Write(downhistoname1) 

                uphistoname2 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2017"+"Up"
                downhistoname2 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2017"+"Down"
                syst_histup2 = file2.Get(uphistoname2)
                syst_histdown2 = file2.Get(downhistoname2)
                
                # systup2 = syst_histup2.Clone()
                # systdown2 = syst_histdown2.Clone()

                syst_histup2.Write(uphistoname2) 
                syst_histdown2.Write(downhistoname2) 
                
                uphistoname3 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2016preVFP"+"Up"
                downhistoname3 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2016preVFP"+"Down"

                syst_histup3 = file3.Get(uphistoname3)
                syst_histdown3 = file3.Get(downhistoname3)

                syst_histup3.Write(uphistoname3) 
                syst_histdown3.Write(downhistoname3) 

                uphistoname4 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2016postVFP"+"Up"
                downhistoname4 = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+decorrelate_syst+"_2016postVFP"+"Down"
                syst_histup4 = file4.Get(uphistoname4)
                syst_histdown4 = file4.Get(downhistoname4)

                syst_histup4.Write(uphistoname4) 
                syst_histdown4.Write(downhistoname4) 

        
        for sys in systlist:

            if sys == "L1Prefiring":

                uphistoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+sys+"Up"
                downhistoname = 'ljets_ge4j_ge3t_'+node+"_node__"+process+"__"+sys+"Down"

                uphist2 = file2.Get(uphistoname)
                downhist2 = file2.Get(downhistoname)

                upcombined_hist = uphist2.Clone()
                downcombined_hist = downhist2.Clone()

                if not uphist2 or not downhist2:
                    print("Error: Unable to load histograms for 17 systematic "+sys)
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

            else:

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











