import os
import sys
import optparse
import pandas as pd
import ROOT


# python combineRun2_script.py -c 5j4b --threeyear 
# python combineRun2_script.py -c 6j4b_2 

usage = "usage=%prog [options] \n"
usage += "USE: python cardmakingscript.py -n True "

parser = optparse.OptionParser(usage=usage)

# parser.add_option("--twoyear", dest="twoyear", action = "store_true", default=False,
#         help="combine 18 with 17", metavar="twoyear")

parser.add_option("-y", "--year", dest="year", default="2017",
        help="year", metavar="year")


parser.add_option("-c", "--category", dest="category",default="5j4b",
        help="selection cateogiry", metavar="category")

(options, args) = parser.parse_args()

processlist = ['ttlf', 'ttcc', 'ttmb', 'ttHH','ttH','ttZ','ttZH','ttZZ','ttnb']
# processlist2 = ['ttbar', 'ttHH','ttH','ttZ','ttZH','ttZZ','ttnb']

# decorrelated_systs = ['effTrigger_mu','effTrigger_e','eff_mu','eff_e','btag_hfstats1','btag_hfstats2','btag_lfstats1','btag_lfstats2','JER']

filedir = os.path.dirname(os.path.realpath(__file__))
combdir = os.path.dirname(filedir)
basedir = os.path.dirname(combdir)
sys.path.append(basedir)


folder_path = basedir + "/workdir/"

files = ['230220','230119','230515','230523']

for file in files:


    filepath = file+"_evaluation_new_"+options.category+"/plots/output_limit.root"

    file = ROOT.TFile.Open(folder_path+filepath, "UPDATE")
    print('file: '+filepath)


    df = pd.read_csv(histofile, index_col=None)

    systlist = df['Uncertainty'].tolist()


    datahistname = 'ljets_ge4j_ge3t_ttbar_node__data_obs'

    datahistname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__data_obs'
    datahistname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__data_obs'
    datahistname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__data_obs'


    datahist_ttlf = file.Get(datahistname_ttlf)
    datahist_ttcc = file.Get(datahistname_ttcc)
    datahist_ttmb = file.Get(datahistname_ttmb)

    datacombined_hist = datahist_ttlf.Clone()
    datacombined_hist.Add(datahist_ttcc)
    datacombined_hist.Add(datahist_ttmb)

    datacombined_hist.Write(datahistname)   

    for node in processlist:

        histoname = 'ljets_ge4j_ge3t_'+node+"_node__ttbar"

        histoname_ttlf = 'ljets_ge4j_ge3t_'+node+"_node__ttlf"
        histoname_ttcc = 'ljets_ge4j_ge3t_'+node+"_node__ttcc"
        histoname_ttmb = 'ljets_ge4j_ge3t_'+node+"_node__ttmb"

        hist_ttlf = file.Get(histoname_ttlf)
        hist_ttcc = file.Get(histoname_ttcc) 
        hist_ttmb = file.Get(histoname_ttmb) 

        combined_hist = hist_ttlf.Clone()
        combined_hist.Add(hist_ttcc)
        combined_hist.Add(hist_ttmb)

        combined_hist.Write(histoname)
            
        for sys in systlist:

            if sys == "L1Prefiring":

                if options.year == "2018":

                    continue

                else:

                    uphistoname = 'ljets_ge4j_ge3t_'+node+"_node__ttbar__"+sys+"Up"
                    downhistoname = 'ljets_ge4j_ge3t_'+node+"_node__ttbar__"+sys+"Down"

                    uphistoname_ttlf = 'ljets_ge4j_ge3t_'+node+"_node__ttlf__"+sys+"Up"
                    downhistoname_ttlf = 'ljets_ge4j_ge3t_'+node+"_node__ttlf__"+sys+"Down"

                    uphistoname_ttcc = 'ljets_ge4j_ge3t_'+node+"_node__ttcc__"+sys+"Up"
                    downhistoname_ttcc = 'ljets_ge4j_ge3t_'+node+"_node__ttcc__"+sys+"Down"

                    uphistoname_ttmb = 'ljets_ge4j_ge3t_'+node+"_node__ttmb__"+sys+"Up"
                    downhistoname_ttmb = 'ljets_ge4j_ge3t_'+node+"_node__ttmb__"+sys+"Down"

                    uphist_ttlf = file.Get(uphistoname_ttlf)
                    uphist_ttcc = file.Get(uphistoname_ttcc)
                    uphist_ttmb = file.Get(uphistoname_ttmb)
                    downhist_ttlf = file.Get(downhistoname_ttlf)
                    downhist_ttcc = file.Get(downhistoname_ttcc)
                    downhist_ttmb = file.Get(downhistoname_ttmb)

                    upcombined_hist = uphist_ttlf.Clone()
                    upcombined_hist.Add(uphist_ttcc)
                    upcombined_hist.Add(uphist_ttmb)

                    downcombined_hist = downhist_ttlf.Clone()
                    downcombined_hist.Add(downhist_ttcc)
                    downcombined_hist.Add(downhist_ttmb)


                    upcombined_hist.Write(uphistoname)
                    downcombined_hist.Write(downhistoname)

            else:

                if df[(df['Uncertainty']==sys)][process].item() == '1':

                    uphistoname = 'ljets_ge4j_ge3t_'+node+"_node__ttbar__"+sys+"Up"
                    downhistoname = 'ljets_ge4j_ge3t_'+node+"_node__ttbar__"+sys+"Down"

                    uphistoname_ttlf = 'ljets_ge4j_ge3t_'+node+"_node__ttlf__"+sys+"Up"
                    downhistoname_ttlf = 'ljets_ge4j_ge3t_'+node+"_node__ttlf__"+sys+"Down"

                    uphistoname_ttcc = 'ljets_ge4j_ge3t_'+node+"_node__ttcc__"+sys+"Up"
                    downhistoname_ttcc = 'ljets_ge4j_ge3t_'+node+"_node__ttcc__"+sys+"Down"

                    uphistoname_ttmb = 'ljets_ge4j_ge3t_'+node+"_node__ttmb__"+sys+"Up"
                    downhistoname_ttmb = 'ljets_ge4j_ge3t_'+node+"_node__ttmb__"+sys+"Down"

                    uphist_ttlf = file.Get(uphistoname_ttlf)
                    uphist_ttcc = file.Get(uphistoname_ttcc)
                    uphist_ttmb = file.Get(uphistoname_ttmb)
                    downhist_ttlf = file.Get(downhistoname_ttlf)
                    downhist_ttcc = file.Get(downhistoname_ttcc)
                    downhist_ttmb = file.Get(downhistoname_ttmb)

                    upcombined_hist = uphist_ttlf.Clone()
                    upcombined_hist.Add(uphist_ttcc)
                    upcombined_hist.Add(uphist_ttmb)

                    downcombined_hist = downhist_ttlf.Clone()
                    downcombined_hist.Add(downhist_ttcc)
                    downcombined_hist.Add(downhist_ttmb)


                    upcombined_hist.Write(uphistoname)
                    downcombined_hist.Write(downhistoname)

    # file.Close()

    print("done combining process")


    # filepath = file+"_evaluation_new_"+options.category+"/plots/output_limit.root"

    # file = ROOT.TFile.Open(folder_path+filepath, "UPDATE")
    # # print('file: '+filepath)


    # # for node2 in processlist2:

    histoname = 'ljets_ge4j_ge3t_ttbar_node__ttbar'

    histoname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__ttbar'
    histoname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__ttbar'
    histoname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__ttbar'

    hist_ttlf = file.Get(histoname_ttlf)
    hist_ttcc = file.Get(histoname_ttcc) 
    hist_ttmb = file.Get(histoname_ttmb) 

    combined_hist = hist_ttlf.Clone()
    combined_hist.Add(hist_ttcc)
    combined_hist.Add(hist_ttmb)

    combined_hist.Write(histoname)
        
    for sys in systlist:

        if sys == "L1Prefiring":

            if options.year == "2018":

                continue

            else:

                uphistoname = 'ljets_ge4j_ge3t_ttbar_node__ttbar__'+sys+"Up"
                downhistoname = 'ljets_ge4j_ge3t_ttbar_node__ttbar__'+sys+"Down"

                uphistoname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__ttbar__'+sys+"Up"
                downhistoname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__ttbar__'+sys+"Down"

                uphistoname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__ttbar__'+sys+"Up"
                downhistoname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__ttbar__'+sys+"Down"

                uphistoname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__ttbar__'+sys+"Up"
                downhistoname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__ttbar__'+sys+"Down"

                uphist_ttlf = file.Get(uphistoname_ttlf)
                uphist_ttcc = file.Get(uphistoname_ttcc)
                uphist_ttmb = file.Get(uphistoname_ttmb)
                downhist_ttlf = file.Get(downhistoname_ttlf)
                downhist_ttcc = file.Get(downhistoname_ttcc)
                downhist_ttmb = file.Get(downhistoname_ttmb)

                upcombined_hist = uphist_ttlf.Clone()
                upcombined_hist.Add(uphist_ttcc)
                upcombined_hist.Add(uphist_ttmb)

                downcombined_hist = downhist_ttlf.Clone()
                downcombined_hist.Add(downhist_ttcc)
                downcombined_hist.Add(downhist_ttmb)


                upcombined_hist.Write(uphistoname)
                downcombined_hist.Write(downhistoname)

        else:

            if df[(df['Uncertainty']==sys)][process].item() == '1':

                uphistoname = 'ljets_ge4j_ge3t_ttbar_node__ttbar__'+sys+"Up"
                downhistoname = 'ljets_ge4j_ge3t_ttbar_node__ttbar__'+sys+"Down"

                uphistoname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__ttbar__'+sys+"Up"
                downhistoname_ttlf = 'ljets_ge4j_ge3t_ttlf_node__ttbar__'+sys+"Down"

                uphistoname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__ttbar__'+sys+"Up"
                downhistoname_ttcc = 'ljets_ge4j_ge3t_ttcc_node__ttbar__'+sys+"Down"

                uphistoname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__ttbar__'+sys+"Up"
                downhistoname_ttmb = 'ljets_ge4j_ge3t_ttmb_node__ttbar__'+sys+"Down"

                uphist_ttlf = file.Get(uphistoname_ttlf)
                uphist_ttcc = file.Get(uphistoname_ttcc)
                uphist_ttmb = file.Get(uphistoname_ttmb)
                downhist_ttlf = file.Get(downhistoname_ttlf)
                downhist_ttcc = file.Get(downhistoname_ttcc)
                downhist_ttmb = file.Get(downhistoname_ttmb)

                upcombined_hist = uphist_ttlf.Clone()
                upcombined_hist.Add(uphist_ttcc)
                upcombined_hist.Add(uphist_ttmb)

                downcombined_hist = downhist_ttlf.Clone()
                downcombined_hist.Add(downhist_ttcc)
                downcombined_hist.Add(downhist_ttmb)

                upcombined_hist.Write(uphistoname)
                downcombined_hist.Write(downhistoname)

    print("done with merging nodes")













