from re import I
import ROOT

class LeptonSF:
    # TODO- modify the basedir, maybe in the dataframe.py
    def __init__(self, dataera='2017', basedir='/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/'):

        self.dataera = dataera
        self.basedir = basedir

        self.electronLowPtRangeCut = 20.0
        self.electronMaxPt = 150.0
        self.electronMinPt = 20.0
        self.electronMinPtLowPt = 10
        self.electronMaxPtLowPt = 19.9
        self.electronMaxPtHigh = 201.0
        self.electronMaxPtHigher = 499.0
        self.electronMaxEta = 2.49
        self.electronMaxEtaLow = 2.19

        self.muonMaxPt = 119.0
        self.muonMaxPtHigh = 1199.
        self.muonMinPt = 20.0
        self.muonMinPtHigh = 29.0
        self.muonMaxEta = 2.39

        IDinputFileBtoF = self.basedir + \
            "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingTight94X.root"

        if (self.dataera == "2017"):
            TRIGGERinputFile = self.basedir + \
                "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2017_v3.root"
            TRIGGERhistName = "ele28_ht150_OR_ele32_ele_pt_ele_sceta"
        elif (self.dataera == "2018"):
            TRIGGERinputFile = self.basedir + \
                "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2018_v3.root"
            TRIGGERhistName = "ele28_ht150_OR_ele32_ele_pt_ele_sceta"
        elif (self.dataera == "2016"):
            TRIGGERinputFile = self.basedir + \
                "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2016_v4.root"
            TRIGGERhistName = "ele27_ele_pt_ele_sceta"

        GFSinputFile = self.basedir + \
            "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root"
        GFSinputFile_lowEt = self.basedir + \
            "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingRECO_lowEt.root"

        f_IDSFBtoF = ROOT.TFile(IDinputFileBtoF, "READ")
        f_TRIGGERSF = ROOT.TFile(TRIGGERinputFile, "READ")
        f_GFSSF = ROOT.TFile(GFSinputFile, "READ")
        f_GFSSF_lowEt = ROOT.TFile(GFSinputFile_lowEt, "READ")

        self.h_ele_ID_abseta_pt_ratioBtoF = f_IDSFBtoF.Get("EGamma_SF2D")
        self.h_ele_TRIGGER_abseta_pt_ratio = f_TRIGGERSF.Get(TRIGGERhistName)
        self.h_ele_GFS_abseta_pt_ratio = f_GFSSF.Get("EGamma_SF2D")
        self.h_ele_GFS_abseta_pt_ratio_lowEt = f_GFSSF_lowEt.Get("EGamma_SF2D")

# TODO - can we modify the Pt and Eta from single value to df?
    def GetElectronSF(self, electronPt, electronEta, syst, type="Trigger"):


        # IDinputFileBtoF = self.basedir + \
        #     "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingTight94X.root"

        # if (self.dataera == "2017"):
        #     TRIGGERinputFile = self.basedir + \
        #         "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2017_v3.root"
        #     TRIGGERhistName = "ele28_ht150_OR_ele32_ele_pt_ele_sceta"
        # elif (self.dataera == "2018"):
        #     TRIGGERinputFile = self.basedir + \
        #         "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2018_v3.root"
        #     TRIGGERhistName = "ele28_ht150_OR_ele32_ele_pt_ele_sceta"
        # elif (self.dataera == "2016"):
        #     TRIGGERinputFile = self.basedir + \
        #         "/data/triggerSFs/SingleEG_JetHT_Trigger_Scale_Factors_ttHbb2016_v4.root"
        #     TRIGGERhistName = "ele27_ele_pt_ele_sceta"

        # GFSinputFile = self.basedir + \
        #     "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root"
        # GFSinputFile_lowEt = self.basedir + \
        #     "/data/LeptonSFs/egammaEffi.txt_EGM2D_runBCDEF_passingRECO_lowEt.root"

        # f_IDSFBtoF = ROOT.TFile(IDinputFileBtoF, "READ")
        # f_TRIGGERSF = ROOT.TFile(TRIGGERinputFile, "READ")
        # f_GFSSF = ROOT.TFile(GFSinputFile, "READ")
        # f_GFSSF_lowEt = ROOT.TFile(GFSinputFile_lowEt, "READ")

        # h_ele_ID_abseta_pt_ratioBtoF = f_IDSFBtoF.Get("EGamma_SF2D")
        # h_ele_TRIGGER_abseta_pt_ratio = f_TRIGGERSF.Get(TRIGGERhistName)
        # h_ele_GFS_abseta_pt_ratio = f_GFSSF.Get("EGamma_SF2D")
        # h_ele_GFS_abseta_pt_ratio_lowEt = f_GFSSF_lowEt.Get("EGamma_SF2D")


        if (electronPt == 0.0):
            return 1.0
        if (electronEta < 0 and electronEta <= -1 * self.electronMaxEta):
            electronEta = -1 * self.electronMaxEta
        if (electronEta > 0 and electronEta >= self.electronMaxEta):
            electronEta = self.electronMaxEta
        if (type == "Trigger"):
            if (electronEta < 0 and electronEta <= -1 * self.electronMaxEtaLow):
                electronEta = -1 * self.electronMaxEtaLow
            if (electronEta > 0 and electronEta >= self.electronMaxEtaLow):
                electronEta = self.electronMaxEtaLow

        if (electronPt > self.electronLowPtRangeCut):
            if (electronPt >= self.electronMaxPtHigher):
                electronPt = self.electronMaxPtHigher
            if (electronPt < self.electronMinPt):
                electronPt = self.electronMinPt
        else:
            if (electronPt >= self.electronMaxPtLowPt):
                electronPt = self.electronMaxPtLowPt
            if (electronPt < self.electronMinPtLowPt):
                electronPt = self.electronMinPtLowPt

        if (type == "Trigger"):

            # print("pt is {}".format(electronPt))
            # print("eta is {}".format(electronEta))
            thisBin = self.h_ele_TRIGGER_abseta_pt_ratio.FindBin(
                electronPt, electronEta)
            # print("bin number is {}".format(thisBin))
            nomval = self.h_ele_TRIGGER_abseta_pt_ratio.GetBinContent(thisBin)
            # error = h_ele_TRIGGER_abseta_pt_ratio.GetBinError(thisBin)
            # upval = nomval+error
            # downval = nomval-error

            # print("electron SF: {}".format(nomval))

            self.nomval = nomval
            return self.nomval


# TODO - modify the basedir
#
class BTagSF:
    def __init__(self, dataera, basedir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/", nHFptBins=5, nLFptBins=4, nLFetaBins=3, jecsysts=None):
        self.dataera = dataera
        self.basedir = basedir
        self.nHFptBins_ = nHFptBins
        self.nLFptBins_ = nLFptBins
        self.nLFetaBins_ = nLFetaBins
        self.jecsysts = jecsysts
        
        if (dataera == "2017"):
            HFfile = self.basedir + "/data/CSV/sfs_deepjet_2017_hf.root"
            LFfile = self.basedir + "/data/CSV/sfs_deepjet_2017_lf.root"
        elif( dataera == "2018" ):
            HFfile=self.basedir + "/data/CSV/sfs_deepjet_2018_hf.root"
            LFfile=self.basedir + "/data/CSV/sfs_deepjet_2018_lf.root"
        elif( dataera == "2016" ):
            HFfile=self.basedir + "/data/CSV/sfs_deepjet_2016_hf.root"
            LFfile=self.basedir + "/data/CSV/sfs_deepjet_2016_lf.root"
        else:
            print("NO VALID DATAERA CHOSEN!!")
            

        self.f_HF = ROOT.TFile(HFfile, "READ")
        self.f_LF = ROOT.TFile(LFfile, "READ")
        self.fillCSVHistos()

    def fillCSVHistos(self):
        syst_suffix = "final"
        for sys in enumerate(self.jecsysts):
            sys = sys.replace("up", "Up") 
            sys = sys.replace("down", "Down")
            sys = sys.replace("CSV", "") 
            sys = sys.replace("Stats", "stats")

            if not sys == None:
                sys = "_"+sys

            self.h_wgt_hf = []
            self.c_wgt_hf = []
            self.h_wgt_lf = []

            for i in range(self.nHFptBins_):
                name_b = "csv_ratio_Pt{}_Eta0_{}".format(i, syst_suffix+sys)
                if (self.f_HF.GetListOfKeys().Contains(name_b)):
                    self.h_wgt_hf.append(self.f_HF.Get(name_b))
                name_c = "c_csv_ratio_Pt{}_Eta0_{}".format(i, syst_suffix+sys)
                if (self.f_HF.GetListOfKeys().Contains(name_c)):
                    self.c_wgt_hf.append(self.f_HF.Get(name_c))

            for i in range(self.nLFptBins_):
                for j in range(self.nLFetaBins_):
                    name_l = "csv_ratio_Pt{}_Eta{}_{}".format(i, j, syst_suffix+sys)
                    if (self.f_LF.GetListOfKeys().Contains(name_l)):
                        self.h_wgt_lf.append(self.f_LF.Get(name_l))

    def getBTagWeight(self, jetPt, jetEta, jetCSV, jetFlavour, index):
        Wgthf = 1.
        WgtC = 1.
        Wgtlf = 1.

        iPt = -1
        iEta = -1
        # pt binning for heavy flavour jets
        if (abs(jetFlavour) > 3):
            if (jetPt >= 19.99 and jetPt <= 30): 
                iPt = 0
            elif (jetPt > 30 and jetPt <= 50):
                iPt = 1
            elif (jetPt > 50 and jetPt <= 70):
                iPt = 2
            elif (jetPt > 70 and jetPt <= 100):
                iPt = 3
            elif (jetPt > 100):
                iPt = 4
            else:
                iPt = 5
        # pt binning for light flavour jets
        else:
            if (jetPt >= 19.99 and jetPt <= 30):
                iPt = 0;
            elif (jetPt > 30 and jetPt <= 40):
                iPt = 1
            elif (jetPt > 40 and jetPt <= 60):
                iPt = 2
            elif (jetPt > 60):
                iPt = 3
            else:
                iPt = 4

    #  light flavour jets also have eta bins
        if (abs(jetEta) >= 0 and abs(jetEta) < 0.8):
            iEta = 0
        elif (abs(jetEta) >= 0.8 and abs(jetEta) < 1.6):
            iEta = 1
        elif (abs(jetEta) >= 1.6 and abs(jetEta) < 2.5):
        # difference between 2016/2017, nut not neccesary since | eta | <2.4 anyway
            iEta = 2

        # b flavour jet
        if (abs(jetFlavour) == 5):
            # RESET iPt to maximum pt bin(only 5 bins for new SFs)
            if (iPt >= self.nHFptBins_):
                iPt = self.nHFptBins_-1
                # [20-30], [30-50], [50-70], [70, 100] and [100-10000] only 5 Pt bins for hf
            # TODO - fix the index thing for histograms
            if (self.h_wgt_hf[index+iPt]):
                iWgtHF = self.h_wgt_hf[index+iPt].Eval(jetCSV)
                if (iWgtHF != 0): Wgthf *= iWgtHF
        
        # c flavour jet
        elif (abs(jetFlavour) == 4):

            if (iPt >= self.nHFptBins_):
                iPt = self.nHFptBins_-1
                # [20-30], [30-50], [50-70], [70, 100] and [100-10000] only 5 Pt bins for hf
            if (self.c_wgt_hf[index+iPt]):
                iWgtC = self.c_wgt_hf[index+iPt].Eval(jetCSV)
            if (iWgtC != 0): WgtC *= iWgtC
        # light flavour jet
        else:
            if (iPt >= self.nLFptBins_):
                iPt = self.nLFptBins_-1
                # [20-30], [30-40], [40-60] and [60-10000] only 4 Pt bins for lf
            if (self.h_wgt_lf[index+iPt+iEta]):
                iWgtLF = self.h_wgt_lf[index+iPt+iEta].Eval(jetCSV)
            if (iWgtLF != 0): Wgtlf *= iWgtLF
        

        self.WgtTotal = Wgthf * WgtC * Wgtlf

        return self.WgtTotal