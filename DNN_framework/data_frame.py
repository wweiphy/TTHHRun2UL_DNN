import os
import sys
import pandas as pd
import numpy as np
# import ROOT
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# local import 
# import GenNormMap
filedir = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir = os.path.dirname(DRACOdir)
sys.path.append(basedir)

internal = pd.read_csv(filedir+"/GenNormMap/internalNorm.csv")
internal_ttbb = pd.read_csv(
    filedir+"/GenNormMap/internalNorm_ttbb.csv")
ttbb = pd.read_csv(filedir+"/GenNormMap/fracttbb.csv")
ttbb_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttbb_ttbb.csv")
ttcc = pd.read_csv(filedir+"/GenNormMap/fracttcc.csv")
ttcc_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttcc_ttbb.csv")
ttlf = pd.read_csv(filedir+"/GenNormMap/fracttlf.csv")
ttlf_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttlf_ttbb.csv")

class Sample:
    def __init__(self, path, label, normalization_weight=1., train_weight=1., test_percentage=0.2, total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom', addSampleSuffix=""):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight
        self.isSignal = None
        self.train_weight = train_weight
        self.test_percentage = test_percentage
        self.min = 0.0
        self.max = 1.0
        self.total_weight_expr = total_weight_expr
        self.addSampleSuffix = addSampleSuffix
        # self.Do_Evaluation = Do_Evaluation

    def load_dataframe(self, event_category, lumi, evenSel="", Do_Evaluation=False, jecsysts=None):
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

        # TODO - move the SF calculation into preprocessing.py 
        



        if Do_Evaluation:

            # print("Do DNN Evaluation")

            if "nominal" in self.path:

            #     # nominal values
            #     df = df.assign(sf_weight=lambda x: (x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
                
            #     # btag SF & uncertainties 
            #     df = df.assign(xs_weight=lambda x: eval(self.total_weight_expr))
            #     xs_weight_sum = sum(df["xs_weight"].values)
            #     print("xs weight sum: {}".format(xs_weight_sum))
            #     df = df.assign(train_weight=lambda x: x.xs_weight /
            #                 xs_weight_sum*self.train_weight)
            #     # df = df.assign(total_weight=lambda x: x.xs_weight * x.extra_weight)
            #     df = df.assign(total_weight=lambda x: x.xs_weight * x.sf_weight)
            #     df = df.assign(total_weight_uplf=lambda x: x.Weight_XS *
            #                 x.Weight_CSV_UL_uplf * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downlf=lambda x: x.Weight_XS *
            #                 x.Weight_CSV_UL_downlf * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_uphf=lambda x: x.Weight_XS *
            #                 x.Weight_CSV_UL_uphf * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downhf=lambda x: x.Weight_XS *
            #                 x.Weight_CSV_UL_downhf * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_uplfstats1=lambda x: x.Weight_XS * x.Weight_CSV_UL_uplfstats1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downlfstats1=lambda x: x.Weight_XS * x.Weight_CSV_UL_downlfstats1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_uplfstats2=lambda x: x.Weight_XS * x.Weight_CSV_UL_uplfstats2 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downlfstats2=lambda x: x.Weight_XS * x.Weight_CSV_UL_downlfstats2 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_uphfstats1=lambda x: x.Weight_XS * x.Weight_CSV_UL_uphfstats1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downhfstats1=lambda x: x.Weight_XS * x.Weight_CSV_UL_downhfstats1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_uphfstats2=lambda x: x.Weight_XS * x.Weight_CSV_UL_uphfstats2 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downhfstats2=lambda x: x.Weight_XS * x.Weight_CSV_UL_downhfstats2 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_upcferr1=lambda x: x.Weight_XS * x.Weight_CSV_UL_upcferr1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downcferr1=lambda x: x.Weight_XS * x.Weight_CSV_UL_downcferr1 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_upcferr2=lambda x: x.Weight_XS * x.Weight_CSV_UL_upcferr2 * x.Weight_GEN_nom * x.sf_weight)
            #     df = df.assign(total_weight_downcferr2=lambda x: x.Weight_XS * x.Weight_CSV_UL_downcferr2 * x.Weight_GEN_nom * x.sf_weight)

            #     # PU weights & uncertainties
            #     df = df.assign(total_weight_upPU=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2Up'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     df = df.assign(total_weight_downPU=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2Down'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     # L1 prefiring
            #     df = df.assign(total_weight_upL1Fire=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefireUp'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     df = df.assign(total_weight_downL1Fire=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefireDown'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
            #     # electron trigger SF & uncertainties
            #     df = df.assign(total_weight_downEleTrigger=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF_Down'] > 0)) * 1. * x['Weight_ElectronTriggerSF_Down'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
            #     df = df.assign(total_weight_upEleTrigger=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF_Up'] > 0)) * 1. * x['Weight_ElectronTriggerSF_Up'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
            #     # muon trigger SF & uncertainties
            #     df = df.assign(total_weight_downMuonTrigger=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF_Down'] > 0.)) * 1. * x['Weight_MuonTriggerSF_Down'])))
                
            #     df = df.assign(total_weight_upMuonTrigger=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF_Up'] > 0.)) * 1. * x['Weight_MuonTriggerSF_Up'])))

            #     # Muon SF & uncertainties
            #     df = df.assign(total_weight_downMuon=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSFDown[0]'] > 0.) & (x['Muon_ReconstructionSFDown[0]'] > 0.) & (x['Muon_IsolationSFDown[0]'] > 0.))*1.*x['Muon_IdentificationSFDown[0]'] * x['Muon_IsolationSFDown[0]'] * x['Muon_ReconstructionSFDown[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     df = df.assign(total_weight_upMuon=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSFUp[0]'] > 0.) & (x['Muon_ReconstructionSFUp[0]'] > 0.) & (x['Muon_IsolationSFUp[0]'] > 0.))*1.*x['Muon_IdentificationSFUp[0]'] * x['Muon_IsolationSFUp[0]'] * x['Muon_ReconstructionSFUp[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     # Electron SF & uncertainties
            #     df = df.assign(total_weight_downEle=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFDown[0]'] > 0.) & (x['Electron_ReconstructionSFDown[0]'] > 0.))*1.*x['Electron_IdentificationSFDown[0]']*x['Electron_ReconstructionSFDown[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            #     df = df.assign(total_weight_upEle=lambda x: (x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFUp[0]'] > 0.) & (x['Electron_ReconstructionSFUp[0]'] > 0.))*1.*x['Electron_IdentificationSFUp[0]']*x['Electron_ReconstructionSFUp[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
                # isr
                # print(GenNormMap.internalNomFactors['isrUp'][4][1])

                df = df.assign(total_preweight=lambda x: (x['lumiWeight']* x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFUp[0]'] > 0.) & (x['Electron_ReconstructionSFUp[0]'] > 0.))*1.*x['Electron_IdentificationSFUp[0]']*x['Electron_ReconstructionSFUp[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                    (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                
                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_0p5'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_1p0'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_2p0'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_0p5'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_1p0'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_2p0'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_0p5'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_1p0'] = 1.
                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_2p0'] = 1.


                if self.label == "ttlf" or self.label == "ttcc" or self.label == "ttb" or self.label == "ttbb" or self.label == "tt2b" or self.label == "ttmb":
                    df['total_preweight'] = df['total_preweight'] * \
                        df['Weight_scale_variation_muR_0p5_muF_0p5'].abs().le(
                            100.) * 1. * df['Weight_scale_variation_muR_0p5_muF_1p0'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_0p5_muF_2p0'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_1p0_muF_0p5'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_1p0_muF_1p0'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_1p0_muF_2p0'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_2p0_muF_0p5'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_2p0_muF_1p0'].abs().le(100.) * 1. * df['Weight_scale_variation_muR_2p0_muF_2p0'].abs().le(100.) * 1.
            
                for x in range(306000, 306001):
                # for x in range(306000, 306103):

                    if (self.label != "ttH" or self.label != "ttlf" or self.label != "ttcc"):
                        df.loc[:,'Weight_pdf_variation_{}'.format(x)] = 0.

                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(0.)

                    df['total_weight_PDF_Weight_{}'.format(x)] = (df['Weight_pdf_variation_{}'.format(
                        x)]*((df['process'] == "ttSL")*1. * float(internal['ttSL'][internal[internal['Name'] == 'Weight_pdf_variation_{}'.format(x)].index]) + (df['process'] == "ttDL")*1. * float(internal['ttDL'][internal[internal['Name'] == 'Weight_pdf_variation_{}'.format(x)].index]) + (df['process'] == "ttH")*1. * float(internal['ttH'][internal[internal['Name'] == 'Weight_pdf_variation_{}'.format(x)].index]))) * df['total_preweight']*df['compare']*1.

                for x in range(320900, 320901):
                # for x in range(320900, 321001):

                    if (self.label != "ttb" or self.label != "ttbb" or self.label != "tt2b" or self.label != "ttmb"):
                        df.loc[:,'Weight_pdf_variation_{}'.format(x)] = 0.

                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(
                        0.)
                    df['total_weight_PDF_Weight_{}'.format(x)] = (df['Weight_pdf_variation_{}'.format(
                        x)]*((df['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_pdf_variation_{}'.format(x)].index]) + (df['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_pdf_variation_{}'.format(x)].index]))) * df['total_preweight']*df['compare']*1.
                

                # if (self.label != "ttb" or self.label != "ttbb" or self.label != "tt2b" or self.label != "ttmb" or self.label != "ttH" or self.label != "ttcc" or self.label != "ttlf"):
                    
                    
                #     df.loc[:, 'Weight_scale_variation_muR_0p5_muF_0p5'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_0p5_muF_1p0'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_0p5_muF_2p0'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_1p0_muF_0p5'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_1p0_muF_1p0'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_1p0_muF_2p0'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_2p0_muF_0p5'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_2p0_muF_1p0'] = 0.
                #     df.loc[:, 'Weight_scale_variation_muR_2p0_muF_2p0'] = 0.


                df = df.assign(total_weight_scaleMuRUp=lambda x: (((x['process'] == "ttSL")*1. * float(internal['ttSL'][internal[internal['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) * x['Weight_scale_variation_muR_2p0_muF_1p0'] + (x['process'] == "ttDL")*1. * float(internal['ttDL'][internal[internal['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) * x['Weight_scale_variation_muR_2p0_muF_1p0'] + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index])) * x.total_preweight))


                df = df.assign(total_weight_scaleMuR_ttbbNLOUp=lambda x: (((x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) * x['Weight_scale_variation_muR_2p0_muF_1p0'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) * x['Weight_scale_variation_muR_2p0_muF_1p0'] + ((self.label == "ttlf") & (
                        x['process'] == "ttSL"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) + ((self.label == "ttlf") & (x['process'] == "ttDL"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttSL"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttDL"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuR_ttHUp=lambda x: (
                        (float(internal['ttH'][internal[internal['Name'] == 'Weight_scale_variation_muR_2p0_muF_1p0'].index]) * x['Weight_scale_variation_muR_2p0_muF_1p0']) * x.total_preweight))

                df=df.assign(total_weight_scaleMuRDown=lambda x: (((x['process'] == "ttSL")*1. * float(internal['ttSL'][internal[internal['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) * x['Weight_scale_variation_muR_0p5_muF_1p0'] + (x['process'] == "ttDL")*1. * float(internal['ttDL'][internal[internal['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) *
                                 x['Weight_scale_variation_muR_0p5_muF_1p0'] + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuR_ttbbNLODown=lambda x: (((x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) * x['Weight_scale_variation_muR_0p5_muF_1p0'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) * x['Weight_scale_variation_muR_0p5_muF_1p0'] + ((self.label == "ttlf") & (
                        x['process'] == "ttSL"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) + ((self.label == "ttlf") & (x['process'] == "ttDL"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttSL"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttDL"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuR_ttHDown=lambda x: (
                    (float(internal['ttH'][internal[internal['Name'] == 'Weight_scale_variation_muR_0p5_muF_1p0'].index]) * x['Weight_scale_variation_muR_0p5_muF_1p0']) * x.total_preweight))

                df=df.assign(total_weight_scaleMuFUp=lambda x: (((x['process'] == "ttSL")*1. * float(internal['ttSL'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) * x['Weight_scale_variation_muR_1p0_muF_2p0'] + (x['process'] == "ttDL")*1. * float(internal['ttDL'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) *
                                x['Weight_scale_variation_muR_1p0_muF_2p0'] + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuF_ttbbNLOUp=lambda x: (((x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) * x['Weight_scale_variation_muR_1p0_muF_2p0'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) * x['Weight_scale_variation_muR_1p0_muF_2p0'] + ((self.label == "ttlf") & (
                    x['process'] == "ttSL"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) + ((self.label == "ttlf") & (x['process'] == "ttDL"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttSL"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) + ((self.label == "ttcc") & (x['process'] == "ttDL"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuF_ttHUp=lambda x: (
                    (float(internal['ttH'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_2p0'].index]) * x['Weight_scale_variation_muR_1p0_muF_2p0']) * x.total_preweight))

                df=df.assign(total_weight_scaleMuFDown=lambda x: (((x['process'] == "ttSL")*1. * float(internal['ttSL'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) * x['Weight_scale_variation_muR_1p0_muF_0p5'] + (x['process'] == "ttDL")*1. * float(internal['ttDL'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) * x['Weight_scale_variation_muR_1p0_muF_0p5'] + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuF_ttbbNLODown=lambda x: (((x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) * x['Weight_scale_variation_muR_1p0_muF_0p5'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) * x['Weight_scale_variation_muR_1p0_muF_0p5'] + ((self.label == "ttlf") & (
                    x['process'] == "ttSL"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) + ((self.label == "ttlf") & (x['process'] == "ttDL"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) + ((self.label == "ttcc") & (x['process'] == "ttSL"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) + ((self.label == "ttcc") & (x['process'] == "ttDL"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index])) * x.total_preweight))

                df=df.assign(total_weight_scaleMuF_ttHDown=lambda x: (
            (float(internal['ttH'][internal[internal['Name'] == 'Weight_scale_variation_muR_1p0_muF_0p5'].index]) * x['Weight_scale_variation_muR_1p0_muF_0p5']) * x.total_preweight))

                df=df.assign(total_weight_upisr_ttH=lambda x: ((x['process'] == "ttH") * 1. * float(
                    internal['ttH'][internal[internal['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up'] * x.total_preweight))

                df=df.assign(total_weight_upisr_ttlf= lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_isr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up'] + (
                    (x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(ttbb_ttbb['ttbbSL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_isr_Def_up'].index]) + (x['process'] == "ttbbDL")*1. * float(ttbb_ttbb['ttbbDL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_isr_Def_up'].index])) * x.total_preweight))

                df= df.assign(total_weight_upisr_ttbb=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_up'].index]) + ((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_up'].index]) + (
                    (x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up']) * x.total_preweight))

                df=df.assign(total_weight_upisr_ttcc=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up'] + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_isr_Def_up'].index]) * x['GenWeight_isr_Def_up'] + (
                    (x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_up'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_up'].index])) * x.total_preweight))

                df=df.assign(total_weight_downisr_ttH=lambda x: (
            (x['process'] == "ttH") * 1. * float(internal['ttH'][internal[internal['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down'] * x.total_preweight))

                df=df.assign(total_weight_downisr_ttlf=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_isr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down'] + (
                    (x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(ttbb_ttbb['ttbbSL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_isr_Def_down'].index]) + (x['process'] == "ttbbDL")*1. * float(ttbb_ttbb['ttbbDL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_isr_Def_down'].index])) * x.total_preweight))

                df=df.assign(total_weight_downisr_ttbb = lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_down'].index]) + ((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_down'].index]) + (
                    (x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_isr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down']) * x.total_preweight))

                df = df.assign(total_weight_downisr_ttcc=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down'] + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_isr_Def_down'].index]) * x['GenWeight_isr_Def_down'] + (
                    (x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_isr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_down'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_isr_Def_down'].index])) * x.total_preweight))



                # fsr

                df=df.assign(total_weight_upfsr_ttH=lambda x: ((x['process'] == "ttH") * 1. * float(
                    internal['ttH'][internal[internal['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up'] * x.total_preweight))

                df=df.assign(total_weight_upfsr_ttlf=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_fsr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up'] + (
                    (x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(ttbb_ttbb['ttbbSL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index]) + (x['process'] == "ttbbDL")*1. * float(ttbb_ttbb['ttbbDL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index])) * x.total_preweight))

                df=df.assign(total_weight_upfsr_ttbb=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_up'].index]) + ((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_up'].index]) + (
                    (x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up']) * x.total_preweight))

                df=df.assign(total_weight_upfsr_ttcc=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up'] + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_fsr_Def_up'].index]) * x['GenWeight_fsr_Def_up'] + (
                    (x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_up'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_up'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_up'].index])) * x.total_preweight))

                df=df.assign(total_weight_downfsr_ttH=lambda x: ((x['process'] == "ttH") * 1. * float(
                    internal['ttH'][internal[internal['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down'] * x.total_preweight))

                df=df.assign(total_weight_downfsr_ttlf=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_fsr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down'] + (
                    (x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(ttbb_ttbb['ttbbSL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index]) + (x['process'] == "ttbbDL")*1. * float(ttbb_ttbb['ttbbDL'][ttbb_ttbb[ttbb_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index])) * x.total_preweight))

                df=df.assign(total_weight_downfsr_ttbb=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_down'].index]) + ((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(ttcc['ttSL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_down'].index]) + (
                    (x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(ttcc['ttDL'][ttcc[ttcc['Name'] == 'GenWeight_fsr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down'] + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down']) * x.total_preweight))

                df=df.assign(total_weight_downfsr_ttcc=lambda x: ((((x['process'] == "ttSL") & (self.label == "ttcc"))*1. * float(internal['ttSL'][internal[internal['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down'] + ((x['process'] == "ttDL") & (self.label == "ttcc"))*1. * float(internal['ttDL'][internal[internal['Name'] == 'GenWeight_fsr_Def_down'].index]) * x['GenWeight_fsr_Def_down'] + (
                    (x['process'] == "ttSL") & (self.label == "ttlf"))*1. * float(ttlf['ttSL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_down'].index]) + ((x['process'] == "ttDL") & (self.label == "ttlf"))*1. * float(ttlf['ttDL'][ttlf[ttlf['Name'] == 'GenWeight_fsr_Def_down'].index]) + (x['process'] == "ttbbSL")*1. * float(internal_ttbb['ttbbSL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index]) + (x['process'] == "ttbbDL")*1. * float(internal_ttbb['ttbbDL'][internal_ttbb[internal_ttbb['Name'] == 'GenWeight_fsr_Def_down'].index])) * x.total_preweight))




                

            # else:

            #     # nominal values
            #     df = df.assign(sf_weight=lambda x: (x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #         (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

            # df = df.assign(sf_weight=lambda x: (x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
            #     (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))


            df = df.assign(xs_weight=lambda x: eval(
                self.total_weight_expr))
            xs_weight_sum = sum(df["xs_weight"].values)
            print("xs weight sum: {}".format(xs_weight_sum))
            df = df.assign(train_weight=lambda x: x.xs_weight /
                            xs_weight_sum*self.train_weight)
            # df = df.assign(total_weight=lambda x: x.xs_weight * x.extra_weight)
            df = df.assign(
                total_weight=lambda x: x.xs_weight * x.sf_weight)
                


        else:
            # print("total weight: ")
            # print("total weight: {}".format(df["total_weight"].values))
            df = df.assign(total_weight=lambda x: eval(self.total_weight_expr))
            print("total weight: {}".format(df["total_weight"].values))
            # assign train weight
            weight_sum = sum(df["total_weight"].values)
            print("weight sum: {}".format(weight_sum))
            print("self train weight: {}".format(self.train_weight))
            df = df.assign(train_weight=lambda x: x.total_weight /
                        weight_sum*self.train_weight)
            print("sum of train weights: {}".format(
                sum(df["train_weight"].values)))

            # add lumi weight

        df = df.assign(lumi_weight=lambda x: x.total_weight *
                        lumi * self.normalization_weight * x.lumiWeight)
        print("sum of lumi weights: {}".format(
            sum(df["lumi_weight"].values)))
        self.data = df
        print("-"*50)

        if self.addSampleSuffix in self.label:
            df["class_label"] = pd.Series(
                [c + self.addSampleSuffix for c in df["class_label"].values], index=df.index)
                


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
    def __init__(self, input_path, test_percentage=0.2, addSampleSuffix=""):
        self.binary_classification = False
        self.input_path = input_path
        self.samples = []
        self.addSampleSuffix = addSampleSuffix
        # print ("test percentage is {}".format(test_percentage))
        self.test_percentage = float(test_percentage)
        if self.test_percentage <= 0. or self.test_percentage >= 1.:
            sys.exit("fraction of events to be used for testing (test_percentage) set to {}. this is not valid. choose something in range (0.,1.)")

    def addSample(self, sample_path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path

        self.samples.append(Sample(sample_path, label, normalization_weight, train_weight,
                            self.test_percentage, total_weight_expr=total_weight_expr, addSampleSuffix=self.addSampleSuffix))
        # print("sample path is "+sample_path)

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
                 addSampleSuffix="",
                 Do_Evaluation = False):

        self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel
        self.save_path = save_path
        self.input_samples = input_samples
        self.train_variables = train_variables
        self.test_percentage = test_percentage
        self.shuffleSeed = shuffleSeed
        self.addSampleSuffix = addSampleSuffix
        self.Do_Evaluation = Do_Evaluation

        self.binary_classification = input_samples.binary_classification
        if self.binary_classification:
            self.bkg_target = input_samples.bkg_target

    def loadDatasets(self):

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in self.input_samples.samples:

            sample.load_dataframe(self.event_category,
                                  self.lumi, self.evenSel, self.Do_Evaluation)
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
                                  
            print("class translation: ")
            print(self.class_translation)

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

        with open(self.save_path+"/output_classes.txt", "w") as txt_file:
            for line in self.output_classes:
                txt_file.write(" ".join(line) + "\n")
        # save this classes dictionary for later evaluation
        json.dump(self.class_translation, open(
            self.save_path+"/class_translation.txt", 'w'))

        print("total events after cuts:  "+ \
              str(self.df_unsplit_preprocessing.shape[0]))
        print("events used for training: "+str(self.df_train.shape[0]))
        print("events used for testing:  "+str(self.df_test.shape[0]))
        del df

        # save dataframe after preprocessing
        # print("save preprocessed events")
        # outFile_df = self.save_path+"/"+"df.h5" 
        # outFile_df_train = self.save_path+"/"+"df_train.h5" 
        # outFile_df_test = self.save_path+"/"+"df_test.h5" 

# TODO - deal with the warning for saving df
# TODO - add ttHH ODD and EVEN selections (I think it's already there)
        # self.saveDatasets(self.df_unsplit_preprocessing, outFile_df)
        # self.saveDatasets(self.df_train, outFile_df_train)
        # self.saveDatasets(self.df_test, outFile_df_test)
        return self

    def saveDatasets(self, df, outFile):
            print("save dataset after preprocessing in {}".format(outFile))
            # df.to_hdf(outFile, key='df', mode='w')
            # print("successfully saved the dataset after preprocessing")

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
        if as_categorical:
            return to_categorical(self.df_train["index_label"].values)
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
        if as_categorical:
            return to_categorical(self.df_test["index_label"].values)
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

    def get_upEle_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upEle"].values
    def get_downEle_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downEle"].values
    def get_upMuon_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upMuon"].values
    def get_downMuon_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downEle"].values
    def get_upEleTrigger_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upEleTrigger"].values
    def get_downEleTrigger_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downEleTrigger"].values
    def get_upMuonTrigger_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upMuonTrigger"].values
    def get_downMuonTrigger_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downMuonTrigger"].values
    def get_upPU_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upPU"].values
    def get_downPU_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downPU"].values
    def get_upL1Fire_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upL1Fire"].values
    def get_downL1Fire_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downL1Fire"].values

    def get_upISR_weights_ttH_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upisr_ttH"].values

    def get_downISR_weights_ttH_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downisr_ttH"].values

    def get_upFSR_weights_ttH_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upfsr_ttH"].values

    def get_downFSR_weights_ttH_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downfsr_ttH"].values

    def get_upISR_weights_ttlf_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upisr_ttlf"].values

    def get_downISR_weights_ttlf_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downisr_ttlf"].values

    def get_upFSR_weights_ttlf_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upfsr_ttlf"].values

    def get_downFSR_weights_ttlf_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downfsr_ttlf"].values

    def get_upISR_weights_ttcc_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upisr_ttcc"].values

    def get_downISR_weights_ttcc_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downisr_ttcc"].values

    def get_upFSR_weights_ttcc_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upfsr_ttcc"].values

    def get_downFSR_weights_ttcc_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downfsr_ttcc"].values

    def get_upISR_weights_ttbb_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upisr_ttbb"].values

    def get_downISR_weights_ttbb_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downisr_ttbb"].values

    def get_upFSR_weights_ttbb_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upfsr_ttbb"].values

    def get_downFSR_weights_ttbb_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downfsr_ttbb"].values

    def get_upMuF_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuFUp"].values
    def get_upMuF_ttbb_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuF_ttbbNLOUp"].values
    def get_upMuF_ttH_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuF_ttHUp"].values

    def get_downMuF_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuFDown"].values
    def get_downMuF_ttbb_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuF_ttbbNLODown"].values

    def get_downMuF_ttH_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuF_ttHDown"].values

    def get_upMuR_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuRUp"].values
    def get_upMuR_ttbb_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuR_ttbbNLOUp"].values
    def get_upMuR_ttH_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuR_ttHUp"].values

    def get_downMuR_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuRDown"].values
    def get_downMuR_ttbb_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuR_ttbbNLODown"].values
    def get_downMuR_ttH_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_scaleMuR_ttHDown"].values

    def get_uplf_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uplf"].values
    def get_downlf_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downlf"].values
    def get_uphf_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uphf"].values
    def get_downhf_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downhf"].values
    def get_uplfstats1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uplfstats1"].values
    def get_downlfstats1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downlfstats1"].values
    def get_uplfstats2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uplfstats2"].values
    def get_downlfstats2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downlfstats2"].values
    def get_uphfstats1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uphfstats1"].values
    def get_downhfstats1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downhfstats1"].values
    def get_uphfstats2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_uphfstats2"].values
    def get_downhfstats2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downhfstats2"].values
    def get_upcferr1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upcferr1"].values
    def get_downcferr1_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downcferr1"].values
    def get_upcferr2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_upcferr2"].values
    def get_downcferr2_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["total_weight_downcferr2"].values
    

    def get_full_lumi_weights_after_preprocessing(self):
        return self.df_unsplit_preprocessing["lumi_weight"].values

    def get_full_train_weights(self):
        return self.df_unsplit_preprocessing["train_weight"].values

    def get_full_labels_after_preprocessing(self, as_categorical=True):
        if as_categorical:
            return to_categorical(self.df_unsplit_preprocessing["index_label"].values)
        else:
            return self.df_unsplit_preprocessing["index_label"].values

    def get_class_flag(self, class_label):
        return pd.Series([1 if c == class_label else 0 for c in self.df_test["class_label"].values], index=self.df_test.index).values
