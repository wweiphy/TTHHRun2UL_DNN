import os
import sys
import optparse

usage = "usage=%prog [options] \n"
usage += "USE: python plottingscript.py -n new_plotting "

# evaluation - discriminators

# 1718

# python plottingscript.py -n new -f TwoYear5j4b -c new_5j4b_sys -j 5 -b 4
# python plottingscript.py -n new -f TwoYear6j4b -c new_6j4b_sys -j 6 -b 4

# 2018

# python plottingscript.py -n new -f 230220_evaluation_new_5j4b -c new_230220_5j4b_sys -j 5 -b 4
# python plottingscript.py -n new -f 230220_evaluation_new_6j4b -c new_230220_6j4b_sys -j 6 -b 4
# python plottingscript.py -n new -f 230220_evaluation_new_5j4b_4FS -c new_230220_5j4b_4FS_sys -j 5 -b 4
# python plottingscript.py -n new -f 230220_evaluation_new_6j4b_4FS -c new_230220_6j4b_4FS_sys -j 6 -b 4



# 2017
# python plottingscript.py -n new -f 230119_evaluation_new_5j4b -c new_230119_5j4b_sys -j 5 -b 4
# python plottingscript.py -n new -f 230119_evaluation_new_6j4b -c new_230119_6j4b_sys -j 6 -b 4
# python plottingscript.py -n new -f 230119_evaluation_new_5j4b_4FS -c new_230119_5j4b_4FS_sys -j 5 -b 4
# python plottingscript.py -n new -f 230119_evaluation_new_6j4b_4FS -c new_230119_6j4b_4FS_sys -j 6 -b 4

# 2016pre
# python plottingscript.py -n new -f 230515_evaluation_new_5j4b_2 -c new_230515_5j4b_sys -j 5 -b 4
# python plottingscript.py -n new -f 230515_evaluation_new_6j4b_2 -c new_230515_6j4b_sys -j 6 -b 4
# python plottingscript.py -n new -f 230515_evaluation_new_2 -c new_230515_sys -j 4 -b 3
# python plottingscript.py -n old -f 230515_evaluation_old_2 -c old_230515_sys -j 4 -b 3

# 2016
# python plottingscript.py -n new -f 230523_evaluation_new_5j4b_2 -c new_230523_5j4b_sys -j 5 -b 4
# python plottingscript.py -n new -f 230523_evaluation_new_6j4b_2 -c new_230523_6j4b_sys -j 6 -b 4
# python plottingscript.py -n new -f 230523_evaluation_new_2 -c new_230523_sys -j 4 -b 3
# python plottingscript.py -n old -f 230523_evaluation_old_2 -c old_230523_sys -j 4 -b 3


# kinematics
# python plottingscript.py -n new_plotting  


parser = optparse.OptionParser(usage=usage)

parser.add_option("-n", "--new", dest="new", default="new",
        help="making datacard for new categorizations, total 9", metavar="new")

parser.add_option("-f", "--filefolder", dest="filefolder", default="230220_evaluation_new",
                  help="file folder name", metavar="filefolder")

parser.add_option("-c", "--cardfolder", dest="cardfolder", default="new_230119_new_sys",
                  help="file folder name", metavar="filefolder")
parser.add_option("-j", "--njets", dest="njets", default=4,
                  help="number of jets selection", metavar="bjets")
parser.add_option("-b", "--nbjets", dest="nbjets", default=3,
                  help="number of bjets selection", metavar="nbjets")

(options, args) = parser.parse_args()

process_new = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttmb', 'ttnb']
process_old = ['ttHH', 'ttH', 'ttZ', 'ttZH',
               'ttZZ', 'ttlf', 'ttcc', 'ttb', 'ttbb', 'tt2b', 'ttbbb', 'tt4b']
# variables = ['N_BTagsM', 'Electron_E[0]', 'Jet_CSV[5]']

filedir = os.path.dirname(os.path.realpath(__file__))
# datacarddir = os.path.dirname(filedir)
basedir = os.path.dirname(filedir)
sys.path.append(basedir)


variables = [
    # 'Evt_CSV_avg',
    # 'Evt_CSV_avg_tagged',
    # 'Evt_CSV_dev',
    # 'Evt_CSV_dev_tagged',
    # 'Evt_CSV_min',
    # 'Evt_CSV_min_tagged',
    # 'Evt_Deta_JetsAverage',
    # 'Evt_Deta_UntaggedJetsAverage',
    # 'Evt_Deta_TaggedJetsAverage',
    # 'Evt_Deta_maxDetaJetJet',
    # 'Evt_Deta_maxDetaJetTag',
    # 'Evt_Deta_maxDetaTagTag',
    # 'Evt_Dr_JetsAverage',
    # 'Evt_Dr_TaggedJetsAverage',
    # 'Evt_Dr_UntaggedJetsAverage',
    # 'Evt_Dr_closestTo91TaggedJets',
    # 'Evt_Dr_maxDrJets',
    # 'Evt_Dr_maxDrTaggedJets',
    # 'Evt_Dr_maxDrUntaggedJets',
    # 'Evt_Dr_minDrJets',
    # 'Evt_Dr_minDrLepJet',
    # 'Evt_Dr_minDrLepTag',
    # 'Evt_Dr_minDrTaggedJets',
    # 'Evt_Dr_minDrUntaggedJets',
    # 'Evt_E_JetsAverage',
    # 'Evt_E_TaggedJetsAverage',
    # 'Evt_Eta_JetsAverage',
    # 'Evt_Eta_TaggedJetsAverage',
    # 'Evt_Eta_UntaggedJetsAverage',
    # 'Evt_HT',
    # 'Evt_HT_jets',
    # 'Evt_HT_tags',
    # 'Evt_HT_wo_MET',
    # 'Evt_JetPt_over_JetE',
    # 'Evt_M2_JetsAverage',
    # 'Evt_M2_TaggedJetsAverage',
    # 'Evt_M2_UntaggedJetsAverage',
    # 'Evt_M2_closestTo125TaggedJets',
    # 'Evt_M2_closestTo91TaggedJets',
    # 'Evt_M2_minDrJets',
    # 'Evt_M2_minDrTaggedJets',
    # 'Evt_M2_minDrUntaggedJets',
    # 'Evt_M3',
    # 'Evt_M3_oneTagged',
    # 'Evt_MHT',
    # 'Evt_MTW',
    # 'Evt_M_JetsAverage',
    # 'Evt_M_TaggedJetsAverage',
    # 'Evt_M_Total',
    # 'Evt_M_UntaggedJetsAverage',
    # 'Evt_M_minDrLepJet',
    # 'Evt_M_minDrLepTag',
    # 'Evt_Pt_JetsAverage',
    # 'Evt_Pt_TaggedJetsAverage',
    # 'Evt_Pt_UntaggedJetsAverage',
    # 'Evt_Pt_minDrJets',
    # 'Evt_Pt_minDrTaggedJets',
    # 'Evt_Pt_minDrUntaggedJets',
    # 'Evt_TaggedJetPt_over_TaggedJetE',
    # 'Evt_aplanarity',
    # 'Evt_aplanarity_jets',
    # 'Evt_aplanarity_tags',
    # 'Evt_blr',
    # 'Evt_blr_transformed',
    # 'Evt_h0',
    # 'Evt_h1',
    # 'Evt_h2',
    # 'Evt_h3',
    # 'Evt_sphericity',
    # 'Evt_sphericity_jets',
    # 'Evt_sphericity_tags',
    # 'Evt_transverse_sphericity',
    # 'Evt_transverse_sphericity_jets',
    # 'Evt_transverse_sphericity_tags',
    # 'RecoHiggs_BJet1_E',
    # #    'RecoHiggs_BJet1_Eta',
    # 'RecoHiggs_BJet1_M',
    # #    'RecoHiggs_BJet1_Phi',
    # 'RecoHiggs_BJet1_Pt',
    # 'RecoHiggs_BJet2_E',
    # #    'RecoHiggs_BJet2_Eta',
    # 'RecoHiggs_BJet2_M',
    # #    'RecoHiggs_BJet2_Phi',
    # 'RecoHiggs_BJet2_Pt',
    # 'RecoHiggs_1_Deta',
    # 'RecoHiggs_1_Dphi',
    # 'RecoHiggs_1_Dr',
    # 'RecoHiggs_1_E',
    # #    'RecoHiggs_1_Eta',
    # 'RecoHiggs_1_M',
    # #    'RecoHiggs_1_Phi',
    # 'RecoHiggs_1_Pt',
    # 'RecoHiggs_1_cosdTheta',
    # 'RecoHiggs_BJet3_E',
    # #    'RecoHiggs_BJet3_Eta',
    # 'RecoHiggs_BJet3_M',
    # #    'RecoHiggs_BJet3_Phi',
    # 'RecoHiggs_BJet3_Pt',
    # 'RecoHiggs_BJet4_E',
    # #    'RecoHiggs_BJet4_Eta',
    # 'RecoHiggs_BJet4_M',
    # #    'RecoHiggs_BJet4_Phi',
    # 'RecoHiggs_BJet4_Pt',
    # 'RecoHiggs_2_Deta',
    # 'RecoHiggs_2_Dphi',
    # 'RecoHiggs_2_Dr',
    # 'RecoHiggs_2_E',
    # #    'RecoHiggs_2_Eta',
    # 'RecoHiggs_2_M',
    # #    'RecoHiggs_2_Phi',
    # 'RecoHiggs_2_Pt',
    # 'RecoHiggs_2_cosdTheta',
    # 'RecoHiggs_Chi2',
    # 'RecoHiggs_logChi2',
    # 'RecoZH_BJet1_E',
    # #    'RecoZH_BJet1_Eta',
    # 'RecoZH_BJet1_M',
    # #    'RecoZH_BJet1_Phi',
    # 'RecoZH_BJet1_Pt',
    # 'RecoZH_BJet2_E',
    # #    'RecoZH_BJet2_Eta',
    # 'RecoZH_BJet2_M',
    # #    'RecoZH_BJet2_Phi',
    # 'RecoZH_BJet2_Pt',
    # 'RecoZH_1_Deta',
    # 'RecoZH_1_Dphi',
    # 'RecoZH_1_Dr',
    # 'RecoZH_1_E',
    # #    'RecoZH_1_Eta',
    # 'RecoZH_1_M',
    # #    'RecoZH_1_Phi',
    # 'RecoZH_1_Pt',
    # 'RecoZH_1_cosdTheta',
    # 'RecoZH_BJet3_E',
    # #    'RecoZH_BJet3_Eta',
    # 'RecoZH_BJet3_M',
    # #    'RecoZH_BJet3_Phi',
    # 'RecoZH_BJet3_Pt',
    # 'RecoZH_BJet4_E',
    # #    'RecoZH_BJet4_Eta',
    # 'RecoZH_BJet4_M',
    # #    'RecoZH_BJet4_Phi',
    # 'RecoZH_BJet4_Pt',
    # 'RecoZH_2_Deta',
    # 'RecoZH_2_Dphi',
    # 'RecoZH_2_Dr',
    # 'RecoZH_2_E',
    # #    'RecoZH_2_Eta',
    # 'RecoZH_2_M',
    # #    'RecoZH_2_Phi',
    # 'RecoZH_2_Pt',
    # 'RecoZH_2_cosdTheta',
    # 'RecoZH_Chi2',
    # 'RecoZH_logChi2',
    # 'RecoZ_BJet1_E',
    # #    'RecoZ_BJet1_Eta',
    # 'RecoZ_BJet1_M',
    # #    'RecoZ_BJet1_Phi',
    # 'RecoZ_BJet1_Pt',
    # 'RecoZ_BJet2_E',
    # #    'RecoZ_BJet2_Eta',
    # 'RecoZ_BJet2_M',
    # #    'RecoZ_BJet2_Phi',
    # 'RecoZ_BJet2_Pt',
    # 'RecoZ_1_Deta',
    # 'RecoZ_1_Dphi',
    # 'RecoZ_1_Dr',
    # 'RecoZ_1_E',
    # #    'RecoZ_1_Eta',
    # 'RecoZ_1_M',
    # #    'RecoZ_1_Phi',
    # 'RecoZ_1_Pt',
    # 'RecoZ_1_cosdTheta',
    # 'RecoZ_BJet3_E',
    # #    'RecoZ_BJet3_Eta',
    # 'RecoZ_BJet3_M',
    # #    'RecoZ_BJet3_Phi',
    # 'RecoZ_BJet3_Pt',
    # 'RecoZ_BJet4_E',
    # #    'RecoZ_BJet4_Eta',
    # 'RecoZ_BJet4_M',
    # #    'RecoZ_BJet4_Phi',
    # 'RecoZ_BJet4_Pt',
    # 'RecoZ_2_Deta',
    # 'RecoZ_2_Dphi',
    # 'RecoZ_2_Dr',
    # 'RecoZ_2_E',
    # #    'RecoZ_2_Eta',
    # 'RecoZ_2_M',
    # #    'RecoZ_2_Phi',
    # 'RecoZ_2_Pt',
    # 'RecoZ_2_cosdTheta',
    # 'RecoZ_Chi2',
    # 'RecoZ_logChi2',
    # 'Reco_JABDT_ttbar_Jet_CSV_btophad',
    # 'Reco_JABDT_ttbar_Jet_CSV_btoplep',
    # 'Reco_JABDT_ttbar_Jet_CSV_whaddau1',
    # 'Reco_JABDT_ttbar_Jet_CSV_whaddau2',
    # 'Reco_JABDT_ttbar_costheta_toplep_tophad',
    # 'Reco_JABDT_ttbar_log_tophad_m',
    # 'Reco_JABDT_ttbar_log_tophad_pt',
    # 'Reco_JABDT_ttbar_log_toplep_m',
    # 'Reco_JABDT_ttbar_log_toplep_pt',
    # 'Reco_JABDT_ttbar_log_whad_m',
    # 'Reco_JABDT_ttbar_tophad_pt__P__toplep_pt__DIV__Evt_HT__P__Evt_Pt_MET__P__Lep_Pt',
    # #    'Reco_LeptonicW_Eta',
    # 'Reco_LeptonicW_M',
    # #    'Reco_LeptonicW_Phi',
    # 'Reco_LeptonicW_Pt',
    # 'Reco_WLep_E',
    # #    'Reco_WLep_Eta',
    # 'Reco_WLep_Mass',
    # #    'Reco_WLep_Phi',
    # 'Reco_WLep_Pt',
    # 'Reco_best_higgs_mass',
    # 'Reco_dEta_fn',
    # 'Reco_ttbar_bestJABDToutput',
    # #    'Reco_ttbar_btophad_eta',
    # 'Reco_ttbar_btophad_m',
    # #    'Reco_ttbar_btophad_phi',
    # 'Reco_ttbar_btophad_pt',
    # 'Reco_ttbar_btophad_w_dr',
    # #    'Reco_ttbar_btoplep_eta',
    # 'Reco_ttbar_btoplep_m',
    # #    'Reco_ttbar_btoplep_phi',
    # 'Reco_ttbar_btoplep_pt',
    # 'Reco_ttbar_btoplep_w_dr',
    # #    'Reco_ttbar_tophad_eta',
    # 'Reco_ttbar_tophad_m',
    # #    'Reco_ttbar_tophad_phi',
    # 'Reco_ttbar_tophad_pt',
    # #    'Reco_ttbar_toplep_eta',
    # 'Reco_ttbar_toplep_m',
    # #    'Reco_ttbar_toplep_phi',
    # 'Reco_ttbar_toplep_pt',
    # 'Reco_ttbar_whad_dr',
    # #    'Reco_ttbar_whad_eta',
    # 'Reco_ttbar_whad_m',
    # #    'Reco_ttbar_whad_phi',
    # 'Reco_ttbar_whad_pt',
    # #    'Reco_ttbar_whaddau_eta1',
    # #    'Reco_ttbar_whaddau_eta2',
    # 'Reco_ttbar_whaddau_m1',
    # 'Reco_ttbar_whaddau_m2',
    # #    'Reco_ttbar_whaddau_phi1',
    # #    'Reco_ttbar_whaddau_phi2',
    # 'Reco_ttbar_whaddau_pt1',
    # 'Reco_ttbar_whaddau_pt2',

    'N_BTagsL',
    'N_BTagsM',
    'N_BTagsT',
    'N_Jets',
    'N_LooseElectrons',
    'N_LooseJets',
    'N_LooseMuons',
    'N_PrimaryVertices',
    'N_TightElectrons',
    'N_TightMuons',
    'CSV[0]',
    'CSV[1]',
    'CSV[2]',
    'CSV[3]',
    'CSV[4]',
    'CSV[5]',
    'CSV[6]',
    'CSV[7]',
    'Electron_E[0]',
    # 'Electron_Eta[0]',
    'Electron_M[0]',
    # 'Electron_Phi[0]',
    'Electron_Pt[0]',
    'Jet_CSV[0]',
    'Jet_CSV[1]',
    'Jet_CSV[2]',
    'Jet_CSV[3]',
    'Jet_CSV[4]',
    'Jet_CSV[5]',
    'Jet_CSV[6]',
    'Jet_CSV[7]',
    'Jet_E[0]',
    'Jet_E[1]',
    'Jet_E[2]',
    'Jet_E[3]',
    'Jet_E[4]',
    'Jet_E[5]',
    'Jet_E[6]',
    'Jet_E[7]',
    'Jet_M[0]',
    'Jet_M[1]',
    'Jet_M[2]',
    'Jet_M[3]',
    'Jet_M[4]',
    'Jet_M[5]',
    'Jet_M[6]',
    'Jet_M[7]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    'Jet_Pt[2]',
    'Jet_Pt[3]',
    'Jet_Pt[4]',
    'Jet_Pt[5]',
    'Jet_Pt[6]',
    'Jet_Pt[7]',
    # 'LooseElectron_E[0]',
    # 'LooseElectron_M[0]',
    # 'LooseElectron_Pt[0]',
    # 'LooseJet_CSV[0]',
    # 'LooseJet_CSV[1]',
    # 'LooseJet_CSV[2]',
    # 'LooseJet_CSV[3]',
    # 'LooseJet_E[0]',
    # 'LooseJet_E[1]',
    # 'LooseJet_E[2]',
    # 'LooseJet_E[3]',
    # 'LooseJet_M[0]',
    # 'LooseJet_M[1]',
    # 'LooseJet_M[2]',
    # 'LooseJet_M[3]',
    # 'LooseJet_Pt[0]',
    # 'LooseJet_Pt[1]',
    # 'LooseJet_Pt[2]',
    # 'LooseJet_Pt[3]',
    # 'LooseLepton_E[0]',
    # 'LooseLepton_M[0]',
    # 'LooseLepton_Pt[0]',
    # 'LooseMuon_E[0]',
    # 'LooseMuon_M[0]',
    # 'LooseMuon_Pt[0]',
    'Muon_E[0]',
    # 'Muon_Eta[0]',
    'Muon_M[0]',
    # 'Muon_Phi[0]',
    'Muon_Pt[0]',
    'TaggedJet_CSV[0]',
    'TaggedJet_CSV[1]',
    'TaggedJet_E[0]',
    'TaggedJet_E[1]',
    'TaggedJet_M[0]',
    'TaggedJet_M[1]',
    'TaggedJet_Pt[0]',
    'TaggedJet_Pt[1]',
    # 'TaggedJet_CSV[2]',
    # 'TaggedJet_CSV[3]',
    # 'TaggedJet_E[0]',
    # 'TaggedJet_E[1]',
    # 'TaggedJet_M[0]',
    # 'TaggedJet_M[1]',
    # 'TaggedJet_Pt[0]',
    # 'TaggedJet_Pt[1]',
    'TightLepton_E[0]',
    # 'TigntLepton_Eta[0]',
    'TightLepton_M[0]',
    # 'TigntLeoton_Phi[0]',
    'TightLepton_Pt[0]',
    # 'Reco_tHH_bestJABDToutput',
    # #    'Reco_tHH_btophad_eta',
    # 'Reco_tHH_btophad_m',
    # #    'Reco_tHH_btophad_phi',
    # 'Reco_tHH_btophad_pt',
    # #    'Reco_tHH_btoplep_eta',
    # 'Reco_tHH_btoplep_m',
    # #    'Reco_tHH_btoplep_phi',
    # 'Reco_tHH_btoplep_pt',
    # 'Reco_tHH_btoplep_w_dr',
    # 'Reco_tHH_h1_dr',
    # #    'Reco_tHH_h1_eta',
    # 'Reco_tHH_h1_m',
    # #    'Reco_tHH_h1_phi',
    # 'Reco_tHH_h1_pt',
    # 'Reco_tHH_h2_dr',
    # #    'Reco_tHH_h2_eta',
    # 'Reco_tHH_h2_m',
    # #    'Reco_tHH_h2_phi',
    # 'Reco_tHH_h2_pt',
    # #    'Reco_tHH_h1dau1_eta',
    # #    'Reco_tHH_h1dau2_eta',
    # 'Reco_tHH_h1dau1_m',
    # 'Reco_tHH_h1dau2_m',
    # #    'Reco_tHH_h1dau1_phi',
    # #    'Reco_tHH_h1dau2_phi',
    # 'Reco_tHH_h1dau1_pt',
    # 'Reco_tHH_h1dau2_pt',
    # # 'Reco_tHH_h2dau3_eta',
    # #    'Reco_tHH_h2dau4_eta',
    # 'Reco_tHH_h2dau3_m',
    # 'Reco_tHH_h2dau4_m',
    # #    'Reco_tHH_h2dau3_phi',
    # #    'Reco_tHH_h2dau4_phi',
    # 'Reco_tHH_h2dau3_pt',
    # 'Reco_tHH_h2dau4_pt',
    # #    'Reco_tHH_toplep_eta',
    # 'Reco_tHH_toplep_m',
    # #    'Reco_tHH_toplep_phi',
    # 'Reco_tHH_toplep_pt',
    # 'Reco_JABDT_tHH_Jet_CSV_btophad',
    # 'Reco_JABDT_tHH_Jet_CSV_btoplep',
    # 'Reco_JABDT_tHH_Jet_CSV_h1dau1',
    # 'Reco_JABDT_tHH_Jet_CSV_h1dau2',
    # 'Reco_JABDT_tHH_Jet_CSV_h2dau3',
    # 'Reco_JABDT_tHH_Jet_CSV_h2dau4',
    # 'Reco_JABDT_tHH_log_h1_m',
    # 'Reco_JABDT_tHH_log_h1_pt',
    # 'Reco_JABDT_tHH_log_h2_m',
    # 'Reco_JABDT_tHH_log_h2_pt',
    # 'Reco_JABDT_tHH_log_toplep_m',
    # 'Reco_JABDT_tHH_log_toplep_pt',
]


if options.new == "new":

    evaluation = True

    workdir = filedir + "/" + options.cardfolder

    for node in process_new:
        
        if "TwoYear" in options.filefolder or "ThreeYear" in options.filefolder:
            rootfile = filedir + "/combineRun2/"+options.filefolder+"/output_limit.root"
        else:
            rootfile = basedir + "/workdir/{}/plots/output_limit.root".format(options.filefolder)
        script = filedir + "/PlotScript.py"
        plotconfig = filedir + "/plotconfig_new.py"
        systematic = filedir + "/systematics.csv"
        # selectionlabel = "\geq {} jets, \geq {} b-tags".format(options.njets, options.nbjets)
        # runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq {} jets, \geq {} b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics_full.csv" --workdir={} --evaluation={}'.format("new", node, options.njets,options.nbjets,rootfile, workdir, evaluation)
        runcommand = 'python {} --plotconfig={}  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq {} jets, \geq {} b-tags" --rootfile={}  --directory={} --systematicfile={} --workdir={} --evaluation={}'.format(script, plotconfig, node, options.njets,options.nbjets,rootfile, filedir, systematic, workdir, evaluation)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

elif options.new == "old":

    evaluation = True

    for node in process_old:

        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/{}/plots/output_limit.root".format(options.filefolder)
        workdir = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/{}".format(
            options.cardfolder)
        # selectionlabel = "\geq {} jets, \geq {} b-tags".format(
            # options.njets, options.nbjets)
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}.py"  --channelname="ljets_ge4j_ge3t_{}_node"  --selectionlabel="\geq {} jets, \geq {} b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir={} --evaluation={}'.format(
            "old", node, options.njets,options.nbjets,rootfile, workdir, evaluation)


        os.system(runcommand)

        print("finish plotting discriminators for process {}".format(node))

else:

    evaluation = False
    for var in variables:

        # rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/output_limit.root"
        # runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, \geq 3 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={}'.format(
        #     "new", var, rootfile,evaluation)
        rootfile = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/workdir/output_limit.root"
        # runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={} --logarithmic=False'.format(
            # "new", var, rootfile,evaluation)
        runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={} --logarithmic=False'.format("new", var, rootfile,evaluation)
        # runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={} --logarithmic=True'.format("new", var, rootfile,evaluation)

        # runcommand = 'python /uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/PlotScript.py --plotconfig="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/plotconfig_{}_plotting.py"  --channelname={}  --selectionlabel="\geq 4 jets, 2 b-tags" --rootfile={}  --directory="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard" --systematicfile="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics.csv" --workdir="/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/newplotting" --evaluation={}'.format(
        #     "new_ttnb", var, rootfile, evaluation)

        os.system(runcommand)

        print("finish plotting variable: "+var)


