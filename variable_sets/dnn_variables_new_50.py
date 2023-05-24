variables = {}
variables["ge4j_ge3t"] = [ 
    'Reco_tHH_bestJABDToutput',
    'Evt_Deta_TaggedJetsAverage',
    'Reco_ttbar_bestJABDToutput',
    'CSV[4]',
    'Reco_JABDT_tHH_Jet_CSV_btophad',
    'Evt_HT',
    'Evt_Deta_JetsAverage',
    'Evt_h1',
    'Evt_Dr_JetsAverage',
    'RecoHiggs_Chi2',
    'Evt_CSV_avg_tagged',
    'CSV[3]',
    'Evt_M2_JetsAverage',
    'RecoZH_logChi2',
    'Evt_Dr_maxDrJets',
    'Evt_M2_minDrJets',
    'Reco_JABDT_ttbar_log_toplep_m',
    'N_BTagsL',
    'N_LooseJets',
    'RecoZH_1_M',
    'TaggedJet_M[4]',
    'CSV[2]',
    'Evt_aplanarity',
    'Jet_Pt[4]',
    'Reco_tHH_btophad_pt',
    'Evt_M2_TaggedJetsAverage',
    'Evt_M_minDrLepJet',
    'TaggedJet_Pt[3]',
    'Evt_blr_transformed',
    'Evt_M2_closestTo91TaggedJets',
    'CSV[5]',
    'Evt_CSV_dev',
    'RecoZ_2_Dr',
    'Reco_tHH_btophad_m',
    'RecoHiggs_1_M',
    'TaggedJet_E[4]',
    'Reco_JABDT_tHH_log_toplep_pt',
    'LooseJet_Pt[2]',
    'Evt_Pt_minDrTaggedJets',
    'Evt_M_Total',
    'Jet_M[0]',
    'Muon_Pt[0]',
    'TaggedJet_M[5]',
    'Evt_aplanarity_jets',
    'Evt_JetPt_over_JetE',
    'Evt_Dr_closestTo91TaggedJets',
    'Evt_Deta_UntaggedJetsAverage',
    'N_HEM_LooseElectrons',
    'Evt_Dr_minDrUntaggedJets',
    'RecoHiggs_BJet2_Pt',

    ]


variables["ge6j_ge4t"] = [ 
    'Reco_tHH_bestJABDToutput',
    'Evt_Deta_JetsAverage',
    'RecoHiggs_logChi2',
    'Reco_dEta_fn',
    'Evt_blr',
    'Reco_ttbar_bestJABDToutput',
    'CSV[4]',
    'RecoZH_1_cosdTheta',
    'Evt_h1',
    'CSV[5]',
    'CSV[2]',
    'Reco_JABDT_ttbar_log_toplep_m',
    'Reco_JABDT_tHH_Jet_CSV_h1dau2',
    'Evt_HT_jets',
    'Evt_CSV_avg',
    'TaggedJet_M[5]',
    'Evt_transverse_sphericity',
    'RecoZ_BJet4_Pt',
    'LooseJet_Pt[3]',
    'Evt_M2_TaggedJetsAverage',
    'Evt_M2_closestTo91TaggedJets',
    'Jet_Pt[5]',
    'Reco_tHH_h2dau3_pt',
    'Evt_M2_JetsAverage',
    'RecoZH_BJet1_M',
    'Reco_JABDT_tHH_log_h1_m',
    'Evt_Dr_closestTo91TaggedJets',
    'Reco_JABDT_tHH_Jet_CSV_btophad',
    'TaggedJet_M[1]',
    'TightLepton_E[0]',
    'Evt_CSV_avg_tagged',
    'N_LooseJets',
    'Evt_HT',
    'TaggedJet_E[0]',
    'Evt_M_UntaggedJetsAverage',
    'CSV[6]',
    'Evt_CSV_dev',
    'CSV[3]',
    'N_BTagsT',
    'N_BTagsM',
    'Reco_JABDT_ttbar_Jet_CSV_btoplep',
    'Reco_JABDT_tHH_log_h2_pt',
    'Evt_M2_UntaggedJetsAverage',
    'RecoZ_1_cosdTheta',
    'RecoZH_1_Dr',
    'RecoZ_BJet4_M',
    'Evt_Deta_UntaggedJetsAverage',
    'RecoHiggs_2_cosdTheta',
    'Reco_WLep_E',
    'Reco_JABDT_tHH_log_toplep_pt',


]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
