# additional_variables = {}
scalefactor_variables = [
    'Electron_IdentificationSF[0]',
    'Electron_Pt_BeforeRun2Calibration[0]',
    'Electron_Eta[0]',
    'Electron_Eta_Supercluster[0]',
    'Electron_IdentificationSF[0]',
    'Electron_ReconstructionSF[0]',
    'Muon_Eta[0]',
    'Muon_IdentificationSF[0]',
    'Muon_IsolationSF[0]',
    # 'Muon_Pt_BeForeRC',
    'Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX',
    'Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX',
    'Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX',
    'Triggered_HLT_IsoMu27_vX',
    'Weight_CSV',
    'Weight_CSVCErr1down',
    'Weight_CSVCErr2down',
    'Weight_CSVCErr2up',
    'Weight_CSVHFStats1down',
    'Weight_CSVHFStats1up',
    'Weight_CSVHFStats2down',
    'Weight_CSVHFStats2up',
    'Weight_CSVHFdown',
    'Weight_CSVHFup',
    'Weight_CSVLFStats1down',
    'Weight_CSVLFStats2down',
    'Weight_CSVLFStats2up',
    'Weight_CSVLFdown',
    'Weight_CSVLFup',
    'Weight_L1ECALPrefire',
    'Weight_MuonTriggerSF',
    'Weight_XS',
    'Weight_pu69p2',
    # 'Weight_scale_variation_muR_0p5_muF_0p5',
    # 'Weight_scale_variation_muR_0p5_muF_1p0',
    # 'Weight_scale_variation_muR_0p5_muF_2p0',
    # 'Weight_scale_variation_muR_1p0_muF_0p5',
    # 'Weight_scale_variation_muR_1p0_muF_1p0',
    # 'Weight_scale_variation_muR_1p0_muF_2p0',
    # 'Weight_scale_variation_muR_2p0_muF_0p5',
    # 'Weight_scale_variation_muR_2p0_muF_1p0',
    # 'Weight_scale_variation_muR_2p0_muF_2p0',
    'GenEvt_I_TTPlusBB',
    'GenEvt_I_TTPlusCC',
    
    'Jet_Flav[0]',
    'Jet_Flav[1]',
    'Jet_Flav[2]',
    'Jet_Flav[3]',
    'Jet_Flav[4]',
    'Jet_Flav[5]',
    'Jet_Flav[6]',
    'Jet_Flav[7]',
    'Jet_Eta[0]',
    'Jet_Eta[1]',
    'Jet_Eta[2]',
    'Jet_Eta[3]',
    'Jet_Eta[4]',
    'Jet_Eta[5]',
    'Jet_Eta[6]',
    'Jet_Eta[7]',
    'Jet_Phi[0]',
    'Jet_Phi[1]',
    'Jet_Phi[2]',
    'Jet_Phi[3]',
    'Jet_Phi[4]',
    'Jet_Phi[5]',
    'Jet_Phi[6]',
    'Jet_Phi[7]',
    'LooseLepton_Eta[0]',
    'LooseLepton_Phi[0]',
    ]

ttbar_variables = [
    'Weight_scale_variation_muR_0p5_muF_0p5',
    'Weight_scale_variation_muR_0p5_muF_1p0',
    'Weight_scale_variation_muR_0p5_muF_2p0',
    'Weight_scale_variation_muR_1p0_muF_0p5',
    'Weight_scale_variation_muR_1p0_muF_1p0',
    'Weight_scale_variation_muR_1p0_muF_2p0',
    'Weight_scale_variation_muR_2p0_muF_0p5',
    'Weight_scale_variation_muR_2p0_muF_1p0',
    'Weight_scale_variation_muR_2p0_muF_2p0',

]

# all_additional_variables = list(set([v for v in additional_variables]))
 

# TODO - check calibrations below
# if (chain -> GetBranch("Electron_Pt_BeforeRun2Calibration") & & chain -> GetBranch("Electron_Eta_Supercluster")){
#                               muonEta = Muon_Eta[0]
#                               }
#     if (N_TightMuons == 1){muonPt = Muon_Pt[0]
#                            muonEta = Muon_Eta[0]
#                            }
#     else {muonPt = 0.0
#           muonEta = 0.0
#           }
#     if (N_TightElectrons == 1){electronPt = Electron_Pt[0]
#                                electronEta = Electron_Eta_Supercluster[0]
#                                }
#     else {electronPt = 0.0
#           electronEta = 0.0
#           }
# }
# else {
#     if (N_TightMuons == 1){muonPt = Muon_Pt[0]
#                            muonEta = Muon_Eta[0]
#                            }
#     else {muonPt = 0.0
#           muonEta = 0.0
#           }
#     if (N_TightElectrons == 1){electronPt = Electron_Pt[0]
#                                electronEta = Electron_Eta[0]
#                                }
#     else {electronPt = 0.0
#           electronEta = 0.0
#           }
# }
