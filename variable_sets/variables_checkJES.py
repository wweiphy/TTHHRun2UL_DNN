variables = {}
variables["ge4j_ge3t"] = [
   'Jet_Eta[0]',
   'Jet_Eta[1]',
   'Jet_Eta[2]',
   'Jet_Eta[3]',
   'Jet_Eta[4]',
   'Jet_Eta[5]',
   'Jet_Eta[6]',
   'Jet_Eta[7]',
#    'Jet_Phi[0]',
#    'Jet_Phi[1]',
#    'Jet_Phi[2]',
#    'Jet_Phi[3]',
#    'Jet_Phi[4]',
#    'Jet_Phi[5]',
#    'Jet_Phi[6]',
#    'Jet_Phi[7]',
    ]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
