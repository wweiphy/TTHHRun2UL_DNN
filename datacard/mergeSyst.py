
import sys
import os
sys.path.append('/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard')
import combine_intermid_systs
filename     = os.getenv('INFILE')
process      = os.getenv('ORIGNAME')
nom_key      = os.getenv('NOMHISTKEY')
syst_key     = os.getenv('SYSTHISTKEY')
separator    = os.getenv('SEPARATOR')
# syst_csvpath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics_full_4FS.csv"
# syst_csvpath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics_full_5FS.csv"
syst_csvpath = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/systematics_full.csv"
print(process)
combine_intermid_systs.combine_intermid_syst(   h_nominal_key   = nom_key, 
                                                h_syst_key      = syst_key, 
                                                rfile_path      = filename,
                                                replace_config  = "/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_11_1_2/src/TTHHRun2UL_DNN/datacard/pdf_relic_names",
                                                processes       = process,
                                                separator       = separator,
                                                syst_csvpath    = syst_csvpath
                                            )
