import pandas as pd


with pd.HDFStore("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/ttlf_dnn.h5", "a") as store:
    store.append("data", pd.read_hdf("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/Eval_0220_UL_nominal_3/ttlf_dnn.h5"),index=False)
    store.append("data", pd.read_hdf("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/Eval_0220_UL_nominal_4/ttlf_dnn.h5"),index=False)
    print("number of events: {}".format(store.get_storer("data").nrows))

print ("finished merging ttlf files")

with pd.HDFStore("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/ttcc_dnn.h5", "a") as store:
    store.append("data", pd.read_hdf("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/Eval_0220_UL_nominal_3/ttcc_dnn.h5"),index=False)
    store.append("data", pd.read_hdf("/uscms/home/wwei/nobackup/SM_TTHH/Summer20UL/CMSSW_12_1_1/src/TTHHRun2UL_DNN/workdir/Eval_0220_UL_nominal_4/ttcc_dnn.h5"),index=False)
    print("number of events: {}".format(store.get_storer("data").nrows))


print ("finished merging ttcc files")