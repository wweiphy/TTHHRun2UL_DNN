import os
import sys
import uproot
import pandas as pd
import numpy as np
import re
import glob
import multiprocessing as mp
from correctionlib import _core


filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)

# internal = pd.read_csv(filedir+"/GenNormMap/internalNorm.csv")
# internal_ttbb = pd.read_csv(
#     filedir+"/GenNormMap/internalNorm_ttbb.csv")
# ttbb = pd.read_csv(filedir+"/GenNormMap/fracttbb.csv")
# ttbb_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttbb_ttbb.csv")
# ttcc = pd.read_csv(filedir+"/GenNormMap/fracttcc.csv")
# ttcc_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttcc_ttbb.csv")
# ttlf = pd.read_csv(filedir+"/GenNormMap/fracttlf.csv")
# ttlf_ttbb = pd.read_csv(filedir+"/GenNormMap/fracttlf_ttbb.csv")
genmap2016 = pd.read_csv(filedir+"/GenNormMap/ratefactors_new_plotscript_2016.csv")
genmap2017 = pd.read_csv(filedir+"/GenNormMap/ratefactors_new_plotscript_2017.csv")
genmap2018 = pd.read_csv(filedir+"/GenNormMap/ratefactors_new_plotscript_2018.csv")

# multi processing magic
# TODO - A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer, col_indexer] = value instead. See the caveats in the documentation: http: // pandas.pydata.org/pandas-docs/stable/indexing.html; self.obj[key] = _infer_fill_value(value)

def processChunk(info):
    info["self"].processChunk(
        info["sample"], info["chunk"], info["chunkNumber"])

class EventCategories:
    def __init__(self):
        self.categories = {}

    def addCategory(self, name, selection=None):
        self.categories[name] = selection

    def getCategorySelections(self):
        selections = []
        for cat in self.categories:
            if self.categories[cat]:
                selections.append(self.categories[cat])
        return selections

class Sample:
    def __init__(self, sampleName, ntuples, categories, process, selections=None, ownVars=[], even_odd=False, lumiWeight=1., islocal = False):
        self.sampleName = sampleName
        self.ntuples = ntuples
        self.selections = selections
        self.categories = categories
        self.process = process
        self.ownVars = ownVars
        self.lumiWeight = lumiWeight
        self.even_odd = even_odd
        self.islocal = islocal
        self.evenOddSplitting()

    def printInfo(self):
        print("\nHANDLING SAMPLE {}\n".format(self.sampleName))
        print("\tntuples: {}".format(self.ntuples))
        print("\tselections: {}".format(self.selections))

    def evenOddSplitting(self):
        if self.even_odd:
            if self.selections:
                self.selections += " and (Evt_Odd == 1)"
            else:
                self.selections = "(Evt_Odd == 1)"



class Dataset:
    def __init__(self, outputdir, tree=['MVATree'], naming='', maxEntries=50000, varName_Run='Evt_Run', varName_LumiBlock='Evt_Lumi', varName_Event='Evt_ID',ncores=1, dataEra = 2017, do_EvalSFs=False):
        # settings for paths
        self.outputdir = outputdir
        self.naming = naming
        self.tree = tree
        self.varName_Run = varName_Run
        self.varName_LumiBlock = varName_LumiBlock
        self.varName_Event = varName_Event
        self.dataEra = dataEra
        self.do_EvalSFs = do_EvalSFs

        genfile = ""
        if dataEra == "2017":
            genfile = genmap2017
        elif dataEra == "2018":
            genfile = genmap2018
        if dataEra == "2016postVFP" or dataEra == "2016preVFP":
            genfile = genmap2016
        else:
            # print("no file matches the dataEra " +dataEra)
            sys.exit("no file matches the dataEra " +dataEra)

        print("handle relative ratio for year "+dataEra)
        self.genfile = genfile

        # generating output dir
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        # settings for dataset
        self.maxEntries = int(maxEntries)

        # default values for some configs
        self.baseSelection = None
        self.samples = {}
        self.variables = []

        self.ncores = int(ncores)

    def addBaseSelection(self, selection):
        self.baseSelection = selection

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
        self.samples[kwargs["sampleName"]] = Sample(**kwargs)

 # variable handling
    def addVariables(self, variables):
        print("adding {} variables.".format(len(variables)))
        self.variables += variables
        self.variables = list(set(self.variables))

    def removeVariables(self, variables):
        n_removed = 0
        for v in variables:
            if v in self.variables:
                    self.variables.remove(v)
                    n_removed += 1
        print("removed {} variables from list.".format(n_removed))

    def gatherTriggerVariables(self):
        # search for all trigger strings
        self.trigger = []

        # search in base selection string
        if self.baseSelection:
            self.trigger.append(self.baseSelection)

        for key in self.samples:
            # collect variables for specific samples
            own_variables = []

            # search in additional selection strings
            if self.samples[key].selections:
                own_variables += self.searchVariablesInTriggerString(
                    self.samples[key].selections)
            # search in category selections
            categorySelections = self.samples[key].categories.getCategorySelections(
            )
            for selection in categorySelections:
                own_variables += self.searchVariablesInTriggerString(selection)
            # save list of variables
            self.samples[key].ownVars = [v for v in list(
                set(own_variables)) if not v in self.variables]

        # list of triggers
        self.trigger = list(set(self.trigger))

        # scan trigger strings for variable names
        self.triggerVariables = []
        for triggerstring in self.trigger:
            self.triggerVariables += self.searchVariablesInTriggerString(
                triggerstring)

        self.triggerVariables = list(set(self.triggerVariables))

        # select variables that only appear in triggerVariables to remove them before saving the final dataframes
        self.removedVariables = [
            v for v in self.triggerVariables if not v in self.variables]

        # test 
        # print("variables to be removed: ")
        # print(self.removedVariables)

        # add trigger variables to variable list
        self.addVariables(self.triggerVariables)


    def searchVariablesInTriggerString(self, string):
        # split trigger string into smaller bits
        splitters = [")", "(", "==", ">=", ">=", ">", "<", "="]

        candidates = string.split(" ")
        for splt in splitters:
            # print("splitter is " + splt)
            candidates = [item for c in candidates for item in c.split(splt)]
            # test
            # print(candidates)

        # remove some entries
        remove_entries = ["", "and", "or", "abs"]
        for entry in remove_entries:
            candidates = [c for c in candidates if not c == entry]

        # remove numbers
        candidates = [c for c in candidates if not c.replace(
            ".", "", 1).isdigit()]

        # the remaining candidates should be variables
        return candidates

    def searchVectorVariables(self):
        # list for variables
        variables = []
        # dictionary for vector variables
        vector_variables = {}

        # loop over variables in list
        for var in self.variables:
            # search for index in name (dummyvar[index])
            found_vector_variable = re.search("\[\d+?\]$", var)
            # append variable to list if not a vector variable
            if not found_vector_variable:
                variables.append(var)
                continue

            # handle vector variable
            index = found_vector_variable.group(0) # return the whole matched string
            var_name = var[:-len(index)]
            var_index = int(index[1:-1])

            # add variable with index to vector_variables dictionary
            if var_name in vector_variables:
                vector_variables[var_name].append(var_index)
            else:
                vector_variables[var_name] = [var_index]

        self.variables = variables
        self.vector_variables = vector_variables

    def runPreprocessing(self):
        # add variables for triggering and event category selection
        self.gatherTriggerVariables()

        # search for vector variables in list of variables and handle them separately
        self.searchVectorVariables()

        print("LOADING {} VARIABLES IN TOTAL.".format(len(self.variables)))
        # remove old files
        self.renameOldFiles()

        sampleList = []

        # start loop over all samples to preprocess them
        for key in self.samples:
            # include own variables of the sample
            self.addVariables(self.samples[key].ownVars)

            # process the sample
            self.processSample(
                sample=self.samples[key],
                # varName_Run=self.varName_Run,
                # varName_LumiBlock=self.varName_LumiBlock,
                # varName_Event=self.varName_Event,
            )

            # remove the own variables
            self.removeVariables(self.samples[key].ownVars)
            self.createSampleList(sampleList, self.samples[key])
            print("done.")
        # write file with preprocessed samples
        self.createSampleFile(self.outputdir, sampleList)

        # handle old files
        self.handleOldFiles()


    def processSample(self, sample):
        # print sample info
        sample.printInfo()

        # collect ntuple files
        # ntuple_files = sample.ntuples
        ntuple_files = sorted(glob.glob(sample.ntuples))

        # initialize loop over ntuple files
        n_entries = 0
        n_files = len(ntuple_files)
        print("number of files: {}".format(n_files))

        while not len(ntuple_files)%self.ncores == 0:
            ntuple_files.append("")
        ntuple_files = np.array(ntuple_files).reshape(self.ncores, -1)

        pool = mp.Pool(self.ncores)
        chunks = [{"self": self, "chunk": c, "sample": sample, "chunkNumber": i+1} for i,c in enumerate(ntuple_files)]
        pool.map(processChunk, chunks)

        # concatenate single thread files
        self.mergeFiles(sample.categories.categories)

    def processChunk(self, sample, files, chunkNumber):

        files = [f for f in files if not f == ""]
        
        n_entries = 0
        concat_df = pd.DataFrame()
        n_files = len(files)
        for iF, f in enumerate(files):
            print("chunk #{}: starting file ({}/{}): {}".format(chunkNumber, iF+1, n_files, f))
            
            if not sample.islocal:
                # add full path for the files in eos space
                f_full = "root://cmseos.fnal.gov/" + f
                file = f.split("/")[-1]
                # print("file for remote: "+file)
                # copy file from eos space into local 
                copyeoscommand = "xrdcp "+f_full+" ."
                # print("copy file {} from eos space".format(copyeoscommand))
                os.system(copyeoscommand)
            else:
                file = f
                # print("file for local: "+file)
            for tr in self.tree:
                # open root file
                with uproot.open(file) as rf:
                    # get TTree
                  try:
                        tree = rf[tr]
                        # print("successfully opened " + str(tr) +" in ROOT file")
                  except:
                        print("could not open "+str(tr)+" in ROOT file")
                        continue


                
                if tree.numentries == 0:
                   print(str(tr)+" has no entries - skipping file")
                   continue
                # test 
                # print(self.variables)

                # convert to dataframe
                # df = tree.arrays(self.variables,library="pd")
                df = tree.pandas.df([v for v in self.variables])
                # print(df["Evt_CSV_avg"])

                # delete subentry index
                try: df = df.reset_index(1, drop = True)
                except: None
                # print(df)
                
                # print("start processing vector variables")
                # print("vector variables list: ")
                # print(self.vector_variables)
                # handle vector variables, loop over them

                for vecvar in self.vector_variables:

                    # load dataframe with vector variable
                    vec_df = tree.pandas.df(vecvar)

                    # loop over inices in vecvar list
                    for idx in self.vector_variables[vecvar]:

                        # slice the index
                        idx_df = vec_df.loc[(slice(None), slice(idx, idx)), :]
                        idx_df = idx_df.reset_index(1, drop=True)

                        # define name for column in df
                        col_name = str(vecvar)+"["+str(idx)+"]"
                        # print("colomn name is: " + col_name)

                        # initialize column in original dataframe
                        df.loc[:, col_name] = 0.
                        # append column to original dataframe
                        df.update(idx_df[vecvar].rename(col_name))

                df.loc[:, "process"] = sample.process

                # apply event selection
                df = self.applySelections(df, sample.selections)

                if self.do_EvalSFs:
                    # for DNN evaluation on data
                    if sample.process == "data":
                        df = df.assign(xs_weight=1.)

                    else:
                    # for DNN evaluation on MC
                        if "nominal" in file:
                            print('Evaluate SFs for nominal files')
                            df = self.CalculateSFsEval(tree, df)

                            if not (sample.process == "ttH" or sample.process == "ttSL" or sample.process == "ttDL" or sample.process == "ttbbSL" or sample.process == "ttbbDL"):
                                df.loc[:, "GenWeight_fsr_Def_down"] = 0.
                                df.loc[:, "GenWeight_fsr_Def_up"] = 0.
                                df.loc[:, "GenWeight_isr_Def_down"] = 0.
                                df.loc[:, "GenWeight_isr_Def_up"] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_0p5'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_1p0'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_0p5_muF_2p0'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_0p5'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_1p0'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_1p0_muF_2p0'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_0p5'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_1p0'] = 0.
                                df.loc[:, 'Weight_scale_variation_muR_2p0_muF_2p0'] = 0.

                            # nominal values
                            df = df.assign(sf_weight=lambda x: (sample.lumiWeight*x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # btag SF & uncertainties
                            df = df.assign(xs_weight=lambda x: x.Weight_XS * x.Weight_GEN_nom)
                            # df = df.assign(xs_weight=lambda x: x.Weight_XS *
                            #             x.Weight_CSV_UL * x.Weight_GEN_nom)

                            df = df.assign(
                                total_weight=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL)
                            df = df.assign(total_weight_uplf=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uplf)
                            df = df.assign(total_weight_downlf=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downlf)
                            df = df.assign(total_weight_uphf=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uphf)
                            df = df.assign(total_weight_downhf=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downhf)
                            df = df.assign(total_weight_uplfstats1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uplfstats1)
                            df = df.assign(total_weight_downlfstats1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downlfstats1)
                            df = df.assign(total_weight_uplfstats2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uplfstats2)
                            df = df.assign(total_weight_downlfstats2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downlfstats2)
                            df = df.assign(total_weight_uphfstats1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uphfstats1)
                            df = df.assign(total_weight_downhfstats1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downhfstats1)
                            df = df.assign(total_weight_uphfstats2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_uphfstats2)
                            df = df.assign(total_weight_downhfstats2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downhfstats2)
                            df = df.assign(total_weight_upcferr1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_upcferr1)
                            df = df.assign(total_weight_downcferr1=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downcferr1)
                            df = df.assign(total_weight_upcferr2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_upcferr2)
                            df = df.assign(total_weight_downcferr2=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL_downcferr2)

                            print("Done with btagging SFs")

                            # PU weights & uncertainties
                            df = df.assign(total_weight_upPU=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2Up'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            df = df.assign(total_weight_downPU=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2Down'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # L1 prefiring
                            df = df.assign(total_weight_upL1Fire=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefireUp'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            df = df.assign(total_weight_downL1Fire=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefireDown'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # electron trigger SF & uncertainties
                            df = df.assign(total_weight_downEleTrigger=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF_Down'] > 0)) * 1. * x['Weight_ElectronTriggerSF_Down'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            df = df.assign(total_weight_upEleTrigger=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF_Up'] > 0)) * 1. * x['Weight_ElectronTriggerSF_Up'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # muon trigger SF & uncertainties
                            df = df.assign(total_weight_downMuonTrigger=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF_Down'] > 0.)) * 1. * x['Weight_MuonTriggerSF_Down'])))

                            df = df.assign(total_weight_upMuonTrigger=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF_Up'] > 0.)) * 1. * x['Weight_MuonTriggerSF_Up'])))

                            # Muon SF & uncertainties
                            df = df.assign(total_weight_downMuon=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSFDown[0]'] > 0.) & (x['Muon_ReconstructionSFDown[0]'] > 0.) & (x['Muon_IsolationSFDown[0]'] > 0.))*1.*x['Muon_IdentificationSFDown[0]'] * x['Muon_IsolationSFDown[0]'] * x['Muon_ReconstructionSFDown[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            df = df.assign(total_weight_upMuon=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSFUp[0]'] > 0.) & (x['Muon_ReconstructionSFUp[0]'] > 0.) & (x['Muon_IsolationSFUp[0]'] > 0.))*1.*x['Muon_IdentificationSFUp[0]'] * x['Muon_IsolationSFUp[0]'] * x['Muon_ReconstructionSFUp[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # Electron SF & uncertainties
                            df = df.assign(total_weight_downEle=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFDown[0]'] > 0.) & (x['Electron_ReconstructionSFDown[0]'] > 0.))*1.*x['Electron_IdentificationSFDown[0]']*x['Electron_ReconstructionSFDown[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            df = df.assign(total_weight_upEle=lambda x: (sample.lumiWeight*x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFUp[0]'] > 0.) & (x['Electron_ReconstructionSFUp[0]'] > 0.))*1.*x['Electron_IdentificationSFUp[0]']*x['Electron_ReconstructionSFUp[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))
                            
                            # df = df.assign(total_preweight=lambda x: (sample.lumiWeight * x['Weight_XS'] * x['Weight_CSV_UL'] * x['Weight_GEN_nom'] * x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSFUp[0]'] > 0.) & (x['Electron_ReconstructionSFUp[0]'] > 0.))*1.*x['Electron_IdentificationSFUp[0]']*x['Electron_ReconstructionSFUp[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                                # (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            print("Done with other SFs")

                            # for x in range(306000, 306001):
                            for x in range(306000, 306103):

                                if (sample.process == "ttDL" or sample.process == "ttSL" or sample.process == "ttH"):
                                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(
                                        0.)
                                    df['total_weight_PDF_Weight_{}'.format(x)] = (df['Weight_pdf_variation_{}'.format(
                                        x)]*((df['process'] == sample.process)*1. * self.genfile[(self.genfile['sample'] == sample.process) & (self.genfile['variation'] == 'Weight_pdf_variation_{}'.format(x))]['final_weight_sl_analysis'].values[0])) * df['total_preweight']*df['compare']*1.
                                
                                else:

                                    df.loc[:, 'Weight_pdf_variation_{}'.format(
                                        x)] = 0.
                                    df.loc[:, 'total_weight_PDF_Weight_{}'.format(
                                        x)] = 0.
                                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(
                                        0.)

                            # for x in range(320900, 320901):
                            for x in range(320900, 321001):

                                if (sample.process == "ttbbSL" or sample.process == "ttbbDL"):

                                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(
                                        0.)
                                    
                                    df['total_weight_PDF_Weight_{}'.format(x)] = (df['Weight_pdf_variation_{}'.format(
                                        x)]*((df['process'] == sample.process)*1. * self.genfile[(self.genfile['sample'] == sample.process) & (self.genfile['variation'] == 'Weight_pdf_variation_{}'.format(x))]['final_weight_sl_analysis'].values[0])) * df['total_preweight']*df['compare']*1.


                                else:
                                    df.loc[:, 'Weight_pdf_variation_{}'.format(
                                        x)] = 0.
                                    df.loc[:, 'total_weight_PDF_Weight_{}'.format(x)] = 0.
                                    df['compare'] = df['Weight_pdf_variation_{}'.format(x)].ge(
                                        0.)

                            print("Done with PDF shape calculations")

                            # print ("*"*50)
                            print("Done with all Uncertainties")

                        else:
                        # for JES & JER variations
                            doJES = False
                            if "JESup" in file:
                                syst = "JESup"
                                doJES = True
                            elif "JESdown" in file:
                                syst = "JESdown"
                                doJES = True
                                 
                            if doJES:
                                print("Evaluate SFs for {} files".format(syst))
                                df = self.CalculateSFsEvalSyst(tree,df,syst)
                            else:
                                print("Evaluate SFs for JER files")
                                df = self.CalculateSFs(tree,df) 

                            df = df.assign(sf_weight=lambda x: (sample.lumiWeight*x['Weight_pu69p2'] * x['Weight_JetPUID'] * x['Weight_L1ECALPrefire'] * (((x['N_TightElectrons'] == 1) & (x['Electron_IdentificationSF[0]'] > 0.) & (x['Electron_ReconstructionSF[0]'] > 0.))*1.*x['Electron_IdentificationSF[0]']*x['Electron_ReconstructionSF[0]'] + ((x['N_TightMuons'] == 1) & (x['Muon_IdentificationSF[0]'] > 0.) & (x['Muon_ReconstructionSF[0]'] > 0.) & (x['Muon_IsolationSF[0]'] > 0.))*1.*x['Muon_IdentificationSF[0]'] * x['Muon_IsolationSF[0]'] * x['Muon_ReconstructionSF[0]']) * ((((x['N_LooseMuons'] == 0) & (x['N_TightElectrons'] == 1)) & ((x['Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX'] == 1) | (
                            (x['Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX'] == 1) & (x['Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX'] == 1))) & (x['Weight_ElectronTriggerSF'] > 0)) * 1. * x['Weight_ElectronTriggerSF'] + (((x['N_LooseElectrons'] == 0) & (x['N_TightMuons'] == 1) & (x['Triggered_HLT_IsoMu27_vX'])) & (x['Weight_MuonTriggerSF'] > 0.)) * 1. * x['Weight_MuonTriggerSF'])))

                            # Weight_CSV_UL here corresponds to btagging SF for JES & JER variations
                            df = df.assign(xs_weight=lambda x: x.Weight_XS *
                                     x.Weight_GEN_nom )
                            df = df.assign(
                                total_weight=lambda x: x.xs_weight * x.sf_weight * x.Weight_CSV_UL)

                else:
                    # for DNN, only Weight_CSV_UL is needed
                    df = self.CalculateSFs(tree,df)
                    # print("df bTag SF: ")
                    # print(df["Weight_CSV_UL"])
                    # print(df["Weight_JetPUID"])
                
                

                if concat_df.empty: concat_df = df
                else: concat_df = concat_df.append(df)            
                n_entries += df.shape[0]
            
                # if number of entries exceeds max threshold, add labels and save dataframe
                if (n_entries > self.maxEntries or f == files[-1]):
                    print("*"*50)
                    print("max entries reached ...")

                    # add class labels
                    concat_df = self.addClassLabels(concat_df, sample.categories.categories)
    
                    # add lumi weight
                    concat_df.loc[:,"lumiWeight"] = sample.lumiWeight

                    # add indexing
                    concat_df.set_index([self.varName_Run, self.varName_LumiBlock, self.varName_Event], inplace=True, drop=True)

                    # remove trigger variables
                    concat_df = self.removeTriggerVariables(concat_df)

                    # write data to file
                    self.createDatasets(concat_df, sample.categories.categories, chunkNumber)
                    print("*"*50)

                    # reset counters
                    n_entries = 0
                    concat_df = pd.DataFrame()
            
            # remove file copied from eos space
            if not sample.islocal:
                rmeoscommand = "rm "+file
                print(rmeoscommand)
                try:
                    os.system(rmeoscommand)
                    print ("successfully removed the file")
                except:
                    print ("failed to remove file")

    # ====================================================================

    def applySelections(self, df, sampleSelection):
        if self.baseSelection:
            df = df.query(self.baseSelection)
        if sampleSelection:
            df = df.query(sampleSelection)
        return df

    def addClassLabels(self, df, categories):
        print("adding class labels to df ...")
        split_dfs = []
        for key in categories:
            if categories[key]:
                tmp_df = df.query(categories[key])
            else:
                tmp_df = df
            tmp_df.loc[:, "class_label"] = key
            split_dfs.append(tmp_df)
        # concatenate the split dataframes again
        df = pd.concat(split_dfs)
        return df


    def removeTriggerVariables(self, df):
        df.drop(self.removedVariables, axis=1, inplace=True)
        return df
        
    def createDatasets(self, df, categories, chunkNumber= None):
        for key in categories:
            if chunkNumber is None:
                outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"
            else:
                outFile = self.outputdir+"/"+key+"_" + \
                    self.naming+"_"+str(chunkNumber)+".h5"

            # create dataframe for category
            cat_df = df.query("(class_label == \""+str(key)+"\")")
            print("creating dataset for class label {} with {} entries".format(
                key, cat_df.shape[0]))

            with pd.HDFStore(outFile, "a") as store:
                store.append("data", cat_df, index=False)


    def mergeFiles(self, categories):
        print("="*50)
        print("merging multicore threading files ...")
        for key in categories:
                print("category: {}".format(key))
                threadFiles = [self.outputdir+"/"+key+"_"+self.naming+"_" +
                    str(chunkNumber+1)+".h5" for chunkNumber in range(self.ncores)]
                outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"

                with pd.HDFStore(outFile, "a") as store:
                    for f in threadFiles:
                        print("merging file {}".format(f))
                        if not os.path.exists(f):
                            print("\t-> does not exist?!")
                            continue
                        store.append("data", pd.read_hdf(f), index=False)
                        os.remove(f)
                    print("number of events: {}".format(
                        store.get_storer("data").nrows))
        print("="*50)

    def renameOldFiles(self):
        for key in self.samples:
            sample = self.samples[key]
            for cat in sample.categories.categories:
                outFile = self.outputdir+"/"+cat+"_"+self.naming+".h5"
                if os.path.exists(outFile):
                    print("renaming file {}".format(outFile))
                    os.rename(outFile, outFile+".old")


    def handleOldFiles(self):
        old = []
        actual = []
        rerename = []
        remo = []
        for filename in os.listdir(self.outputdir):
                if filename.endswith(".old"):
                    old.append(filename.split(".")[0])
                else:
                    actual.append(filename.split(".")[0])
        for name in old:
            if name in actual:
                remo.append(name)
            else:
                rerename.append(name)
        for filename in os.listdir(self.outputdir):
            if filename.endswith(".old") and filename.split(".")[0] in remo:
                print("removing file {}".format(filename))
                os.remove(self.outputdir+"/"+filename)
            if filename.endswith(".old") and filename.split(".")[0] in rerename:
                print("re-renaming file {}".format(filename))
                os.rename(self.outputdir+"/"+filename,
                        self.outputdir+"/"+filename[:-4])
    # function to append a list with sample, label and normalization_weight to a list samples

    def createSampleList(self, sList, sample, label=None, nWeight=1):
        """ takes a List a sample and appends a list with category, label and weight. Checks if even/odd splitting was made and therefore adjusts the normalization weight """
        if sample.even_odd:
            nWeight *= 2.
        for cat in sample.categories.categories:
            if label == None:
                sList.append([cat, cat, nWeight])
            else:
                sList.append([cat, label, nWeight])
        return sList

    # function to create a file with all preprocessed samples


    def createSampleFile(self, outPath, sampleList):
        # create file
        samples = []
        processedSamples = ""
        # write samplenames in file
        for sample in sampleList:
            if str(sample[0]) not in samples:
                samples.append(sample[0])
                processedSamples += str(sample[0])+" " + \
                    str(sample[1])+" "+str(sample[2])+"\n"
        with open(outPath+"/sampleFile.dat", "w") as sampleFile:
            sampleFile.write(processedSamples)


    def CalculateSFs(self, tree, df):

        bsfDir = os.path.join(basedir, "data", "BTV", "{}_UL".format(self.dataEra))
        bsfName = os.path.join(bsfDir, "btagging.json.gz")
        print("eataEra is: {}".format(self.dataEra))
        print("btv file is: "+bsfName)

        PUIDsfDir = os.path.join(
            basedir, "data", "PUJetIDSFs", "{}".format(self.dataEra))
        PUIDsfName = os.path.join(PUIDsfDir, "jmar.json.gz")
        
        if bsfName.endswith(".gz"):
            import gzip
            with gzip.open(bsfName, "rt") as f:
                data = f.read().strip()
            btvjson = _core.CorrectionSet.from_string(data)
        else:
            btvjson = _core.CorrectionSet.from_file(bsfName)

        if PUIDsfName.endswith(".gz"):
            import gzip
            with gzip.open(PUIDsfName, "rt") as f:
                data = f.read().strip()
            PUIDjson = _core.CorrectionSet.from_string(data)
        else:
            PUIDjson = _core.CorrectionSet.from_file(PUIDsfName)

        jet_flavor = tree.pandas.df("Jet_Flav")
        jet_eta = tree.pandas.df("Jet_Eta")
        jet_pt = tree.pandas.df("Jet_Pt")
        jet_bTag = tree.pandas.df("Jet_CSV")
        njet = tree.pandas.df("N_Jets")

        # https: // cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_2016postVFP_UL.html
        jet_PUIDsf = []
        jet_btagsf = []
        for i in range(njet.size):

            jet_PUIDsf_perevent = 1.
            jet_btagsf_perevent = 1.

            for j in range(jet_pt["Jet_Pt"][i].size):	


                # btagging SF	
                jet_btagsf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))

                if float(jet_pt['Jet_Pt'][i][j]) < 50.:	
                    jet_PUIDsf_perevent *= PUIDjson["PUJetID_eff"].evaluate(	
                        float(jet_eta['Jet_Eta'][i][j]), float(jet_pt['Jet_Pt'][i][j]), "nom", "L")

            jet_PUIDsf.append(jet_PUIDsf_perevent)
            jet_btagsf.append(jet_btagsf_perevent)

        df.loc[:, "Weight_CSV_UL"] = 0.
        df.loc[:, "Weight_JetPUID"] = 0.
        jet_btagsf = pd.DataFrame(jet_btagsf, columns=["Weight_CSV_UL"])
        jet_PUIDsf = pd.DataFrame(jet_PUIDsf, columns=["Weight_JetPUID"])

        df.update(jet_btagsf)
        df.update(jet_PUIDsf)
        return df


    def CalculateSFsEval(self, tree, df):

        bsfDir = os.path.join(basedir, "data", "BTV",
                              "{}_UL".format(self.dataEra))
        bsfName = os.path.join(bsfDir, "btagging.json.gz")

        print("btv file is: "+bsfName)

        PUIDsfDir = os.path.join(
            basedir, "data", "PUJetIDSFs", "{}".format(self.dataEra))
        PUIDsfName = os.path.join(PUIDsfDir, "jmar.json.gz")

        if bsfName.endswith(".gz"):
            import gzip
            with gzip.open(bsfName, "rt") as f:
                data = f.read().strip()
            btvjson = _core.CorrectionSet.from_string(data)
        else:
            btvjson = _core.CorrectionSet.from_file(bsfName)

        if PUIDsfName.endswith(".gz"):
            import gzip
            with gzip.open(PUIDsfName, "rt") as f:
                data = f.read().strip()
            PUIDjson = _core.CorrectionSet.from_string(data)
        else:
            PUIDjson = _core.CorrectionSet.from_file(PUIDsfName)

        jet_flavor = tree.pandas.df("Jet_Flav")
        jet_eta = tree.pandas.df("Jet_Eta")
        jet_pt = tree.pandas.df("Jet_Pt")
        jet_bTag = tree.pandas.df("Jet_CSV")
        njet = tree.pandas.df("N_Jets")

        # https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_2016postVFP_UL.html

        jet_PUIDsf = []
        jet_PUIDsfup = []
        jet_PUIDsfdown = []

        jet_btagsf = []
        jet_btagsf_uplf = []
        jet_btagsf_downlf = []
        jet_btagsf_uphf = []
        jet_btagsf_downhf = []
        jet_btagsf_uphfstats1 = []
        jet_btagsf_downhfstats1 = []
        jet_btagsf_uphfstats2 = []
        jet_btagsf_downhfstats2 = []
        jet_btagsf_uplfstats1 = []
        jet_btagsf_downlfstats1 = []
        jet_btagsf_uplfstats2 = []
        jet_btagsf_downlfstats2 = []
        jet_btagsf_upcferr1 = []
        jet_btagsf_downcferr1 = []
        jet_btagsf_upcferr2 = []
        jet_btagsf_downcferr2 = []

        # loop over event
        for i in range(njet.size):

            jet_PUIDsf_perevent = 1.
            jet_PUIDsfup_perevent = 1.
            jet_PUIDsfdown_perevent = 1.

            jet_btagsf_perevent = 1.
            jet_btagsf_uplf_perevent = 1.	            
            jet_btagsf_downlf_perevent = 1.	            
            jet_btagsf_uphf_perevent = 1.	            
            jet_btagsf_downhf_perevent = 1.	                    
            jet_btagsf_uphfstats1_perevent = 1.	            
            jet_btagsf_downhfstats1_perevent = 1.	                    
            jet_btagsf_uphfstats2_perevent = 1.	            
            jet_btagsf_downhfstats2_perevent = 1.	                    
            jet_btagsf_uplfstats1_perevent = 1.	
            jet_btagsf_downlfstats1_perevent = 1.	
            jet_btagsf_uplfstats2_perevent = 1.	
            jet_btagsf_downlfstats2_perevent = 1.	
            jet_btagsf_upcferr1_perevent = 1.	
            jet_btagsf_downcferr1_perevent = 1.	
            jet_btagsf_upcferr2_perevent = 1.	
            jet_btagsf_downcferr2_perevent = 1.

            for j in range(jet_pt["Jet_Pt"][i].size):

                if float(jet_pt['Jet_Pt'][i][j]) < 50.:	

                    jet_PUIDsf_perevent *= PUIDjson["PUJetID_eff"].evaluate(	
                        float(jet_eta['Jet_Eta'][i][j]), float(jet_pt['Jet_Pt'][i][j]), "nom", "L")	
                    jet_PUIDsfup_perevent *= PUIDjson["PUJetID_eff"].evaluate(	
                        float(jet_eta['Jet_Eta'][i][j]), float(jet_pt['Jet_Pt'][i][j]), "up", "L")	
                    jet_PUIDsfdown_perevent *= PUIDjson["PUJetID_eff"].evaluate(	
                        float(jet_eta['Jet_Eta'][i][j]), float(jet_pt['Jet_Pt'][i][j]), "down", "L")


                # btagging SF	
                jet_btagsf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))

                if jet_flavor['Jet_Flav'][i][j] != 4: 	

                    jet_btagsf_uphf_perevent *= btvjson["deepJet_shape"].evaluate("up_hf", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhf_perevent *= btvjson["deepJet_shape"].evaluate("down_hf", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	

                    jet_btagsf_uphfstats1_perevent *= btvjson["deepJet_shape"].evaluate("up_hfstats1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhfstats1_perevent *= btvjson["deepJet_shape"].evaluate("down_hfstats1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uphfstats2_perevent *= btvjson["deepJet_shape"].evaluate("up_hfstats2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhfstats2_perevent *= btvjson["deepJet_shape"].evaluate("down_hfstats2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	


                    jet_btagsf_uplf_perevent *= btvjson["deepJet_shape"].evaluate("up_lf", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlf_perevent *= btvjson["deepJet_shape"].evaluate("down_lf", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uplfstats1_perevent *= btvjson["deepJet_shape"].evaluate("up_lfstats1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlfstats1_perevent *= btvjson["deepJet_shape"].evaluate("down_lfstats1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uplfstats2_perevent *= btvjson["deepJet_shape"].evaluate("up_lfstats2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlfstats2_perevent *= btvjson["deepJet_shape"].evaluate("down_lfstats2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    
                    jet_btagsf_upcferr1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downcferr1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_upcferr2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downcferr2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	

                # c jets	
                elif jet_flavor['Jet_Flav'][i][j] == 4:	
                    jet_btagsf_upcferr1_perevent *= btvjson["deepJet_shape"].evaluate("up_cferr1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downcferr1_perevent *= btvjson["deepJet_shape"].evaluate("down_cferr1", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_upcferr2_perevent *= btvjson["deepJet_shape"].evaluate("up_cferr2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downcferr2_perevent *= btvjson["deepJet_shape"].evaluate("down_cferr2", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    
                    jet_btagsf_uphf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	

                    jet_btagsf_uphfstats1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhfstats1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uphfstats2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downhfstats2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	


                    jet_btagsf_uplf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uplfstats1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlfstats1_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_uplfstats2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	
                    jet_btagsf_downlfstats2_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(	
                        float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))	

            
            
            jet_PUIDsf.append(jet_PUIDsf_perevent)
            jet_PUIDsfup.append(jet_PUIDsfup_perevent)
            jet_PUIDsfdown.append(jet_PUIDsfdown_perevent)
            jet_btagsf.append(jet_btagsf_perevent)
            jet_btagsf_uplf.append(jet_btagsf_uplf_perevent)
            jet_btagsf_downlf.append(jet_btagsf_downlf_perevent)
            jet_btagsf_uphf.append(jet_btagsf_uphf_perevent)
            jet_btagsf_downhf.append(jet_btagsf_downhf_perevent)
            jet_btagsf_uphfstats1.append(jet_btagsf_uphfstats1_perevent)
            jet_btagsf_downhfstats1.append(jet_btagsf_downhfstats1_perevent)
            jet_btagsf_uphfstats2.append(jet_btagsf_uphfstats2_perevent)
            jet_btagsf_downhfstats2.append(jet_btagsf_downhfstats2_perevent)
            jet_btagsf_uplfstats1.append(jet_btagsf_uplfstats1_perevent)
            jet_btagsf_downlfstats1.append(jet_btagsf_downlfstats1_perevent)
            jet_btagsf_uplfstats2.append(jet_btagsf_uplfstats2_perevent)
            jet_btagsf_downlfstats2.append(jet_btagsf_downlfstats2_perevent)
            jet_btagsf_upcferr1.append(jet_btagsf_upcferr1_perevent)
            jet_btagsf_downcferr1.append(jet_btagsf_downcferr1_perevent)
            jet_btagsf_upcferr2.append(jet_btagsf_upcferr2_perevent)
            jet_btagsf_downcferr2.append(jet_btagsf_downcferr2_perevent)


        df.loc[:, "Weight_CSV_UL"] = 0.
        df.loc[:, "Weight_CSV_UL_uplf"] = 0.
        df.loc[:, "Weight_CSV_UL_downlf"] = 0.
        df.loc[:, "Weight_CSV_UL_uphf"] = 0.
        df.loc[:, "Weight_CSV_UL_downhf"] = 0.
        df.loc[:, "Weight_CSV_UL_uphfstats1"] = 0.
        df.loc[:, "Weight_CSV_UL_downhfstats1"] = 0.
        df.loc[:, "Weight_CSV_UL_uphfstats2"] = 0.
        df.loc[:, "Weight_CSV_UL_downhfstats2"] = 0.
        df.loc[:, "Weight_CSV_UL_uplfstats1"] = 0.
        df.loc[:, "Weight_CSV_UL_downlfstats1"] = 0.
        df.loc[:, "Weight_CSV_UL_uplfstats2"] = 0.
        df.loc[:, "Weight_CSV_UL_downlfstats2"] = 0.
        df.loc[:, "Weight_CSV_UL_upcferr1"] = 0.
        df.loc[:, "Weight_CSV_UL_downcferr1"] = 0.
        df.loc[:, "Weight_CSV_UL_upcferr2"] = 0.
        df.loc[:, "Weight_CSV_UL_downcferr2"] = 0.

        df.loc[:, "Weight_JetPUID"] = 0.
        df.loc[:, "Weight_JetPUID_up"] = 0.
        df.loc[:, "Weight_JetPUID_down"] = 0.

        jet_btagsf = pd.DataFrame(jet_btagsf, columns=["Weight_CSV_UL"])
        jet_btagsf_uplf = pd.DataFrame(
            jet_btagsf_uplf, columns=["Weight_CSV_UL_uplf"])
        jet_btagsf_downlf = pd.DataFrame(
            jet_btagsf_downlf, columns=["Weight_CSV_UL_downlf"])
        jet_btagsf_uphf = pd.DataFrame(
            jet_btagsf_uphf, columns=["Weight_CSV_UL_uphf"])
        jet_btagsf_downhf = pd.DataFrame(
            jet_btagsf_downhf, columns=["Weight_CSV_UL_downhf"])
        jet_btagsf_uplfstats1 = pd.DataFrame(jet_btagsf_uplfstats1, columns=[
                                             "Weight_CSV_UL_uplfstats1"])
        jet_btagsf_downlfstats1 = pd.DataFrame(
            jet_btagsf_downlfstats1, columns=["Weight_CSV_UL_downlfstats1"])
        jet_btagsf_uplfstats2 = pd.DataFrame(jet_btagsf_uplfstats2, columns=[
                                             "Weight_CSV_UL_uplfstats2"])
        jet_btagsf_downlfstats2 = pd.DataFrame(jet_btagsf_downlfstats2, columns=[
                                               "Weight_CSV_UL_downlfstats2"])
        jet_btagsf_uphfstats1 = pd.DataFrame(jet_btagsf_uphfstats1, columns=[
                                             "Weight_CSV_UL_uphfstats1"])
        jet_btagsf_downhfstats1 = pd.DataFrame(jet_btagsf_downhfstats1, columns=[
                                               "Weight_CSV_UL_downhfstats1"])
        jet_btagsf_uphfstats2 = pd.DataFrame(jet_btagsf_uphfstats2, columns=[
                                             "Weight_CSV_UL_uphfstats2"])
        jet_btagsf_downhfstats2 = pd.DataFrame(jet_btagsf_downhfstats2, columns=[
                                               "Weight_CSV_UL_downhfstats2"])
        jet_btagsf_upcferr1 = pd.DataFrame(jet_btagsf_upcferr1, columns=[
                                           "Weight_CSV_UL_upcferr1"])
        jet_btagsf_downcferr1 = pd.DataFrame(jet_btagsf_downcferr1, columns=[
                                             "Weight_CSV_UL_downcferr1"])
        jet_btagsf_upcferr2 = pd.DataFrame(jet_btagsf_upcferr2, columns=[
                                           "Weight_CSV_UL_upcferr2"])
        jet_btagsf_downcferr2 = pd.DataFrame(jet_btagsf_downcferr2, columns=[
                                             "Weight_CSV_UL_downcferr2"])

        jet_PUIDsf = pd.DataFrame(jet_PUIDsf, columns=["Weight_JetPUID"]) 
        jet_PUIDsfup = pd.DataFrame(jet_PUIDsfup, columns=["Weight_JetPUID_up"])
        jet_PUIDsfdown = pd.DataFrame(
            jet_PUIDsfdown, columns=["Weight_JetPUID_down"])

        df.update(jet_btagsf)
        df.update(jet_btagsf_uplf)
        df.update(jet_btagsf_downlf)
        df.update(jet_btagsf_uphf)
        df.update(jet_btagsf_downhf)
        df.update(jet_btagsf_uplfstats1)
        df.update(jet_btagsf_downlfstats1)
        df.update(jet_btagsf_uplfstats2)
        df.update(jet_btagsf_downlfstats2)
        df.update(jet_btagsf_uphfstats1)
        df.update(jet_btagsf_downhfstats1)
        df.update(jet_btagsf_uphfstats2)
        df.update(jet_btagsf_downhfstats2)
        df.update(jet_btagsf_upcferr1)
        df.update(jet_btagsf_downcferr1)
        df.update(jet_btagsf_upcferr2)
        df.update(jet_btagsf_downcferr2)

        df.update(jet_PUIDsf)
        df.update(jet_PUIDsfup)
        df.update(jet_PUIDsfdown)
        return df



    def CalculateSFsEvalSyst(self, tree, df, syst):

        bsfDir = os.path.join(basedir, "data", "BTV", "{}_UL".format(self.dataEra))
        bsfName = os.path.join(bsfDir, "btagging.json.gz")
        print("eataEra is: {}".format(self.dataEra))
        print("btv file is: "+bsfName)

        PUIDsfDir = os.path.join(
            basedir, "data", "PUJetIDSFs", "{}".format(self.dataEra))
        PUIDsfName = os.path.join(PUIDsfDir, "jmar.json.gz")
        
        if bsfName.endswith(".gz"):
            import gzip
            with gzip.open(bsfName, "rt") as f:
                data = f.read().strip()
            btvjson = _core.CorrectionSet.from_string(data)
        else:
            btvjson = _core.CorrectionSet.from_file(bsfName)

        if PUIDsfName.endswith(".gz"):
            import gzip
            with gzip.open(PUIDsfName, "rt") as f:
                data = f.read().strip()
            PUIDjson = _core.CorrectionSet.from_string(data)
        else:
            PUIDjson = _core.CorrectionSet.from_file(PUIDsfName)

        jet_flavor = tree.pandas.df("Jet_Flav")
        jet_eta = tree.pandas.df("Jet_Eta")
        jet_pt = tree.pandas.df("Jet_Pt")
        jet_bTag = tree.pandas.df("Jet_CSV")
        njet = tree.pandas.df("N_Jets")

        # https: // cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_2016postVFP_UL.html
        jet_PUIDsf = []
        jet_btagsf = []

        if syst == "JESup":
            dosyst = "up_jes"
        if syst == "JESdown":
            dosyst = "down_jes"

        for i in range(njet.size):

            jet_PUIDsf_perevent = 1.
            jet_btagsf_perevent = 1.

            for j in range(jet_pt["Jet_Pt"][i].size):	


                # btagging SF
                if jet_flavor['Jet_Flav'][i][j] != 4: 

                    jet_btagsf_perevent *= btvjson["deepJet_shape"].evaluate(dosyst, jet_flavor['Jet_Flav'][i][j], abs(float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))

                else:
                    
                    jet_btagsf_perevent *= btvjson["deepJet_shape"].evaluate("central", jet_flavor['Jet_Flav'][i][j], abs(float(jet_eta['Jet_Eta'][i][j])), float(jet_pt['Jet_Pt'][i][j]), float(jet_bTag['Jet_CSV'][i][j]))

                if float(jet_pt['Jet_Pt'][i][j]) < 50.:	
                    jet_PUIDsf_perevent *= PUIDjson["PUJetID_eff"].evaluate(	
                        float(jet_eta['Jet_Eta'][i][j]), float(jet_pt['Jet_Pt'][i][j]), "nom", "L")

            jet_PUIDsf.append(jet_PUIDsf_perevent)
            jet_btagsf.append(jet_btagsf_perevent)

        df.loc[:, "Weight_CSV_UL"] = 0.
        df.loc[:, "Weight_JetPUID"] = 0.
        jet_btagsf = pd.DataFrame(jet_btagsf, columns=["Weight_CSV_UL"])
        jet_PUIDsf = pd.DataFrame(jet_PUIDsf, columns=["Weight_JetPUID"])

        df.update(jet_btagsf)
        df.update(jet_PUIDsf)
        return df