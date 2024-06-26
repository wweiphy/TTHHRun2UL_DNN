import os
import uproot
import pandas as pd
import numpy as np
import re
import glob
import multiprocessing as mp


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
    def __init__(self, sampleName, ntuples, categories, selections=None, ownVars=[], even_odd=False, lumiWeight=1.):
        self.sampleName = sampleName
        self.ntuples = ntuples
        self.selections = selections
        self.categories = categories
        self.ownVars = ownVars
        self.lumiWeight = lumiWeight
        self.even_odd = even_odd
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
    def __init__(self, outputdir, tree=['MVATree'], naming='', maxEntries=50000, varName_Run='Evt_Run', varName_LumiBlock='Evt_Lumi', varName_Event='Evt_ID',ncores=1):
        # settings for paths
        self.outputdir = outputdir
        self.naming = naming
        self.tree = tree
        self.varName_Run = varName_Run
        self.varName_LumiBlock = varName_LumiBlock
        self.varName_Event = varName_Event

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

        # add trigger variables to variable list
        self.addVariables(self.triggerVariables)


    def searchVariablesInTriggerString(self, string):
        # split trigger string into smaller bits
        splitters = [")", "(", "==", ">=", ">=", ">", "<", "="]

        candidates = string.split(" ")
        for splt in splitters:
            candidates = [item for c in candidates for item in c.split(splt)]

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
                varName_Run=self.varName_Run,
                varName_LumiBlock=self.varName_LumiBlock,
                varName_Event=self.varName_Event,
            )

            # remove the own variables
            self.removeVariables(self.samples[key].ownVars)
            self.createSampleList(sampleList, self.samples[key])
            print("done.")
        # write file with preprocessed samples
        self.createSampleFile(self.outputdir, sampleList)

        # handle old files
        self.handleOldFiles()


    def processSample(self, sample, varName_Run, varName_LumiBlock, varName_Event):
        # print sample info
        sample.printInfo()

        # collect ntuple files
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
        pool.map(self.processChunk, chunks)

        # concatenate single thread files
        self.mergeFiles(sample.categories.categories)

    def processChunk(self, sample, files, chunkNumber):

        files = [f for f in files if not f == ""]
        
        n_entries = 0
        concat_df = pd.DataFrame()
        n_files = len(files)
        for iF, f in enumerate(files):
            print("chunk #{}: starting file ({}/{}): {}".format(chunkNumber, iF+1, n_files, f))
            
            # add full path for the files in eos space
            f_full = "root://cmseos.fnal.gov/" + f
            file = f.split("/")[-1]
            print(file)
            
            # copy file from eos space into local 
            copyeoscommand = "xrdcp "+f_full+" ."
            print("copy file {} from eos space".format(copyeoscommand))
            os.system(copyeoscommand)
            for tr in self.tree:
                # open root file
                with uproot.open(file) as rf:
                    # get TTree
                  try:
                        tree = rf[tr]
                  except:
                        print("could not open "+str(tr)+" in ROOT file")
                        continue

                if tree.numentries == 0:
                   print(str(tr)+" has no entries - skipping file")
                   continue

                # convert to dataframe
                df = tree.pandas.df([v for v in self.variables])

                # delete subentry index
                try: df = df.reset_index(1, drop = True)
                except: None
                print(df)
                
                # handle vector variables, loop over them
                for vecvar in self.vector_variables:

                    # load dataframe with vector variable
                    vec_df = tree.pandas.df(vecvar)

                    # loop over inices in vecvar list
                    for idx in self.vector_variables[vecvar]:

                        # slice the index
                        idx_df = vec_df.loc[ (slice(None), slice(idx,idx)), :]
                        idx_df = idx_df.reset_index(1, drop = True)

                        # define name for column in df
                        col_name = str(vecvar)+"["+str(idx)+"]"

                        # initialize column in original dataframe
                        df.loc[:,col_name] = 0.
                        # append column to original dataframe
                        df.update( idx_df[vecvar].rename(col_name) )


                # apply event selection
                df = self.applySelections(df, sample.selections)

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
            
            # remove file copied in local space 
            rmeoscommand = "rm "+file
            print(rmeoscommand)
            try:
                os.system(rmeoscommand)
                print ("successfully removed the file")
            except:
                print ("failed to remove file")
                continue
    # ====================================================================

    def applySelections(self, df, sampleSelection):
        if self.baseSelection:
            df = df.query(self.baseSelection)
        if sampleSelection:
            df = df.query(sampleSelection)
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

    # function to append a list with sample, label and normalization_weight to a list samples

    def createSampleList(self, sList, label=None, nWeight=1):
        """ takes a List a sample and appends a list with category, label and weight. Checks if even/odd splitting was made and therefore adjusts the normalization weight """
        if self.sample.even_odd:
            nWeight *= 2.
        for cat in self.sample.categories.categories:
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
