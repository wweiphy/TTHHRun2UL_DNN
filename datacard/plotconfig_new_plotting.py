#samples named in the rootfile
samples = {  

    'tt': {'info': {'color': 633L,
                    'label': 't#bar{t}',
                    'typ': 'bkg'},
           'plot': True},

    'tt4b': {   'info': {   'color': 636L,
                            'label': 't#bar{t}+4b',
                            'typ': 'bkg'},
                'plot': True},
    'ttH': {   'info': {   'color': 417L,
                           'label': 't#bar{t}+H',
                           'typ': 'bkg'},
               'plot': True},
    'ttHH': {   'info': {   'color': 600L,
                            'label': 't#bar{t}+HH',
                            'typ': 'signal'},
                'plot': True},
    'ttZ': {   'info': {   'color': 393L,
                           'label': 't#bar{t}+Z',
                           'typ': 'bkg'},
               'plot': True},
    'ttZH': {   'info': {   'color': 800L,
                            'label': 't#bar{t}+ZH',
                            'typ': 'bkg'},
                'plot': True},
    'ttZZ': {   'info': {   'color': 860L,
                            'label': 't#bar{t}+ZZ',
                            'typ': 'bkg'},
                'plot': True},
    'ttbb': {   'info': {   'color': 634L,
                            'label': 't#bar{t}+b#bar{b}',
                            'typ': 'bkg'},
                'plot': True},

    }

#combined samples
plottingsamples = { 

    'tt': {'addSamples': ['tt'],
           'color': 633L,
           'label': 't#bar{t}',
           'typ': 'bkg'},
    
    'tt4b': {   'addSamples': [   'tt4b'],
                'color': 636L,
                'label': 't#bar{t}4b',
                'typ': 'bkg'},
    'ttH': {   'addSamples': [   'ttH'],
               'color': 417L,
               'label': 't#bar{t}H',
               'typ': 'bkg'},
    'ttHH': {   'addSamples': [   'ttHH'],
                'color': 600L,
                'label': 't#bar{t}HH',
                'typ': 'signal'},
    'ttZ': {   'addSamples': [   'ttZ'],
               'color': 393L,
               'label': 't#bar{t}Z',
               'typ': 'bkg'},
    'ttZH': {   'addSamples': [   'ttZH'],
                'color': 800L,
                'label': 't#bar{t}ZH',
                'typ': 'bkg'},
    'ttZZ': {   'addSamples': [   'ttZZ'],
                'color': 860L,
                'label': 't#bar{t}ZZ',
                'typ': 'bkg'},
    'ttbb': {   'addSamples': [   'ttbb'],
                'color': 634L,
                'label': 't#bar{t}bb',
                'typ': 'bkg'},

    }

#systematics to be plotted
systematics = [   ]

# order of the stack processes, descending from top to bottom
sortedprocesses = [   
    'ttHH',
    'tt',
    'ttbb',
    'tt4b',
    'ttH',
    'ttZZ',
    'ttZH',
    'ttZ'
    ]

#options for the plotting style
plotoptions = {   'cmslabel': 'private Work',
    'data': 'data_obs',
    'datalabel': 'data',
    'logarithmic': False,
    'lumiLabel': '59.8',
    # 'lumiLabel': '41.5',
    # 'lumiLabel': '101.3',
    # 'lumiLabel': '16.81', # post
    # 'lumiLabel': '19.52',  # pre
    'nominalKey': '$CHANNEL__$PROCESS',
    'normalize': False,
    'ratio': '#frac{data}{MC Background}',
    'shape': False,
    'signalScaling': -1,
    'splitLegend': True,
    'statErrorband': True,
    'systematicKey': '$CHANNEL__$PROCESS__$SYSTEMATIC',
    # "combineflag" : "shapes_prefit"/"shapes_fit_s",
    # "signallabel" : "Signal"
}
    
