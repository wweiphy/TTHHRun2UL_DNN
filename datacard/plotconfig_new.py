#samples named in the rootfile
samples = {   'ttmb': {   'info': {   'color': 633L,
                            'label': 't#bar{t}+2b',
                            'typ': 'bkg'},
                'plot': True},
    'ttnb': {   'info': {   'color': 636L,
                            'label': 't#bar{t}+4b',
                            'typ': 'bkg'},
                'plot': True},
    'ttH': {   'info': {   'color': 416L,
                           'label': 't#bar{t}+H',
                           'typ': 'bkg'},
               'plot': True},
    'ttHH': {   'info': {   'color': 601L,
                            'label': 't#bar{t}+HH',
                            'typ': 'signal'},
                'plot': True},
    'ttZ': {   'info': {   'color': 416L,
                           'label': 't#bar{t}+Z',
                           'typ': 'bkg'},
               'plot': True},
    'ttZH': {   'info': {   'color': 416L,
                            'label': 't#bar{t}+ZH',
                            'typ': 'bkg'},
                'plot': True},
    'ttZZ': {   'info': {   'color': 416L,
                            'label': 't#bar{t}+ZZ',
                            'typ': 'bkg'},
                'plot': True},
    'ttcc': {   'info': {   'color': 633L,
                            'label': 't#bar{t}+c#bar{c}',
                            'typ': 'bkg'},
                'plot': True},
    'ttlf': {   'info': {   'color': 625L,
                            'label': 't#bar{t}+lf',
                            'typ': 'bkg'},
                'plot': True}}

#combined samples
plottingsamples = {   'ttmb': {   'addSamples': [   'tt2b'],
                'color': 633L,
                'label': 't#bar{t}2b',
                'typ': 'bkg'},
    'ttnb': {   'addSamples': [   'tt4b'],
                'color': 636L,
                'label': 't#bar{t}4b',
                'typ': 'bkg'},
    'ttH': {   'addSamples': [   'ttH'],
               'color': 416L,
               'label': 't#bar{t}H',
               'typ': 'bkg'},
    'ttHH': {   'addSamples': [   'ttHH'],
                'color': 601L,
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
    'ttcc': {   'addSamples': [   'ttcc'],
                'color': 633L,
                'label': 't#bar{t}cc',
                'typ': 'bkg'},
    'ttlf': {   'addSamples': [   'ttlf'],
                'color': 625L,
                'label': 't#bar{t}lf',
                'typ': 'bkg'}}

#systematics to be plotted
systematics = [   ]

# order of the stack processes, descending from top to bottom
sortedprocesses = [   'ttHH',
    'ttlf',
    'ttcc',
    'ttmb',
    'ttnb',
    'ttH',
    'ttZZ',
    'ttZH',
    'ttZ']

#options for the plotting style
plotoptions = {   'cmslabel': 'private Work',
    'data': 'data_obs',
    'datalabel': 'data',
    'logarithmic': False,
    'lumiLabel': '41.5',
    'nominalKey': '$CHANNEL__$PROCESS',
    'normalize': False,
    'ratio': '#frac{S+B}{B}',
    'shape': False,
    'signalScaling': -1,
    'splitLegend': True,
    'statErrorband': True,
    'systematicKey': '$CHANNEL__$PROCESS__$SYSTEMATIC',
    # "combineflag" : "shapes_prefit"/"shapes_fit_s",
    # "signallabel" : "Signal"
}
    