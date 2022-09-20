import ROOT
import numpy as np


def GetPlotColor(cls):
    color_dict = {
        "ttHH":         ROOT.kBlue,
        "ttZ":          ROOT.kYellow-7,
        "ttZbb":        ROOT.kYellow-7,
        "ttZH":         ROOT.kOrange,
        "ttZZ":         ROOT.kAzure,
        "ttH":          ROOT.kGreen+1,
        "ttHbb":        ROOT.kGreen+1,
        "ttHnonbb":     ROOT.kYellow-7,
        "ttlf":         ROOT.kRed-7,
        "ttcc":         ROOT.kRed+1,
        "ttbbb":        ROOT.kRed+3,
        "tt4b":         ROOT.kRed+4,
        "ttbb":         ROOT.kRed+2,
        "tt2b":         ROOT.kRed+1,
        "ttb":          ROOT.kRed-2,
        "ttmb":         ROOT.kRed-1,
        "ttnb":         ROOT.kRed+4,
        "sig":          ROOT.kCyan,
        "bkg":          ROOT.kOrange,
    }

    return color_dict[cls]


def GetyTitle(privateWork=False):
    # if privateWork flag is enabled, normalize plots to unit area
    if privateWork:
        return "normalized to unit area"
    return "Events expected"


# ===============================================
# SETUP OF HISTOGRAMS
# ===============================================

def setupHistogram(
        values, weights,
        nbins, bin_range,
        xtitle, ytitle,
        color=ROOT.kBlack, filled=True):
    # define histogram
    histogram = ROOT.TH1D(xtitle, "", nbins, *bin_range)
    histogram.Sumw2(True)

    for v, w in zip(values, weights):
        histogram.Fill(v, w)

    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.055)
    histogram.GetXaxis().SetLabelSize(0.055)

    histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor(ROOT.kBlack)
        histogram.SetFillColor(color)
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor(color)
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    return histogram


def setupConfusionMatrix(matrix, ncls, xtitle, ytitle, binlabel, errors=None):
    # check if errors for matrix are given
    has_errors = isinstance(errors, np.ndarray)
    #print(has_errors)

    # init histogram
    cm = ROOT.TH2D("confusionMatrix", "", ncls, 0, ncls, ncls, 0, ncls)
    cm.SetStats(False)
    ROOT.gStyle.SetPaintTextFormat(".3f")

    for xit in range(cm.GetNbinsX()):
        for yit in range(cm.GetNbinsY()):
            cm.SetBinContent(xit+1, yit+1, matrix[xit, yit])
            if has_errors:
                cm.SetBinError(xit+1, yit+1, errors[xit, yit])

    cm.GetXaxis().SetTitle(xtitle)
    cm.GetYaxis().SetTitle(ytitle)

    cm.SetMarkerColor(ROOT.kWhite)

    minimum = np.min(matrix)
    maximum = np.max(matrix)

    cm.GetZaxis().SetRangeUser(minimum, maximum)

    for xit in range(ncls):
        cm.GetXaxis().SetBinLabel(xit+1, binlabel[xit])
    for yit in range(ncls):
        cm.GetYaxis().SetBinLabel(yit+1, binlabel[yit])

    cm.GetXaxis().SetLabelSize(0.05)
    cm.GetYaxis().SetLabelSize(0.05)
    cm.SetMarkerSize(2.)
    if cm.GetNbinsX() > 6:
        cm.SetMarkerSize(1.5)
    if cm.GetNbinsX() > 8:
        cm.SetMarkerSize(1.)

    return cm
