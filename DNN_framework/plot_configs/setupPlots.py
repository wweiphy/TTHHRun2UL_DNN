import ROOT

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
