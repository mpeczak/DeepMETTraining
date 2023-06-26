#!/usr/bin/env python

import sys
import uproot
import numpy as np
import h5py
import progressbar
import os

widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]

def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


import optparse

#configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input file', default='', type='string')
parser.add_option('-o', '--output', dest='output', help='output file', default='', type='string')
parser.add_option("-N", "--maxevents", dest='maxevents', help='max number of events', default=-1, type='int')
parser.add_option("--data", dest="data", action="store_true", default=False, help="input is data. The default is MC")
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

varList = [
    'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi',
    'nPFCands', 'PFCands_pt', 'PFCands_eta', 'PFCands_phi',
    'PFCands_pdgId', 'PFCands_charge', 'PFCands_mass',
    'PFCands_d0', 'PFCands_dz', 'PFCands_fromPV',
    'PFCands_puppiWeightNoLep', 'PFCands_puppiWeight',
]

# event-level variables
varList_evt = [
    'fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo',
    'PV_npvs', 'PV_npvsGood', 'nMuon'
]

varList_mc = [
    'GenMET_pt', 'GenMET_phi',
]

d_encoding = {
    b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
    b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
    b'PFCands_fromPV':{0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3}
}

if not opt.data:
    varList = varList + varList_mc
varList = varList + varList_evt
    
uptree = uproot.open(opt.input + ':Events')
tree = uptree.arrays( varList )

# general setup
maxNPF = 4500
nFeatures = 14

maxEntries = len(tree['nPFCands']) if opt.maxevents==-1 else opt.maxevents
# input PF candidates
X = np.zeros(shape=(maxEntries,maxNPF,nFeatures), dtype=float, order='F')
# recoil estimators
Y = np.zeros(shape=(maxEntries,2), dtype=float, order='F')
# leptons 
XLep = np.zeros(shape=(maxEntries, 2, nFeatures), dtype=float, order='F')
# event-level information
EVT = np.zeros(shape=(maxEntries,len(varList_evt)), dtype=float, order='F')

print(X.shape)

# loop over events
for e in progressbar.progressbar(range(maxEntries), widgets=widgets):
    Leptons = []
    for ilep in range(min(2, tree['nMuon'][e])):
        Leptons.append( (tree['Muon_pt'][e][ilep], tree['Muon_eta'][e][ilep], tree['Muon_phi'][e][ilep]) )
        
    # get momenta
    ipf = 0
    ilep = 0
    for j in range(tree['nPFCands'][e]):
        if ipf == maxNPF:
            break

        pt = tree['PFCands_pt'][e][j]
        #if pt < 0.5:
        #    continue
        eta = tree['PFCands_eta'][e][j]
        phi = tree['PFCands_phi'][e][j]
       
        pf = X[e][ipf]

        isLep = False
        for lep in Leptons:
            if deltaR( eta, phi, lep[1], lep[2] )<0.001 and abs(pt/lep[0]-1.0)<0.4:
                # pfcand matched to the muon
                # fill into XLep instead
                pf = XLep[e][ilep]
                ilep += 1
                Leptons.remove(lep)
                break
        if not isLep:
            ipf += 1
                 
        # 4-momentum
        pf[0] = pt
        pf[1] = pt * np.cos(phi)
        pf[2] = pt * np.sin(phi)
        pf[3] = eta
        pf[4] = phi
        pf[5] = tree['PFCands_d0'][e][j]
        pf[6] = tree['PFCands_dz'][e][j]
        pf[7] = tree['PFCands_puppiWeightNoLep'][e][j]
        pf[8] = tree['PFCands_mass'][e][j]
        #pf[9] = tree[b'PF_hcalFraction'][e][j]
        pf[9] = 0. # variable not found in input data, but also not used in training
        pf[10] = tree['PFCands_puppiWeight'][e][j]
        # encoding
        pf[11] = d_encoding[b'PFCands_pdgId' ][float(tree['PFCands_pdgId' ][e][j])]
        pf[12] = d_encoding[b'PFCands_charge'][float(tree['PFCands_charge'][e][j])]
        pf[13] = d_encoding[b'PFCands_fromPV'][float(tree['PFCands_fromPV'][e][j])]

    # truth info
    Y[e][0] += tree['GenMET_pt'][e] * np.cos(tree['GenMET_phi'][e])
    Y[e][1] += tree['GenMET_pt'][e] * np.sin(tree['GenMET_phi'][e])

    EVT[e][0] = tree['fixedGridRhoFastjetAll'][e]
    EVT[e][1] = tree['fixedGridRhoFastjetCentralCalo'][e]
    EVT[e][2] = tree['PV_npvs'][e]
    EVT[e][3] = tree['PV_npvsGood'][e]
    EVT[e][4] = tree['nMuon'][e]

with h5py.File(opt.output, 'w') as h5f:
    h5f.create_dataset('X',    data=X,   compression='lzf')
    h5f.create_dataset('Y',    data=Y,   compression='lzf')
    h5f.create_dataset('EVT',  data=EVT, compression='lzf')
    h5f.create_dataset('XLep', data=XLep, compression='lzf')
