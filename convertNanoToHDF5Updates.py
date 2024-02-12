import sys
import logging
import warnings
import argparse
import numpy as np
import awkward as ak
import h5py
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import PFNanoAODSchema
import os
import concurrent.futures

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='<inputfile>', type=str,
        help='input NanoAOD file.')
    parser.add_argument('outputfile', metavar='<outputfile>', type=str,
        help='output HDF5 file')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='show logs')
    parser.add_argument('-l', '--leptons',
        metavar='<leptons>', type=int, default=2,
        help='number of leptons to remove from pfcands (default is 2)')
    return parser.parse_args()

def remove_lepton(pfcands, lepton, r_max=0.001):
    """
    Remove deltaR matched lepton from pfcands. A lepton is matched to the
    nearest pfcand if they are closer than a deltaR of r_max.
    """
    dr = pfcands.delta_r(lepton)
    ipf = ak.local_index(dr)
    imin = ak.argmin(dr, axis=1, mask_identity=False)
    match = (ipf == imin) & (dr < r_max)
    return pfcands[~match]

### Main program ###

def convert(args):
    """Get training data from a NanoAOD and save it to an HDF5 file"""
    # Configure logger if enabled
    if args.verbose:
        logger.setLevel(level=logging.INFO)

    # Supress warnings about names of unused branches from coffea
    warnings.filterwarnings('ignore', message='Found duplicate branch .*Jet_')

    # Get events from NanoAOD
    logger.info('Fetching events')
    events = NanoEventsFactory.from_root(
        args.inputfile,
        schemaclass=PFNanoAODSchema
    ).events()

    # Event selection
    logger.info(f'Num events before selection: {len(events)}')
    n_lep = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_lep >= args.leptons]
    logger.info(f'Num events after selection:  {len(events)}')

    # Get training data collections and leading leptons
    pfcands = events.PFCands
    genMET = events.GenMET
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:,:args.leptons]

    # DeltaR matching
    logger.info('Removing leading leptons from pfcand list')
    for ilep in range(args.leptons):
        logger.info(f'Removing lepton {ilep+1}')
        pfcands = remove_lepton(pfcands, leptons[:,ilep])
    logger.info('Lepton matching completed')

    # px and py are not computed or saved until they are initialized
    logger.info('Computing additional quantities')
    pfcands['px'] = pfcands.px
    pfcands['py'] = pfcands.py
    genMET['px'] = genMET.px
    genMET['py'] = genMET.py
    logger.info(f'Additional computations completed')

    # Format training data
    logger.info('Preparing training inputs')
    pfcands_fields = []
    npf = ak.max(ak.num(pfcands)) if args.auto_npf else args.npf

    for field_name in input_fields:
        logger.info(f'Processing PFCands_{field_name}')
        field = pfcands[field_name]
        if field_name in list(labels):
            pass # todo: conversion to labels
        field = ak.pad_none(field, npf, axis=-1, clip=True)
        field = ak.fill_none(field, args.fill)
        pfcands_fields.append(field)

    logger.info('Preparing training outputs')
    genMET_fields = ak.unzip(genMET[output_fields])

    # Save data to file
    logger.info('Converting to numpy arrays')
    X = np.array(pfcands_fields)
    Y = np.array(genMET_fields)

    logger.info('Saving to HDF5 file')
    with h5py.File(args.outputfile, 'w') as h5f:
        h5f.create_dataset('X', data=X, compression='lzf')
        h5f.create_dataset('Y', data=Y, compression='lzf')

    logger.info(f'Inputs shape:  {np.shape(X)}')   # (nfields,nevents,npf)
    logger.info(f'Outputs shape: {np.shape(Y)}')   # (nfields,nevents)
    logger.info(f'Training data saved to {args.outputfile}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='<inputfile>', type=str,
        help='input NanoAOD file.')
    parser.add_argument('outputfile', metavar='<outputfile>', type=str,
        help='output HDF5 file')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='show logs')
    parser.add_argument('-l', '--leptons',
        metavar='<leptons>', type=int, default=2,
        help='number of leptons to remove from pfcands (default is 2)')




    ### Parameters ###

    # PFCands and GenMET fields, respectively, to be saved in HDF5 file
    input_fields = [
        'd0',
        'dz',
        'eta',
        'mass',
        'pt',
        'puppiWeight',
        'px',
        'py',
        'pdgId',
        'charge',
        #'fromPV'
    ]
    output_fields = ['px', 'py']

    # Labels for fields with discrete values
    labels = {
    'charge':{  -1.0: 0,   0.0: 1,   1.0: 2},
    'pdgId': {-211.0: 0, -13.0: 1, -11.0: 2,   0.0: 3,   1.0: 4,  2.0: 5,
                11.0: 6,  13.0: 7,  22.0: 8, 130.0: 9, 211.0:10},
    'fromPV':{   0.0: 0,   1.0: 1,   2.0: 2,   3.0: 3}
    }

    auto_npf = False    # If true, overwrites npf_max at runtime
    npf_max = 4500      # Number of PFCands entries in output file;
                        # empty values are padded
    fill = -999         # Padding value used to fill empty PFCands entries
    
    nworkers = 2


    with open(args.inputfile) as file_in:
        old_file_names = []
        new_file_names = []
        i = 0
        for line in file_in:
            old_file_names.append(line)
            new_file_names.append("file_" + str(i) + ".root")
            i = i+1
    

    for i in range(len(old_file_names)):
        os.rename(old_file_names[i], new_file_names[i])

    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = set()

        futures.update(executor.submit(convert, line) for line in new_file_names)
        
        try:
            total = len(futures)
            processed = 0
            while len(futures) > 0:
                finished = set(job for job in futures if job.done())
                for job in finished:
                    X,Y,line = job.result()
                    #line[:-5]+'.h5'
                    with h5py.File(get_args.outputfile, 'w') as h5f:
                        h5f.create_dataset('X',    data=X,   compression='lzf')
                        h5f.create_dataset('Y',    data=Y,   compression='lzf')
                    print (X,Y)
                    processed += 1
                futures -= finished
            del finished
        except KeyboardInterrupt:
            print("Ok quitter")
            for job in futures: job.cancel()
        except:
            for job in futures: job.cancel()
            raise
