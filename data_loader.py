import numpy as np
import keras
import h5py
import os
from tqdm import tqdm
import glob


def read_file(fname):
    h5f = h5py.File( fname, 'r')
    Xarr = h5f['X'][:]
    Yarr = h5f['Y'][:]
    h5f.close()
    return (Xarr,Yarr)

def read_processed_file(fname):
    h5f = h5py.File(fname, 'r')
    X = h5f['X'][:]
    inputs_cat0 = h5f['cat0'][:]
    inputs_cat1 = h5f['cat1'][:]
    inputs_cat2 = h5f['cat2'][:]
    Y = h5f['Y'][:]
    h5f.close()
    return (X,inputs_cat0,inputs_cat1,inputs_cat2,Y)

def preProcessing(X, Y, EVT=None):
    """ pre-processing input """
    norm = 50.0

    dxy = X[:,:,5:6]
    dz  = X[:,:,6:7].clip(-100, 100)
    eta = X[:,:,3:4]
    mass = X[:,:,8:9]
    pt = X[:,:,0:1] / norm
    puppi = X[:,:,7:8]
    px = X[:,:,1:2] / norm
    py = X[:,:,2:3] / norm

    # remove outliers
    pt[ np.where(np.abs(pt>200)) ] = 0.
    px[ np.where(np.abs(px>200)) ] = 0.
    py[ np.where(np.abs(py>200)) ] = 0.

    if EVT is not None:
        # environment variables
        evt = EVT[:,0:4]
        evt_expanded = np.expand_dims(evt, axis=1)
        evt_expanded = np.repeat(evt_expanded, X.shape[1], axis=1)
        # px py has to be in the last two columns
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, evt_expanded, px, py), axis=2)
    else:
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, px, py), axis=2)

    inputs_cat0 = X[:,:,11:12] # encoded PF pdgId
    inputs_cat1 = X[:,:,12:13] # encoded PF charge
    inputs_cat2 = X[:,:,13:14] # encoded PF fromPV

    return inputs, inputs_cat0, inputs_cat1, inputs_cat2

def save_processed_file(fname,inputs,cat0,cat1,cat2,Y):
    with h5py.File(fname, 'w') as h5f:
        h5f.create_dataset('X', data=inputs,   compression='lzf')
        h5f.create_dataset('cat0',   data=cat0,   compression='lzf')
        h5f.create_dataset('cat1',  data=cat1, compression='lzf')
        h5f.create_dataset('cat2', data=cat2, compression='lzf')
        h5f.create_dataset('Y', data=Y, compression='lzf')

class Dataset(keras.utils.Sequence):
    def __init__(self,input_dir, batch_size = 16):
        #default assume <= 1000 events per file
        self.processed_file_names = sorted(glob.glob(input_dir+'/processed/*.h5'))
        self.raw_file_names = sorted(glob.glob(input_dir+'/raw/*.h5'))
        self.batch_size = batch_size
        self.idx_dict = dict()
        self.num_events = 0
        self.num_batches = 0
        self.curr_file = None
        self.curr_X = None
        self.curr_cat0 = None
        self.curr_cat1 = None
        self.curr_cat2 = None
        self.curr_Y = None
        self.train_Dataloader = None
        self.test_Dataloader = None

        def initialise(self):
            batch_idx = 0
            for i,raw_file_name in enumerate(tqdm(self.raw_file_names)):
                curr_file = self.raw_file_names[i]
                processed_name = curr_file.split('.')[0].split('/')[:-2] #+'_processed.h5'
                name  = curr_file.split(".")[0].split("/")[-1]
                processed_name = "/".join(processed_name)+'/processed/'+name+'_processed.h5'
                
                if processed_name in self.processed_file_names :
                    try:
                        (inputs,inputs_cat0,inputs_cat1,inputs_cat2,Y) = read_processed_file(processed_name)
                    except:
                        print(f"failed on {processed_name}")
                        continue
                    self.num_events += len(inputs)
                    file_idx = 0

                    assert(len(inputs)==len(inputs_cat1))

                    while (file_idx + self.batch_size) < len(inputs):
                        self.idx_dict[batch_idx] = (processed_name,file_idx,file_idx+self.batch_size)
                        file_idx += self.batch_size
                        batch_idx += 1

                    if file_idx < len(inputs):
                        self.idx_dict[batch_idx] = (processed_name,file_idx,len(inputs))
                        file_idx += len(inputs)
                        batch_idx += 1
                else:
                    print('false')
                    try:
                        (X,Y) = read_file(curr_file)
                    except:
                        print(f"failed on {curr_file}")
                        continue
                    (inputs,inputs_cat0,inputs_cat1,inputs_cat2) = preProcessing(X,Y)
                    save_processed_file(processed_name,inputs,inputs_cat0,inputs_cat1,inputs_cat2,Y)
                    
                    self.processed_file_names.append(processed_name)                   
                    file_idx = 0
                    while (file_idx + self.batch_size) < len(inputs):
                        self.idx_dict[batch_idx] = (processed_name,file_idx,file_idx+self.batch_size)
                        file_idx += self.batch_size
                        batch_idx += 1

                    if file_idx < len(inputs):
                        self.idx_dict[batch_idx] = (processed_name,file_idx,len(inputs))
                        file_idx += len(inputs)
                        batch_idx += 1
                    
            self.num_batches = batch_idx-1
            self.curr_file = self.processed_file_names[0]
            self.curr_X,self.curr_cat0,self.curr_cat1, self.curr_cat2,self.curr_Y = read_processed_file(self.curr_file)

                
        initialise(self)

    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self,idx):
        file, start_i, end_i = self.idx_dict[idx]
        if file != self.curr_file:
            self.curr_X,self.curr_cat0,self.curr_cat1,self.curr_cat2, self.curr_Y = read_processed_file(file)
            self.curr_file = file
            
        X = self.curr_X[start_i:end_i]
        cat0 = self.curr_cat0[start_i:end_i]
        cat1 = self.curr_cat1[start_i:end_i]
        cat2 = self.curr_cat2[start_i:end_i]
        input = [X,cat0,cat1,cat2]
        Y = self.curr_Y[start_i:end_i]
        return (input,Y)
    
    def fetch_dataloaders(self,validation_split, batch_size=16):
        split = int(np.floor(validation_split * len(self)))
        batch_ids = np.arange(len(self))
        #np.random.shuffle(batch_ids)
        train_ids = batch_ids[:split]
        test_ids = batch_ids[split:]
        
        train_loader = Dataloader(self,train_ids,batch_size=batch_size)
        test_loader = Dataloader(self,test_ids,batch_size=batch_size)
        return(train_loader,test_loader)
        
    
class Dataloader(keras.utils.Sequence):
        def __init__(self,dataset,batch_ids, batch_size = 16):
            self.dataset = dataset
            self.batch_ids = batch_ids
            self.batch_size = batch_size 

        def __len__(self):
            return len(self.batch_ids)
        
        def __getitem__(self,idx):
            return self.dataset.__getitem__(self.batch_ids[idx])
