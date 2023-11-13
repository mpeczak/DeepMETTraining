import uproot
import numpy as np
import optparse
import os
import progressbar

widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]
# Import the input parameters
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input file', default='/hildafs/projects/phy230010p/share/DYJetsToLL/output_nano_10.root', type='string')
parser.add_option('-o', '--output', dest='output', help='output directory', default='/hildafs/projects/phy230010p/xiea/hdf5s/dy/split_nano/', type='string')
parser.add_option('-n', '--num_events', dest='n_events_per_file', help='number of events per file', default=1000, type=int)
(opt, args) = parser.parse_args()


old_filename = opt.input
new_dir = opt.output
tree_name = 'Events'
n_events_per_file = opt.n_events_per_file


print ('###########################################################################')
print ('# Input filename: ', old_filename)
print ('# Directory for split files: ', new_dir)
print ('# Number of events per file: ', n_events_per_file)
print ('###########################################################################')

branchNames = [
    'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi',
    'nPFCands', 'PFCands_pt', 'PFCands_eta', 'PFCands_phi', 'PFCands_pdgId',
    'PFCands_charge', 'PFCands_mass',
    'PFCands_d0', 'PFCands_dz', 'PFCands_fromPV', 'PFCands_puppiWeightNoLep', 'PFCands_puppiWeight',
    'fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo',
    'PV_npvs', 'PV_npvsGood','nMuon',
    'GenMET_pt', 'GenMET_phi',
]

fname = 'test.root'
input_file = uproot.open(old_filename)
input_tree = input_file['Events']
num_events = input_tree.num_entries
print("num events in root file:",num_events)

splitindex = 0
if num_events%n_events_per_file==0:
    num_splits = num_events//n_events_per_file
else:
	num_splits = (num_events//n_events_per_file) + 1


master_Branches = dict()  
for branch in branchNames:
	master_Branches[branch] = input_tree[branch].array()
print("master branch created")

for i in progressbar.progressbar(range(num_splits),widges=widgets):
#for i in range(num_splits):

	new_file_name = f'output_nano_10_splitnum_{i}.root'
	#print(f'file {i}:',new_file_name)
	
	new_file = uproot.recreate(os.path.join(new_dir,new_file_name))
	
	start = i*n_events_per_file
	end = (i+1)*n_events_per_file
	if end > num_events:
		end = num_events
	
	branches = dict()
	for branch in branchNames:
		
		branches[branch] = master_Branches[branch][i*n_events_per_file:(i+1)*n_events_per_file]

	new_file['Events'] = branches
	#print(f'{new_file_name} created with {end-start} events ')
	new_file.close()

print("-------------> DONE!")
