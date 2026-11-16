#!/bin/bash

: ' to streamline debugging and testing PF unpacker. 
This shell script should be used to run the runScoutingPhase2PFUnpacker.py config file with a base configuration involving:
- 1 stream of barrel PF candidates containing all 18 timeslices
- 1 stream of endcap PF candidates containing all 18 timeslices

The minbias dump files (PF_Barrel.dumo, PF_Endcap.dump) for such configuration are already available inside the data/ 
directory of the simple-clients repository. In case it is necessary to generate the .raw files from special .dump files 
(e.g., for signal studies) they can be generated using the following commands:

./p2scout-generator CMSSW \
    /path/to/PF_barrel_signal.dump \
    /dev/shm/$USER/raw/run000038/run000038_ls0001_index000001_stream00.raw \
    --orbits 500 -T 1 -O 16384 --seed 0

./p2scout-generator CMSSW \
    /path/to/PF_endcap_signal.dump \
    /dev/shm/$USER/raw/run000038/run000038_ls0001_index000001_stream01.raw \
    --orbits 500 -T 1 -O 16384 --seed 0

The options for p2scout-generator are available inside the simple-client repository.

More complicated configurations involving a different number of timeslices, barrel/endcap split factor and stream ID
assignment should be handled case-by-case using directly the options available for runScoutingPhase2PFUnpacker.py.

usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2PFUnpacker.sh <backend> <num_events> <run_number> <num_threads> <num_streams>
example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2PFUnpacker.sh cuda 100 1 1 40
'

function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/Phase2/test/runScoutingPhase2PFUnpacker.py"
DATA="/mnt/ramdisk/gizago/raw"

if [ "$#" != "5" ]; then
    die "Need exactly 5 arguments: 1st ('cpu', 'cuda', or 'rocm'), 2nd ('num_events'), 3rd ('num_threads'), 4th ('num_streams'), 5th ('run_number'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" runNumber=$5 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 numThreads=$3 numFwkStreams=$4 maxEvents=$2 backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running GPU-only test"
  cmsRun "${SCRIPT}" runNumber=$5 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 numThreads=$3 numFwkStreams=$4 maxEvents=$2 backend=cuda_async
fi