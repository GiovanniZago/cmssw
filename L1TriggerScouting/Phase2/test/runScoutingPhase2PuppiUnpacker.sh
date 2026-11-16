#!/bin/bash

: ' to streamline debugging and testing PF unpacker. 
This shell script should be used to run the runScoutingPhase2PuppiUnpacker.py config file with a base configuration involving 
1 stream of Puppi candidates containing all 6 timeslices.

The minbias dump file (Puppi.dump) for such configuration is already available inside the data/ directory of the simple-clients 
repository. In case it is necessary to generate the .raw file from a special .dump file (e.g., for signal studies) it can be 
generated using the following commands:

./p2scout-generator CMSSW \
    /path/to/Puppi_signal.dump \
    /dev/shm/$USER/raw/run000038/run000038_ls0001_index000001_stream00.raw \
    --orbits 1000 -T 1 -O 16384 --seed 0

The options for p2scout-generator are available inside the simple-client repository.

More complicated configurations involving a different number of timeslices, split factor and stream ID assignment
should be handled case-by-case using directly the options available for runScoutingPhase2PuppiUnpacker.py.

usage: bash L1TriggerScouting/Phase2/test/runScoutingPhasePuppiUnpacker.sh <backend> <num_events> <run_number>
example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2PuppiUnpacker.sh cuda 100 40
'

function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/Phase2/test/runScoutingPhase2PuppiUnpacker.py"
DATA="/mnt/ramdisk/gizago/raw"

if [ "$#" != "3" ]; then
    die "Need exactly 3 arguments: 1st ('cpu', 'cuda', or 'rocm'), 2nd ('num_events'), 3rd ('num_threads'), 4th ('num_streams'), 5th ('run_number'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" runNumber=$5 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 puppiStreamIDs=0 numThreads=$3 numFwkStreams=$4 maxEvents=$2 backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running GPU-only test"
  cmsRun "${SCRIPT}" runNumber=$5 buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=1 puppiStreamIDs=0 numThreads=$3 numFwkStreams=$4 maxEvents=$2 backend=cuda_async
fi