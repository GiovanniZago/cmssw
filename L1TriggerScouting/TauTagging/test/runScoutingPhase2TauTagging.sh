#!/bin/bash

: ' to streamline debugging and testing of the Soft Tau reconstruction and identification pipeline. 
This shell script should be used to run the runScoutingPhase2TauTagging.py config file with a base configuration involving:
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
assignment should be handled case-by-case using directly the options available for runScoutingPhase2TauTagging.py.

usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2TauTagging.sh <backend(-dump)> <num_events> <num_threads> <num_streams> <run_number> <pipeline_step> trace [optional]
example usage: bash L1TriggerScouting/Phase2/test/runScoutingPhase2PFUnpacker.sh cuda 100 40
'

function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.py"
DATA="/mnt/ramdisk/gizago/raw"
TRACE="False"

if [ "$#" -lt "6" ]; then
    die "Need at least 6 arguments: 1st ('cpu', 'cpu-dump', 'cuda', 'cuda-dump'), 2nd ('num_events'), 3rd ('num_threads'), 4th ('num_streams'), 5rd ('run_number'), 6th ('unpacking', 'clustering', 'sorting', 'reshaping', 'tagging'), 7th ('trace' [optional]) got $#" 1
elif [ "$#" -eq "7" ]; then
    if [[ "$7" =~ ^(trace)$ ]]; then
        TRACE="True"
        if [ "$2" -gt "40" ]; then
            die "Trying to run a Perfetto-traced job with more than 40 events, not allowed!"
        fi
    else
        die "Argument needs to be 'trace'; got '$7'" 1
    fi
elif [ "$#" -gt "7" ]; then
    die "Need at most 7 arguments: 1st ('cpu', 'cpu-dump', 'cuda', 'cuda-dump'), 2nd ('num_events'), 3rd ('num_threads'), 4th ('num_streams'), 5rd ('run_number'), 6th ('unpacking', 'clustering', 'sorting', 'reshaping', 'tagging'), 7th ('trace' [optional]) got $#" 1
fi
if [[ "$1" =~ ^(cpu|cpu-dump|cuda|cuda-dump|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cpu-dump', 'cuda', 'cuda-dump' or 'rocm'; got '$1'" 1
fi
if [[ "$6" =~ ^(unpacking|clustering|sorting|reshaping|tagging)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'unpacking', 'clustering', 'sorting', 'reshaping' or 'tagging'; got '$6'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cpu-dump" ]; then
  echo "Running CPU-only test with dumps"
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cuda-dump" ]; then
  echo "Running CUDA test with dumps"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs pfBarrelStreamIDs=0 pfEndcapStreamIDs=1 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "rocm" ]; then
  die "rocm not curently supported" 1 
fi