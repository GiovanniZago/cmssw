#!/bin/bash
# to streamline debugging and testing of SoftTauTagging
# usage: bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh <backend(-dump)> <num_events> <num_threads> <num_streams> <run_number> <step> [sofie] [trace]
# example usage: bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh cuda 10 4 4 40 tagging sofie trace
function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.py"
DATA="/mnt/ramdisk/gizago/raw"
TRACE="False"
USE_SOFIE="False"
USAGE="Usage: $0 <backend(-dump)> <num_events> <num_threads> <num_streams> <run_number> <step> [sofie] [trace]"

if [ "$#" -lt "6" ] || [ "$#" -gt "8" ]; then
    die "Need 6 to 8 arguments; got $#. ${USAGE}" 1
fi
if [ "$#" -ge "7" ]; then
    if [ "$7" == "sofie" ]; then
        USE_SOFIE="True"
    elif [ "$7" == "trace" ] && [ "$#" -eq "7" ]; then
        TRACE="True"
    else
        die "Expected optional arguments [sofie] [trace]. ${USAGE}" 1
    fi
fi
if [ "$#" -eq "8" ]; then
    if [ "$8" == "trace" ]; then
        TRACE="True"
    else
        die "Expected 'trace' after 'sofie'; got '$8'" 1
    fi
fi
if [ "${TRACE}" == "True" ] && [ "$2" -gt "40" ]; then
    die "Trying to run a Perfetto-traced job with more than 40 events, not allowed!" 1
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
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 useSofie="${USE_SOFIE}" tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cpu-dump" ]; then
  echo "Running CPU-only test with dumps"
  cmsRun "${SCRIPT}" backend=serial_sync buBaseDir=${DATA} fuBaseDir=${DATA} dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 useSofie="${USE_SOFIE}" tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 useSofie="${USE_SOFIE}" tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "cuda-dump" ]; then
  echo "Running CUDA test with dumps"
  cmsRun "${SCRIPT}" backend=cuda_async buBaseDir=${DATA} fuBaseDir=${DATA} dump=candidates,cluster_indexes,soft_tau_inputs,soft_tau_outputs buNumStreams=2 maxEvents=$2 numThreads=$3 numFwkStreams=$4 runNumber=$5 pipelineStep=$6 useSofie="${USE_SOFIE}" tracePerfetto="${TRACE}"
elif [ "${TARGET}" == "rocm" ]; then
  die "rocm not curently supported" 1 
fi
