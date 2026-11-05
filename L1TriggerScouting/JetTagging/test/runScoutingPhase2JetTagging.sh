#!/bin/bash
# Launch from CMSSW's src directory, like runScoutingPhase2TauTagging.sh.
# Usage: ./L1TriggerScouting/JetTagging/test/runScoutingPhase2JetTagging.sh \
#          <backend(-dump)> <num_events> <num_threads> <num_streams> <run_number> \
#          <unpacking|clustering|tagging> [trt] [trace] [name=value ...]
# Example: ./L1TriggerScouting/JetTagging/test/runScoutingPhase2JetTagging.sh \
#            cuda 15 1 1 50 tagging trt trace batchSize=512

set -euo pipefail

function die { printf 'Failed: %s\n' "$*" >&2; exit 1; }

USAGE="Usage: $0 <backend(-dump)> <num_events> <num_threads> <num_streams> <run_number> <unpacking|clustering|tagging> [trt] [trace] [name=value ...]"
if [ "$#" -lt 6 ]; then
    die "Need at least 6 arguments; got $#. $USAGE"
fi

TARGET=$1
NUM_EVENTS=$2
NUM_THREADS=$3
NUM_STREAMS=$4
RUN_NUMBER=$5
STEP=$6
shift 6

SCRIPT="$(dirname -- "${BASH_SOURCE[0]}")/runScoutingPhase2JetTagging.py"
DATA="${SCOUTING_DATA:-/mnt/ramdisk/gizago/raw}"
TRACE=False
USE_TRT=False
DUMP=False

case "$TARGET" in
    cpu|cpu-dump) BACKEND=serial_sync ;;
    cuda|cuda-dump) BACKEND=cuda_async ;;
    rocm) die "rocm is not supported by the current JetTagging build" ;;
    *) die "Backend must be cpu, cpu-dump, cuda, cuda-dump or rocm; got '$TARGET'" ;;
esac
if [[ "$TARGET" == *-dump ]]; then
    DUMP=True
fi
case "$STEP" in
    unpacking|clustering|tagging) ;;
    *) die "Step must be unpacking, clustering or tagging; got '$STEP'" ;;
esac
if [[ ! "$NUM_EVENTS" =~ ^(-1|[0-9]+)$ ]]; then
    die "num_events must be -1 or a nonnegative integer"
fi
for value in "$NUM_THREADS" "$NUM_STREAMS" "$RUN_NUMBER"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        die "num_threads, num_streams and run_number must be nonnegative integers"
    fi
done

if [ "${1:-}" = trt ]; then
    USE_TRT=True
    shift
fi
if [ "${1:-}" = trace ]; then
    TRACE=True
    shift
fi
for argument in "$@"; do
    if [[ "$argument" != *=* ]]; then
        die "Expected [trt] [trace] followed by cmsRun name=value options; got '$argument'"
    fi
done
if [ "$TRACE" = True ] && { [ "$NUM_EVENTS" -lt 0 ] || [ "$NUM_EVENTS" -gt 40 ]; }; then
    die "Perfetto tracing requires a finite run of at most 40 events"
fi
if [ "$USE_TRT" = True ] && [ "$BACKEND" != cuda_async ] && [ "$STEP" = tagging ]; then
    die "TensorRT tagging requires the cuda or cuda-dump backend"
fi

cmsrun_args=(
    "backend=$BACKEND"
    "buBaseDir=$DATA" "fuBaseDir=$DATA" buNumStreams=2
    "maxEvents=$NUM_EVENTS" "numThreads=$NUM_THREADS" "numFwkStreams=$NUM_STREAMS"
    "runNumber=$RUN_NUMBER" "pipelineStep=$STEP"
    "useTRT=$USE_TRT" "tracePerfetto=$TRACE"
)
if [ "$DUMP" = True ]; then
    cmsrun_args+=(dump=candidates,cluster_indexes,jets)
fi
# Explicit name=value options replace launcher defaults, including list options.
for argument in "$@"; do
    name=${argument%%=*}
    for index in "${!cmsrun_args[@]}"; do
        if [ "${cmsrun_args[index]%%=*}" = "$name" ]; then
            unset 'cmsrun_args[index]'
        fi
    done
done

printf 'Running soft-jet %s on %s (TensorRT=%s, trace=%s)\n' "$STEP" "$BACKEND" "$USE_TRT" "$TRACE"
exec cmsRun "$SCRIPT" "${cmsrun_args[@]}" "$@"
