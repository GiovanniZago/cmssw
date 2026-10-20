# Tau tagging with SoftTauIdML (Phase-2 L1 Data Scouting)

This package runs the soft-tau tagging pipeline on live-style scouting orbit
data: unpack PUPPI V2 TM data from the DAQ stream, cluster the candidates with
CLUE (CLUEstering), reshape each cluster into a 16×10 input tensor, and run
the `STT_jit.pt` transformer model on it in batches. Everything runs through
alpaka, so each step works on both CPU (`serial_sync`) and GPU (`cuda_async`).

The steps below are ordered as you need them: first produce the input data,
then run the tagger.

## Environment setup

```bash
# after having checked out this branch
cmsenv
scram b -j
```

## 1. Generating the input files (simple-clients generator)

The cmsRun job does not read a normal ROOT file: it behaves like an online
processor and consumes raw orbit files from a mini-DAQ directory on disk. You
produce those files with the `p2scout-generator` from the simple-clients
repository:

<https://gitlab.cern.ch/scouting-demonstrator/phase-2/simple-clients>

### What the job expects on disk

Inside the data directory (`DATA`, currently `/mnt/ramdisk/gizago/raw` in the
run script) one directory per run, containing the raw streams:

```
/mnt/ramdisk/$USER/raw/
└── run000038/                                        # "run" + 6-digit run number
    └── run000038_ls0001_index000001_stream00.raw
```

- File names are `runNNNNNN_lsNNNN_indexNNNNNN_streamNN.raw`.
- With the default configuration (`broker=none`, file-list mode) the job starts
  from exactly `runNNNNNN_ls0001_index000001_stream00.raw`, i.e. lumisection 1
  and file index 1 — name your generated file accordingly (or pass a different
  `lumiNumber=`).
- The run directory itself, the FU directory and the `fu.lock` file are created
  **automatically by the CMSSW job**, so you only have to make sure the `.raw`
  file is in place before starting.
- The file format is the scouting "CMSSW" format: Native64 events with the FRD
  per-file and per-orbit headers — that is what the `DAQSource` in
  `ScoutingPhase2` mode understands.

### Building the generator

```bash
git clone https://gitlab.cern.ch/scouting-demonstrator/phase-2/simple-clients.git
cd simple-clients
make -j 8
```

It only needs a recent gcc. If you prefer not to build it, prebuilt binaries
ship in a container image (you still need to bind the ramdisk/data dirs into
it):

```bash
apptainer shell -B /mnt/ramdisk -B $PWD:/src \
  docker://gitlab-registry.cern.ch/scouting-demonstrator/phase-2/simple-clients/el9:latest
```

### Generating a raw stream file

The generator resamples events from an existing Native64 dump (e.g. a Puppi
dump taken from the demonstrator, `data/Puppi.dump` in the repo) and emits a
stream of orbits. Use the `CMSSW` mode so the FRD headers are included. The following commands show
how to generate the barrel + endcap raw files corresponding to the whole set of L1 PF candidates:

```bash
mkdir -p /mnt/ramdisk/gizago/raw/run000038
./p2scout-generator CMSSW \
                    data/PF_barrel.dump \
                    /mnt/ramdisk/$USER/raw/run000038/run000038_ls0001_index000001_stream00.raw \
                    --orbits 2000 -T 1 -O 16384 --seed 0
./p2scout-generator CMSSW \
                    data/PF_endcap.dump \
                    /mnt/ramdisk/$USER/raw/run000038/run000038_ls0001_index000001_stream01.raw \
                    --orbits 2000 -T 1 -O 16384 --seed 0
```

Useful options:

- `--orbits N` — how many orbits to generate (pay attention that with ~2000 orbits
    the PF barrel raw file size is approximately 12 GB);
- `--seed 0` — replay the dump in order instead of random resampling (useful for
    reproducible tests);

If needed, a more complete set of options is available in the `simple-client`'s README, 
but pay attention that the CMSSW source could need different settings.

This generator is only needed when you have no real data coming over the
network. In a live setup the same file layout is produced by the receivers
(`p2scout-client DTHRollingReceive256MM ... --cmssw --run NNNNNN`), and with
more than one file you typically serve them over the HTTP file broker
(`p2scout-fileBroker`) and run with `broker=host:port` instead of file-list
mode.

## 2. The configuration: `test/runScoutingPhase2TauTagging.py`

A cmsRun config, driven by CMSSW's `VarParsing` (so all options are passed
directly on the command line, no editing needed). What it does:

- **Inputs and DAQ plumbing** — through `setupPhase2Scouting(process, options)`
  (`L1TriggerScouting/Phase2/python/L1TScPhase2RunScouting_cff.py`): an
  `EvFDaqDirector` service pointing at the BU/FU directories
  (`buBaseDir`/`fuBaseDir`, run number, streams) and a `DAQSource` in
  `ScoutingPhase2` mode reading the `.raw` files described above. Framework
  threads/streams/maxEvents come from `numThreads`, `numFwkStreams`,
  `maxEvents`.
- **The pipeline**, enabled step by step with `pipelineStep`:

  | step | adds |
  |---|---|
  | `unpacking` | `L1TScPhase2PuppiRawToDigiV2@alpaka`: TM6 PUPPI V2 data → per-BX candidate collections |
  | `clustering` | `L1TScPhase2CLUEJets@alpaka`: CLUE clustering of the candidates, produces the bx→clusters and clusters→candidates maps |
  | `sorting` | `SoftTauIdML@alpaka` set to `sorting`: sorts the cluster→candidates map by pT |
  | `reshaping` | + builds the 16×10 input tensor (`SoftTauInputTensor`) per cluster |
  | `tagging` | + runs the batched `STT_jit.pt` inference (default `batchSize=2048` clusters, float16) and writes the `cls`/`pt`/`charge` per-cluster outputs |

  Stop earlier in the pipeline with a smaller `pipelineStep` to debug in
  isolation, e.g. `pipelineStep=clustering`.
- **The model** is taken from `model` (default
  `L1TriggerScouting/TauTagging/data/STT_jit.pt`, a TorchScript file).
  `PyTorchService` disables internal torch threading so the framework controls
  the parallelism.
- **Timing** comes free: `FastTimerService` prints the event summary and writes
  `Phase2Timing_resources.json` in the working directory.
- **Dumps** (`dump=` list): any combination of `candidates`, `cluster_indexes`,
  `soft_tau_inputs`, `soft_tau_outputs` — each dumps a flat orbit table via the
  dedicated `*ToOrbitFlatTable` producers (entries incompatible with the chosen
  `pipelineStep` are dropped with a message). If any dump is requested, an
  `OrbitNanoAODOutputModule` writes a NanoAOD `.root` file. These tables
  are also the inputs of the standalone cross-check script
  `test/validateSoftTauIdML.py`.
- **Tracing**: `tracePerfetto=True` starts a Perfetto trace (`trace.pftrace`)
  with device memory allocations. Keep it for ≤ 40 events.

## 3. Running it: `test/runScoutingPhase2TauTagging.sh`

A small bash wrapper around the config that already has the working directory
setup in it (edit the `DATA=` line at the top if your ramdisk lives elsewhere;
default `/mnt/ramdisk/gizago/raw`):

```bash
bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh \
    <cpu|cpu-dump|cuda|cuda-dump> <num_events> <num_threads> <num_streams> <run_number> <step> [trace]
```

| argument | values | meaning |
|---|---|---|
| 1st | `cpu`, `cpu-dump`, `cuda`, `cuda-dump` (takes `rocm` too, currently unsupported) | alpaka backend: `serial_sync` or `cuda_async`; `-dump` also enables all four flat-table dumps + the output file |
| 2nd | number | events (orbits) to process, `maxEvents` |
| 3rd | number | framework threads |
| 4th | number | framework streams |
| 5th | run number | must match the `runNNNNNN` directory on the ramdisk |
| 6th | `unpacking`, `clustering`, `sorting`, `reshaping`, `tagging` | how far to run the pipeline (see the table above) |
| 7th (optional) | `trace` | enable Perfetto tracing (max 40 events) |

Examples:

```bash
# GPU, full pipeline, 10 000 orbits, 8 threads/streams, run 37
bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh cuda 10000 8 8 37 tagging

# CPU, stop after clustering, small debug run
bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh cpu 100 1 1 37 clustering

# GPU with all dumps on (tables + orbit NanoAOD output), 100 orbits
bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh cuda-dump 100 8 8 37 tagging

# Traced run (Perfetto, trace.pftrace)
bash L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.sh cuda 40 8 8 37 tagging trace
```

You can equally run cmsRun directly; the wrapper is only sugar. The equivalent
of the GPU example above is:

```bash
cmsRun L1TriggerScouting/TauTagging/test/runScoutingPhase2TauTagging.py \
    backend=cuda_async buBaseDir=/mnt/ramdisk/gizago/raw fuBaseDir=/mnt/ramdisk/gizago/raw \
    buNumStreams=2 maxEvents=10000 numThreads=8 numFwkStreams=8 \
    runNumber=37 pipelineStep=tagging
```
