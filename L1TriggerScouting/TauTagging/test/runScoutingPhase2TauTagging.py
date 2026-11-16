import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2TauTagging")

# import TauTagging options
from L1TriggerScouting.TauTagging.options_cff import options, VarParsing

# tracing option
options.register ("pfBarrelSplitFactor", 
    1, 
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of input streams in which a timeslice of barrel PF candidates is split."
)

options.register ("pfEndcapSplitFactor", 
    1, 
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of input streams in which a timeslice of endcap PF candidates is split."
)

options.register ("tracePerfetto",
    False,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool, 
    "Generate trace with Perfetto"
)

# parse arguments
options.parseArguments()

# check arguments consistency
if options.pipelineStep not in ["unpacking", "clustering", "sorting", "reshaping", "tagging"]:
    raise ValueError("pipelineStep must be one among unpacking, clustering, sorting, reshaping, tagging")

step_mapper = {
    "unpacking": 0, 
    "clustering": 1, 
    "sorting": 2, 
    "reshaping": 3, 
    "tagging": 4, 
    "candidates": 0, 
    "cluster_indexes": 1, 
    "soft_tau_inputs": 3, 
    "soft_tau_outputs": 4
}

dump = [] # keep only dump requests compatible with the pipeline step chosen
if options.dump != []:
    for d in options.dump:
        if d not in ["candidates", "cluster_indexes", "soft_tau_inputs", "soft_tau_outputs"]:
            raise ValueError("dump must be one among candidates, cluster_indexes, soft_tau_inputs, soft_tau_outputs")
        if step_mapper[d] > step_mapper[options.pipelineStep]:
            print(f"Requested object dump ({d}) is not compatible with specified pipelineStep ({options.pipelineStep}) and will be removed.")
            continue
        dump.append(d)
    
    print("Dumps requested and compatible with pipeline step: ", dump)

# framework setup
process.load("Configuration.StandardSequences.Accelerators_cff")
process.options.numberOfThreads = cms.untracked.uint32(options.numThreads)
process.options.numberOfStreams = cms.untracked.uint32(options.numFwkStreams)
process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

# common framework options, accelerator support, and DAQ input
from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
setupPhase2Scouting(process, options)

# disable pytorch inner threading
process.PyTorchService = cms.Service("PyTorchService")

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f"resources_runScoutingPhase2TauTagging.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
process.options.wantSummary = cms.untracked.bool(True)

# define pipeline path
process.p_pipeline = cms.Path()

# unpacking
process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    src = cms.InputTag("rawDataCollector"),
    streams = cms.vuint32(
        *list(range(options.buNumStreams[0]))
        if options.pfBarrelStreamIDs == [] else options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
    ),
    splitFactor = cms.uint32(options.pfBarrelSplitFactor + options.pfEndcapSplitFactor) 
)
process.p_pipeline += process.unpacker

# check orbits
process.check_orbits = cms.EDFilter("GoodOrbitNBxSelector", 
    unpackers=cms.VInputTag(),
    unpackersAlpaka=cms.VInputTag(cms.InputTag("unpacker")),
    nbxMin=cms.uint32(3564 * options.timeslices // options.tmuxPeriod)
)
process.p_pipeline += process.check_orbits

# clustering
process.load(
    "L1TriggerScouting.Phase2.L1TScPhase2CLUEJets_cff"
)
process.L1TScPhase2CLUEJetsProducer.alpaka = cms.untracked.PSet(
    backend = cms.untracked.string(options.backend)
)
process.L1TScPhase2CLUEJetsProducer.candidates = cms.InputTag("unpacker", "candidates")
process.L1TScPhase2CLUEJetsProducer.bxSizes = cms.InputTag("unpacker", "bxSizes")

# sorting, reshaping, tagging
process.softTaus = cms.EDProducer("l1sc::SoftTauIdML@alpaka", 
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    ),
    srcCandidates = cms.InputTag("unpacker", "candidates"), 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcClustersCandsMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clustersCandsMap"), 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"),
    model = cms.FileInPath(options.model), 
    substep = cms.string(options.pipelineStep), 
    batchSize = cms.uint32(options.batchSize)
)

if step_mapper[options.pipelineStep] >= step_mapper["clustering"]:
    process.p_pipeline += process.L1TScPhase2CLUEJetsProducer
if step_mapper[options.pipelineStep] >= step_mapper["sorting"]:
    process.p_pipeline += process.softTaus

# define table path
process.p_tables = cms.Path()

process.candOrbitTable = cms.EDProducer("PFCandidateSoAToOrbitFlatTable", 
    srcBx = cms.InputTag("unpacker", "bxLookup"), 
    srcCandidates = cms.InputTag("unpacker", "candidates"), 
    name = cms.string("L1PF")
)
process.clusterOrbitTable = cms.EDProducer("ClusterSoAToOrbitFlatTable", 
    srcBx = cms.InputTag("unpacker", "bxLookup"), 
    srcClusters = cms.InputTag("L1TScPhase2CLUEJetsProducer", "clusters"), 
    clustering_name = cms.string("CLUEstering"),
    name = cms.string("L1PF"), 
    extension = cms.bool(True) # extends candOrbitTable, set same name as candOrbitTable
)
process.softTauInputsOrbitTable = cms.EDProducer("SoftTauInputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcInputs = cms.InputTag("softTaus", "softTauInputs"), 
    name = cms.string("SoftTauInputs")
)
process.softTauOutputsOrbitTable = cms.EDProducer("SoftTauOutputTensorToOrbitFlatTable", 
    srcBxClustersMap = cms.InputTag("L1TScPhase2CLUEJetsProducer", "bxClustersMap"), 
    srcOutputs = cms.InputTag("softTaus", "softTauOutputs"), 
    name = cms.string("SoftTauOutputs")
)

if "candidates" in dump:
    process.p_tables += process.candOrbitTable
    if "cluster_indexes" in dump:
        process.p_tables += process.clusterOrbitTable

if "soft_tau_inputs" in dump:
    process.p_tables += process.softTauInputsOrbitTable

if "soft_tau_outputs" in dump:
    process.p_tables += process.softTauOutputsOrbitTable

# define endpath
process.end = cms.EndPath()

process.out = cms.OutputModule("OrbitNanoAODOutputModule",
    fileName = cms.untracked.string("tau_tagging.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring()),
    outputCommands = cms.untracked.vstring("drop *", "keep l1ScoutingRun3OrbitFlatTable_*_*_*"),
)

if dump:
    process.end += process.out

# schedule
process.schedule = cms.Schedule(
    process.p_pipeline, 
    process.p_tables,
    process.end
)

# tracing with Perfetto
if options.tracePerfetto:
    from PerfTools.Perfetto.customisePerfetto import customisePerfetto
    customisePerfetto(process,
                fileName="trace.pftrace",
                traceFunctions=True,
                traceGpuKernels=True,
                traceAllocations=True)