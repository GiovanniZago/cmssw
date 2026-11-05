"""Run Phase-2 PF/vertex unpacking, seeded-cone clustering and soft-jet tagging."""

import FWCore.ParameterSet.Config as cms

from L1TriggerScouting.Phase2.options_cff import options, VarParsing
from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting


# Match the standalone tau workflow, retaining the SoftJets clustering options.
singleton = VarParsing.VarParsing.multiplicity.singleton
var_type = VarParsing.VarParsing.varType
for name, default, value_type, description in (
    ("pipelineStep", "tagging", var_type.string, "Last pipeline stage: unpacking, clustering, tagging"),
    ("tracePerfetto", False, var_type.bool, "Generate trace.pftrace with Perfetto"),
    ("useTRT", False, var_type.bool, "Use SoftJetIdONNX with a TensorRT EPContext model"),
    ("model", "", var_type.string, "Model FileInPath; defaults to TorchScript or TensorRT according to useTRT"),
    ("batchSize", 0, var_type.int, "Inference batch size; 0 selects 32 for TorchScript or 512 for TensorRT"),
    ("splitFactor", 0, var_type.int, "PF streams per BX; 0 derives it from the streams and timeslices"),
    ("vertexSplitFactor", 1, var_type.int, "Vertex streams per BX"),
    ("reportEvery", 100, var_type.int, "Report every N scouting orbits"),
    ("jetAlgo", "SCGreedy", var_type.string, "SCGreedy, SCNMS, SCNMSWeighted, SCNMSWeightedMultiIter, LinkTree"),
    ("njets", 16, var_type.int, "Maximum number of SCGreedy jets per BX"),
    ("jetR", 0.4, var_type.float, "SCGreedy cone radius"),
    ("minSeedPt", 0.0, var_type.float, "Minimum seed pt for SCNMSWeightedMultiIter and LinkTree"),
    ("RSeed", 0.3, var_type.float, "Seed-finding radius"),
    ("RCen", 0.4, var_type.float, "SCNMSWeightedMultiIter centroid radius"),
    ("RClu", 0.4, var_type.float, "Final particle-assignment radius"),
    ("RLink", 0.3, var_type.float, "LinkTree link radius"),
    ("alphaSeed", 2.0, var_type.float, "SCNMSWeightedMultiIter seed-pt exponent"),
    ("nCentroidIters", 1, var_type.int, "SCNMSWeightedMultiIter centroid iterations"),
    ("minConstituents", 0, var_type.int, "Minimum number of jet constituents"),
):
    options.register(name, default, singleton, value_type, description)

options.register("vertexStreamIDs", [1], VarParsing.VarParsing.multiplicity.list,
                 var_type.int, "Raw stream IDs containing vertices")
options.register("dump", [], VarParsing.VarParsing.multiplicity.list,
                 var_type.string, "Orbit NanoAOD tables: candidates, cluster_indexes, jets")
options.outFile = "softJetOrbitNano.root"
options.parseArguments()

# Validate options and keep only dumps available at the selected stage.
step_mapper = {"unpacking": 0, "clustering": 1, "tagging": 2}
if options.pipelineStep not in step_mapper:
    raise ValueError("pipelineStep must be unpacking, clustering or tagging")
if options.jetAlgo not in ("SCGreedy", "SCNMS", "SCNMSWeighted", "SCNMSWeightedMultiIter", "LinkTree"):
    raise ValueError("Unsupported jetAlgo: %s" % options.jetAlgo)
if options.batchSize < 0 or options.splitFactor < 0 or options.vertexSplitFactor <= 0:
    raise ValueError("batchSize and splitFactor must be nonnegative; vertexSplitFactor must be positive")
if options.timeslices <= 0 or options.tmuxPeriod <= 0:
    raise ValueError("timeslices and tmuxPeriod must be positive")
if min(options.njets, options.nCentroidIters, options.minConstituents) < 0:
    raise ValueError("njets, nCentroidIters and minConstituents must be nonnegative")
if options.useTRT and options.pipelineStep == "tagging" and options.backend != "cuda_async":
    raise ValueError("TensorRT tagging requires backend=cuda_async")

if not options.buNumStreams and len(options.buBaseDir) == 1:
    options.buNumStreams = [2]
pf_stream_ids = options.pfBarrelStreamIDs + options.pfEndcapStreamIDs
if not pf_stream_ids:
    pf_stream_ids = list(range(sum(options.buNumStreams)))
if not pf_stream_ids or not options.vertexStreamIDs:
    raise ValueError("PF and vertex stream lists must not be empty")
split_factor = options.splitFactor
if not split_factor:
    if len(pf_stream_ids) % options.timeslices:
        raise ValueError("PF stream count must be divisible by timeslices, or specify splitFactor")
    split_factor = len(pf_stream_ids) // options.timeslices

dump_steps = {"candidates": 0, "cluster_indexes": 1, "jets": 1}
dump = []
for name in options.dump:
    if name not in dump_steps:
        raise ValueError("dump must contain candidates, cluster_indexes or jets")
    if dump_steps[name] > step_mapper[options.pipelineStep]:
        print("Skipping dump %s: unavailable at pipelineStep=%s" % (name, options.pipelineStep))
    elif name not in dump:
        dump.append(name)

process = cms.Process("ScoutingPhase2JetTagging")
setupPhase2Scouting(process, options)

# Timing and pipeline paths follow runScoutingPhase2TauTagging.py.
process.load("HLTrigger.Timer.FastTimerService_cfi")
process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = "Phase2JetTaggingTiming_resources.json"

# Unpacking: preserve the PF and vertex inputs used by the original SoftJets job.
process.unpacker = cms.EDProducer("l1sc::L1TScPhase2PuppiRawToDigi@alpaka",
    alpaka=cms.untracked.PSet(backend=cms.untracked.string(options.backend)),
    src=cms.InputTag("rawDataCollector"),
    streams=cms.vuint32(*pf_stream_ids),
    splitFactor=cms.uint32(split_factor),
    environment=cms.untracked.int32(options.environment),
)
process.vertexUnpacker = cms.EDProducer("l1sc::L1TScPhase2VertexRawToDigi@alpaka",
    alpaka=cms.untracked.PSet(backend=cms.untracked.string(options.backend)),
    src=cms.InputTag("rawDataCollector"),
    streams=cms.vuint32(*options.vertexStreamIDs),
    splitFactor=cms.uint32(options.vertexSplitFactor),
    environment=cms.untracked.int32(options.environment),
)
process.goodOrbitsByNBX = cms.EDFilter("GoodOrbitNBxSelector",
    unpackers=cms.VInputTag(),
    unpackersAlpaka=cms.VInputTag(cms.InputTag("unpacker")),
    nbxMin=cms.uint32(3564 * options.timeslices // options.tmuxPeriod),
)
process.p_pipeline = cms.Path(process.unpacker + process.vertexUnpacker + process.goodOrbitsByNBX)

# Clustering: use only the parameters belonging to the selected algorithm.
if step_mapper[options.pipelineStep] >= step_mapper["clustering"]:
    clustering_parameters = {
        "SCGreedy": dict(rParam=cms.double(options.jetR), nJets=cms.uint32(options.njets)),
        "SCNMS": dict(RSeed=cms.double(options.RSeed), RClu=cms.double(options.RClu)),
        "SCNMSWeighted": dict(RSeed=cms.double(options.RSeed), RClu=cms.double(options.RClu)),
        "SCNMSWeightedMultiIter": dict(
            RSeed=cms.double(options.RSeed), RCen=cms.double(options.RCen),
            RClu=cms.double(options.RClu), alphaSeed=cms.double(options.alphaSeed),
            minSeedPt=cms.double(options.minSeedPt), nCentroidIters=cms.uint32(options.nCentroidIters),
        ),
        "LinkTree": dict(RLink=cms.double(options.RLink), minSeedPt=cms.double(options.minSeedPt)),
    }
    process.scJets = cms.EDProducer("l1sc::L1TScPhase2SCJets@alpaka",
        alpaka=cms.untracked.PSet(backend=cms.untracked.string(options.backend)),
        srcCandidates=cms.InputTag("unpacker", "candidates"),
        srcBxLookup=cms.InputTag("unpacker", "bxLookup"),
        algo=cms.string(options.jetAlgo),
        minConstituents=cms.uint32(options.minConstituents),
        **clustering_parameters[options.jetAlgo],
    )
    process.p_pipeline += process.scJets

# Tagging: both producers consume the same clusters, PF candidates and vertices.
if options.pipelineStep == "tagging":
    if options.useTRT:
        process.ONNXService = cms.Service("ONNXService")
        producer = "l1sc::SoftJetIdONNX@alpaka"
        default_model = "L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.int8.fallback.onnx"
        default_batch_size = 512
    else:
        process.PyTorchService = cms.Service("PyTorchService")
        producer = "l1sc::SoftJetIdML@alpaka"
        default_model = "L1TriggerScouting/JetTagging/data/btagging/softjet_part_unique.pt"
        default_batch_size = 32

    process.softJets = cms.EDProducer(producer,
        alpaka=cms.untracked.PSet(backend=cms.untracked.string(options.backend)),
        pf=cms.InputTag("unpacker", "candidates"),
        clusters=cms.InputTag("scJets"),
        jetBxLookup=cms.InputTag("scJets"),
        vertices=cms.InputTag("vertexUnpacker"),
        vertexBxLookup=cms.InputTag("vertexUnpacker", "bxLookup"),
        model=cms.FileInPath(options.model or default_model),
        maxBatchSize=cms.uint32(options.batchSize or default_batch_size),
    )
    process.p_pipeline += process.softJets

# Optional dumps use the existing PF, cluster-index and jet OrbitFlatTable producers.
process.p_tables = cms.Path()
process.end = cms.EndPath()
if dump:
    process.p_tables += process.unpacker + process.vertexUnpacker + process.goodOrbitsByNBX
    if "candidates" in dump or "cluster_indexes" in dump:
        process.candOrbitTable = cms.EDProducer("PFCandidateSoAToOrbitFlatTable",
            srcBx=cms.InputTag("unpacker", "bxLookup"),
            srcCandidates=cms.InputTag("unpacker", "candidates"),
            name=cms.string("L1PF"),
        )
        process.p_tables += process.candOrbitTable
    if "cluster_indexes" in dump:
        process.clusterOrbitTable = cms.EDProducer("ClusterSoAToOrbitFlatTable",
            srcBx=cms.InputTag("unpacker", "bxLookup"),
            srcClusters=cms.InputTag("scJets"),
            clustering_name=cms.string(options.jetAlgo),
            name=cms.string("L1PF"),
            extension=cms.bool(True),
        )
        process.p_tables += process.clusterOrbitTable
    if "jets" in dump:
        process.jetOrbitTable = cms.EDProducer("ClusterObjSoAToOrbitFlatTable",
            srcBx=cms.InputTag("scJets"),
            srcClusters=cms.InputTag("scJets"),
            name=cms.string("SoftJets"),
            doc=cms.string("Phase-2 %s jets" % options.jetAlgo),
        )
        process.p_tables += process.jetOrbitTable
    process.out = cms.OutputModule("OrbitNanoAODOutputModule",
        fileName=cms.untracked.string(options.outFile),
        SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring("p_pipeline")),
        outputCommands=cms.untracked.vstring("drop *", "keep l1ScoutingRun3OrbitFlatTable_*_*_*"),
    )
    process.end += process.out

process.schedule = cms.Schedule(process.p_pipeline, process.p_tables, process.end)

if options.tracePerfetto:
    from PerfTools.Perfetto.customisePerfetto import customisePerfetto
    customisePerfetto(process,
        fileName="trace.pftrace",
        traceFunctions=True,
        traceGpuKernels=True,
        traceAllocations=True,
    )
