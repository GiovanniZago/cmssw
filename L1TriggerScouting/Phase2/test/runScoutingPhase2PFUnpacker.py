# define process
import FWCore.ParameterSet.Config as cms
process = cms.Process("ScoutingPhase2PFUnpackers")

# import L1 scouting options
from L1TriggerScouting.Phase2.options_cff import options, VarParsing

# extra options
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

# parse options
options.parseArguments()

# framework setup
process.load("Configuration.StandardSequences.Accelerators_cff")
process.options.numberOfThreads = cms.untracked.uint32(options.numThreads)
process.options.numberOfStreams = cms.untracked.uint32(options.numFwkStreams)
process.options.numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

# setup DAQ director and DAQ source
from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
setupPhase2Scouting(process, options)

# timing
process.load( "HLTrigger.Timer.FastTimerService_cfi" )
process.FastTimerService.printEventSummary = cms.untracked.bool(True)
process.FastTimerService.writeJSONSummary = cms.untracked.bool(True)
process.FastTimerService.jsonFileName = cms.untracked.string(f"resources_runScoutingPhase2PFUnpacker.json")
process.FastTimerService.enableTimingPaths = cms.untracked.bool(True)
process.FastTimerService.enableTimingModules = cms.untracked.bool(True)
process.FastTimerService.useRealTimeClock = cms.untracked.bool(True)

# throughput
# process.ThroughputService = cms.Service("ThroughputService",
#     enableDQM = cms.untracked.bool(False),
#     eventResolution = cms.untracked.uint32(10),
#     printEventSummary = cms.untracked.bool(True)
# )
# if options.maxEvents >= 0:
#     process.ThroughputService.eventRange = cms.untracked.uint32(options.maxEvents)

# report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
# process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1)
# )
process.options.wantSummary = cms.untracked.bool(True)

# define pipeline path
process.p_unpacking = cms.Path()

# unpacker
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

process.p_unpacking += process.unpacker

# check orbits
process.check_orbits = cms.EDFilter("GoodOrbitNBxSelector", 
    unpackers=cms.VInputTag(),
    unpackersAlpaka=cms.VInputTag(cms.InputTag("unpacker")),
    nbxMin=cms.uint32(3564 * options.timeslices // options.tmuxPeriod)
)

process.p_unpacking += process.check_orbits

# schedule 
process.schedule = cms.Schedule(
    process.p_unpacking
)
