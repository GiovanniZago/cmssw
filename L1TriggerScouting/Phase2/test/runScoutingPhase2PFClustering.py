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

options.register ('njets',
                  16,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  'Number of jets to reconstruct with SCGreedy (iterative seeded cone)'
)

options.register ('minSeedPt',
                  0.0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.float,
                  'Minimum pt cut for seed candidates (used by SCNMSWeightedMultiIter / LinkTree)'
)

options.register ('jetR',
                  0.4,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.float,
                  'Legacy single-radius parameter used by auto / SCGreedy fallback / demo modules'
)
# Additional algorithms
# canonical names:
#   SCGreedy                = iterative greedy seeded cone
#   SCNMS                   = old non-iterative seeded cone with split radii:
#                             RSeed for seed finding + old centroid accumulation,
#                             RClu for final nearest-axis assignment
#   SCNMSWeighted           = same old non-iterative seeded cone logic, but final
#                             assignment uses old weighted metric
#   SCNMSWeightedMultiIter  = newer weighted NMS-style seeded cone with explicit
#                             centroid iterations and separate RCen
#   LinkTree                = link-based clustering
#
# backward-compatible aliases kept:
#   iterative               -> SCGreedy
#   seededCone              -> SCNMS
#   seededConeNMSWeighted   -> SCNMSWeighted
#   linkTree                -> LinkTree
options.register('jetAlgo',
                 'CLUE',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Jet algorithm: CLUE | SCGreedy | SCNMS | SCNMSWeighted | SCNMSWeightedMultiIter | LinkTree')

options.register('RSeed',
                 0.3,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 'Seed-finding radius; for SCNMS/SCNMSWeighted also used for old centroid accumulation')

options.register('RCen',
                 0.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 'Centroid radius (used only by SCNMSWeightedMultiIter)')

options.register('RClu',
                 0.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 'Final particle-assignment radius for SCNMS / SCNMSWeighted / SCNMSWeightedMultiIter')

options.register('RLink',
                 0.3,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 'Link radius for LinkTree')

options.register('alphaSeed',
                 2.0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 'alpha in score = dr2 / pt_seed^alpha (used by SCNMSWeightedMultiIter)')

options.register('nCentroidIters',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 'Number of centroid iterations for SCNMSWeightedMultiIter: 0=use seed axis, >0 refine axis')

options.register('minConstituents',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 'Minimum number of constituents for a jet')

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
process.ThroughputService = cms.Service("ThroughputService",
    enableDQM = cms.untracked.bool(False),
    eventRange = cms.untracked.uint32(options.maxEvents),
    eventResolution = cms.untracked.uint32(10),
    printEventSummary = cms.untracked.bool(True)
)

# report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.options.wantSummary = cms.untracked.bool(True)

# define pipeline path
process.p_clustering = cms.Path()

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

process.p_clustering += process.unpacker

# check orbits
process.check_orbits = cms.EDFilter("GoodOrbitNBxSelector", 
    unpackers=cms.VInputTag(),
    unpackersAlpaka=cms.VInputTag(cms.InputTag("unpacker")),
    nbxMin=cms.uint32(3564 * options.timeslices // options.tmuxPeriod)
)

process.p_clustering += process.check_orbits

# clustering
sc4Alpaka_kwargs = {
    "alpaka": cms.untracked.PSet( backend = cms.untracked.string(options.backend) ),
    "srcCandidates": cms.InputTag("unpacker", "candidates"),
    "srcBxLookup": cms.InputTag("unpacker", "bxLookup"),
    "algo": cms.string(options.jetAlgo),
    "minConstituents": cms.uint32(options.minConstituents),
}

if options.jetAlgo == "CLUE":
    process.load(
        "L1TriggerScouting.Phase2.L1TScPhase2CLUEJets_cff"
    )
    process.L1TScPhase2CLUEJetsProducer.alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(options.backend)
    )
    process.L1TScPhase2CLUEJetsProducer.candidates = cms.InputTag("unpacker", "candidates")
    process.L1TScPhase2CLUEJetsProducer.bxSizes = cms.InputTag("unpacker", "bxSizes")

    process.p_clustering += process.L1TScPhase2CLUEJetsProducer

else:
    from L1TriggerScouting.Phase2.modules import l1sc_L1TScPhase2SCJets_alpaka

    if options.jetAlgo == "SCGreedy":
        sc4Alpaka_kwargs.update({
            "rParam": cms.double(options.jetR),
            "nJets": cms.uint32(options.njets),
        })

    elif options.jetAlgo == "SCNMS" or options.jetAlgo == "SCNMSWeighted":
        sc4Alpaka_kwargs.update({
            "RSeed": cms.double(options.RSeed),
            "RClu": cms.double(options.RClu),
        })

    elif options.jetAlgo == "SCNMSWeightedMultiIter":
        sc4Alpaka_kwargs.update({
            "RSeed": cms.double(options.RSeed),
            "RCen": cms.double(options.RCen),
            "RClu": cms.double(options.RClu),
            "alphaSeed": cms.double(options.alphaSeed),
            "minSeedPt": cms.double(options.minSeedPt),
            "nCentroidIters": cms.uint32(options.nCentroidIters),
        })

    elif options.jetAlgo == "LinkTree":
        sc4Alpaka_kwargs.update({
            "RLink": cms.double(options.RLink),
            "minSeedPt": cms.double(options.minSeedPt),
        })
    
    else:
        raise ValueError("jetAlgo not recognized")

    process.scPhase2SC4PFAlpaka = l1sc_L1TScPhase2SCJets_alpaka(
      **sc4Alpaka_kwargs
    ) 

    process.p_clustering += process.scPhase2SC4PFAlpaka

# schedule 
process.schedule = cms.Schedule(
    process.p_clustering
)