"""Common framework and DAQ input setup for Phase-2 scouting jobs.

Call after parsing Phase2.options_cff (or options derived from it)::

    from L1TriggerScouting.Phase2.L1TScPhase2RunScouting_cff import setupPhase2Scouting
    setupPhase2Scouting(process, options)

Then configure and schedule the desired unpacker with src="rawDataCollector".
Importing this module does not parse arguments or create directories. The setup
call creates the FU/BU run directories and the first BU's fu.lock file, following
the standalone scouting test configurations. Source settings can be overridden
on process.source afterwards.
"""

import os
import FWCore.ParameterSet.Config as cms

def setupPhase2Scouting(process, options):
    if not options.buBaseDir or len(options.buNumStreams) != len(options.buBaseDir):
        raise ValueError("Specify one buNumStreams entry per buBaseDir (at least one BU directory)")
    if any(n <= 0 for n in options.buNumStreams):
        raise ValueError("Each buNumStreams entry must be positive")

    process.EvFDaqDirector = cms.Service("EvFDaqDirector",
        useFileBroker = cms.untracked.bool(options.broker != "none"),
        fileBrokerHostFromCfg = cms.untracked.bool(False),
        fileBrokerHost = cms.untracked.string(options.broker.split(":")[0] if options.broker != "none" else "htcp40.cern.ch"),
        fileBrokerPort = cms.untracked.string(options.broker.split(":")[1] if options.broker != "none" else "8080"),
        runNumber = cms.untracked.uint32(options.runNumber),
        baseDir = cms.untracked.string(options.fuBaseDir),
        buBaseDir = cms.untracked.string(options.buBaseDir[0]),
        buBaseDirsAll = cms.untracked.vstring(*options.buBaseDir),
        buBaseDirsNumStreams = cms.untracked.vint32(*options.buNumStreams),
        directorIsBU = cms.untracked.bool(False),
    )
    process.FastMonitoringService = cms.Service("FastMonitoringService")

    runDir = "run%06d" % options.runNumber
    fuDir = os.path.join(options.fuBaseDir, runDir)
    buDirs = [os.path.join(baseDir, runDir) for baseDir in options.buBaseDir]
    for directory in [fuDir, options.fuBaseDir] + buDirs + options.buBaseDir:
        os.makedirs(directory, exist_ok=True)

    # process.source = cms.Source("DAQSource",
    #     testing = cms.untracked.bool(True),
    #     dataMode = cms.untracked.string(options.daqSourceMode),
    #     verifyChecksum = cms.untracked.bool(True),
    #     useL1EventID = cms.untracked.bool(False),
    #     eventChunkBlock = cms.untracked.uint32(64), # Keep each read(2) request well below Linux's per-call transfer limit.
    #     eventChunkSize = cms.untracked.uint32(8 * 1024), 
    #     maxChunkSize = cms.untracked.uint32(128 * 1024),
    #     numBuffers = cms.untracked.uint32(4),
    #     maxBufferedFiles = cms.untracked.uint32(4),
    #     fileListMode = cms.untracked.bool(options.broker == "none"),
    #     fileNames = cms.untracked.vstring(
    #         # buDirs[0] + "/" + "run%06d_ls%04d_index%012d_stream00.raw" % (options.runNumber, options.lumiNumber, 1), # 8194
    #         buDirs[0] + "/" + "run%06d_ls%04d_index%012d_stream00.raw" % (options.runNumber, options.lumiNumber, 1),
    #     ),
    # )
    process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2 * 1024),
    eventChunkSize = cms.untracked.uint32(2 * 1024),
    maxChunkSize = cms.untracked.uint32(4 * 1024),
    numBuffers = cms.untracked.uint32(4),
    maxBufferedFiles = cms.untracked.uint32(4),
    fileListMode = cms.untracked.bool(options.broker == "none"),
    fileNames = cms.untracked.vstring(
        # buDirs[0] + "/" + "run%06d_ls%04d_index%06d_stream00.raw" % (options.runNumber, options.lumiNumber, 1),
        buDirs[0] + "/" + "run%06d_ls%04d_index%012d_stream00.raw" % (options.runNumber, options.lumiNumber, 1),
    )
)

    os.system("touch " + buDirs[0] + "/" + "fu.lock")
    
    return process
