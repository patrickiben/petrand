"""
PetRand: Physical Chaos Hardware Entropy Generator & Monte Carlo Seeder
"""

from petrand.health import (
    NISTHealthMonitor,
    NISTStartupHealthTest,
    HealthError,
    RepetitionCountFailure,
    AdaptiveProportionFailure,
    StartupHealthFailure,
)
from petrand.entropy_estimator import (
    MostCommonValueEstimator,
    CollisionEstimator,
    MarkovEstimator,
    NISTEntropyAssessor,
)
from petrand.self_test import (
    SelfTestEngine,
    KnownAnswerSelfTestError,
)
from petrand.sts import (
    NISTStatisticalTestSuite,
)
from petrand.conditioning_bound import (
    NISTConditioningAssessor,
)
from petrand.engine import (
    PetRandEngine,
    RawPhysicalEvent,
    EngineHaltedError,
    InsufficientEntropyError,
    combine_entropy_estimates,
    intervals_to_von_neumann_bits,
)
from petrand.lifecycle import (
    BallProfile,
    BallStatus,
    TwoBallManager,
    COLOR_PRESETS,
)
from petrand.optical import (
    OverheadCameraTracker,
    OpticalTrackingResult,
)
from petrand.sensors import (
    BaseSensor,
    CameraMotionSensor,
    AudioImpactSensor,
    SyntheticArenaSimulator,
)
from petrand.adapters import (
    MonteCarloSeedFactory,
    uint64_to_uniform_float64,
    SeedSpooler,
    PetRandRandom,
    PetRandSecrets,
)
from petrand.benchmarks import (
    run_coverage_test,
    run_hypersphere_10d_benchmark,
    run_autocorrelation_test,
    run_tail_risk_benchmark,
)
from petrand.server import start_server
from petrand.toolkit import EverydaySeedToolkit
from petrand.zoomies import (
    FORM_CARDS,
    ZoomiesScorer,
    start_zoomies_server,
    write_standalone_html,
)
from petrand.antibot import (
    ZoomiesBotAudit,
)

__version__ = "1.7.0"
__all__ = [
    "NISTHealthMonitor",
    "NISTStartupHealthTest",
    "HealthError",
    "RepetitionCountFailure",
    "AdaptiveProportionFailure",
    "StartupHealthFailure",
    "MostCommonValueEstimator",
    "CollisionEstimator",
    "MarkovEstimator",
    "NISTEntropyAssessor",
    "SelfTestEngine",
    "KnownAnswerSelfTestError",
    "NISTStatisticalTestSuite",
    "NISTConditioningAssessor",
    "PetRandEngine",
    "RawPhysicalEvent",
    "EngineHaltedError",
    "InsufficientEntropyError",
    "combine_entropy_estimates",
    "intervals_to_von_neumann_bits",
    "BallProfile",
    "BallStatus",
    "TwoBallManager",
    "COLOR_PRESETS",
    "OverheadCameraTracker",
    "OpticalTrackingResult",
    "BaseSensor",
    "CameraMotionSensor",
    "AudioImpactSensor",
    "SyntheticArenaSimulator",
    "MonteCarloSeedFactory",
    "uint64_to_uniform_float64",
    "SeedSpooler",
    "PetRandRandom",
    "PetRandSecrets",
    "run_coverage_test",
    "run_hypersphere_10d_benchmark",
    "run_autocorrelation_test",
    "run_tail_risk_benchmark",
    "start_server",
    "EverydaySeedToolkit",
    "FORM_CARDS",
    "ZoomiesScorer",
    "start_zoomies_server",
    "write_standalone_html",
    "ZoomiesBotAudit",
]
