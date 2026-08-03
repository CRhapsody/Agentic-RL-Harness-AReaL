"""Production training adapters built on validated joint trajectory records."""

from .areal_policy_candidate import (
    AREAL_POLICY_CANDIDATE_SCHEMA,
    PINNED_AREAL_COMMIT,
    ArealPolicyCandidateError,
    ValidatedArealPolicyCandidate,
    checkpoint_manifest,
    run_areal_policy_candidate_update,
    validate_areal_policy_candidate,
)
from .areal_policy_optimizer import (
    AREAL_EXTERNAL_ADVANTAGE_BATCH_SCHEMA,
    ArealExternalAdvantageBatchError,
    ValidatedArealExternalAdvantageBatch,
    build_areal_external_advantage_batch,
    materialize_areal_ppo_update_tensors,
    validate_areal_external_advantage_batch,
    validate_m0_areal_actor_config,
)

__all__ = [
    "AREAL_EXTERNAL_ADVANTAGE_BATCH_SCHEMA",
    "AREAL_POLICY_CANDIDATE_SCHEMA",
    "PINNED_AREAL_COMMIT",
    "ArealExternalAdvantageBatchError",
    "ArealPolicyCandidateError",
    "ValidatedArealExternalAdvantageBatch",
    "ValidatedArealPolicyCandidate",
    "build_areal_external_advantage_batch",
    "checkpoint_manifest",
    "materialize_areal_ppo_update_tensors",
    "run_areal_policy_candidate_update",
    "validate_areal_external_advantage_batch",
    "validate_areal_policy_candidate",
    "validate_m0_areal_actor_config",
]
