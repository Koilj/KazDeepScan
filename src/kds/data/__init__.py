"""Dataset manifests, assets, group splits, and PyTorch datasets."""

from kds.data.assets import AssetValidationError, validate_assets
from kds.data.consents import (
    ConsentRegistryEntry,
    ConsentRegistryError,
    load_consent_registry,
    product_eligible_speaker_ids,
)
from kds.data.ksc_slr102 import KscIngestionError
from kds.data.licenses import (
    LicenseLedgerError,
    TrainingProtocolError,
    TrainingProtocolReport,
    load_license_ledger,
    validate_manifest_licenses,
    validate_training_protocol,
)
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.source_matrix import (
    SourceMatrixError,
    SourceMixedResearchMatrix,
    SourceMixedResearchMatrixReport,
    load_source_mixed_research_matrix,
    validate_source_mixed_research_matrix,
)
from kds.data.split import GroupSplitter, SplitConfig

__all__ = [
    "AssetValidationError",
    "ConsentRegistryEntry",
    "ConsentRegistryError",
    "GroupSplitter",
    "KscIngestionError",
    "LicenseLedgerError",
    "ManifestError",
    "ManifestRow",
    "SplitConfig",
    "SourceMatrixError",
    "SourceMixedResearchMatrix",
    "SourceMixedResearchMatrixReport",
    "TrainingProtocolError",
    "TrainingProtocolReport",
    "load_manifest",
    "load_source_mixed_research_matrix",
    "load_consent_registry",
    "load_license_ledger",
    "validate_assets",
    "validate_manifest",
    "validate_manifest_licenses",
    "validate_training_protocol",
    "validate_source_mixed_research_matrix",
    "product_eligible_speaker_ids",
    "write_manifest",
]
