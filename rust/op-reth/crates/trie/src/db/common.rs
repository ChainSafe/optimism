use alloy_eips::NumHash;

pub(crate) struct ProofWindowValue {
    pub earliest: NumHash,
    pub latest: NumHash,
}
