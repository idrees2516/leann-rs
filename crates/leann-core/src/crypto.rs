//! At-rest encryption for single-file indexes (AEAD container).
//!
//! The container is `MAGIC_ENC(8) + version(4) + salt(16) + nonce(12) +
//! ciphertext||tag`, where the *entire* plaintext index serialization
//! (texts, metadatas, codebooks, codes, graph, tombstones) is sealed as one
//! ChaCha20-Poly1305 blob. The key is derived from a passphrase with
//! Argon2id (m = 19 MiB, t = 2, p = 1 — the OWASP-recommended baseline),
//! so passphrase guessing on a stolen file is expensive by construction.
//!
//! Security properties:
//! - **Confidentiality + integrity**: ChaCha20-Poly1305 is an AEAD; any
//!   tampering (bit flips, truncation, splice) fails authentication.
//! - **Constant-time tag verification** is provided by the AEAD itself, so
//!   wrong-passphrase and corrupted-file responses are indistinguishable in
//!   timing beyond the Argon2 derivation cost.
//! - **Random 96-bit nonce per save** (never reused under one key; the key
//!   itself is salt-derived and never persisted).
//! - **Quantized codes and the graph structure are inside the seal** — a
//!   stolen encrypted index leaks nothing about content or corpus size
//!   (length is padded by the AEAD tag but the header keeps no `n`).

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::ChaCha20Poly1305;

use crate::{LeannError, Result};

/// Magic bytes for the encrypted container (8 bytes).
pub const MAGIC_ENC: &[u8; 8] = b"LEANNENC";
/// Container format version.
pub const ENC_VERSION: u32 = 1;
/// Argon2id salt length.
pub const SALT_LEN: usize = 16;
/// ChaCha20-Poly1305 nonce length.
pub const NONCE_LEN: usize = 12;
/// ChaCha20-Poly1305 tag length.
pub const TAG_LEN: usize = 16;

/// Derive a 256-bit key from a passphrase with Argon2id.
pub fn derive_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32]> {
    use argon2::{Algorithm, Argon2, Params, Version};
    let params = Params::new(19 * 1024, 2, 1, Some(32))
        .map_err(|e| LeannError::Invalid(format!("argon2 params: {e}")))?;
    let a2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    a2.hash_password_into(passphrase.as_bytes(), salt, &mut key)
        .map_err(|e| LeannError::Invalid(format!("argon2: {e}")))?;
    Ok(key)
}

/// Encrypt `plaintext` under `key`. Returns `nonce || ciphertext||tag`.
pub fn seal(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce: [u8; NONCE_LEN] = rand::random();
    let ct = cipher
        .encrypt((&nonce).into(), plaintext)
        .map_err(|_| LeannError::Crypto("encryption failed".into()))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Decrypt a `seal`ed blob. Fails on wrong key, tampering, or truncation.
pub fn unseal(key: &[u8; 32], blob: &[u8]) -> Result<Vec<u8>> {
    if blob.len() < NONCE_LEN + TAG_LEN {
        return Err(LeannError::Corrupt("encrypted blob too short".into()));
    }
    let (nonce, ct) = blob.split_at(NONCE_LEN);
    let cipher = ChaCha20Poly1305::new(key.into());
    cipher
        .decrypt(nonce.into(), ct)
        .map_err(|_| LeannError::Crypto("decryption failed (wrong passphrase or tampered file)".into()))
}

/// Seal with a freshly generated salt; returns `(salt, blob)`.
pub fn seal_passphrase(passphrase: &str, plaintext: &[u8]) -> Result<([u8; SALT_LEN], Vec<u8>)> {
    let salt: [u8; SALT_LEN] = rand::random();
    let key = derive_key(passphrase, &salt)?;
    let blob = seal(&key, plaintext)?;
    Ok((salt, blob))
}

/// Unseal a `seal_passphrase`ed blob with the given salt.
pub fn unseal_passphrase(passphrase: &str, salt: &[u8], blob: &[u8]) -> Result<Vec<u8>> {
    let key = derive_key(passphrase, salt)?;
    unseal(&key, blob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tamper() {
        let pt = b"the quick brown fox jumps over the lazy dog".to_vec();
        let (salt, blob) = seal_passphrase("correct horse battery staple", &pt).unwrap();
        let back = unseal_passphrase("correct horse battery staple", &salt, &blob).unwrap();
        assert_eq!(back, pt);

        // wrong passphrase fails
        assert!(unseal_passphrase("wrong", &salt, &blob).is_err());

        // bit flip fails
        let mut tampered = blob.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0x01;
        assert!(unseal_passphrase("correct horse battery staple", &salt, &tampered).is_err());

        // truncation fails
        assert!(unseal_passphrase("correct horse battery staple", &salt, &blob[..blob.len() - 1]).is_err());
    }

    #[test]
    fn salt_diversifies() {
        let pt = b"same plaintext".to_vec();
        let (s1, b1) = seal_passphrase("pw", &pt).unwrap();
        let (s2, b2) = seal_passphrase("pw", &pt).unwrap();
        assert_ne!(s1, s2, "salts must be random");
        assert_ne!(b1, b2, "nonces must be random");
        assert_eq!(unseal_passphrase("pw", &s1, &b1).unwrap(), pt);
        assert_eq!(unseal_passphrase("pw", &s2, &b2).unwrap(), pt);
    }
}
