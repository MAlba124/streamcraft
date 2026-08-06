//! MD5 (RFC 1321), implemented from the RFC because this tree takes no
//! crypto crates. It exists solely for RTSP Digest authentication — RFC 2617
//! §3.2.1 makes "MD5" the default (and, in practice, only deployed) digest
//! algorithm, and §3.1.3 represents every digest as 32 lowercase hex digits,
//! which is what [`md5_hex`] returns.
//!
//! MD5 is cryptographically broken (practical collisions); it is never used
//! here for content integrity, only to answer Digest challenges.
//!
//! The layout follows RFC 1321 §3 step by step: pad to 448 mod 512 bits
//! (§3.1), append the 64-bit little-endian bit count (§3.2), initialize the
//! four-word state (§3.3), digest 16-word blocks through 4 rounds × 16
//! operations (§3.4), and emit the state little-endian, A first (§3.5).

// COLD: Digest-auth primitive, run once per RTSP challenge — never per media
// packet; the padding buffer and hex string are one-time per hash.
#![allow(clippy::disallowed_methods)]

/// Per-operation left-rotate amounts (RFC 1321 §3.4: the `s` constants of
/// the four rounds — 7/12/17/22, 5/9/14/20, 4/11/16/23, 6/10/15/21).
#[rustfmt::skip]
const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// The sine-derived additive constants (RFC 1321 §3.4: `T[i+1] =
/// floor(4294967296 × abs(sin(i+1)))`, i+1 in radians; values as tabulated
/// in the RFC's appendix A.3 reference implementation).
#[rustfmt::skip]
const T: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// The MD5 digest of `msg` (RFC 1321).
pub fn md5(msg: &[u8]) -> [u8; 16] {
    // §3.3: the initial state, given in the RFC as little-endian byte
    // sequences `01 23 45 67`, `89 ab cd ef`, `fe dc ba 98`, `76 54 32 10`.
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];

    // §3.1: append a single '1' bit (0x80) then '0' bits until the length is
    // congruent to 448 mod 512 bits (56 mod 64 bytes) — padding is always
    // performed, even if the length already matches. §3.2: then append the
    // original length in bits as a 64-bit little-endian ("low-order word
    // first") quantity, taken modulo 2^64.
    let mut data = Vec::with_capacity(msg.len() + 72);
    data.extend_from_slice(msg);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&(msg.len() as u64).wrapping_mul(8).to_le_bytes());

    // §3.4: process each 16-word (64-byte) block. Words are little-endian.
    for block in data.as_chunks::<64>().0 {
        let mut x = [0u32; 16];
        for (i, w) in block.as_chunks::<4>().0.iter().enumerate() {
            x[i] = u32::from_le_bytes(*w);
        }
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            // §3.4: one auxiliary function and message-word order per round —
            //   round 1: F(X,Y,Z) = XY v not(X)Z,      word x[i]
            //   round 2: G(X,Y,Z) = XZ v Y not(Z),     word x[(1+5i) mod 16]
            //   round 3: H(X,Y,Z) = X xor Y xor Z,     word x[(5+3i) mod 16]
            //   round 4: I(X,Y,Z) = Y xor (X v not(Z)), word x[(7i) mod 16]
            let (f, k) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((b & d) | (c & !d), (1 + 5 * i) % 16),
                2 => (b ^ c ^ d, (5 + 3 * i) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            // §3.4: a = b + ((a + f(b,c,d) + X[k] + T[i]) <<< s), then rotate
            // the roles (d→a→b→c→d) — expressed here by shuffling variables.
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(x[k])
                .wrapping_add(T[i])
                .rotate_left(S[i]);
            (a, b, c, d) = (d, b.wrapping_add(rotated), b, c);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }

    // §3.5: output is A,B,C,D each low-order byte first.
    let mut out = [0u8; 16];
    for (o, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *o = w.to_le_bytes();
    }
    out
}

/// The MD5 digest as 32 lowercase hex digits — the representation Digest
/// auth hashes and transmits (RFC 2617 §3.1.3: "an ASCII hex representation
/// ... using lowercase").
pub fn md5_hex(msg: &[u8]) -> String {
    let mut s = String::with_capacity(32);
    for b in md5(msg) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::md5_hex;

    /// The full RFC 1321 appendix A.5 test suite.
    #[test]
    fn rfc1321_a5_test_suite() {
        for (msg, want) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ] {
            assert_eq!(md5_hex(msg.as_bytes()), want, "MD5({msg:?})");
        }
    }

    /// Padding edge cases around the 56 mod 64 boundary (§3.1: padding is
    /// 1..=64 bytes; a 56-byte message forces a second block).
    #[test]
    fn padding_boundaries_digest_without_panicking() {
        for n in [55usize, 56, 57, 63, 64, 65, 119, 120] {
            let msg = vec![0xA5u8; n];
            assert_eq!(md5_hex(&msg).len(), 32);
        }
    }
}
