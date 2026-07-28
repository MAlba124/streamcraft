//! ITU-T H.273 code points ↔ the pipeline's categorical colorimetry names
//! (spec: Formats; RFC 9559 §5.1.4.1.31 — mkv's `Colour` element carries these
//! integers). mkv stays dependency-free, so the closed mapping is transcribed
//! here; the names match `streamcraft-video`'s color vocabulary and are pinned
//! by that crate's tests.

/// H.273 §8.3 `MatrixCoefficients` → `matrix` name. `None` = unspecified/exotic.
pub(crate) fn h273_matrix_name(cp: u8) -> Option<&'static str> {
    Some(match cp {
        0 => "identity",
        1 => "bt709",
        5 | 6 => "bt601",
        9 | 10 => "bt2020",
        _ => return None,
    })
}

/// `matrix` name → the canonical H.273 §8.3 code point to write.
pub(crate) fn matrix_name_h273(name: &str) -> Option<u8> {
    Some(match name {
        "identity" => 0,
        "bt709" => 1,
        "bt601" => 6, // SMPTE 170M — the canonical SD choice
        "bt2020" => 9, // non-constant luminance
        _ => return None,
    })
}

/// mkv `Colour\Range` → `range` name (1 = broadcast/limited, 2 = full).
pub(crate) fn mkv_range_name(v: u8) -> Option<&'static str> {
    Some(match v {
        1 => "limited",
        2 => "full",
        _ => return None,
    })
}

/// `range` name → mkv `Colour\Range` value.
pub(crate) fn range_name_mkv(name: &str) -> Option<u8> {
    Some(match name {
        "limited" => 1,
        "full" => 2,
        _ => return None,
    })
}

/// H.273 §8.2 `TransferCharacteristics` → `transfer` name.
pub(crate) fn h273_transfer_name(cp: u8) -> Option<&'static str> {
    Some(match cp {
        1 | 6 | 14 | 15 => "bt709",
        13 => "srgb",
        16 => "pq",
        18 => "hlg",
        8 => "linear",
        _ => return None,
    })
}

/// `transfer` name → the canonical H.273 §8.2 code point to write.
pub(crate) fn transfer_name_h273(name: &str) -> Option<u8> {
    Some(match name {
        "bt709" => 1,
        "srgb" => 13,
        "pq" => 16,
        "hlg" => 18,
        "linear" => 8,
        _ => return None,
    })
}

/// H.273 §8.1 `ColourPrimaries` → `primaries` name.
pub(crate) fn h273_primaries_name(cp: u8) -> Option<&'static str> {
    Some(match cp {
        1 => "bt709",
        5 | 6 => "bt601",
        9 => "bt2020",
        11 | 12 => "dci-p3",
        _ => return None,
    })
}

/// `primaries` name → the canonical H.273 §8.1 code point to write.
pub(crate) fn primaries_name_h273(name: &str) -> Option<u8> {
    Some(match name {
        "bt709" => 1,
        "bt601" => 6,
        "bt2020" => 9,
        "dci-p3" => 12, // D65 display variant
        _ => return None,
    })
}

/// Read announced colorimetry names from a fixed format into H.273 code points
/// for the writer's `Colour` element (spec: Formats — remux preserves color).
// COLD: once per stream at writer header setup; owns each color-name string briefly.
#[allow(clippy::disallowed_methods)]
pub(crate) fn colour_from_format(
    ctx: &streamcraft_core::ctx::Ctx,
    f: &streamcraft_core::format::FixedFormat,
) -> Option<crate::writer::ColourConfig> {
    let name = |field: &str| -> Option<String> {
        let id = ctx.field_id(field)?;
        match f.get(id)? {
            streamcraft_core::format::Value::Id(v) => ctx.value_name(v).map(str::to_owned),
            _ => None,
        }
    };
    // Unannounced fields stay at the ColourConfig defaults (RFC 9559
    // "unspecified" — not written), so an announced identity matrix (H.273
    // code point 0) still round-trips.
    let dflt = crate::writer::ColourConfig::default();
    let c = crate::writer::ColourConfig {
        matrix: name("matrix").as_deref().and_then(matrix_name_h273).unwrap_or(dflt.matrix),
        range: name("range").as_deref().and_then(range_name_mkv).unwrap_or(dflt.range),
        transfer: name("transfer").as_deref().and_then(transfer_name_h273).unwrap_or(dflt.transfer),
        primaries: name("primaries").as_deref().and_then(primaries_name_h273).unwrap_or(dflt.primaries),
    };
    (!c.is_empty()).then_some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_canonical_code_points() {
        for name in ["identity", "bt709", "bt601", "bt2020"] {
            assert_eq!(h273_matrix_name(matrix_name_h273(name).unwrap()), Some(name));
        }
        for name in ["limited", "full"] {
            assert_eq!(mkv_range_name(range_name_mkv(name).unwrap()), Some(name));
        }
        for name in ["bt709", "srgb", "pq", "hlg", "linear"] {
            assert_eq!(h273_transfer_name(transfer_name_h273(name).unwrap()), Some(name));
        }
        for name in ["bt709", "bt601", "bt2020", "dci-p3"] {
            assert_eq!(h273_primaries_name(primaries_name_h273(name).unwrap()), Some(name));
        }
    }
}
