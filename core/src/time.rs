//! Nanosecond timestamps. All timeline math in core happens in this unit; external
//! timebases (RTP, device clocks, container timestamps) convert at the edges only.

/// Nanoseconds since an unspecified epoch (running time, PTS, durations).
///
/// [`Timestamp::NONE`] is the "unset" sentinel (`u64::MAX`). Note that the derived
/// `Ord` sorts `NONE` as the *largest* value — treat "unknown" as "infinitely late"
/// and guard comparisons where that is wrong.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// The "unset" sentinel.
    pub const NONE: Self = Self(u64::MAX);
    pub const ZERO: Self = Self(0);
    /// Largest representable *valid* time; [`Timestamp::NONE`] sits one above this,
    /// so saturating arithmetic can never accidentally produce the sentinel.
    pub const MAX: Self = Self(u64::MAX - 1);

    pub const fn from_nanos(ns: u64) -> Self {
        Self(ns)
    }

    pub const fn from_micros(us: u64) -> Self {
        Self(us.saturating_mul(1_000))
    }

    pub const fn from_millis(ms: u64) -> Self {
        Self(ms.saturating_mul(1_000_000))
    }

    pub const fn from_secs(s: u64) -> Self {
        Self(s.saturating_mul(1_000_000_000))
    }

    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }

    pub const fn is_some(self) -> bool {
        self.0 != u64::MAX
    }

    /// Raw nanoseconds, or `None` for the sentinel.
    pub const fn nanos(self) -> Option<u64> {
        if self.is_none() {
            None
        } else {
            Some(self.0)
        }
    }

    /// Saturating add that propagates `NONE` and never overflows into the sentinel.
    pub const fn saturating_add(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() {
            return Self::NONE;
        }
        match self.0.checked_add(rhs.0) {
            Some(sum) if sum < u64::MAX => Self(sum),
            _ => Self::MAX,
        }
    }

    /// Saturating subtract; propagates `NONE`, clamps at [`Timestamp::ZERO`].
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() {
            return Self::NONE;
        }
        Self(self.0.saturating_sub(rhs.0))
    }
}

/// Exact rational, for rates and durations that must not accumulate error
/// (e.g. 30000/1001 fps).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rational {
    pub num: i32,
    pub den: i32,
}

impl Rational {
    pub const fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    pub const fn is_valid(self) -> bool {
        self.den != 0
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(Timestamp::from_micros(1).0, 1_000);
        assert_eq!(Timestamp::from_millis(1).0, 1_000_000);
        assert_eq!(Timestamp::from_secs(1).0, 1_000_000_000);
    }

    #[test]
    fn none_sentinel() {
        assert!(Timestamp::NONE.is_none());
        assert!(!Timestamp::NONE.is_some());
        assert!(Timestamp::ZERO.is_some());
        assert_eq!(Timestamp::NONE.nanos(), None);
        assert_eq!(Timestamp::from_nanos(5).nanos(), Some(5));
    }

    #[test]
    fn saturating_add_propagates_none_and_never_hits_sentinel() {
        assert_eq!(
            Timestamp::from_nanos(2).saturating_add(Timestamp::from_nanos(3)),
            Timestamp::from_nanos(5)
        );
        assert!(Timestamp::NONE.saturating_add(Timestamp::ZERO).is_none());
        assert!(Timestamp::ZERO.saturating_add(Timestamp::NONE).is_none());
        // Overflow clamps to MAX (a valid time), not into NONE.
        let big = Timestamp::MAX.saturating_add(Timestamp::MAX);
        assert_eq!(big, Timestamp::MAX);
        assert!(big.is_some());
    }

    #[test]
    fn saturating_sub_clamps_at_zero() {
        assert_eq!(
            Timestamp::from_nanos(3).saturating_sub(Timestamp::from_nanos(5)),
            Timestamp::ZERO
        );
        assert_eq!(
            Timestamp::from_nanos(5).saturating_sub(Timestamp::from_nanos(3)),
            Timestamp::from_nanos(2)
        );
        assert!(Timestamp::from_nanos(5).saturating_sub(Timestamp::NONE).is_none());
    }

    #[test]
    fn rational() {
        assert_eq!(Rational::new(1, 2).as_f64(), 0.5);
        assert!(Rational::new(30000, 1001).is_valid());
        assert!(!Rational::new(1, 0).is_valid());
    }
}
