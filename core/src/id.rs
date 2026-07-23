//! Interned identifiers and the interning table (spec: Formats — open vocabulary,
//! closed representation). Ids compare as integers everywhere; strings are resolved
//! only for logging, dumps, and errors.

use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FormatId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FieldId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MetaId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ElementId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PadId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LinkId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GroupId(pub u32);

/// A string ↔ `u32` interning table. One instance per domain (formats, fields,
/// values, …); callers wrap the returned `u32` in the appropriate typed id.
///
/// Interning is idempotent: the same string always maps to the same id, and ids are
/// dense (`0..len`), so they double as indices into side tables.
pub struct Interner {
    map: HashMap<Box<str>, u32>,
    names: Vec<Box<str>>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }

    /// Intern `s`, returning its id (assigning a fresh one on first sighting).
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        let boxed: Box<str> = s.into();
        self.names.push(boxed.clone());
        self.map.insert(boxed, id);
        id
    }

    /// The id for `s` if it has been interned, without interning it.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }

    /// The string an id resolves to, for logging / dumps / errors.
    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(|s| &**s)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interns_and_dedups() {
        let mut i = Interner::new();
        let a = i.intern("video/raw");
        let b = i.intern("audio/raw");
        let a2 = i.intern("video/raw");
        assert_eq!(a, a2, "same string interns to same id");
        assert_ne!(a, b);
        assert_eq!(i.len(), 2, "duplicate did not grow the table");
    }

    #[test]
    fn ids_are_dense_and_ordered() {
        let mut i = Interner::new();
        assert_eq!(i.intern("a"), 0);
        assert_eq!(i.intern("b"), 1);
        assert_eq!(i.intern("c"), 2);
        assert_eq!(i.intern("b"), 1);
    }

    #[test]
    fn resolves_round_trip() {
        let mut i = Interner::new();
        let id = i.intern("h264/annexb");
        assert_eq!(i.resolve(id), Some("h264/annexb"));
        assert_eq!(i.get("h264/annexb"), Some(id));
        assert_eq!(i.resolve(999), None);
        assert_eq!(i.get("never-interned"), None);
    }

    #[test]
    fn empty_interner() {
        let i = Interner::new();
        assert!(i.is_empty());
        assert_eq!(i.len(), 0);
    }
}
