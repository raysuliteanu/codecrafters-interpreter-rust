use std::fmt::Display;

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Span {
    offset: usize,
    len: usize,
    line: usize,
}

impl Span {
    pub fn new(line: usize, offset: usize, len: usize) -> Self {
        Span { offset, len, line }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn line(&self) -> usize {
        self.line
    }
}

impl From<(usize, usize, usize)> for Span {
    fn from(value: (usize, usize, usize)) -> Self {
        Span {
            line: value.0,
            offset: value.1,
            len: value.2,
        }
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line: {}, ({}, {})", self.line, self.offset, self.len)
    }
}
