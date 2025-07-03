#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub(crate) struct Span {
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
