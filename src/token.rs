use std::fmt::Display;
use thiserror::Error;

use crate::span::Span;

#[derive(Error, Debug)]
pub enum TokenError {
    #[error("[line {}] Error: Unexpected character: {}", .span.line(),
    .src[.span.offset()..(.span.offset() + .span.len())].to_string())]
    InvalidToken { src: String, span: Span },

    // CC/Book wants error message exactly like this even though easy
    // enough to have the part of the unterminated string in the error
    #[error("[line {}] Error: Unterminated string.", .span.line())]
    UnterminatedString { src: String, span: Span },
}

pub type TokenResult<T> = Result<T, u8>;

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub(crate) lexeme: Lexeme,
    pub(crate) span: Span,
}

impl Token {
    pub fn new(lexeme: Lexeme, span: Span) -> Token {
        Token { lexeme, span }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lexeme = &self.lexeme;
        match lexeme {
            Lexeme::String(value) => {
                write!(f, "{lexeme} \"{value}\" {value}")
            }
            Lexeme::Number(raw, value) => {
                if *value == value.trunc() {
                    write!(f, "NUMBER {raw} {value}.0")
                } else {
                    write!(f, "NUMBER {raw} {value}")
                }
            }
            Lexeme::Identifier(value) => {
                write!(f, "{lexeme} {value} null")
            }
            Lexeme::LeftParen
            | Lexeme::RightParen
            | Lexeme::LeftBrace
            | Lexeme::RightBrace
            | Lexeme::Dot
            | Lexeme::Comma
            | Lexeme::Minus
            | Lexeme::Plus
            | Lexeme::SemiColon
            | Lexeme::Star
            | Lexeme::Eq
            | Lexeme::Bang
            | Lexeme::Less
            | Lexeme::Greater
            | Lexeme::Slash => write!(f, "{lexeme} {} null", lexeme.lexeme_str()),
            Lexeme::EqEq
            | Lexeme::BangEq
            | Lexeme::LessEq
            | Lexeme::GreaterEq => write!(f, "{lexeme} {} null", lexeme.lexeme_str()),
            Lexeme::True
            | Lexeme::False
            | Lexeme::Nil
            | Lexeme::And
            | Lexeme::Or
            | Lexeme::Class
            | Lexeme::For
            | Lexeme::Fun
            | Lexeme::If
            | Lexeme::Else
            | Lexeme::Return
            | Lexeme::Super
            | Lexeme::This
            | Lexeme::Var
            | Lexeme::While
            | Lexeme::Print => {
                let v = lexeme.lexeme_str();
                write!(f, "{lexeme} {v} null")
            }
            Lexeme::Eof => {
                write!(f, "{lexeme}  null")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Lexeme {
    // keywords
    True,
    False,
    Nil,
    And,
    Or,
    Class,
    For,
    Fun,
    If,
    Else,
    Return,
    Super,
    This,
    Var,
    While,
    Print,

    // literals
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    SemiColon,
    Star,
    Eq,
    EqEq,
    Bang,
    BangEq,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Slash,

    // value holders
    Number(String, f64),
    Identifier(String),
    String(String),

    // symbolic placeholder
    Eof,
}

impl Lexeme {
    /// Returns the lexeme as a string (the actual source representation)
    pub fn lexeme_str(&self) -> &'static str {
        match self {
            Lexeme::LeftParen => "(",
            Lexeme::RightParen => ")",
            Lexeme::LeftBrace => "{",
            Lexeme::RightBrace => "}",
            Lexeme::Comma => ",",
            Lexeme::Dot => ".",
            Lexeme::Minus => "-",
            Lexeme::Plus => "+",
            Lexeme::SemiColon => ";",
            Lexeme::Star => "*",
            Lexeme::Eq => "=",
            Lexeme::EqEq => "==",
            Lexeme::Bang => "!",
            Lexeme::BangEq => "!=",
            Lexeme::Less => "<",
            Lexeme::LessEq => "<=",
            Lexeme::Greater => ">",
            Lexeme::GreaterEq => ">=",
            Lexeme::Slash => "/",
            Lexeme::True => "true",
            Lexeme::False => "false",
            Lexeme::Nil => "nil",
            Lexeme::And => "and",
            Lexeme::Or => "or",
            Lexeme::Class => "class",
            Lexeme::For => "for",
            Lexeme::Fun => "fun",
            Lexeme::If => "if",
            Lexeme::Else => "else",
            Lexeme::Return => "return",
            Lexeme::Super => "super",
            Lexeme::This => "this",
            Lexeme::Var => "var",
            Lexeme::While => "while",
            Lexeme::Print => "print",
            Lexeme::Eof => "eof",
            // Variable lexemes don't have constant representations
            Lexeme::Number(_, _) | Lexeme::Identifier(_) | Lexeme::String(_) => 
                panic!("Variable lexemes don't have constant string representations"),
        }
    }

    /// Returns the display name for the lexeme (uppercase token type)
    pub fn display_name(&self) -> &'static str {
        match self {
            Lexeme::True => "TRUE",
            Lexeme::False => "FALSE",
            Lexeme::Nil => "NIL",
            Lexeme::And => "AND",
            Lexeme::Or => "OR",
            Lexeme::Class => "CLASS",
            Lexeme::For => "FOR",
            Lexeme::Fun => "FUN",
            Lexeme::If => "IF",
            Lexeme::Else => "ELSE",
            Lexeme::Return => "RETURN",
            Lexeme::Super => "SUPER",
            Lexeme::This => "THIS",
            Lexeme::Var => "VAR",
            Lexeme::While => "WHILE",
            Lexeme::Print => "PRINT",
            Lexeme::Eof => "EOF",
            _ => self.lexeme_str(),
        }
    }
}

impl From<&Lexeme> for String {
    fn from(value: &Lexeme) -> Self {
        match value {
            Lexeme::Identifier(s) | Lexeme::String(s) => String::from(s),
            Lexeme::Number(s, _) => String::from(s),
            // For all constant lexemes, use the lexeme_str() method
            _ => value.lexeme_str().to_string(),
        }
    }
}

impl From<&str> for Lexeme {
    fn from(value: &str) -> Self {
        match value {
            "(" => Lexeme::LeftParen,
            ")" => Lexeme::RightParen,
            "{" => Lexeme::LeftBrace,
            "}" => Lexeme::RightBrace,
            "," => Lexeme::Comma,
            "." => Lexeme::Dot,
            ";" => Lexeme::SemiColon,
            "+" => Lexeme::Plus,
            "-" => Lexeme::Minus,
            "*" => Lexeme::Star,
            "/" => Lexeme::Slash,
            "=" => Lexeme::Eq,
            "!" => Lexeme::Bang,
            "<" => Lexeme::Less,
            ">" => Lexeme::Greater,
            "==" => Lexeme::EqEq,
            "!=" => Lexeme::BangEq,
            "<=" => Lexeme::LessEq,
            ">=" => Lexeme::GreaterEq,
            "print" => Lexeme::Print,
            "true" => Lexeme::True,
            "false" => Lexeme::False,
            "nil" => Lexeme::Nil,
            "and" => Lexeme::And,
            "or" => Lexeme::Or,
            "fun" => Lexeme::Fun,
            "return" => Lexeme::Return,
            "if" => Lexeme::If,
            "else" => Lexeme::Else,
            "for" => Lexeme::For,
            "while" => Lexeme::While,
            "class" => Lexeme::Class,
            "super" => Lexeme::Super,
            "this" => Lexeme::This,
            "var" => Lexeme::Var,
            "eof" => Lexeme::Eof,
            _ => panic!("invalid token {value}"),
        }
    }
}

impl Display for Lexeme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lexeme::Number(_, _) => write!(f, "NUMBER"),
            Lexeme::Identifier(_) => write!(f, "IDENTIFIER"),
            Lexeme::String(_) => write!(f, "STRING"),
            Lexeme::LeftParen => write!(f, "LEFT_PAREN"),
            Lexeme::RightParen => write!(f, "RIGHT_PAREN"),
            Lexeme::LeftBrace => write!(f, "LEFT_BRACE"),
            Lexeme::RightBrace => write!(f, "RIGHT_BRACE"),
            Lexeme::Dot => write!(f, "DOT"),
            Lexeme::Comma => write!(f, "COMMA"),
            Lexeme::Minus => write!(f, "MINUS"),
            Lexeme::Plus => write!(f, "PLUS"),
            Lexeme::SemiColon => write!(f, "SEMICOLON"),
            Lexeme::Star => write!(f, "STAR"),
            Lexeme::Eq => write!(f, "EQUAL"),
            Lexeme::EqEq => write!(f, "EQUAL_EQUAL"),
            Lexeme::Bang => write!(f, "BANG"),
            Lexeme::BangEq => write!(f, "BANG_EQUAL"),
            Lexeme::Less => write!(f, "LESS"),
            Lexeme::LessEq => write!(f, "LESS_EQUAL"),
            Lexeme::Greater => write!(f, "GREATER"),
            Lexeme::GreaterEq => write!(f, "GREATER_EQUAL"),
            Lexeme::Slash => write!(f, "SLASH"),
            // For keywords and EOF, use the display_name method
            _ => write!(f, "{}", self.display_name()),
        }
    }
}

pub struct Scanner<'scanner> {
    source: &'scanner str,
    print_tokens: bool,
}

impl<'scanner> Scanner<'scanner> {
    pub fn new(source: &'scanner str, print_tokens: bool) -> Self {
        Scanner {
            source,
            print_tokens,
        }
    }

    pub fn scan(self) -> TokenResult<Vec<Token>> {
        let mut line: usize = 1;
        let mut tokens: Vec<Token> = Vec::new();
        let mut peekable_iter = self.source.char_indices().peekable();
        let mut has_error = false;

        #[allow(clippy::while_let_loop)]
        loop {
            match peekable_iter.next() {
                Some((i, c)) => match c {
                    c if one_of(c, "\n\r") => {
                        line += 1;
                        continue;
                    }
                    c if c.is_whitespace() => continue,
                    '(' => tokens.push(Token::new(lexeme_from("("), (line, i, 1).into())),
                    ')' => tokens.push(Token::new(lexeme_from(")"), (line, i, 1).into())),
                    '{' => tokens.push(Token::new(lexeme_from("{"), (line, i, 1).into())),
                    '}' => tokens.push(Token::new(lexeme_from("}"), (line, i, 1).into())),
                    ',' => tokens.push(Token::new(lexeme_from(","), (line, i, 1).into())),
                    '.' => tokens.push(Token::new(lexeme_from("."), (line, i, 1).into())),
                    '+' => tokens.push(Token::new(lexeme_from("+"), (line, i, 1).into())),
                    '-' => tokens.push(Token::new(lexeme_from("-"), (line, i, 1).into())),
                    ';' => tokens.push(Token::new(lexeme_from(";"), (line, i, 1).into())),
                    '*' => tokens.push(Token::new(lexeme_from("*"), (line, i, 1).into())),
                    '=' => {
                        if peekable_iter.peek().is_some_and(|(_, l)| *l == '=') {
                            peekable_iter.next();
                            tokens.push(Token::new(lexeme_from("=="), (line, i, 2).into()))
                        } else {
                            tokens.push(Token::new(lexeme_from("="), (line, i, 1).into()))
                        }
                    }
                    '<' => {
                        if peekable_iter.peek().is_some_and(|(_, l)| *l == '=') {
                            peekable_iter.next();
                            tokens.push(Token::new(lexeme_from("<="), (line, i, 2).into()))
                        } else {
                            tokens.push(Token::new(lexeme_from("<"), (line, i, 1).into()))
                        }
                    }
                    '>' => {
                        if peekable_iter.peek().is_some_and(|(_, l)| *l == '=') {
                            peekable_iter.next();
                            tokens.push(Token::new(lexeme_from(">="), (line, i, 2).into()))
                        } else {
                            tokens.push(Token::new(lexeme_from(">"), (line, i, 1).into()))
                        }
                    }
                    '!' => {
                        if peekable_iter.peek().is_some_and(|(_, l)| *l == '=') {
                            peekable_iter.next();
                            tokens.push(Token::new(lexeme_from("!="), (line, i, 2).into()))
                        } else {
                            tokens.push(Token::new(lexeme_from("!"), (line, i, 1).into()))
                        }
                    }
                    '\"' => {
                        let mut str = String::new();

                        while peekable_iter
                            .peek()
                            .is_some_and(|(_, l)| !matches!(*l, '"'))
                        {
                            let (_, ch) = peekable_iter.next().unwrap();
                            str.push(ch);
                        }

                        if peekable_iter.peek().is_none() {
                            let error = TokenError::UnterminatedString {
                                src: self.source.to_string(),
                                span: Span::new(line, i, str.len()),
                            };
                            eprintln!("{error}");
                            has_error = true;
                        } else {
                            // consume terminating "
                            let (l, _) = peekable_iter.next().unwrap();
                            tokens
                                .push(Token::new(Lexeme::String(str), (line, i + 1, l - 1).into()))
                        }
                    }
                    c if c.is_ascii_digit() => {
                        let mut number = String::from(c);
                        while peekable_iter
                            .peek()
                            .is_some_and(|(_, l)| l.is_ascii_digit())
                        {
                            let (_, ch) = peekable_iter.next().unwrap();
                            number.push(ch);
                        }

                        if peekable_iter.peek().is_some_and(|(_, l)| matches!(*l, '.')) {
                            let (_, ch) = peekable_iter.next().unwrap();
                            number.push(ch);
                        }

                        while peekable_iter
                            .peek()
                            .is_some_and(|(_, l)| l.is_ascii_digit())
                        {
                            let (_, ch) = peekable_iter.next().unwrap();
                            number.push(ch);
                        }

                        let mut num_literal = number.as_str();

                        // There are 3 possibilities now. The number can be one of
                        // 1. just digits e.g. 123
                        // 2. digits plus a trailing . e.g. 123.
                        // 3. digits before and after a . e.g. 123.45
                        // So doing the splitn(3, '.') will result in
                        // 1. Some, None, None => the _ case
                        // 2. Some, Some, None => the 123. case
                        // 3. Some, Some, Some => the 123.45 case
                        let mut split = num_literal.splitn(3, '.');
                        match (split.next(), split.next(), split.next()) {
                            (Some(first), Some(second), Some(_)) => {
                                num_literal = &num_literal[..first.len() + 1 + second.len()]; // +1 for the dot sep
                            }
                            (Some(first), Some(second), None) => {
                                if second.is_empty() {
                                    num_literal = &num_literal[..first.len()];
                                }
                            }
                            _ => {}
                        }

                        if let Ok(num) = num_literal.parse::<f64>() {
                            tokens.push(Token::new(
                                Lexeme::Number(num_literal.to_string(), num),
                                (line, i, num_literal.len()).into(),
                            ));
                        } else {
                            let error = TokenError::InvalidToken {
                                src: number.clone(),
                                span: Span::new(line, i, num_literal.len()),
                            };
                            eprintln!("{error}");
                            has_error = true;
                        }
                    }
                    c if c.is_alphabetic() | (c == '_') => {
                        let mut s = String::from(c);

                        #[allow(clippy::almost_complete_range)]
                        while peekable_iter.peek().is_some_and(
                            |(_, l)| matches!(*l, '_' | 'a'..='z'|'A'..='Z' | '0'..'9'),
                        ) {
                            let (_, ch) = peekable_iter.next().unwrap();
                            s.push(ch);
                        }

                        if let Some(keyword) = keyword_token(&s) {
                            tokens.push(Token::new(keyword, (line, i, s.len()).into()))
                        } else {
                            let len = s.len();
                            tokens.push(Token::new(
                                Lexeme::Identifier(s.clone()),
                                (line, i, len).into(),
                            ));
                        }
                    }
                    '/' => {
                        if peekable_iter.peek().is_some_and(|(_, l)| *l == '/') {
                            // line comment so each chars till EOL chars
                            while peekable_iter
                                .peek()
                                .is_some_and(|(_i, c)| !one_of(*c, "\n\r"))
                            {
                                peekable_iter.next();
                            }
                        } else {
                            tokens.push(Token::new(lexeme_from("/"), (line, i, 1).into()))
                        }
                    }
                    _ => {
                        let error = TokenError::InvalidToken {
                            src: self.source.to_string(),
                            span: Span::new(line, i, 1),
                        };
                        eprintln!("{error}");
                        has_error = true;
                    }
                },
                None => {
                    break;
                }
            }
        }

        tokens.push(Token::new(
            lexeme_from("eof"),
            Span::new(line, self.source.len(), 0),
        ));

        if self.print_tokens {
            tokens.iter().for_each(|t| println!("{t}"));
        }

        if has_error { Err(65) } else { Ok(tokens) }
    }
}

pub fn lexeme_from(s: &str) -> Lexeme {
    s.into()
}

fn keyword_token(s: &str) -> Option<Lexeme> {
    match s {
        "true" | "false" | "nil" | "and" | "or" | "class" | "for" | "fun" | "if" | "else"
        | "return" | "super" | "this" | "var" | "while" | "print" => Some(lexeme_from(s)),
        _ => None,
    }
}

fn one_of(c: char, chars: &str) -> bool {
    chars.chars().any(|possible| possible == c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_new() {
        let span = Span::new(1, 0, 1);
        let lexeme = Lexeme::LeftParen;
        let token = Token::new(lexeme.clone(), span.clone());

        assert_eq!(token.lexeme, lexeme);
        assert_eq!(token.span, span);
    }

    #[test]
    fn test_token_display_string() {
        let token = Token::new(Lexeme::String("hello".to_string()), Span::new(1, 0, 7));
        assert_eq!(format!("{token}"), "STRING \"hello\" hello");
    }

    #[test]
    fn test_token_display_number_integer() {
        let token = Token::new(Lexeme::Number("42".to_string(), 42.0), Span::new(1, 0, 2));
        assert_eq!(format!("{token}"), "NUMBER 42 42.0");
    }

    #[test]
    fn test_token_display_number_float() {
        let token = Token::new(Lexeme::Number("1.23".to_string(), 1.23), Span::new(1, 0, 4));
        assert_eq!(format!("{token}"), "NUMBER 1.23 1.23");
    }

    #[test]
    fn test_token_display_identifier() {
        let token = Token::new(
            Lexeme::Identifier("variable".to_string()),
            Span::new(1, 0, 8),
        );
        assert_eq!(format!("{token}"), "IDENTIFIER variable null");
    }

    #[test]
    fn test_token_display_single_char() {
        let token = Token::new(Lexeme::LeftParen, Span::new(1, 0, 1));
        assert_eq!(format!("{token}"), "LEFT_PAREN ( null");
    }

    #[test]
    fn test_token_display_keyword() {
        let token = Token::new(Lexeme::True, Span::new(1, 0, 4));
        assert_eq!(format!("{token}"), "TRUE true null");
    }

    #[test]
    fn test_token_display_eof() {
        let token = Token::new(Lexeme::Eof, Span::new(1, 0, 0));
        assert_eq!(format!("{token}"), "EOF  null");
    }

    #[test]
    fn test_lexeme_from_single_chars() {
        assert_eq!(Lexeme::from("("), Lexeme::LeftParen);
        assert_eq!(Lexeme::from(")"), Lexeme::RightParen);
        assert_eq!(Lexeme::from("{"), Lexeme::LeftBrace);
        assert_eq!(Lexeme::from("}"), Lexeme::RightBrace);
        assert_eq!(Lexeme::from(","), Lexeme::Comma);
        assert_eq!(Lexeme::from("."), Lexeme::Dot);
        assert_eq!(Lexeme::from(";"), Lexeme::SemiColon);
        assert_eq!(Lexeme::from("+"), Lexeme::Plus);
        assert_eq!(Lexeme::from("-"), Lexeme::Minus);
        assert_eq!(Lexeme::from("*"), Lexeme::Star);
        assert_eq!(Lexeme::from("/"), Lexeme::Slash);
        assert_eq!(Lexeme::from("="), Lexeme::Eq);
        assert_eq!(Lexeme::from("!"), Lexeme::Bang);
        assert_eq!(Lexeme::from("<"), Lexeme::Less);
        assert_eq!(Lexeme::from(">"), Lexeme::Greater);
    }

    #[test]
    fn test_lexeme_from_double_chars() {
        assert_eq!(Lexeme::from("=="), Lexeme::EqEq);
        assert_eq!(Lexeme::from("!="), Lexeme::BangEq);
        assert_eq!(Lexeme::from("<="), Lexeme::LessEq);
        assert_eq!(Lexeme::from(">="), Lexeme::GreaterEq);
    }

    #[test]
    fn test_lexeme_from_keywords() {
        assert_eq!(Lexeme::from("print"), Lexeme::Print);
        assert_eq!(Lexeme::from("true"), Lexeme::True);
        assert_eq!(Lexeme::from("false"), Lexeme::False);
        assert_eq!(Lexeme::from("nil"), Lexeme::Nil);
        assert_eq!(Lexeme::from("and"), Lexeme::And);
        assert_eq!(Lexeme::from("or"), Lexeme::Or);
        assert_eq!(Lexeme::from("fun"), Lexeme::Fun);
        assert_eq!(Lexeme::from("return"), Lexeme::Return);
        assert_eq!(Lexeme::from("if"), Lexeme::If);
        assert_eq!(Lexeme::from("else"), Lexeme::Else);
        assert_eq!(Lexeme::from("for"), Lexeme::For);
        assert_eq!(Lexeme::from("while"), Lexeme::While);
        assert_eq!(Lexeme::from("class"), Lexeme::Class);
        assert_eq!(Lexeme::from("super"), Lexeme::Super);
        assert_eq!(Lexeme::from("this"), Lexeme::This);
        assert_eq!(Lexeme::from("var"), Lexeme::Var);
        assert_eq!(Lexeme::from("eof"), Lexeme::Eof);
    }

    #[test]
    #[should_panic(expected = "invalid token invalid")]
    fn test_lexeme_from_invalid() {
        let _ = Lexeme::from("invalid");
    }

    #[test]
    fn test_lexeme_display() {
        assert_eq!(
            format!("{}", Lexeme::Number("42".to_string(), 42.0)),
            "NUMBER"
        );
        assert_eq!(
            format!("{}", Lexeme::Identifier("var".to_string())),
            "IDENTIFIER"
        );
        assert_eq!(format!("{}", Lexeme::String("hello".to_string())), "STRING");
        assert_eq!(format!("{}", Lexeme::LeftParen), "LEFT_PAREN");
        assert_eq!(format!("{}", Lexeme::True), "TRUE");
        assert_eq!(format!("{}", Lexeme::Eof), "EOF");
    }

    #[test]
    fn test_scanner_new() {
        let source = "test source";
        let scanner = Scanner::new(source, true);
        assert_eq!(scanner.source, source);
    }

    #[test]
    fn test_scanner_empty_source() {
        let scanner = Scanner::new("", true);
        let tokens = scanner.scan().unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_single_tokens() {
        let scanner = Scanner::new("(){},.+-;*", true);
        let tokens = scanner.scan().unwrap();

        let expected_lexemes = vec![
            Lexeme::LeftParen,
            Lexeme::RightParen,
            Lexeme::LeftBrace,
            Lexeme::RightBrace,
            Lexeme::Comma,
            Lexeme::Dot,
            Lexeme::Plus,
            Lexeme::Minus,
            Lexeme::SemiColon,
            Lexeme::Star,
            Lexeme::Eof,
        ];

        assert_eq!(tokens.len(), expected_lexemes.len());
        for (token, expected) in tokens.iter().zip(expected_lexemes.iter()) {
            assert_eq!(&token.lexeme, expected);
        }
    }

    #[test]
    fn test_scanner_comparison_operators() {
        let scanner = Scanner::new("== ! != < <= > >=", true);
        let tokens = scanner.scan().unwrap();

        let expected_lexemes = vec![
            Lexeme::EqEq,
            Lexeme::Bang,
            Lexeme::BangEq,
            Lexeme::Less,
            Lexeme::LessEq,
            Lexeme::Greater,
            Lexeme::GreaterEq,
            Lexeme::Eof,
        ];

        assert_eq!(tokens.len(), expected_lexemes.len());
        for (token, expected) in tokens.iter().zip(expected_lexemes.iter()) {
            assert_eq!(&token.lexeme, expected);
        }
    }

    #[test]
    fn test_scanner_string_literal() {
        let scanner = Scanner::new("\"hello world\"", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::String("hello world".to_string()));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_unterminated_string() {
        let scanner = Scanner::new("\"unterminated", true);
        let result = scanner.scan();

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error, 65);
    }

    #[test]
    fn test_scanner_numbers() {
        let scanner = Scanner::new("123 1.23 42.0", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("123".to_string(), 123.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Number("1.23".to_string(), 1.23));
        assert_eq!(tokens[2].lexeme, Lexeme::Number("42.0".to_string(), 42.0));
        assert_eq!(tokens[3].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_identifiers() {
        let scanner = Scanner::new("variable _123private camelCase", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].lexeme, Lexeme::Identifier("variable".to_string()));
        assert_eq!(
            tokens[1].lexeme,
            Lexeme::Identifier("_123private".to_string())
        );
        assert_eq!(
            tokens[2].lexeme,
            Lexeme::Identifier("camelCase".to_string())
        );
        assert_eq!(tokens[3].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_keywords() {
        let scanner = Scanner::new("true false nil and or", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].lexeme, Lexeme::True);
        assert_eq!(tokens[1].lexeme, Lexeme::False);
        assert_eq!(tokens[2].lexeme, Lexeme::Nil);
        assert_eq!(tokens[3].lexeme, Lexeme::And);
        assert_eq!(tokens[4].lexeme, Lexeme::Or);
        assert_eq!(tokens[5].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_comments() {
        let scanner = Scanner::new("// this is a comment\n42", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("42".to_string(), 42.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_whitespace() {
        let scanner = Scanner::new("  \t\n  42  \r\n  ", true);
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("42".to_string(), 42.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof);
    }

    #[test]
    fn test_scanner_invalid_character() {
        let scanner = Scanner::new("@", true);
        let result = scanner.scan();

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error, 65);
    }

    #[test]
    fn test_scanner_complex_program() {
        let source = r#"
            fun fibonacci(n) {
                if (n <= 1) return n;
                return fibonacci(n - 1) + fibonacci(n - 2);
            }

            print fibonacci(10);
        "#;

        let scanner = Scanner::new(source, true);
        let tokens = scanner.scan().unwrap();

        // Should contain fun, identifier, (, identifier, ), {, if, (, etc.
        assert!(tokens.len() > 20);
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Fun)));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::If)));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Return)));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Print)));
    }

    #[test]
    fn test_lexeme_from_helper() {
        assert_eq!(lexeme_from("("), Lexeme::LeftParen);
        assert_eq!(lexeme_from("true"), Lexeme::True);
    }

    #[test]
    fn test_keyword_token_helper() {
        assert_eq!(
            keyword_token("true"),
            Some(Lexeme::True)
        );
        assert_eq!(
            keyword_token("false"),
            Some(Lexeme::False)
        );
        assert_eq!(keyword_token("identifier"), None);
    }

    #[test]
    fn test_one_of_helper() {
        assert!(one_of('a', "abc"));
        assert!(one_of('\n', "\n\r"));
        assert!(!one_of('x', "abc"));
        assert!(!one_of('a', ""));
    }
}
