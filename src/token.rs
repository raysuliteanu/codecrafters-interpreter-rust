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
            Lexeme::LeftParen(v)
            | Lexeme::RightParen(v)
            | Lexeme::LeftBrace(v)
            | Lexeme::RightBrace(v)
            | Lexeme::Dot(v)
            | Lexeme::Comma(v)
            | Lexeme::Minus(v)
            | Lexeme::Plus(v)
            | Lexeme::SemiColon(v)
            | Lexeme::Star(v)
            | Lexeme::Eq(v)
            | Lexeme::Bang(v)
            | Lexeme::Less(v)
            | Lexeme::Greater(v)
            | Lexeme::Slash(v) => write!(f, "{lexeme} {v} null"),
            Lexeme::EqEq(v)
            | Lexeme::BangEq(v)
            | Lexeme::LessEq(v)
            | Lexeme::GreaterEq(v)
            | Lexeme::True(v)
            | Lexeme::False(v)
            | Lexeme::Nil(v)
            | Lexeme::And(v)
            | Lexeme::Or(v)
            | Lexeme::Class(v)
            | Lexeme::For(v)
            | Lexeme::Fun(v)
            | Lexeme::If(v)
            | Lexeme::Else(v)
            | Lexeme::Return(v)
            | Lexeme::Super(v)
            | Lexeme::This(v)
            | Lexeme::Var(v)
            | Lexeme::While(v)
            | Lexeme::Print(v) => {
                let v = v.to_lowercase();
                write!(f, "{lexeme} {v} null")
            }
            Lexeme::Eof(_) => {
                write!(f, "{lexeme}  null")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Lexeme {
    // keywords
    True(String),
    False(String),
    Nil(String),
    And(String),
    Or(String),
    Class(String),
    For(String),
    Fun(String),
    If(String),
    Else(String),
    Return(String),
    Super(String),
    This(String),
    Var(String),
    While(String),
    Print(String),

    // literals
    LeftParen(char),
    RightParen(char),
    LeftBrace(char),
    RightBrace(char),
    Comma(char),
    Dot(char),
    Minus(char),
    Plus(char),
    SemiColon(char),
    Star(char),
    Eq(char),
    EqEq(String),
    Bang(char),
    BangEq(String),
    Less(char),
    LessEq(String),
    Greater(char),
    GreaterEq(String),
    Slash(char),

    // value holders
    Number(String, f64),
    Identifier(String),
    String(String),

    // symbolic placeholder
    Eof(String),
}

impl From<&str> for Lexeme {
    fn from(value: &str) -> Self {
        match value {
            "(" => Lexeme::LeftParen('('),
            ")" => Lexeme::RightParen(')'),
            "{" => Lexeme::LeftBrace('{'),
            "}" => Lexeme::RightBrace('}'),
            "," => Lexeme::Comma(','),
            "." => Lexeme::Dot('.'),
            ";" => Lexeme::SemiColon(';'),
            "+" => Lexeme::Plus('+'),
            "-" => Lexeme::Minus('-'),
            "*" => Lexeme::Star('*'),
            "/" => Lexeme::Slash('/'),
            "=" => Lexeme::Eq('='),
            "!" => Lexeme::Bang('!'),
            "<" => Lexeme::Less('<'),
            ">" => Lexeme::Greater('>'),
            "==" => Lexeme::EqEq("==".to_string()),
            "!=" => Lexeme::BangEq("!=".to_string()),
            "<=" => Lexeme::LessEq("<=".to_string()),
            ">=" => Lexeme::GreaterEq(">=".to_string()),
            "print" => Lexeme::Print("PRINT".to_string()),
            "true" => Lexeme::True("TRUE".to_string()),
            "false" => Lexeme::False("FALSE".to_string()),
            "nil" => Lexeme::Nil("NIL".to_string()),
            "and" => Lexeme::And("AND".to_string()),
            "or" => Lexeme::Or("OR".to_string()),
            "fun" => Lexeme::Fun("FUN".to_string()),
            "return" => Lexeme::Return("RETURN".to_string()),
            "if" => Lexeme::If("IF".to_string()),
            "else" => Lexeme::Else("ELSE".to_string()),
            "for" => Lexeme::For("FOR".to_string()),
            "while" => Lexeme::While("WHILE".to_string()),
            "class" => Lexeme::Class("CLASS".to_string()),
            "super" => Lexeme::Super("SUPER".to_string()),
            "this" => Lexeme::This("THIS".to_string()),
            "var" => Lexeme::Var("VAR".to_string()),
            "eof" => Lexeme::Eof("EOF".to_string()),
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
            Lexeme::LeftParen(_) => write!(f, "LEFT_PAREN"),
            Lexeme::RightParen(_) => write!(f, "RIGHT_PAREN"),
            Lexeme::LeftBrace(_) => write!(f, "LEFT_BRACE"),
            Lexeme::RightBrace(_) => write!(f, "RIGHT_BRACE"),
            Lexeme::Dot(_) => write!(f, "DOT"),
            Lexeme::Comma(_) => write!(f, "COMMA"),
            Lexeme::Minus(_) => write!(f, "MINUS"),
            Lexeme::Plus(_) => write!(f, "PLUS"),
            Lexeme::SemiColon(_) => write!(f, "SEMICOLON"),
            Lexeme::Star(_) => write!(f, "STAR"),
            Lexeme::Eq(_) => write!(f, "EQUAL"),
            Lexeme::EqEq(_) => write!(f, "EQUAL_EQUAL"),
            Lexeme::Bang(_) => write!(f, "BANG"),
            Lexeme::BangEq(_) => write!(f, "BANG_EQUAL"),
            Lexeme::Less(_) => write!(f, "LESS"),
            Lexeme::LessEq(_) => write!(f, "LESS_EQUAL"),
            Lexeme::Greater(_) => write!(f, "GREATER"),
            Lexeme::GreaterEq(_) => write!(f, "GREATER_EQUAL"),
            Lexeme::Slash(_) => write!(f, "SLASH"),
            Lexeme::True(v)
            | Lexeme::False(v)
            | Lexeme::Nil(v)
            | Lexeme::And(v)
            | Lexeme::Or(v)
            | Lexeme::Class(v)
            | Lexeme::For(v)
            | Lexeme::Fun(v)
            | Lexeme::If(v)
            | Lexeme::Else(v)
            | Lexeme::Return(v)
            | Lexeme::Super(v)
            | Lexeme::This(v)
            | Lexeme::Var(v)
            | Lexeme::While(v)
            | Lexeme::Print(v)
            | Lexeme::Eof(v) => write!(f, "{v}"),
        }
    }
}

pub struct Scanner<'scanner> {
    source: &'scanner str,
}

impl<'scanner> Scanner<'scanner> {
    pub fn new(source: &'scanner str) -> Self {
        Scanner { source }
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

        tokens.iter().for_each(|t| println!("{t}"));

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
        let lexeme = Lexeme::LeftParen('(');
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
        let token = Token::new(Lexeme::LeftParen('('), Span::new(1, 0, 1));
        assert_eq!(format!("{token}"), "LEFT_PAREN ( null");
    }

    #[test]
    fn test_token_display_keyword() {
        let token = Token::new(Lexeme::True("TRUE".to_string()), Span::new(1, 0, 4));
        assert_eq!(format!("{token}"), "TRUE true null");
    }

    #[test]
    fn test_token_display_eof() {
        let token = Token::new(Lexeme::Eof("EOF".to_string()), Span::new(1, 0, 0));
        assert_eq!(format!("{token}"), "EOF  null");
    }

    #[test]
    fn test_lexeme_from_single_chars() {
        assert_eq!(Lexeme::from("("), Lexeme::LeftParen('('));
        assert_eq!(Lexeme::from(")"), Lexeme::RightParen(')'));
        assert_eq!(Lexeme::from("{"), Lexeme::LeftBrace('{'));
        assert_eq!(Lexeme::from("}"), Lexeme::RightBrace('}'));
        assert_eq!(Lexeme::from(","), Lexeme::Comma(','));
        assert_eq!(Lexeme::from("."), Lexeme::Dot('.'));
        assert_eq!(Lexeme::from(";"), Lexeme::SemiColon(';'));
        assert_eq!(Lexeme::from("+"), Lexeme::Plus('+'));
        assert_eq!(Lexeme::from("-"), Lexeme::Minus('-'));
        assert_eq!(Lexeme::from("*"), Lexeme::Star('*'));
        assert_eq!(Lexeme::from("/"), Lexeme::Slash('/'));
        assert_eq!(Lexeme::from("="), Lexeme::Eq('='));
        assert_eq!(Lexeme::from("!"), Lexeme::Bang('!'));
        assert_eq!(Lexeme::from("<"), Lexeme::Less('<'));
        assert_eq!(Lexeme::from(">"), Lexeme::Greater('>'));
    }

    #[test]
    fn test_lexeme_from_double_chars() {
        assert_eq!(Lexeme::from("=="), Lexeme::EqEq("==".to_string()));
        assert_eq!(Lexeme::from("!="), Lexeme::BangEq("!=".to_string()));
        assert_eq!(Lexeme::from("<="), Lexeme::LessEq("<=".to_string()));
        assert_eq!(Lexeme::from(">="), Lexeme::GreaterEq(">=".to_string()));
    }

    #[test]
    fn test_lexeme_from_keywords() {
        assert_eq!(Lexeme::from("print"), Lexeme::Print("PRINT".to_string()));
        assert_eq!(Lexeme::from("true"), Lexeme::True("TRUE".to_string()));
        assert_eq!(Lexeme::from("false"), Lexeme::False("FALSE".to_string()));
        assert_eq!(Lexeme::from("nil"), Lexeme::Nil("NIL".to_string()));
        assert_eq!(Lexeme::from("and"), Lexeme::And("AND".to_string()));
        assert_eq!(Lexeme::from("or"), Lexeme::Or("OR".to_string()));
        assert_eq!(Lexeme::from("fun"), Lexeme::Fun("FUN".to_string()));
        assert_eq!(Lexeme::from("return"), Lexeme::Return("RETURN".to_string()));
        assert_eq!(Lexeme::from("if"), Lexeme::If("IF".to_string()));
        assert_eq!(Lexeme::from("else"), Lexeme::Else("ELSE".to_string()));
        assert_eq!(Lexeme::from("for"), Lexeme::For("FOR".to_string()));
        assert_eq!(Lexeme::from("while"), Lexeme::While("WHILE".to_string()));
        assert_eq!(Lexeme::from("class"), Lexeme::Class("CLASS".to_string()));
        assert_eq!(Lexeme::from("super"), Lexeme::Super("SUPER".to_string()));
        assert_eq!(Lexeme::from("this"), Lexeme::This("THIS".to_string()));
        assert_eq!(Lexeme::from("var"), Lexeme::Var("VAR".to_string()));
        assert_eq!(Lexeme::from("eof"), Lexeme::Eof("EOF".to_string()));
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
        assert_eq!(format!("{}", Lexeme::LeftParen('(')), "LEFT_PAREN");
        assert_eq!(format!("{}", Lexeme::True("TRUE".to_string())), "TRUE");
        assert_eq!(format!("{}", Lexeme::Eof("EOF".to_string())), "EOF");
    }

    #[test]
    fn test_scanner_new() {
        let source = "test source";
        let scanner = Scanner::new(source);
        assert_eq!(scanner.source, source);
    }

    #[test]
    fn test_scanner_empty_source() {
        let scanner = Scanner::new("");
        let tokens = scanner.scan().unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_single_tokens() {
        let scanner = Scanner::new("(){},.+-;*");
        let tokens = scanner.scan().unwrap();

        let expected_lexemes = vec![
            Lexeme::LeftParen('('),
            Lexeme::RightParen(')'),
            Lexeme::LeftBrace('{'),
            Lexeme::RightBrace('}'),
            Lexeme::Comma(','),
            Lexeme::Dot('.'),
            Lexeme::Plus('+'),
            Lexeme::Minus('-'),
            Lexeme::SemiColon(';'),
            Lexeme::Star('*'),
            Lexeme::Eof("EOF".to_string()),
        ];

        assert_eq!(tokens.len(), expected_lexemes.len());
        for (token, expected) in tokens.iter().zip(expected_lexemes.iter()) {
            assert_eq!(&token.lexeme, expected);
        }
    }

    #[test]
    fn test_scanner_comparison_operators() {
        let scanner = Scanner::new("= == ! != < <= > >=");
        let tokens = scanner.scan().unwrap();

        let expected_lexemes = vec![
            Lexeme::Eq('='),
            Lexeme::EqEq("==".to_string()),
            Lexeme::Bang('!'),
            Lexeme::BangEq("!=".to_string()),
            Lexeme::Less('<'),
            Lexeme::LessEq("<=".to_string()),
            Lexeme::Greater('>'),
            Lexeme::GreaterEq(">=".to_string()),
            Lexeme::Eof("EOF".to_string()),
        ];

        assert_eq!(tokens.len(), expected_lexemes.len());
        for (token, expected) in tokens.iter().zip(expected_lexemes.iter()) {
            assert_eq!(&token.lexeme, expected);
        }
    }

    #[test]
    fn test_scanner_string_literal() {
        let scanner = Scanner::new("\"hello world\"");
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::String("hello world".to_string()));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_unterminated_string() {
        let scanner = Scanner::new("\"unterminated");
        let result = scanner.scan();

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(error, 65);
    }

    #[test]
    fn test_scanner_numbers() {
        let scanner = Scanner::new("123 1.23 42.0");
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("123".to_string(), 123.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Number("1.23".to_string(), 1.23));
        assert_eq!(tokens[2].lexeme, Lexeme::Number("42.0".to_string(), 42.0));
        assert_eq!(tokens[3].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_identifiers() {
        let scanner = Scanner::new("variable _123private camelCase");
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
        assert_eq!(tokens[3].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_keywords() {
        let scanner = Scanner::new("true false nil and or");
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].lexeme, Lexeme::True("TRUE".to_string()));
        assert_eq!(tokens[1].lexeme, Lexeme::False("FALSE".to_string()));
        assert_eq!(tokens[2].lexeme, Lexeme::Nil("NIL".to_string()));
        assert_eq!(tokens[3].lexeme, Lexeme::And("AND".to_string()));
        assert_eq!(tokens[4].lexeme, Lexeme::Or("OR".to_string()));
        assert_eq!(tokens[5].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_comments() {
        let scanner = Scanner::new("// this is a comment\n42");
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("42".to_string(), 42.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_whitespace() {
        let scanner = Scanner::new("  \t\n  42  \r\n  ");
        let tokens = scanner.scan().unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].lexeme, Lexeme::Number("42".to_string(), 42.0));
        assert_eq!(tokens[1].lexeme, Lexeme::Eof("EOF".to_string()));
    }

    #[test]
    fn test_scanner_invalid_character() {
        let scanner = Scanner::new("@");
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

        let scanner = Scanner::new(source);
        let tokens = scanner.scan().unwrap();

        // Should contain fun, identifier, (, identifier, ), {, if, (, etc.
        assert!(tokens.len() > 20);
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Fun(_))));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::If(_))));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Return(_))));
        assert!(tokens.iter().any(|t| matches!(t.lexeme, Lexeme::Print(_))));
    }

    #[test]
    fn test_lexeme_from_helper() {
        assert_eq!(lexeme_from("("), Lexeme::LeftParen('('));
        assert_eq!(lexeme_from("true"), Lexeme::True("TRUE".to_string()));
    }

    #[test]
    fn test_keyword_token_helper() {
        assert_eq!(
            keyword_token("true"),
            Some(Lexeme::True("TRUE".to_string()))
        );
        assert_eq!(
            keyword_token("false"),
            Some(Lexeme::False("FALSE".to_string()))
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
