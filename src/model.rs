use std::fmt::Display;

use crate::span::Span;

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
            Lexeme::EqEq | Lexeme::BangEq | Lexeme::LessEq | Lexeme::GreaterEq => {
                write!(f, "{lexeme} {} null", lexeme.lexeme_str())
            }
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
            Lexeme::Number(_, _) | Lexeme::Identifier(_) | Lexeme::String(_) => {
                panic!("{self} doesn't have a constant repr")
            }
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

#[derive(Debug, PartialEq, Clone)]
pub enum Ast {
    Class,
    Function {
        name: String,
        params: Vec<String>,
        body: Box<Ast>,
    },
    Variable {
        name: Token,
        initializer: Option<Box<AstExpr>>,
    },
    Block(Vec<Ast>),
    Statement(AstStmt),
    Expression(AstExpr),
}

#[derive(Debug, PartialEq, Clone)]
pub enum AstStmt {
    // condition, then, else
    If {
        condition: Box<AstExpr>,
        then: Box<Ast>,
        or_else: Option<Box<Ast>>,
    },
    // cond, body
    While(Box<AstExpr>, Box<Ast>),
    Return(Option<Box<AstExpr>>),
    Print(AstExpr),
    Expression(AstExpr),
}

#[derive(Debug, PartialEq, Clone)]
pub enum AstExpr {
    Call {
        func: String,
        args: Vec<AstExpr>,
        site: Span,
    },
    Assignment {
        id: String,
        expr: Box<AstExpr>,
    },
    // expr AND/OR expr
    Logical {
        op: Token,
        left: Box<AstExpr>,
        right: Box<AstExpr>,
    },
    Terminal(Token),
    Group(Box<AstExpr>),
    Unary {
        op: Token,
        exp: Box<AstExpr>,
    },
    Binary {
        op: Token,
        left: Box<AstExpr>,
        right: Box<AstExpr>,
    },
}

impl Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ast::Class => todo!(),
            Ast::Function { name, params, body } => {
                writeln!(f, "fun {name} ({}) {{", params.join(", "))?;
                writeln!(f, "{body} }}")
            }
            Ast::Variable { name, initializer } => {
                let name = match &name.lexeme {
                    Lexeme::Identifier(name) => name,
                    _ => panic!("Variable declaration must have an identifier token"),
                };
                write!(f, "var {name}")?;
                if let Some(ast) = initializer {
                    write!(f, " = {ast}")?;
                }
                write!(f, ";")
            }
            Ast::Block(block) => {
                writeln!(f, "{{")?;
                for s in block {
                    writeln!(f, "{s}")?;
                }
                write!(f, "}}")
            }
            Ast::Statement(s) => write!(f, "{s}"),
            Ast::Expression(e) => write!(f, "{e}"),
        }
    }
}

impl Display for AstExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstExpr::Call {
                func: id,
                args,
                site: _,
            } => {
                let args_str = args
                    .iter()
                    .map(|arg| arg.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{id}([{args_str}])")
            }
            AstExpr::Assignment { id, expr } => write!(f, "{id} = {expr}"),
            AstExpr::Logical { op, left, right } => {
                write!(f, "({} {left} {right})", print_ast_token(op))
            }
            AstExpr::Unary { op, exp } => write!(f, "({} {exp})", print_ast_token(op)),
            AstExpr::Binary { op, left, right } => {
                write!(f, "({} {left} {right})", print_ast_token(op))
            }
            AstExpr::Group(ast) => write!(f, "(group {ast})"),
            AstExpr::Terminal(token) => write!(f, "{}", print_ast_token(token)),
        }
    }
}

fn print_ast_token(token: &Token) -> String {
    match &token.lexeme {
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
        | Lexeme::Print
        | Lexeme::EqEq
        | Lexeme::BangEq
        | Lexeme::LessEq
        | Lexeme::GreaterEq => token.lexeme.lexeme_str().to_lowercase(),
        Lexeme::LeftParen
        | Lexeme::RightParen
        | Lexeme::LeftBrace
        | Lexeme::RightBrace
        | Lexeme::Comma
        | Lexeme::Dot
        | Lexeme::Minus
        | Lexeme::Plus
        | Lexeme::SemiColon
        | Lexeme::Star
        | Lexeme::Eq
        | Lexeme::Bang
        | Lexeme::Less
        | Lexeme::Greater
        | Lexeme::Slash => token.lexeme.lexeme_str().to_string(),
        Lexeme::Identifier(v) | Lexeme::String(v) => v.to_lowercase(),
        Lexeme::Number(_, v) => {
            if *v == v.trunc() {
                format!("{v}.0")
            } else {
                format!("{v}")
            }
        }
        Lexeme::Eof => unreachable!(),
    }
}

impl Display for AstStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstStmt::If {
                condition,
                then,
                or_else,
            } => {
                write!(f, "if {condition} {then}")?;
                if let Some(else_stmt) = or_else {
                    write!(f, " else {else_stmt}")?;
                }
                Ok(())
            }
            AstStmt::While(cond, body) => {
                write!(f, "while {cond} {body}")
            }
            AstStmt::Return(ast) => {
                write!(f, "return")?;
                if let Some(ast) = ast {
                    write!(f, " {ast}")?;
                }
                write!(f, ";")
            }
            AstStmt::Print(ast) => write!(f, "print {ast};"),
            AstStmt::Expression(e) => write!(f, "{e}"),
        }
    }
}
