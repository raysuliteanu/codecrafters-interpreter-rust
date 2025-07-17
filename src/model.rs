use std::fmt::Display;

use crate::token::{Lexeme, Token};

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum Ast {
    Class,
    Function,
    // name, initializer
    Variable(Token, Option<Box<AstExpr>>),
    Block(Vec<Ast>),
    Statement(AstStmt),
    Expression(AstExpr),
}

#[derive(Debug, PartialEq)]
pub enum AstStmt {
    // condition, then, else
    If(Box<AstExpr>, Box<Ast>, Option<Box<Ast>>),
    // cond, body
    While(Box<AstExpr>, Box<Ast>),
    Return(Option<Box<AstExpr>>),
    Print(AstExpr),
    Expression(AstExpr),
}

#[derive(Debug, PartialEq)]
pub enum AstExpr {
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
            Ast::Function => todo!(),
            Ast::Variable(ident, expr) => {
                let name = match &ident.lexeme {
                    Lexeme::Identifier(name) => name,
                    _ => panic!("Variable declaration must have an identifier token"),
                };
                write!(f, "var {name}")?;
                if let Some(ast) = expr {
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
            AstStmt::If(cond, then_stmt, else_stmt) => {
                write!(f, "if {cond} {then_stmt}")?;
                if let Some(else_stmt) = else_stmt {
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
