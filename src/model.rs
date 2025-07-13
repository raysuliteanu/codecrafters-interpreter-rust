use std::fmt::Display;

use crate::token::{Lexeme, Token};

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum Ast {
    Class,
    Function,
    // name, initializer
    Variable(Token, Option<Box<AstExpr>>),
    Block(Vec<AstStmt>),
    Statement(AstStmt),
    Expression(AstExpr),
}

#[derive(Debug, PartialEq)]
pub enum AstStmt {
    // condition, then, else
    If(Box<AstExpr>, Box<Ast>, Option<Box<Ast>>),
    // cond, body
    While(Box<AstExpr>, Box<Ast>),
    For,
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
                    write!(f, "{s}")?;
                }
                writeln!(f, "}}")
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
        | Lexeme::EqEq(v)
        | Lexeme::BangEq(v)
        | Lexeme::LessEq(v)
        | Lexeme::GreaterEq(v)
        | Lexeme::Identifier(v)
        | Lexeme::String(v) => v.to_lowercase(),
        Lexeme::LeftParen(v)
        | Lexeme::RightParen(v)
        | Lexeme::LeftBrace(v)
        | Lexeme::RightBrace(v)
        | Lexeme::Comma(v)
        | Lexeme::Dot(v)
        | Lexeme::Minus(v)
        | Lexeme::Plus(v)
        | Lexeme::SemiColon(v)
        | Lexeme::Star(v)
        | Lexeme::Eq(v)
        | Lexeme::Bang(v)
        | Lexeme::Less(v)
        | Lexeme::Greater(v)
        | Lexeme::Slash(v) => format!("{v}"),
        Lexeme::Number(_, v) => {
            if *v == v.trunc() {
                format!("{v}.0")
            } else {
                format!("{v}")
            }
        }
        Lexeme::Eof(_) => unreachable!(),
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
            AstStmt::For => todo!(),
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
