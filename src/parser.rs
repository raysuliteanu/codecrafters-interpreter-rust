use std::{fmt::Display, iter::Peekable};

use log::trace;
use thiserror::Error;

use crate::token::{Lexeme, Scanner, Token, lexeme_from};

type PeekableTokenIter<'a> = Peekable<std::slice::Iter<'a, Token>>;
pub type ParseResult<T> = Result<T, u8>;

#[derive(Debug, PartialEq)]
pub struct Ast {
    ty: AstType,
}

impl Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ty)
    }
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum AstType {
    Class,
    Function,
    // name, initializer
    Variable(Token, Option<Box<Ast>>),
    ExprStatement,
    ForStatement,
    // condition, then, else
    IfStatement(Box<Ast>, Box<Ast>, Option<Box<Ast>>),
    PrintStatement(Box<Ast>),
    ReturnStatement(Option<Box<Ast>>),
    // cond, body
    WhileStatement(Box<Ast>, Box<Ast>),
    Block,
    Group(Box<Ast>),
    Expression(ExpressionType),
    Terminal(Token),
}

impl Display for AstType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstType::Class => todo!(),
            AstType::Function => todo!(),
            AstType::Variable(ident, expr) => {
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
            AstType::ExprStatement => todo!(),
            AstType::ForStatement => todo!(),
            AstType::IfStatement(cond, then_stmt, else_stmt) => {
                writeln!(f, "if {cond} {{ {then_stmt} }}")?;
                if let Some(else_stmt) = else_stmt {
                    write!(f, "else {{ {else_stmt} }}")?;
                }
                Ok(())
            }
            AstType::PrintStatement(ast) => {
                write!(f, "print {ast};")
            }
            AstType::ReturnStatement(ast) => {
                write!(f, "return")?;
                if let Some(ast) = ast {
                    write!(f, " {ast}")?;
                }
                write!(f, ";")
            }
            AstType::WhileStatement(cond, body) => {
                writeln!(f, "while {cond} {{")?;
                writeln!(f, "{body} }}")
            }
            AstType::Block => todo!(),
            AstType::Group(ast) => {
                write!(f, "(group {ast})")
            }
            AstType::Expression(e) => match e {
                ExpressionType::Unary { op, exp } => {
                    write!(f, "({op} {exp})")
                }
                ExpressionType::Binary { op, left, right } => {
                    write!(f, "({op} {left} {right})")
                }
            },
            AstType::Terminal(t) => match &t.lexeme {
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
                | Lexeme::String(v) => write!(f, "{v}"),
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
                | Lexeme::Slash(v) => write!(f, "{v}"),
                Lexeme::Number(_, v) => write!(f, "{v}"),
                Lexeme::Eof(_) => unreachable!(),
            },
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ExpressionType {
    Unary {
        op: Box<Ast>,
        exp: Box<Ast>,
    },
    Binary {
        op: Box<Ast>,
        left: Box<Ast>,
        right: Box<Ast>,
    },
}

impl Display for ExpressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpressionType::Unary { op, exp } => {
                write!(f, "{op}{exp}")
            }
            ExpressionType::Binary { op, left, right } => {
                write!(f, "{left} {op} {right}")
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("missing token '{}' got '{}'", expected, actual)]
    MissingToken { expected: Lexeme, actual: Lexeme },

    #[error("unexpected token '{actual}'")]
    UnexpectedToken { actual: Token },

    #[error("Unexpected EOF")]
    UnexpectedEof,
}

macro_rules! ast_missing_token {
    ($e: expr, $a: expr) => {
        ParseError::MissingToken {
            expected: $e,
            actual: $a,
        }
        .into()
    };
}

/// macro for handling error case when next token was not what was expected.
/// it could be either because the next token was some other "real" token or
/// it could because the next "token" was actually no more tokens i.e. "eof"
macro_rules! ast_expected_token {
    ($t: expr, $e: expr) => {
        Err(ParseError::MissingToken {
            expected: $e,
            actual: $t.lexeme.clone(),
        }
        .into())
    };
}

/// convenience macro for creating a new binary expression AST node
macro_rules! ast_binary {
    ($op: expr, $l: expr, $r: expr) => {
        Ast {
            ty: AstType::Expression(ExpressionType::Binary {
                op: Box::new($op),
                left: Box::new($l),
                right: Box::new($r),
            }),
        }
    };
}

/// convenience macro for creating a new expression group AST node
macro_rules! ast_group {
    ($e: expr) => {
        Ast {
            ty: AstType::Group(Box::new($e)),
        }
    };
}

/// convenience macro for creating a new unary expression AST node
macro_rules! ast_unary {
    ($op: expr, $exp: expr) => {
        Ast {
            ty: AstType::Expression(ExpressionType::Unary {
                op: Box::new(Ast {
                    ty: AstType::Terminal($op.clone()),
                }),
                exp: Box::new($exp),
            }),
        }
    };
}

/// convenience macro for creating a new terminal AST node
macro_rules! ast_terminal {
    ($t: expr) => {
        Ast {
            ty: AstType::Terminal($t.clone()),
        }
    };
}

pub struct Parser<'parser> {
    source: &'parser str,
}

impl<'parser> Parser<'parser> {
    pub fn new(source: &'parser str) -> Parser<'parser> {
        Parser { source }
    }

    pub fn parse(&mut self) -> ParseResult<Vec<Ast>> {
        let scanner = Scanner::new(self.source);
        if let Ok(tokens) = scanner.scan() {
            if tokens.is_empty() {
                trace!("no tokens scanned");
                return Ok(vec![]);
            }
            trace!("parsing {} tokens", tokens.len());
            match Parser::program(&mut tokens.iter().peekable()) {
                Ok(v) => {
                    v.iter().for_each(|node| println!("{node}"));
                    Ok(v)
                }
                Err(e) => {
                    eprintln!("{e}");
                    Err(65)
                }
            }
        } else {
            Err(65)
        }
    }

    fn program(tokens: &mut PeekableTokenIter) -> anyhow::Result<Vec<Ast>> {
        let mut ast: Vec<Ast> = Vec::new();

        while let Some(token) = tokens.peek()
            && token.lexeme != lexeme_from("eof")
        {
            let statement = Parser::declaration(tokens)?;
            ast.push(statement);
        }

        trace!("parsed AST: {ast:?}");

        Ok(ast)
    }

    fn declaration(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        if let Some(token) = tokens.peek() {
            match token.lexeme {
                Lexeme::Class(_) => Parser::class_decl(tokens),
                Lexeme::Fun(_) => Parser::fun_decl(tokens),
                Lexeme::Var(_) => Parser::var_decl(tokens),
                _ => Parser::statement(tokens),
            }
        } else {
            Err(ParseError::UnexpectedEof.into())
        }
    }

    fn class_decl(_tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        todo!("class decl")
    }

    fn fun_decl(_tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        todo!("fun decl")
    }

    // varDecl → "var" IDENTIFIER ( "=" expression )? ";" ;
    fn var_decl(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("var_decl: {:?}", tokens.peek());

        // eat 'var' token
        let var_token = tokens.next().unwrap();
        assert_eq!(var_token.lexeme, lexeme_from("var"));

        match tokens.peek() {
            Some(t) => match &t.lexeme {
                Lexeme::Identifier(_) => {
                    trace!("found identifier: {t}");
                    let identifier_token = tokens.next().unwrap().clone();
                    let expr = if let Some(_e) = tokens.next_if(|t| t.lexeme == lexeme_from("=")) {
                        let expr = Parser::expression(tokens)?;
                        Some(Box::new(expr))
                    } else {
                        None
                    };

                    if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
                        Ok(Ast {
                            ty: AstType::Variable(identifier_token, expr),
                        })
                    } else if tokens.peek().is_some() {
                        ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
                    } else {
                        Err(ParseError::UnexpectedEof.into())
                    }
                }
                _ => ast_expected_token!(t, lexeme_from("identifier")),
            },
            _ => todo!("unexpected eof"),
        }
    }

    fn statement(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("statement: {:?}", tokens.peek());
        let r = match tokens.peek() {
            Some(token) => match token.lexeme {
                // left brace token indicates block start
                crate::token::Lexeme::LeftBrace(_) => Parser::parse_block(tokens),
                crate::token::Lexeme::For(_) => Parser::for_stmt(tokens),
                crate::token::Lexeme::If(_) => Parser::if_stmt(tokens),
                crate::token::Lexeme::Print(_) => Parser::print_stmt(tokens),
                crate::token::Lexeme::Return(_) => Parser::return_stmt(tokens),
                crate::token::Lexeme::While(_) => Parser::while_stmt(tokens),
                _ => Parser::expression_statement(tokens),
            },
            None => Err(ParseError::UnexpectedEof.into()),
        }?;
        Ok(r)
    }

    fn parse_block(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("block_stmt: {:?}", tokens.peek());
        todo!("parse block")
    }

    fn for_stmt(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("for_stmt: {:?}", tokens.peek());
        todo!("parse for stmt")
    }

    fn if_stmt(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("if_stmt: {:?}", tokens.peek());
        let if_token = tokens.next().unwrap();
        assert_eq!(if_token.lexeme, lexeme_from("if"));

        let cond = Parser::expression(tokens)?;
        let then_stmt = Parser::statement(tokens)?;
        let else_stmt = if tokens
            .next_if(|t| t.lexeme == lexeme_from("else"))
            .is_some()
        {
            Some(Box::new(Parser::statement(tokens)?))
        } else {
            None
        };

        Ok(Ast {
            ty: AstType::IfStatement(Box::new(cond), Box::new(then_stmt), else_stmt),
        })
    }

    fn while_stmt(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("while_stmt: {:?}", tokens.peek());
        let while_token = tokens.next().unwrap();
        assert_eq!(while_token.lexeme, lexeme_from("while"));

        let cond = Parser::expression(tokens)?;
        let body = Parser::statement(tokens)?;

        Ok(Ast {
            ty: AstType::WhileStatement(Box::new(cond), Box::new(body)),
        })
    }

    fn print_stmt(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("print_stmt: {:?}", tokens.peek());

        let print_token = tokens.next().unwrap();
        assert_eq!(print_token.lexeme, lexeme_from("print"));

        let expr = Parser::expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            Ok(Ast {
                ty: AstType::PrintStatement(Box::new(expr)),
            })
        } else if tokens.peek().is_some() {
            ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
        } else {
            Err(ParseError::UnexpectedEof.into())
        }
    }

    fn return_stmt(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("return_stmt: {:?}", tokens.peek());

        // Consume the 'return' token first
        let return_token = tokens.next().unwrap();

        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            // a "naked" 'return' without expression i.e. "return;"
            Ok(Ast {
                ty: AstType::ReturnStatement(None),
            })
        } else if tokens.peek().is_some_and(|t| {
            !matches!(
                t.lexeme,
                Lexeme::LeftBrace(_)
                    | Lexeme::For(_)
                    | Lexeme::While(_)
                    | Lexeme::If(_)
                    | Lexeme::Print(_)
                    | Lexeme::Return(_)
            )
        }) {
            let ast = Parser::expression(tokens)?;
            if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
                Ok(Ast {
                    ty: AstType::ReturnStatement(Some(Box::new(ast))),
                })
            } else if tokens.peek().is_some() {
                ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
            } else {
                Err(ast_missing_token!(lexeme_from(";"), lexeme_from("eof")))
            }
        } else {
            ast_expected_token!(return_token, lexeme_from(";"))
        }
    }

    fn expression_statement(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        let token = tokens.peek();
        trace!("expr_stmt: {token:?}");
        let ast = Parser::expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            Ok(ast)
        } else if tokens.peek().is_some() {
            ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
        } else {
            Err(ast_missing_token!(lexeme_from(";"), lexeme_from("eof")))
        }
    }

    fn expression(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("expr: {:?}", tokens.peek());
        Parser::equality(tokens)
    }

    fn equality(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("equality: {:?}", tokens.peek());
        let mut left = Parser::comparison(tokens)?;

        while let Some(t) =
            tokens.next_if(|t| matches!(t.lexeme, Lexeme::BangEq(_) | Lexeme::EqEq(_)))
        {
            let right = Parser::comparison(tokens)?;
            let op = ast_terminal!(t);
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn comparison(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("comparison: {:?}", tokens.peek());
        let mut left = Parser::term(tokens)?;

        while let Some(t) = tokens.next_if(|t| {
            matches!(
                t.lexeme,
                Lexeme::Greater(_) | Lexeme::GreaterEq(_) | Lexeme::Less(_) | Lexeme::LessEq(_)
            )
        }) {
            let right = Parser::term(tokens)?;
            let op = ast_terminal!(t);
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn term(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("term: {:?}", tokens.peek());
        let mut left = Parser::factor(tokens)?;

        while let Some(t) =
            tokens.next_if(|t| matches!(t.lexeme, Lexeme::Plus(_) | Lexeme::Minus(_)))
        {
            let right = Parser::factor(tokens)?;
            let op = ast_terminal!(t);
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn factor(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("factor: {:?}", tokens.peek());
        let mut left = Parser::unary(tokens)?;

        while let Some(t) =
            tokens.next_if(|t| matches!(t.lexeme, Lexeme::Star(_) | Lexeme::Slash(_)))
        {
            let right = Parser::unary(tokens)?;
            let op = ast_terminal!(t);
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn unary(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("unary: {:?}", tokens.peek());
        if let Some(t) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Minus(_) | Lexeme::Bang(_)))
        {
            let right = Parser::unary(tokens)?;
            Ok(ast_unary!(t, right))
        } else {
            Parser::primary(tokens)
        }
    }

    fn primary(tokens: &mut PeekableTokenIter) -> anyhow::Result<Ast> {
        trace!("primary: {:?}", tokens.peek());
        if let Some(token) = tokens.next_if(|t| {
            matches!(
                t.lexeme,
                Lexeme::True(_) | Lexeme::False(_) | Lexeme::Nil(_)
            )
        }) {
            Ok(ast_terminal!(token))
        } else if let Some(_left_paren) = tokens.next_if(|t| t.lexeme == lexeme_from("(")) {
            let expr = Parser::expression(tokens)?;
            if let Some(_right_paren) = tokens.next_if(|t| t.lexeme == lexeme_from(")")) {
                Ok(ast_group!(expr))
            } else if tokens.peek().is_some() {
                // something other than a closing ')'
                Err(ParseError::MissingToken {
                    expected: lexeme_from(")"),
                    actual: tokens.next().unwrap().lexeme.clone(),
                }
                .into())
            } else {
                Err(ParseError::UnexpectedEof.into())
            }
        } else if let Some(token) = tokens.next_if(|t| {
            matches!(
                t.lexeme,
                Lexeme::Number { .. } | Lexeme::String { .. } | Lexeme::Identifier { .. }
            )
        }) {
            Ok(ast_terminal!(token.clone()))
        } else {
            Err(ParseError::UnexpectedToken {
                actual: tokens.next().unwrap().clone(),
            }
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;

    fn create_token(lexeme: Lexeme) -> Token {
        Token::new(lexeme, Span::new(1, 0, 1))
    }

    #[test]
    fn test_ast_display_print_statement() {
        let expr = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("true"))),
        };
        let print_stmt = Ast {
            ty: AstType::PrintStatement(Box::new(expr)),
        };
        assert_eq!(print_stmt.to_string(), "print TRUE;");
    }

    #[test]
    fn test_ast_display_return_statement_with_value() {
        let expr = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("42".to_string(), 42.0))),
        };
        let return_stmt = Ast {
            ty: AstType::ReturnStatement(Some(Box::new(expr))),
        };
        assert_eq!(return_stmt.to_string(), "return 42;");
    }

    #[test]
    fn test_ast_display_return_statement_without_value() {
        let return_stmt = Ast {
            ty: AstType::ReturnStatement(None),
        };
        assert_eq!(return_stmt.to_string(), "return;");
    }

    #[test]
    fn test_ast_display_group() {
        let expr = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("true"))),
        };
        let group = Ast {
            ty: AstType::Group(Box::new(expr)),
        };
        assert_eq!(group.to_string(), "(group TRUE)");
    }

    #[test]
    fn test_ast_display_binary_expression() {
        let left = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("1".to_string(), 1.0))),
        };
        let right = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("2".to_string(), 2.0))),
        };
        let op = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("+"))),
        };
        let binary = Ast {
            ty: AstType::Expression(ExpressionType::Binary {
                op: Box::new(op),
                left: Box::new(left),
                right: Box::new(right),
            }),
        };
        assert_eq!(binary.to_string(), "(+ 1 2)");
    }

    #[test]
    fn test_ast_display_unary_expression() {
        let expr = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("5".to_string(), 5.0))),
        };
        let op = create_token(lexeme_from("-"));
        let unary = Ast {
            ty: AstType::Expression(ExpressionType::Unary {
                op: Box::new(Ast {
                    ty: AstType::Terminal(op),
                }),
                exp: Box::new(expr),
            }),
        };
        assert_eq!(unary.to_string(), "(- 5)");
    }

    #[test]
    fn test_ast_display_terminal_literals() {
        let true_ast = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("true"))),
        };
        assert_eq!(true_ast.to_string(), "TRUE");

        let false_ast = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("false"))),
        };
        assert_eq!(false_ast.to_string(), "FALSE");

        let nil_ast = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("nil"))),
        };
        assert_eq!(nil_ast.to_string(), "NIL");

        let number_ast = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("1.23".to_string(), 1.23))),
        };
        assert_eq!(number_ast.to_string(), "1.23");

        let string_ast = Ast {
            ty: AstType::Terminal(create_token(Lexeme::String("hello".to_string()))),
        };
        assert_eq!(string_ast.to_string(), "hello");

        let identifier_ast = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Identifier("var_name".to_string()))),
        };
        assert_eq!(identifier_ast.to_string(), "var_name");
    }

    #[test]
    fn test_parser_new() {
        let source = "print 42;";
        let parser = Parser::new(source);
        assert_eq!(parser.source, source);
    }

    #[test]
    fn test_parse_empty_source() {
        let mut parser = Parser::new("");
        let result = parser.parse();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![]);
    }

    #[test]
    fn test_parse_simple_print_statement() {
        let mut parser = Parser::new("print 42;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "print 42;");
    }

    #[test]
    fn test_parse_return_statement_with_value() {
        let mut parser = Parser::new("return 123;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "return 123;");
    }

    #[test]
    fn test_parse_return_statement_without_value() {
        let mut parser = Parser::new("return;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "return;");
    }

    #[test]
    fn test_parse_expression_statement() {
        let mut parser = Parser::new("42;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "42");
    }

    #[test]
    fn test_parse_binary_expression() {
        let mut parser = Parser::new("1 + 2;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(+ 1 2)");
    }

    #[test]
    fn test_parse_unary_expression() {
        let mut parser = Parser::new("-5;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(- 5)");
    }

    #[test]
    fn test_parse_grouped_expression() {
        let mut parser = Parser::new("(42);");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(group 42)");
    }

    #[test]
    fn test_parse_equality_expression() {
        let mut parser = Parser::new("1 == 2;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(== 1 2)");

        let mut parser = Parser::new("true != false;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(!= TRUE FALSE)");
    }

    #[test]
    fn test_parse_comparison_expression() {
        let mut parser = Parser::new("5 > 3;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(> 5 3)");

        let mut parser = Parser::new("2 <= 4;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(<= 2 4)");
    }

    #[test]
    fn test_parse_factor_expression() {
        let mut parser = Parser::new("6 * 7;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(* 6 7)");

        let mut parser = Parser::new("8 / 2;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(/ 8 2)");
    }

    #[test]
    fn test_parse_complex_expression() {
        let mut parser = Parser::new("1 + 2 * 3;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(+ 1 (* 2 3))");
    }

    #[test]
    fn test_parse_multiple_statements() {
        let mut parser = Parser::new("print 1; return 2;");
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 2);
        assert_eq!(ast[0].to_string(), "print 1;");
        assert_eq!(ast[1].to_string(), "return 2;");
    }

    #[test]
    fn test_parse_error_missing_semicolon() {
        let mut parser = Parser::new("print 42");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_missing_closing_paren() {
        let mut parser = Parser::new("(42;");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let mut parser = Parser::new("print");
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_expression_type_display() {
        let left = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("1".to_string(), 1.0))),
        };
        let right = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("2".to_string(), 2.0))),
        };
        let op = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("+"))),
        };
        let binary = ExpressionType::Binary {
            op: Box::new(op),
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(binary.to_string(), "1 + 2");

        let expr = Ast {
            ty: AstType::Terminal(create_token(Lexeme::Number("5".to_string(), 5.0))),
        };
        let op = Ast {
            ty: AstType::Terminal(create_token(lexeme_from("-"))),
        };
        let unary = ExpressionType::Unary {
            op: Box::new(op),
            exp: Box::new(expr),
        };
        assert_eq!(unary.to_string(), "-5");
    }
}
