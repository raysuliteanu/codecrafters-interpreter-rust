use anyhow::Result;
use std::iter::Peekable;

use log::trace;
use thiserror::Error;

use crate::model::{Ast, AstExpr, AstStmt};
use crate::token::{Lexeme, Scanner, Token, lexeme_from};

type PeekableTokenIter<'a> = Peekable<std::slice::Iter<'a, Token>>;

/// Pretty print the AST as a tree structure with proper indentation
pub fn print_ast(ast: &[Ast]) {
    fn print_ast_node(ast: &Ast, level: usize) {
        let indent = "    ".repeat(level);
        match ast {
            Ast::Class => println!("{indent}Class"),
            Ast::Function => println!("{indent}Function"),
            Ast::Variable(token, initializer) => {
                println!("{indent}Variable({token:?}, {initializer:?}),");
            }
            Ast::Block(nodes) => {
                println!("{indent}Block(");
                for node in nodes {
                    print_ast_node(node, level + 1);
                }
                println!("{indent})");
            }
            Ast::Statement(stmt) => {
                println!("{indent}Statement(");
                print_ast_stmt(stmt, level + 1);
                println!("{indent})");
            }
            Ast::Expression(expr) => {
                println!("{indent}Expression(");
                print_ast_expr(expr, level + 1);
                println!("{indent})");
            }
        }
    }

    fn print_ast_stmt(stmt: &AstStmt, level: usize) {
        let indent = "    ".repeat(level);
        match stmt {
            AstStmt::If(cond, then_block, else_block) => {
                println!("{indent}If(");
                print_ast_expr(cond, level + 1);
                println!("{indent},");
                print_ast_node(then_block, level + 1);
                if let Some(else_block) = else_block {
                    println!("{indent},");
                    print_ast_node(else_block, level + 1);
                }
                println!("{indent})");
            }
            AstStmt::While(cond, body) => {
                println!("{indent}While(");
                print_ast_expr(cond, level + 1);
                println!("{indent},");
                print_ast_node(body, level + 1);
                println!("{indent})");
            }
            AstStmt::Return(expr) => {
                println!("{indent}Return({expr:?})");
            }
            AstStmt::Print(expr) => {
                println!("{indent}Print(");
                print_ast_expr(expr, level + 1);
                println!("{indent})");
            }
            AstStmt::Expression(expr) => {
                print_ast_expr(expr, level);
            }
        }
    }

    fn print_ast_expr(expr: &AstExpr, level: usize) {
        let indent = "    ".repeat(level);
        match expr {
            AstExpr::Assignment { id, expr } => {
                println!("{indent}Assignment {{ id: \"{id}\", expr: {expr:?} }}");
            }
            AstExpr::Logical { op, left, right } => {
                println!("{indent}Logical {{ op: {op:?}, left: {left:?}, right: {right:?} }}");
            }
            AstExpr::Terminal(token) => {
                println!("{indent}Terminal({token:?})");
            }
            AstExpr::Group(expr) => {
                println!("{indent}Group(");
                print_ast_expr(expr, level + 1);
                println!("{indent})");
            }
            AstExpr::Unary { op, exp } => {
                println!("{indent}Unary {{ op: {op:?}, exp: {exp:?} }}");
            }
            AstExpr::Binary { op, left, right } => {
                println!("{indent}Binary {{ op: {op:?}, left: {left:?}, right: {right:?} }}");
            }
        }
    }

    println!("AST:");
    for node in ast {
        print_ast_node(node, 0);
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

pub type ParseResult<T> = Result<T>;

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
        crate::model::AstExpr::Binary {
            op: $op,
            left: Box::new($l),
            right: Box::new($r),
        }
    };
}

/// convenience macro for creating a new expression group AST node
macro_rules! ast_group {
    ($e: expr) => {
        crate::model::AstExpr::Group(Box::new($e))
    };
}

/// convenience macro for creating a new unary expression AST node
macro_rules! ast_unary {
    ($op_token: expr, $exp: expr) => {
        crate::model::AstExpr::Unary {
            op: $op_token.clone(),
            exp: Box::new($exp),
        }
    };
}

/// convenience macro for creating a new terminal AST node
macro_rules! ast_terminal {
    ($t: expr) => {
        crate::model::AstExpr::Terminal($t.clone())
    };
}

pub struct Parser<'parser> {
    source: &'parser str,
    expression_mode: bool,
    pretty_print: bool,
    print_ast: bool,
}

#[derive(Default, Clone)]
pub struct ParserBuilder<'parser> {
    source: &'parser str,
    expression_mode: Option<bool>,
    pretty_print: Option<bool>,
    print_ast: Option<bool>,
}

impl<'parser> ParserBuilder<'parser> {
    pub fn new(source: &'parser str) -> Self {
        ParserBuilder {
            source,
            ..Default::default()
        }
    }

    pub fn expression_mode(&mut self, expression_mode: bool) -> &mut Self {
        self.expression_mode = Some(expression_mode);
        self
    }

    pub fn pretty_print(&mut self, pretty_print: bool) -> &mut Self {
        self.pretty_print = Some(pretty_print);
        self
    }

    #[allow(dead_code)]
    pub fn print_ast(&mut self, print_ast: bool) -> &mut Self {
        self.print_ast = Some(print_ast);
        self
    }

    pub fn build(&self) -> Parser<'parser> {
        Parser {
            source: self.source,
            expression_mode: self.expression_mode.unwrap_or(true),
            pretty_print: self.pretty_print.unwrap_or(false),
            print_ast: self.print_ast.unwrap_or(true),
        }
    }
}

impl<'parser> Parser<'parser> {
    pub fn new(source: &'parser str, expression_mode: bool, print_ast: bool) -> Parser<'parser> {
        Parser {
            source,
            expression_mode,
            print_ast,
            pretty_print: false,
        }
    }

    pub fn parse(&self) -> ParseResult<Vec<Ast>> {
        let scanner = Scanner::new(self.source, false);
        if let Ok(tokens) = scanner.scan() {
            if tokens.is_empty() {
                trace!("no tokens scanned");
                return Ok(vec![]);
            }

            trace!("parsing {} tokens", tokens.len());

            match self.program(&mut tokens.iter().peekable()) {
                Ok(v) => {
                    if self.print_ast {
                        if self.pretty_print {
                            print_ast(&v);
                        } else {
                            v.iter().for_each(|node| println!("{node}"));
                        }
                    }

                    Ok(v)
                }
                Err(e) => {
                    eprintln!("{e}");
                    Err(e)
                }
            }
        } else {
            Err(ParseError::UnexpectedEof.into())
        }
    }

    fn program(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Vec<Ast>> {
        let mut ast: Vec<Ast> = Vec::new();

        while let Some(token) = tokens.peek()
            && token.lexeme != lexeme_from("eof")
        {
            let statement = self.declaration(tokens)?;
            ast.push(statement);
        }

        trace!("parsed AST: {ast:?}");

        Ok(ast)
    }

    fn declaration(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        if let Some(token) = tokens.peek() {
            match token.lexeme {
                Lexeme::Class => self.class_decl(tokens),
                Lexeme::Fun => self.fun_decl(tokens),
                Lexeme::Var => self.var_decl(tokens),
                _ => {
                    let ast = if self.expression_mode {
                        Ast::Expression(self.expression(tokens)?)
                    } else {
                        self.statement(tokens)?
                    };

                    Ok(ast)
                }
            }
        } else {
            Err(ParseError::UnexpectedEof.into())
        }
    }

    fn class_decl(&self, _tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        todo!("class decl")
    }

    fn fun_decl(&self, _tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        todo!("fun decl")
    }

    // varDecl → "var" IDENTIFIER ( "=" expression )? ";" ;
    fn var_decl(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
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
                        let expr = self.expression(tokens)?;
                        Some(Box::new(expr))
                    } else {
                        None
                    };

                    if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
                        Ok(Ast::Variable(identifier_token, expr))
                    } else if tokens.peek().is_some() {
                        ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
                    } else {
                        Err(ParseError::UnexpectedEof.into())
                    }
                }
                _ => ast_expected_token!(t, lexeme_from("identifier")),
            },
            _ => Err(ParseError::UnexpectedEof.into()),
        }
    }

    fn statement(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("statement: {:?}", tokens.peek());
        let r = match tokens.peek() {
            Some(token) => match token.lexeme {
                // left brace token indicates block start
                crate::token::Lexeme::LeftBrace => self.parse_block(tokens),
                crate::token::Lexeme::For => self.for_stmt(tokens),
                crate::token::Lexeme::If => self.if_stmt(tokens),
                crate::token::Lexeme::Print => self.print_stmt(tokens),
                crate::token::Lexeme::Return => self.return_stmt(tokens),
                crate::token::Lexeme::While => self.while_stmt(tokens),
                _ => {
                    let ast = self.expression_statement(tokens)?;
                    Ok(ast)
                }
            },
            None => Err(ParseError::UnexpectedEof.into()),
        }?;
        Ok(r)
    }

    fn parse_block(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        let left_brace_token = tokens.next().unwrap();
        assert_eq!(left_brace_token.lexeme, lexeme_from("{"));
        trace!("block start");

        let mut stmts = vec![];
        while tokens.peek().is_some_and(|t| t.lexeme != lexeme_from("}")) {
            let stmt = self.declaration(tokens)?;
            stmts.push(stmt);
        }

        let right_brace_token = tokens.next().unwrap();
        assert_eq!(right_brace_token.lexeme, lexeme_from("}"));
        trace!("block end");

        Ok(Ast::Block(stmts))
    }

    // forStmt        → "for" "(" ( varDecl | exprStmt | ";" )
    //                              expression? ";"
    //                              expression? ")" statement ;
    // NOTE: for loops can desugar to while loops
    fn for_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("for_stmt");

        let for_token = tokens.next().unwrap();
        assert_eq!(for_token.lexeme, lexeme_from("for"));

        if tokens.next_if(|t| t.lexeme == lexeme_from("(")).is_some() {
            // handle initializer, if any
            let init_expr = match tokens.peek() {
                // no initializer
                // e.g. for ( ; cond; incr)
                Some(t) if t.lexeme == lexeme_from(";") => {
                    trace!("for_stmt: no initializer");
                    let for_token = tokens.next().unwrap();
                    assert_eq!(for_token.lexeme, lexeme_from(";"));

                    None
                }
                // var decl initializer
                // e.g. for (var init; cond; incr)
                Some(t) if t.lexeme == lexeme_from("var") => {
                    let var = self.var_decl(tokens)?;
                    trace!("for_stmt: init: {var}");
                    Some(var)
                }
                // expr initializer
                // e.g. var a; for (a = 1; cond; incr)
                Some(_) => {
                    let expr = self.expression_statement(tokens)?;
                    trace!("for_stmt: init: {expr}");
                    Some(expr)
                }
                // unexpected eof
                None => return Err(ParseError::UnexpectedEof.into()),
            };

            let semicolon = tokens.next_if(|t| t.lexeme == lexeme_from(";"));
            let cond_expr = if semicolon.is_some() {
                // for (init ; ; incr)
                trace!("for_stmt: no cond");
                ast_terminal!(Token {
                    lexeme: lexeme_from("true"),
                    span: (0, 0, 0).into()
                })
            } else {
                // for (_ ; cond ; _)
                let expr = self.expression(tokens)?;
                // must be semicolon after cond
                let _semicolon = tokens.next();
                assert_eq!(_semicolon.expect(";").lexeme, lexeme_from(";"));
                trace!("for_stmt: cond: {expr}");
                expr
            };

            let close_paren = tokens.next_if(|t| t.lexeme == lexeme_from(")"));
            let incr_expr = if close_paren.is_some() {
                // for (init ; cond ; )
                trace!("for_stmt: no incr expr");
                None
            } else {
                // for (_ ; _ ; incr)
                let expr = self.expression(tokens)?;
                let _closing_paren = tokens.next();
                assert_eq!(_closing_paren.expect(";").lexeme, lexeme_from(")"));
                trace!("for_stmt: incr: {expr}");
                Some(expr)
            };

            let body = if tokens.peek().is_some_and(|t| t.lexeme == lexeme_from("{")) {
                trace!("for_stmt: body block start");
                let block = self.parse_block(tokens)?;
                trace!("for_stmt: body block end");
                block
            } else {
                trace!("for_stmt: body expression");
                self.statement(tokens)?
            };

            // build while loop body
            // {
            //    (incr;)
            //    for_loop_body
            // }
            let body = if let Some(incr) = incr_expr {
                Ast::Block(vec![body, Ast::Expression(incr)])
            } else {
                body
            };

            // build while loop condition
            // while (cond)
            //     body
            let body = Ast::Statement(AstStmt::While(Box::new(cond_expr), Box::new(body)));

            // build while loop cond initializer
            // init
            // while (cond)
            //     body
            let body = if let Some(init) = init_expr {
                Ast::Block(vec![init, body])
            } else {
                body
            };

            Ok(body)
        } else {
            todo!("missing opening paren");
        }
    }

    fn if_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("if_stmt: {:?}", tokens.peek());
        let if_token = tokens.next().unwrap();
        assert_eq!(if_token.lexeme, lexeme_from("if"));

        let cond = self.expression(tokens)?;
        let then_stmt = self.statement(tokens)?;
        let else_stmt = if tokens
            .next_if(|t| t.lexeme == lexeme_from("else"))
            .is_some()
        {
            Some(Box::new(self.statement(tokens)?))
        } else {
            None
        };

        Ok(Ast::Statement(AstStmt::If(
            Box::new(cond),
            Box::new(then_stmt),
            else_stmt,
        )))
    }

    fn while_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("while_stmt: {:?}", tokens.peek());
        let while_token = tokens.next().unwrap();
        assert_eq!(while_token.lexeme, lexeme_from("while"));

        let cond = self.expression(tokens)?;
        let body = self.statement(tokens)?;

        Ok(Ast::Statement(AstStmt::While(
            Box::new(cond),
            Box::new(body),
        )))
    }

    fn print_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("print_stmt: {:?}", tokens.peek());

        let print_token = tokens.next().unwrap();
        assert_eq!(print_token.lexeme, lexeme_from("print"));

        let expr = self.expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            Ok(Ast::Statement(AstStmt::Print(expr)))
        } else if tokens.peek().is_some() {
            ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
        } else {
            Err(ParseError::UnexpectedEof.into())
        }
    }

    fn return_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("return_stmt: {:?}", tokens.peek());

        // Consume the 'return' token first
        let return_token = tokens.next().unwrap();

        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            // a "naked" 'return' without expression i.e. "return;"
            Ok(Ast::Statement(AstStmt::Return(None)))
        } else if tokens.peek().is_some_and(|t| {
            !matches!(
                t.lexeme,
                Lexeme::LeftBrace
                    | Lexeme::For
                    | Lexeme::While
                    | Lexeme::If
                    | Lexeme::Print
                    | Lexeme::Return
            )
        }) {
            let ast = self.expression(tokens)?;
            if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
                Ok(Ast::Statement(AstStmt::Return(Some(Box::new(ast)))))
            } else if tokens.peek().is_some() {
                ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
            } else {
                Err(ast_missing_token!(lexeme_from(";"), lexeme_from("eof")))
            }
        } else {
            ast_expected_token!(return_token, lexeme_from(";"))
        }
    }

    fn expression_statement(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        let token = tokens.peek();
        trace!("expr_stmt: {token:?}");
        let expr = self.expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == lexeme_from(";")).is_some() {
            Ok(Ast::Statement(AstStmt::Expression(expr)))
        } else if tokens.peek().is_some() {
            ast_expected_token!(tokens.peek().unwrap(), lexeme_from(";"))
        } else {
            Err(ast_missing_token!(lexeme_from(";"), lexeme_from("eof")))
        }
    }

    fn expression(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("expr: {:?}", tokens.peek());
        self.assignment(tokens)
    }

    fn assignment(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("assignment: {:?}", tokens.peek());

        // Clone the token info we need before borrowing mutably
        let token_lexeme = if let Some(token) = tokens.peek() {
            match &token.lexeme {
                Lexeme::Identifier(name) => name.clone(),
                _ => token.lexeme.to_string(),
            }
        } else {
            return Err(ParseError::UnexpectedEof.into());
        };

        let left = self.logical_or(tokens)?;
        // after parsing tokens, if the next token is '=' then ...
        if tokens.next_if(|t| t.lexeme == lexeme_from("=")).is_some() {
            // ... it's an assignment i.e. 'token' is lvalue, so parse rvalue
            trace!("assignment is assignment");
            let rvalue = self.assignment(tokens)?;
            Ok(AstExpr::Assignment {
                id: token_lexeme,
                expr: Box::new(rvalue),
            })
        } else {
            trace!("assignment -> expression");
            // ... otherwise it was just an expression so return that
            Ok(left)
        }
    }

    fn logical_or(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("logical_or: {:?}", tokens.peek());
        let mut left = self.logical_and(tokens)?;
        while let Some(t) = tokens.next_if(|t| t.lexeme == lexeme_from("or")) {
            let right = self.logical_and(tokens)?;
            let op = t.clone();
            left = AstExpr::Logical {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }

        Ok(left)
    }

    fn logical_and(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("logical_and: {:?}", tokens.peek());
        let mut left = self.equality(tokens)?;
        while let Some(t) = tokens.next_if(|t| t.lexeme == lexeme_from("and")) {
            let right = self.equality(tokens)?;
            let op = t.clone();
            left = AstExpr::Logical {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }

        Ok(left)
    }

    fn equality(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("equality: {:?}", tokens.peek());
        let mut left = self.comparison(tokens)?;

        while let Some(t) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::BangEq | Lexeme::EqEq)) {
            let right = self.comparison(tokens)?;
            let op = t.clone();
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn comparison(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("comparison: {:?}", tokens.peek());
        let mut left = self.term(tokens)?;

        while let Some(t) = tokens.next_if(|t| {
            matches!(
                t.lexeme,
                Lexeme::Greater | Lexeme::GreaterEq | Lexeme::Less | Lexeme::LessEq
            )
        }) {
            let right = self.term(tokens)?;
            let op = t.clone();
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn term(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("term: {:?}", tokens.peek());
        let mut left = self.factor(tokens)?;

        while let Some(t) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Plus | Lexeme::Minus)) {
            let right = self.factor(tokens)?;
            let op = t.clone();
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn factor(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("factor: {:?}", tokens.peek());
        let mut left = self.unary(tokens)?;

        while let Some(t) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Star | Lexeme::Slash)) {
            let right = self.unary(tokens)?;
            let op = t.clone();
            left = ast_binary!(op, left, right);
        }

        Ok(left)
    }

    fn unary(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("unary: {:?}", tokens.peek());
        if let Some(op_token) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Minus | Lexeme::Bang))
        {
            let right = self.unary(tokens)?;
            Ok(ast_unary!(op_token, right))
        } else {
            self.primary(tokens)
        }
    }

    fn primary(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("primary: {:?}", tokens.peek());
        if let Some(token) =
            tokens.next_if(|t| matches!(t.lexeme, Lexeme::True | Lexeme::False | Lexeme::Nil))
        {
            Ok(ast_terminal!(token))
        } else if let Some(_left_paren) = tokens.next_if(|t| t.lexeme == lexeme_from("(")) {
            let expr = self.expression(tokens)?;
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
    use crate::{model::AstExpr, span::Span};

    fn create_token(lexeme: Lexeme) -> Token {
        Token::new(lexeme, Span::new(1, 0, 1))
    }

    #[test]
    fn test_ast_display_print_statement() {
        let expr = AstExpr::Terminal(create_token(lexeme_from("true")));
        let print_stmt = Ast::Statement(AstStmt::Print(expr));
        assert_eq!(print_stmt.to_string(), "print true;");
    }

    #[test]
    fn test_ast_display_return_statement_with_value() {
        let expr = AstExpr::Terminal(create_token(Lexeme::Number("42".to_string(), 42.0)));
        let return_stmt = Ast::Statement(AstStmt::Return(Some(Box::new(expr))));
        assert_eq!(return_stmt.to_string(), "return 42.0;");
    }

    #[test]
    fn test_ast_display_return_statement_without_value() {
        let return_stmt = Ast::Statement(AstStmt::Return(None));
        assert_eq!(return_stmt.to_string(), "return;");
    }

    #[test]
    fn test_ast_display_group() {
        let expr = AstExpr::Terminal(create_token(lexeme_from("true")));
        let group = Ast::Expression(AstExpr::Group(Box::new(expr)));
        assert_eq!(group.to_string(), "(group true)");
    }

    #[test]
    fn test_ast_display_binary_expression() {
        let left = AstExpr::Terminal(create_token(Lexeme::Number("1".to_string(), 1.0)));
        let right = AstExpr::Terminal(create_token(Lexeme::Number("2".to_string(), 2.0)));
        let op = create_token(lexeme_from("+"));
        let binary = AstExpr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(binary.to_string(), "(+ 1.0 2.0)");
    }

    #[test]
    fn test_ast_display_unary_expression() {
        let expr = AstExpr::Terminal(create_token(Lexeme::Number("5".to_string(), 5.0)));
        let op = create_token(lexeme_from("-"));
        let unary = Ast::Expression(AstExpr::Unary {
            op,
            exp: Box::new(expr),
        });
        assert_eq!(unary.to_string(), "(- 5.0)");
    }

    #[test]
    fn test_ast_display_terminal_literals() {
        let true_ast = Ast::Expression(AstExpr::Terminal(create_token(lexeme_from("true"))));
        assert_eq!(true_ast.to_string(), "true");

        let false_ast = Ast::Expression(AstExpr::Terminal(create_token(lexeme_from("false"))));
        assert_eq!(false_ast.to_string(), "false");

        let nil_ast = Ast::Expression(AstExpr::Terminal(create_token(lexeme_from("nil"))));
        assert_eq!(nil_ast.to_string(), "nil");

        let number_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::Number(
            "1.23".to_string(),
            1.23,
        ))));
        assert_eq!(number_ast.to_string(), "1.23");

        let string_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::String(
            "hello".to_string(),
        ))));
        assert_eq!(string_ast.to_string(), "hello");

        let identifier_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::Identifier(
            "var_name".to_string(),
        ))));
        assert_eq!(identifier_ast.to_string(), "var_name");
    }

    #[test]
    fn test_parser_new() {
        let source = "print 42;";
        let parser = Parser::new(source, true, true);
        assert_eq!(parser.source, source);
    }

    #[test]
    fn test_parse_empty_source() {
        let parser = Parser::new("", true, true);
        let result = parser.parse();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![]);
    }

    #[test]
    fn test_parse_simple_print_statement() {
        let parser = Parser::new("print 42;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "print 42.0;");
    }

    #[test]
    fn test_parse_return_statement_with_value() {
        let parser = Parser::new("return 123;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "return 123.0;");
    }

    #[test]
    fn test_parse_return_statement_without_value() {
        let parser = Parser::new("return;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "return;");
    }

    #[test]
    fn test_parse_expression_statement() {
        let parser = Parser::new("42;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "42.0");
    }

    #[test]
    fn test_parse_binary_expression() {
        let parser = Parser::new("1 + 2;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(+ 1.0 2.0)");
    }

    #[test]
    fn test_parse_unary_expression() {
        let parser = Parser::new("-5;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(- 5.0)");
    }

    #[test]
    fn test_parse_grouped_expression() {
        let parser = Parser::new("(42);", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(group 42.0)");
    }

    #[test]
    fn test_parse_equality_expression() {
        let parser = Parser::new("1 == 2;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(== 1.0 2.0)");

        let parser = Parser::new("true != false;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(!= true false)");
    }

    #[test]
    fn test_parse_comparison_expression() {
        let parser = Parser::new("5 > 3;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(> 5.0 3.0)");

        let parser = Parser::new("2 <= 4;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(<= 2.0 4.0)");
    }

    #[test]
    fn test_parse_factor_expression() {
        let parser = Parser::new("6 * 7;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(* 6.0 7.0)");

        let parser = Parser::new("8 / 2;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(/ 8.0 2.0)");
    }

    #[test]
    fn test_parse_complex_expression() {
        let parser = Parser::new("1 + 2 * 3;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(+ 1.0 (* 2.0 3.0))");
    }

    #[test]
    fn test_parse_multiple_statements() {
        let parser = Parser::new("print 1; return 2;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 2);
        assert_eq!(ast[0].to_string(), "print 1.0;");
        assert_eq!(ast[1].to_string(), "return 2.0;");
    }

    #[test]
    fn test_parse_error_missing_semicolon() {
        let parser = Parser::new("print 42", true, true);
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_missing_closing_paren() {
        let parser = Parser::new("(42;", true, true);
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let parser = Parser::new("print", true, true);
        let result = parser.parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_expression_type_display() {
        let left = AstExpr::Terminal(create_token(Lexeme::Number("1".to_string(), 1.0)));
        let right = AstExpr::Terminal(create_token(Lexeme::Number("2".to_string(), 2.0)));
        let op = create_token(lexeme_from("+"));
        let binary = AstExpr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(binary.to_string(), "(+ 1.0 2.0)");

        let expr = AstExpr::Terminal(create_token(Lexeme::Number("5".to_string(), 5.0)));
        let op = create_token(lexeme_from("-"));
        let unary = AstExpr::Unary {
            op,
            exp: Box::new(expr),
        };
        assert_eq!(unary.to_string(), "(- 5.0)");
    }

    #[test]
    fn test_parse_variable_declaration_without_initializer() {
        let parser = Parser::new("var x;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "var x;");
    }

    #[test]
    fn test_parse_variable_declaration_with_initializer() {
        let parser = Parser::new("var x = 42;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "var x = 42.0;");
    }

    #[test]
    fn test_parse_assignment_expression() {
        let parser = Parser::new("x = 10;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "x = 10.0");
    }

    #[test]
    fn test_parse_logical_or_expression() {
        let parser = Parser::new("true or false;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(or true false)");
    }

    #[test]
    fn test_parse_logical_and_expression() {
        let parser = Parser::new("true and false;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "(and true false)");
    }

    #[test]
    fn test_parse_if_statement() {
        let parser = Parser::new("if (true) print 42;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "if (group true) print 42.0;");
    }

    #[test]
    fn test_parse_if_else_statement() {
        let parser = Parser::new("if (false) print 1; else print 2;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(
            ast[0].to_string(),
            "if (group false) print 1.0; else print 2.0;"
        );
    }

    #[test]
    fn test_parse_while_statement() {
        let parser = Parser::new("while (true) print 42;", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "while (group true) print 42.0;");
    }

    #[test]
    fn test_parse_empty_block() {
        let parser = Parser::new("{}", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\n}\n");
    }

    #[test]
    fn test_parse_block_with_statements() {
        let parser = Parser::new("{ print 1; print 2; }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\nprint 1.0;print 2.0;}\n");
    }

    #[test]
    fn test_parse_nested_blocks() {
        let parser = Parser::new("{ { print 42; } }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\n{\nprint 42.0;}\n}\n");
    }

    #[test]
    fn test_parse_block_with_variable_declaration() {
        let parser = Parser::new("{ var x = 10; print x; }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\nvar x = 10.0;print x;}\n");
    }
}
