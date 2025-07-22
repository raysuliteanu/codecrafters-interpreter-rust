use anyhow::Result;
use std::iter::Peekable;

use log::trace;
use thiserror::Error;

use crate::model::{Ast, AstExpr, AstStmt, Lexeme, Token};
use crate::token::Scanner;

type PeekableTokenIter<'a> = Peekable<std::slice::Iter<'a, Token>>;

const MAX_FUNC_ARGS: u8 = u8::MAX;

#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("missing token '{}' got '{}'\n[line {line}]", expected.lexeme_str(), actual.lexeme_str())]
    MissingToken {
        expected: Lexeme,
        actual: Lexeme,
        line: usize,
    },
    #[error("[line {}] unexpected token '{}'", actual.span.line(), actual.lexeme.lexeme_str())]
    UnexpectedToken { actual: Token },
    #[error("Unexpected EOF")]
    UnexpectedEof,
    #[error("[line {}] Error at '{}'. Expect expression.\n{source}", token.span.line(), token.lexeme.lexeme_str())]
    ExpectedExpressionError {
        token: Token,
        source: Box<ParseError>,
    },
    #[error("Can't have more than {MAX_FUNC_ARGS} arguments.\n[line {0}]")]
    TooManyFunctionArgs(usize),
}

pub type ParseResult<T> = Result<T, ParseError>;

macro_rules! ast_call_expression {
    ($id: expr, $args: expr, $site: expr) => {
        crate::model::AstExpr::Call {
            func: $id,
            args: $args,
            site: $site,
        }
    };
}
macro_rules! ast_expression_expected {
    ($t: expr, $s: expr) => {{
        crate::parser::ParseError::ExpectedExpressionError {
            token: $t,
            source: Box::new($s),
        }
    }};
}

macro_rules! ast_expected_token {
    (token $a: expr, $e: expr) => {
        crate::parser::ParseError::MissingToken {
            expected: $e,
            actual: $a.lexeme.clone(),
            line: $a.span.line(),
        }
    };
    (lexeme $a: expr, $e: expr) => {
        crate::parser::ParseError::MissingToken {
            expected: $e,
            actual: $a,
            line: 0,
        }
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
                            crate::util::print_ast(&v);
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
            Err(ParseError::UnexpectedEof)
        }
    }

    fn program(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Vec<Ast>> {
        let mut ast: Vec<Ast> = Vec::new();

        while let Some(token) = tokens.peek()
            && token.lexeme != Lexeme::Eof
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
            Err(ParseError::UnexpectedEof)
        }
    }

    fn class_decl(&self, _tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        todo!("class decl")
    }

    // function   → IDENTIFIER "(" parameters? ")" block ;
    // parameters → IDENTIFIER ( "," IDENTIFIER )* ;
    fn fun_decl(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("fun_decl");

        // consume 'fun' token
        let fun_token = tokens.next().unwrap();
        assert_eq!(fun_token.lexeme, Lexeme::Fun);

        let name = self.parse_fun_name(tokens)?;

        let params = if tokens.next_if(|t| t.lexeme == Lexeme::LeftParen).is_some() {
            self.parse_fun_params(tokens)?
        } else if let Some(token) = tokens.peek() {
            return Err(ast_expected_token!(token token, Lexeme::LeftParen));
        } else {
            return Err(ParseError::UnexpectedEof);
        };

        let body = self.parse_block(tokens)?;

        Ok(Ast::Function {
            name,
            params,
            body: Box::new(body),
        })
    }

    fn parse_fun_name(&self, tokens: &mut PeekableTokenIter) -> ParseResult<String> {
        if let Some(id) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Identifier { .. })) {
            match &id.lexeme {
                Lexeme::Identifier(i) => Ok(i.clone()),
                // TODO: how can we do this better?
                _ => panic!("matched {id} but next_if() said it was an Lexeme::Identifier"),
            }
        } else if let Some(token) = tokens.peek() {
            Err(ast_expected_token!(token token, Lexeme::Identifier("id".to_owned())))
        } else {
            Err(ParseError::UnexpectedEof)
        }
    }

    // parameters → IDENTIFIER ( "," IDENTIFIER )* ;
    fn parse_fun_params(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Vec<String>> {
        let mut params = vec![];

        while tokens
            .peek()
            .is_some_and(|f| f.lexeme != Lexeme::RightParen)
        {
            let next_token = tokens.next().expect("peeked already");
            if let Lexeme::Identifier(i) = &next_token.lexeme {
                trace!("adding param {i}");
                params.push(i.clone());

                match tokens.peek() {
                    // identifier identifier
                    Some(t) if matches!(t.lexeme, Lexeme::Identifier { .. }) => {
                        return Err(ast_expected_token!(token t, Lexeme::Comma));
                    }
                    // identifier ','
                    Some(t) if matches!(t.lexeme, Lexeme::Comma) => {
                        tokens.next();
                        continue;
                    }
                    Some(_) | None => continue,
                }
            }
        }

        // consume ')' token
        let rparen = tokens.next().expect("loop already verified closing paren");
        assert_eq!(rparen.lexeme, Lexeme::RightParen);

        Ok(params)
    }

    // varDecl → "var" IDENTIFIER ( "=" expression )? ";" ;
    fn var_decl(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("var_decl: {:?}", tokens.peek());

        // eat 'var' token
        let var_token = tokens.next().unwrap();
        assert_eq!(var_token.lexeme, Lexeme::Var);

        match tokens.peek() {
            Some(t) => match &t.lexeme {
                Lexeme::Identifier(_) => {
                    trace!("found identifier: {t}");
                    let name = tokens.next().unwrap().clone();
                    let initializer = if let Some(_e) = tokens.next_if(|t| t.lexeme == Lexeme::Eq) {
                        let expr = self.expression(tokens)?;
                        Some(Box::new(expr))
                    } else {
                        None
                    };

                    if tokens.next_if(|t| t.lexeme == Lexeme::SemiColon).is_some() {
                        Ok(Ast::Variable { name, initializer })
                    } else if tokens.peek().is_some() {
                        Err(ast_expected_token!(token
                            tokens.peek().unwrap(),
                            Lexeme::SemiColon
                        ))
                    } else {
                        Err(ParseError::UnexpectedEof)
                    }
                }
                _ => Err(ast_expected_token!(token t, Lexeme::from("identifier"))),
            },
            _ => Err(ParseError::UnexpectedEof),
        }
    }

    fn statement(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("statement: {:?}", tokens.peek());
        let r = match tokens.peek() {
            Some(token) => match token.lexeme {
                // left brace token indicates block start
                Lexeme::LeftBrace => self.parse_block(tokens),
                Lexeme::For => self.for_stmt(tokens),
                Lexeme::If => self.if_stmt(tokens),
                Lexeme::Print => self.print_stmt(tokens),
                Lexeme::Return => self.return_stmt(tokens),
                Lexeme::While => self.while_stmt(tokens),
                _ => {
                    let ast = self.expression_statement(tokens)?;
                    Ok(ast)
                }
            },
            None => Err(ParseError::UnexpectedEof),
        }?;
        Ok(r)
    }

    fn parse_block(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("block start");

        let expect_left_brace = tokens.next().unwrap();
        if expect_left_brace.lexeme != Lexeme::LeftBrace {
            return Err(ast_expected_token!(token expect_left_brace, Lexeme::LeftBrace));
        }

        let mut stmts = vec![];
        while tokens
            .peek()
            .is_some_and(|t| t.lexeme != Lexeme::RightBrace)
        {
            let stmt = self.declaration(tokens)?;
            stmts.push(stmt);
        }

        let right_brace_token = tokens.next().unwrap();
        assert_eq!(right_brace_token.lexeme, Lexeme::RightBrace);
        trace!("block end");

        Ok(Ast::Block(stmts))
    }

    // forStmt        → "for" "(" ( varDecl | exprStmt | ";" )
    //                              expression? ";"
    //                              expression? ")" statement ;
    // NOTE: for loops can desugar to while loops
    fn for_stmt(&self, tokens: &mut PeekableTokenIter) -> Result<Ast, ParseError> {
        trace!("for_stmt");

        let for_token = tokens.next().unwrap();
        assert_eq!(for_token.lexeme, Lexeme::For);

        if tokens.next_if(|t| t.lexeme == Lexeme::LeftParen).is_some() {
            // handle initializer, if any
            let init_expr = match tokens.peek().cloned() {
                // no initializer
                // e.g. for ( ; cond; incr)
                Some(t) if t.lexeme == Lexeme::SemiColon => {
                    trace!("for_stmt: no initializer");
                    let for_token = tokens.next().unwrap();
                    assert_eq!(for_token.lexeme, Lexeme::SemiColon);

                    None
                }
                // var decl initializer
                // e.g. for (var init; cond; incr)
                Some(token) if token.lexeme == Lexeme::Var => match self.var_decl(tokens) {
                    Ok(expr) => Some(expr),
                    Err(e) => return Err(ast_expression_expected!(token.clone(), e)),
                },
                // expr initializer
                // e.g. var a; for (a = 1; cond; incr)
                Some(token) => match self.expression_statement(tokens) {
                    Ok(expr) => {
                        trace!("for_stmt: init: {expr}");
                        Some(expr)
                    }
                    Err(e) => return Err(ast_expression_expected!(token.clone(), e)),
                },
                // unexpected eof
                None => return Err(ParseError::UnexpectedEof),
            };

            let semicolon = tokens.next_if(|t| t.lexeme == Lexeme::SemiColon);
            let cond_expr = if semicolon.is_some() {
                // for (init ; ; incr)
                trace!("for_stmt: no cond");
                ast_terminal!(Token {
                    lexeme: Lexeme::True,
                    span: (0, 0, 0).into()
                })
            } else {
                // for (_ ; cond ; _)
                let t = (*tokens.peek().unwrap()).clone();
                let expr = match self.expression(tokens) {
                    Ok(expr) => expr,
                    Err(e) => return Err(ast_expression_expected!(t.clone(), e)),
                };

                // must be semicolon after cond
                let _semicolon = tokens.next();
                assert_eq!(_semicolon.expect(";").lexeme, Lexeme::SemiColon);
                trace!("for_stmt: cond: {expr}");
                expr
            };

            let close_paren = tokens.next_if(|t| t.lexeme == Lexeme::RightParen);
            let incr_expr = if close_paren.is_some() {
                // for (init ; cond ; )
                trace!("for_stmt: no incr expr");
                None
            } else {
                // for (_ ; _ ; incr)
                let t = (*tokens.peek().unwrap()).clone();
                let expr = match self.expression(tokens) {
                    Ok(expr) => expr,
                    Err(e) => return Err(ast_expression_expected!(t.clone(), e)),
                };

                let _closing_paren = tokens.next();
                assert_eq!(_closing_paren.expect(";").lexeme, Lexeme::RightParen);
                trace!("for_stmt: incr: {expr}");
                Some(expr)
            };

            let body = if tokens.peek().is_some_and(|t| t.lexeme == Lexeme::LeftBrace) {
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
        assert_eq!(if_token.lexeme, Lexeme::If);

        let cond = self.expression(tokens)?;
        let then_stmt = self.statement(tokens)?;
        let else_stmt = if tokens.next_if(|t| t.lexeme == Lexeme::Else).is_some() {
            Some(Box::new(self.statement(tokens)?))
        } else {
            None
        };

        Ok(Ast::Statement(AstStmt::If {
            condition: Box::new(cond),
            then: Box::new(then_stmt),
            or_else: else_stmt,
        }))
    }

    fn while_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("while_stmt: {:?}", tokens.peek());
        let while_token = tokens.next().unwrap();
        assert_eq!(while_token.lexeme, Lexeme::While);

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
        assert_eq!(print_token.lexeme, Lexeme::Print);

        let expr = self.expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == Lexeme::SemiColon).is_some() {
            Ok(Ast::Statement(AstStmt::Print(expr)))
        } else if tokens.peek().is_some() {
            Err(ast_expected_token!(token tokens.peek().unwrap(), Lexeme::SemiColon))
        } else {
            Err(ParseError::UnexpectedEof)
        }
    }

    fn return_stmt(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        trace!("return_stmt: {:?}", tokens.peek());

        // Consume the 'return' token first
        let return_token = tokens.next().unwrap();
        assert_eq!(return_token.lexeme, Lexeme::Return);

        if tokens.next_if(|t| t.lexeme == Lexeme::SemiColon).is_some() {
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
            if tokens.next_if(|t| t.lexeme == Lexeme::SemiColon).is_some() {
                Ok(Ast::Statement(AstStmt::Return(Some(Box::new(ast)))))
            } else if tokens.peek().is_some() {
                Err(ast_expected_token!(token tokens.peek().unwrap(), Lexeme::SemiColon))
            } else {
                Err(ast_expected_token!(lexeme Lexeme::SemiColon, Lexeme::Eof))
            }
        } else {
            Err(ast_expected_token!(token return_token, Lexeme::SemiColon))
        }
    }

    fn expression_statement(&self, tokens: &mut PeekableTokenIter) -> ParseResult<Ast> {
        let token = tokens.peek();
        trace!("expr_stmt: {token:?}");
        let expr = self.expression(tokens)?;
        if tokens.next_if(|t| t.lexeme == Lexeme::SemiColon).is_some() {
            Ok(Ast::Statement(AstStmt::Expression(expr)))
        } else if tokens.peek().is_some() {
            Err(ast_expected_token!(token tokens.peek().unwrap(), Lexeme::SemiColon))
        } else {
            Err(ast_expected_token!(lexeme Lexeme::SemiColon, Lexeme::Eof))
        }
    }

    fn expression(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("expr: {:?}", tokens.peek());
        self.assignment(tokens)
    }

    // assignment → ( call "." )? IDENTIFIER "=" assignment
    //              | logic_or ;
    fn assignment(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("assignment: {:?}", tokens.peek());

        // Clone the token info we need before borrowing mutably
        let token_lexeme = if let Some(token) = tokens.peek() {
            match &token.lexeme {
                Lexeme::Identifier(name) => name.clone(),
                _ => token.lexeme.to_string(),
            }
        } else {
            return Err(ParseError::UnexpectedEof);
        };

        let left = self.logical_or(tokens)?;
        // after parsing tokens, if the next token is '=' then ...
        if tokens.next_if(|t| t.lexeme == Lexeme::Eq).is_some() {
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

    // logic_or       → logic_and ( "or" logic_and )* ;
    fn logical_or(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("logical_or: {:?}", tokens.peek());
        let mut left = self.logical_and(tokens)?;
        while let Some(t) = tokens.next_if(|t| t.lexeme == Lexeme::Or) {
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

    // logic_and      → equality ( "and" equality )* ;
    fn logical_and(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("logical_and: {:?}", tokens.peek());
        let mut left = self.equality(tokens)?;
        while let Some(t) = tokens.next_if(|t| t.lexeme == Lexeme::And) {
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

    // equality → comparison ( ( "!=" | "==" ) comparison )* ;
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

    // comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
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

    // term → factor ( ( "-" | "+" ) factor )* ;
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

    // factor → unary ( ( "/" | "*" ) unary )* ;
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

    // unary → ( "!" | "-" ) unary | call ;
    fn unary(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("unary: {:?}", tokens.peek());
        if let Some(op_token) = tokens.next_if(|t| matches!(t.lexeme, Lexeme::Minus | Lexeme::Bang))
        {
            let right = self.unary(tokens)?;
            Ok(ast_unary!(op_token, right))
        } else {
            self.call(tokens)
        }
    }

    // call → primary ( "(" arguments? ")" | "." IDENTIFIER )* ;
    // arguments → expression ( "," expression )* ;
    fn call(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("call");

        let mut expr = self.primary(tokens)?;

        let result = match &expr {
            AstExpr::Terminal(token) => match &token.lexeme {
                Lexeme::Identifier(id) => match tokens.peek() {
                    Some(_t) if _t.lexeme == Lexeme::LeftParen => {
                        trace!("call: args start");
                        let _open_paren = tokens.next();

                        let mut args = vec![];

                        // while not closing paren ...
                        while let Some(t) = tokens.peek().cloned() {
                            // if closing paren, then done with args processing
                            if t.lexeme == Lexeme::RightParen {
                                let _close_paren = tokens.next();
                                expr = ast_call_expression!((*id).clone(), args, t.span.clone());
                                trace!("call: args end");
                                break;
                            }

                            let arg = self.expression(tokens)?;
                            args.push(arg);

                            // limit number of args to 256 (per Crafting Interpeters book, Ch 10)
                            if args.len() >= MAX_FUNC_ARGS as usize {
                                return Err(ParseError::TooManyFunctionArgs(t.span.line()));
                            }

                            if tokens.peek().is_some_and(|t| t.lexeme == Lexeme::Comma) {
                                let _comma = tokens.next();
                            }
                        }

                        expr
                    }
                    Some(_) | None => expr,
                },
                _ => expr,
            },
            _ => expr,
        };

        Ok(result)

        // // if expr is an Identifier, (e.g. 'foo'), then if next is an open paren then this is a
        // // call, otherewise just a normal expr
        // let result = match tokens.peek() {
        //     Some(_t) if _t.lexeme == Lexeme::LeftParen => {
        //         trace!("call: args start");
        //         let _open_paren = tokens.next();
        //
        //         let mut args = vec![];
        //
        //         // while not closing paren ...
        //         while let Some(t) = tokens.peek().cloned() {
        //             // if closing paren, then done with args processing
        //             if t.lexeme == Lexeme::RightParen {
        //                 let _close_paren = tokens.next();
        //                 expr = ast_call_expression!(expr, args, t.span.clone());
        //                 trace!("call: args end");
        //                 break;
        //             }
        //
        //             let arg = self.expression(tokens)?;
        //             args.push(arg);
        //
        //             // limit number of args to 256 (per Crafting Interpeters book, Ch 10)
        //             if args.len() >= MAX_FUNC_ARGS as usize {
        //                 return Err(ParseError::TooManyFunctionArgs(t.span.line()));
        //             }
        //
        //             if tokens.peek().is_some_and(|t| t.lexeme == Lexeme::Comma) {
        //                 let _comma = tokens.next();
        //             }
        //         }
        //         expr
        //     }
        //     Some(_) => expr,
        //     None => return Err(ParseError::UnexpectedEof),
        // };
        //
        // Ok(result)
    }

    // primary → "true" | "false" | "nil" | "this"
    //         | NUMBER | STRING | IDENTIFIER | "(" expression ")"
    //         | "super" "." IDENTIFIER ;
    fn primary(&self, tokens: &mut PeekableTokenIter) -> ParseResult<AstExpr> {
        trace!("primary: {:?}", tokens.peek());
        if let Some(token) =
            tokens.next_if(|t| matches!(t.lexeme, Lexeme::True | Lexeme::False | Lexeme::Nil))
        {
            Ok(ast_terminal!(token))
        } else if let Some(_left_paren) = tokens.next_if(|t| t.lexeme == Lexeme::LeftParen) {
            let expr = self.expression(tokens)?;
            if let Some(_right_paren) = tokens.next_if(|t| t.lexeme == Lexeme::RightParen) {
                Ok(ast_group!(expr))
            } else if tokens.peek().is_some() {
                // something other than a closing ')'
                let token = tokens.next().unwrap().lexeme.clone();
                Err(ast_expected_token!(lexeme Lexeme::RightParen, token))
            } else {
                Err(ParseError::UnexpectedEof)
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
            })
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
        let expr = AstExpr::Terminal(create_token(Lexeme::True));
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
        let expr = AstExpr::Terminal(create_token(Lexeme::True));
        let group = Ast::Expression(AstExpr::Group(Box::new(expr)));
        assert_eq!(group.to_string(), "(group true)");
    }

    #[test]
    fn test_ast_display_binary_expression() {
        let left = AstExpr::Terminal(create_token(Lexeme::Number("1".to_string(), 1.0)));
        let right = AstExpr::Terminal(create_token(Lexeme::Number("2".to_string(), 2.0)));
        let op = create_token(Lexeme::Plus);
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
        let op = create_token(Lexeme::Minus);
        let unary = Ast::Expression(AstExpr::Unary {
            op,
            exp: Box::new(expr),
        });
        assert_eq!(unary.to_string(), "(- 5.0)");
    }

    #[test]
    fn test_ast_display_terminal_literals() {
        let true_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::True)));
        assert_eq!(true_ast.to_string(), "true");

        let false_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::False)));
        assert_eq!(false_ast.to_string(), "false");

        let nil_ast = Ast::Expression(AstExpr::Terminal(create_token(Lexeme::Nil)));
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
        let op = create_token(Lexeme::Plus);
        let binary = AstExpr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(binary.to_string(), "(+ 1.0 2.0)");

        let expr = AstExpr::Terminal(create_token(Lexeme::Number("5".to_string(), 5.0)));
        let op = create_token(Lexeme::Minus);
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
        assert_eq!(ast[0].to_string(), "{\n}");
    }

    #[test]
    fn test_parse_block_with_statements() {
        let parser = Parser::new("{ print 1; print 2; }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\nprint 1.0;\nprint 2.0;\n}");
    }

    #[test]
    fn test_parse_nested_blocks() {
        let parser = Parser::new("{ { print 42; } }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\n{\nprint 42.0;\n}\n}");
    }

    #[test]
    fn test_parse_block_with_variable_declaration() {
        let parser = Parser::new("{ var x = 10; print x; }", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "{\nvar x = 10.0;\nprint x;\n}");
    }

    #[test]
    fn test_parse_function_call_no_args() {
        let parser = Parser::new("foo();", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "foo([])");
    }

    #[test]
    fn test_parse_function_call_single_arg() {
        let parser = Parser::new("foo(42);", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "foo([42.0])");
    }

    #[test]
    fn test_parse_function_call_multiple_args() {
        let parser = Parser::new("foo(42, \"hello\", true);", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "foo([42.0, hello, true])");
    }

    #[test]
    fn test_parse_function_call_nested() {
        let parser = Parser::new("foo(bar(baz));", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "foo([bar([baz])])");
    }

    #[test]
    fn test_parse_function_call_complex_args() {
        let parser = Parser::new("foo(1 + 2, bar(3), \"test\");", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);
        assert_eq!(ast[0].to_string(), "foo([(+ 1.0 2.0), bar([3.0]), test])");
    }

    #[test]
    fn test_ast_display_call_expression() {
        let args = vec![
            AstExpr::Terminal(create_token(Lexeme::Number("42".to_string(), 42.0))),
            AstExpr::Terminal(create_token(Lexeme::String("hello".to_string()))),
        ];
        let call = AstExpr::Call {
            func: "foo".to_string(),
            args,
            site: Span::new(1, 0, 1),
        };
        assert_eq!(call.to_string(), "foo([42.0, hello])");
    }

    #[test]
    fn test_ast_display_call_no_args() {
        let call = AstExpr::Call {
            func: "foo".to_string(),
            args: vec![],
            site: Span::new(1, 0, 1),
        };
        assert_eq!(call.to_string(), "foo([])");
    }

    #[test]
    fn test_ast_display_nested_call() {
        let inner_args = vec![AstExpr::Terminal(create_token(Lexeme::Number(
            "5".to_string(),
            5.0,
        )))];
        let inner_call = AstExpr::Call {
            func: "bar".to_string(),
            args: inner_args,
            site: Span::new(1, 0, 1),
        };

        let outer_args = vec![inner_call];
        let outer_call = AstExpr::Call {
            func: "foo".to_string(),
            args: outer_args,
            site: Span::new(1, 0, 1),
        };

        assert_eq!(outer_call.to_string(), "foo([bar([5.0])])");
    }

    #[test]
    fn test_parse_function_call_as_expression_statement() {
        let parser = Parser::new("foo(42, \"hello\");", false, true);
        let result = parser.parse();
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 1);

        // Check that it's parsed as an expression statement
        match &ast[0] {
            Ast::Statement(AstStmt::Expression(expr)) => match expr {
                AstExpr::Call {
                    func: _,
                    args,
                    site: _,
                } => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(expr.to_string(), "foo([42.0, hello])");
                }
                _ => panic!("Expected call expression, got {expr:?}"),
            },
            _ => panic!("Expected Expression Statement, got {:?}", ast[0]),
        }
    }

    #[test]
    fn test_parse_chained_function_calls() {
        let parser = Parser::new("foo().bar().baz();", false, true);
        let result = parser.parse();
        // This should fail currently as chained calls aren't implemented
        // But the test documents expected behavior
        assert!(result.is_err());
    }
}
