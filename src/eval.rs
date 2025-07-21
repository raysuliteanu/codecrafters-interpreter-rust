use anyhow::Result;
use log::trace;
use std::collections::{HashMap, VecDeque};
use std::fmt::Display;
use thiserror::Error;

use crate::func;
use crate::model::{Ast, AstExpr, AstStmt};
use crate::model::{Lexeme, Token};
use crate::parser::Parser;
use crate::span::Span;

pub type Callable = fn(&[EvalValue]) -> EvalResult<EvalValue>;

#[derive(PartialEq, Debug, Clone)]
pub enum EvalValue {
    FunDecl(LoxFunction),
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

impl Display for EvalValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalValue::Number(v) => write!(f, "{v}"),
            EvalValue::String(s) => write!(f, "{s}"),
            EvalValue::Boolean(b) => write!(f, "{b}"),
            EvalValue::Nil => write!(f, "nil"),
            EvalValue::FunDecl(func) => write!(f, "{func}"),
        }
    }
}

#[derive(Error, Debug)]
pub enum EvalErrors {
    #[error("invalid op {} for {}\n[line {line}]", <&Lexeme as Into<String>>::into(op), val)]
    InvalidUnaryOp {
        op: Lexeme,
        val: Box<EvalValue>,
        line: usize,
    },
    #[error("invalid operation {op} {}\n[line {line}]", <&Lexeme as Into<String>>::into(op))]
    InvalidBinaryOp { op: Lexeme, line: usize },
    #[error("Operands must be two numbers or two strings.\n[line {0}]")]
    StringsOrNumbers(usize),
    #[error("Undefined variable '{0}'.\n[line {1}]")]
    UndefinedVar(String, usize),
    // #[error("bad function call")]
    // FunctionCall,
}

macro_rules! invalid_unary_op {
    ($token: expr, $val: expr) => {
        EvalErrors::InvalidUnaryOp {
            op: $token.lexeme.clone(),
            val: Box::new($val),
            line: $token.span.line(),
        }
    };
}

macro_rules! invalid_binary_op {
    ($token: expr) => {
        EvalErrors::InvalidBinaryOp {
            op: $token.lexeme.clone(),
            line: $token.span.line(),
        }
    };
}

pub type EvalResult<T> = Result<T, EvalErrors>;

#[derive(Debug, Clone, PartialEq)]
pub struct LoxFunction {
    name: String,
    params: Option<Vec<String>>,
    fn_type: LoxFunctionType,
}

impl LoxFunction {
    fn arity(&self) -> usize {
        self.params.as_ref().map_or(0, |f| f.len())
    }
}

impl Display for LoxFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<fn {}>", self.name)
    }
}

#[derive(Debug, PartialEq)]
enum LoxFunctionType {
    System(Callable),
    UserDefined(Ast),
}

impl Clone for LoxFunctionType {
    fn clone(&self) -> Self {
        match self {
            LoxFunctionType::System(callable) => LoxFunctionType::System(*callable),
            LoxFunctionType::UserDefined(ast) => LoxFunctionType::UserDefined(ast.clone()),
        }
    }
}

#[derive(Default)]
struct EvalEnv {
    vars: HashMap<String, EvalValue>,
    fns: HashMap<String, EvalValue>,
}

impl EvalEnv {
    fn new() -> Self {
        EvalEnv::default()
    }

    fn upsert_var(&mut self, id: String, initializer: Option<EvalValue>) -> Option<EvalValue> {
        let init = if let Some(ev) = initializer {
            ev
        } else {
            EvalValue::Nil
        };
        self.vars.insert(id, init)
    }

    fn lookup_var(&self, id: &str) -> Option<&EvalValue> {
        self.vars.get(id)
    }

    fn lookup_var_mut(&mut self, id: &str) -> Option<&mut EvalValue> {
        self.vars.get_mut(id)
    }

    fn add_fn(&mut self, func: LoxFunction) -> Option<EvalValue> {
        self.fns.insert(func.name.clone(), EvalValue::FunDecl(func))
    }

    fn lookup_lox_fn(&self, id: String) -> Option<&EvalValue> {
        self.fns.get(&id)
    }
}

type Stack<T> = VecDeque<T>;

struct EvalState {
    env: Stack<EvalEnv>,
}

impl EvalState {
    fn new() -> Self {
        let mut global_env = EvalEnv::new();
        global_env.add_fn(LoxFunction {
            name: "clock".to_string(),
            params: None,
            fn_type: LoxFunctionType::System(func::clock),
        });

        let mut env = Stack::new();
        env.push_front(global_env);

        EvalState { env }
    }

    fn add_lox_fn(&mut self, func: LoxFunction) -> Option<EvalValue> {
        self.env
            .front_mut()
            .expect("always at least a global env")
            .add_fn(func)
    }

    fn lookup_lox_fn(&self, id: &str) -> Option<&EvalValue> {
        for env in &self.env {
            if env.fns.contains_key(id) {
                return env.lookup_lox_fn(id.to_string());
            }
        }

        None
    }

    fn add_var(&mut self, id: String, initializer: Option<EvalValue>) -> Option<EvalValue> {
        self.env
            .front_mut()
            .expect("always at least a global env")
            .upsert_var(id, initializer)
    }

    fn var_value(&self, id: &str) -> Option<&EvalValue> {
        for env in &self.env {
            if env.vars.contains_key(id) {
                return env.lookup_var(id);
            }
        }

        None
    }

    fn var_value_mut(&mut self, id: &str) -> Option<&mut EvalValue> {
        for env in &mut self.env {
            if env.vars.contains_key(id) {
                return env.lookup_var_mut(id);
            }
        }

        None
    }

    fn var_exists(&self, id: &str) -> bool {
        for env in &self.env {
            if env.vars.contains_key(id) {
                return true;
            }
        }

        false
    }

    fn push(&mut self) {
        self.env.push_front(EvalEnv::new());
    }

    fn pop(&mut self) {
        self.env.pop_front();
    }
}

pub struct Eval<'eval> {
    source: &'eval str,
    expression_mode: bool,
    state: EvalState,
}

impl<'eval> Eval<'_> {
    pub fn new(source: &str, expression_mode: bool) -> Eval {
        Eval {
            source,
            expression_mode,
            state: EvalState::new(),
        }
    }

    pub fn evaluate(&mut self) -> anyhow::Result<EvalValue> {
        let parser = Parser::new(self.source, self.expression_mode, false);
        let tree = parser.parse()?;

        match self.eval(tree.iter()) {
            Ok(v) => Ok(v),
            Err(e) => {
                eprintln!("{e}");
                Err(e.into())
            }
        }
    }

    fn eval(&mut self, tree: std::slice::Iter<'_, Ast>) -> EvalResult<EvalValue> {
        let mut value = EvalValue::Nil;
        for ast in tree {
            value = self.eval_ast(ast)?;
            trace!("eval = {value}");
        }

        Ok(value)
    }

    fn eval_ast(&mut self, ast: &'eval Ast) -> EvalResult<EvalValue> {
        trace!("eval_ast");
        match ast {
            Ast::Class => todo!("class decl"),
            Ast::Function { name, params, body } => self.eval_fun_decl(name, params, body),
            Ast::Variable { name, initializer } => self.eval_var_decl(name, initializer),
            Ast::Statement(stmt) => self.eval_stmt(stmt),
            Ast::Block(block) => self.eval_block(block),
            Ast::Expression(e) => self.eval_expr(e),
        }
    }

    fn eval_stmt(&mut self, stmt: &'eval AstStmt) -> EvalResult<EvalValue> {
        trace!("eval_stmt");
        match stmt {
            AstStmt::Expression(expr) => self.eval_expr(expr),
            AstStmt::Print(ast) => self.eval_print_stmt(ast),
            AstStmt::If {
                condition,
                then,
                or_else,
            } => self.eval_if_stmt(condition, then, or_else),
            AstStmt::Return(_ast) => todo!("return stmts"),
            AstStmt::While(cond, body) => self.eval_while(cond, body),
        }
    }

    fn eval_expr(&mut self, expr: &'eval AstExpr) -> EvalResult<EvalValue> {
        trace!("eval_expr");
        match expr {
            AstExpr::Terminal(token) => self.eval_terminal(token),
            AstExpr::Group(expr) => self.eval_expr(expr),
            AstExpr::Unary { op, exp } => self.eval_unary(op, exp),
            AstExpr::Binary { op, left, right } => self.eval_binary(op, left, right),
            AstExpr::Assignment { id, expr } => self.eval_assignment(id, expr),
            AstExpr::Call { func, args, site } => self.eval_call(func, args, site),
            AstExpr::Logical { op, left, right } => self.eval_logical(op, left, right),
        }
    }

    fn eval_print_stmt(&mut self, expr: &'eval AstExpr) -> EvalResult<EvalValue> {
        trace!("eval_print");
        let val = self.eval_expr(expr)?;
        trace!("print = {val}");
        println!("{val}");
        Ok(EvalValue::Nil)
    }

    fn eval_terminal(&self, token: &'eval Token) -> EvalResult<EvalValue> {
        trace!("eval_terminal");
        let val = match &token.lexeme {
            Lexeme::Number(_, v) => EvalValue::Number(*v),
            Lexeme::String(s) => EvalValue::String(s.to_string()),
            Lexeme::Identifier(id) => {
                if let Some(value) = self.eval_identifier(id) {
                    value.clone()
                } else {
                    return Err(EvalErrors::UndefinedVar(id.clone(), token.span.line()));
                }
            }
            Lexeme::True => EvalValue::Boolean(true),
            Lexeme::False => EvalValue::Boolean(false),
            Lexeme::Nil => EvalValue::Nil,
            _ => unimplemented!("{}", token.lexeme),
        };

        Ok(val)
    }

    fn eval_unary(&mut self, op: &Token, expr: &AstExpr) -> EvalResult<EvalValue> {
        trace!("eval_unary");
        let val = self.eval_expr(expr)?;
        let result = match op.lexeme {
            Lexeme::Bang => match val {
                EvalValue::Number(_) => EvalValue::Boolean(false),
                EvalValue::Boolean(v) => EvalValue::Boolean(!v),
                EvalValue::Nil => EvalValue::Boolean(true),
                _ => return Err(invalid_unary_op!(op, val)),
            },
            Lexeme::Minus => match val {
                EvalValue::Number(v) => EvalValue::Number(-v),
                _ => return Err(invalid_unary_op!(op, val)),
            },
            _ => return Err(invalid_unary_op!(op, val)),
        };

        Ok(result)
    }

    fn eval_binary(
        &mut self,
        op: &Token,
        left: &AstExpr,
        right: &AstExpr,
    ) -> EvalResult<EvalValue> {
        trace!("eval_binary");
        let left_expr = self.eval_expr(left)?;
        let right_expr = self.eval_expr(right)?;
        let result = match op.lexeme {
            Lexeme::Plus => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l + r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::String(l + &r),
                /* TODO: I think this should be valid but not for CC
                    (EvalValue::String(l), EvalValue::Number(r)) => {
                        EvalValue::String(l + &r.to_string())
                    }
                    (EvalValue::Number(l), EvalValue::String(r)) => {
                        EvalValue::String(l.to_string() + &r)
                    }
                */
                _ => return Err(EvalErrors::StringsOrNumbers(op.span.line())),
            },
            Lexeme::Minus => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l - r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::Star => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l * r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::Slash => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l / r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::EqEq => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l == r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::Boolean(l == r),
                (EvalValue::Boolean(l), EvalValue::Boolean(r)) => EvalValue::Boolean(l == r),
                _ => EvalValue::Boolean(false),
            },
            Lexeme::BangEq => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l != r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::Boolean(l != r),
                (EvalValue::Boolean(l), EvalValue::Boolean(r)) => EvalValue::Boolean(l != r),
                _ => EvalValue::Boolean(true),
            },
            Lexeme::Less => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l < r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::LessEq => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l <= r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::Greater => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l > r),
                _ => return Err(invalid_binary_op!(op)),
            },
            Lexeme::GreaterEq => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l >= r),
                _ => return Err(invalid_binary_op!(op)),
            },
            _ => return Err(invalid_binary_op!(op)),
        };

        Ok(result)
    }

    // var some_var [= expr] ;
    fn eval_var_decl(
        &mut self,
        token: &Token,
        ast: &Option<Box<AstExpr>>,
    ) -> EvalResult<EvalValue> {
        let initializer = if let Some(expr) = ast {
            Some(self.eval_expr(expr)?)
        } else {
            None
        };

        if let Lexeme::Identifier(id) = &token.lexeme {
            self.state.add_var(id.clone(), initializer);
        } else {
            panic!("invalid token {token}");
        }

        Ok(EvalValue::Nil)
    }

    fn eval_identifier(&self, id: &str) -> Option<&EvalValue> {
        if let Some(val) = self.state.var_value(id) {
            Some(val)
        } else if let Some(func) = self.state.lookup_lox_fn(id) {
            Some(func)
        } else {
            None
        }
    }

    // some_var = expr
    fn eval_assignment(&mut self, id: &str, expr: &AstExpr) -> EvalResult<EvalValue> {
        if !self.state.var_exists(id) {
            Err(EvalErrors::UndefinedVar(id.to_string(), 0))
        } else {
            let new_val = self.eval_expr(expr)?;
            let val = self
                .state
                .var_value_mut(id)
                .expect("already checked the var exists");
            *val = new_val;
            Ok(val.clone())
        }
    }

    fn eval_logical(
        &mut self,
        op: &Token,
        left: &AstExpr,
        right: &AstExpr,
    ) -> EvalResult<EvalValue> {
        trace!("eval_logical: {}", op.lexeme);
        let left_val = self.eval_expr(left)?;
        trace!("left = {left_val}");
        match op.lexeme {
            Lexeme::Or if Eval::is_truthy(&left_val) => {
                trace!("OR returning {left_val}");
                Ok(left_val)
            }
            Lexeme::And if !Eval::is_truthy(&left_val) => {
                trace!("AND returning {left_val}");
                Ok(left_val)
            }
            _ => {
                trace!("evaluating right expr of {}", op.lexeme);
                self.eval_expr(right)
            }
        }
    }

    fn is_truthy(val: &EvalValue) -> bool {
        match val {
            EvalValue::Boolean(b) => *b,
            EvalValue::Nil => false,
            _ => true,
        }
    }

    fn eval_block(&mut self, block: &[Ast]) -> EvalResult<EvalValue> {
        trace!("eval_block");
        self.state.push();

        // Use a closure to ensure pop() is always called
        let result = (|| {
            let mut value = EvalValue::Nil;
            for ast in block {
                value = self.eval_ast(ast)?;
            }
            Ok(value)
        })();

        self.state.pop();
        result

        // Alternative approach using RAII guard pattern for future reference:
        //
        // self.state.push();
        //
        // // Create a guard that will pop on drop
        // struct ScopeGuard<'a>(&'a mut EvalState);
        // impl Drop for ScopeGuard<'_> {
        //     fn drop(&mut self) {
        //         self.0.pop();
        //     }
        // }
        // let _guard = ScopeGuard(&mut self.state);
        //
        // let mut result = EvalValue::Nil;
        // for ast in block {
        //     result = self.eval_ast(&ast)?;
        // }
        // Ok(result)
    }

    fn eval_if_stmt(
        &mut self,
        cond: &AstExpr,
        then_block: &Ast,
        else_block: &Option<Box<Ast>>,
    ) -> EvalResult<EvalValue> {
        trace!("eval_if");
        let cond_result = self.eval_expr(cond)?;
        if Eval::is_truthy(&cond_result) {
            trace!("eval_if:then");
            match then_block {
                Ast::Block(asts) => self.eval_block(asts),
                Ast::Statement(ast_stmt) => self.eval_stmt(ast_stmt),
                _ => todo!("then block not block or statement"),
            }
        } else if let Some(ast) = else_block {
            trace!("eval_if:else");
            match ast.as_ref() {
                Ast::Block(block) => self.eval_block(block),
                Ast::Statement(stmt) => self.eval_stmt(stmt),
                _ => todo!("then block not block or statement"),
            }
        } else {
            Ok(EvalValue::Nil)
        }
    }

    fn eval_while(&mut self, cond: &AstExpr, body: &Ast) -> EvalResult<EvalValue> {
        trace!("eval_while");
        loop {
            if Eval::is_truthy(&self.eval_expr(cond)?) {
                match body {
                    Ast::Block(asts) => self.eval_block(asts)?,
                    Ast::Statement(ast_stmt) => self.eval_stmt(ast_stmt)?,
                    _ => todo!("then block not block or statement"),
                }
            } else {
                break;
            };
        }

        Ok(EvalValue::Nil)
    }

    fn eval_call(&mut self, func: &str, args: &[AstExpr], site: &Span) -> EvalResult<EvalValue> {
        let lox_func = match self.state.lookup_lox_fn(func) {
            Some(value) => match value {
                EvalValue::FunDecl(lox_function) => lox_function.clone(),
                _ => todo!(),
            },
            None => todo!("no such function at {site}"),
        };

        let val = if lox_func.arity() == args.len() {
            self.do_fn_call(&lox_func, args)?
        } else {
            todo!("arity mismatch");
        };

        Ok(val)
    }

    fn do_fn_call(&mut self, lox_func: &LoxFunction, args: &[AstExpr]) -> EvalResult<EvalValue> {
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            let val = self.eval_expr(arg)?;
            vals.push(val.clone());
        }

        match &lox_func.fn_type {
            LoxFunctionType::System(system) => system(&vals),
            LoxFunctionType::UserDefined(body) => {
                self.state.push();

                lox_func
                    .params
                    .iter()
                    .flatten()
                    .zip(vals)
                    .for_each(|(p, i)| {
                        self.state.add_var(p.to_string(), Some(i));
                    });

                let result = self.eval_ast(body);

                self.state.pop();

                result
            }
        }
    }

    fn eval_fun_decl(
        &mut self,
        name: &str,
        params: &[String],
        body: &Ast,
    ) -> EvalResult<EvalValue> {
        self.state.add_lox_fn(LoxFunction {
            name: name.to_string(),
            params: Some(params.to_vec()),
            fn_type: LoxFunctionType::UserDefined((*body).clone()),
        });
        Ok(EvalValue::Nil)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;
    use crate::util::print_ast;

    #[test]
    fn test_eval_value_display() {
        assert_eq!(format!("{}", EvalValue::Number(42.0)), "42");
        assert_eq!(format!("{}", EvalValue::Number(1.23)), "1.23");
        assert_eq!(
            format!("{}", EvalValue::String("hello".to_string())),
            "hello"
        );
        assert_eq!(format!("{}", EvalValue::Boolean(true)), "true");
        assert_eq!(format!("{}", EvalValue::Boolean(false)), "false");
        assert_eq!(format!("{}", EvalValue::Nil), "nil");
    }

    #[test]
    fn test_eval_value_equality() {
        assert_eq!(EvalValue::Number(42.0), EvalValue::Number(42.0));
        assert_eq!(
            EvalValue::String("test".to_string()),
            EvalValue::String("test".to_string())
        );
        assert_eq!(EvalValue::Boolean(true), EvalValue::Boolean(true));
        assert_eq!(EvalValue::Nil, EvalValue::Nil);

        assert_ne!(EvalValue::Number(42.0), EvalValue::Number(43.0));
        assert_ne!(
            EvalValue::String("test".to_string()),
            EvalValue::String("other".to_string())
        );
        assert_ne!(EvalValue::Boolean(true), EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_new() {
        let source = "print 42;";
        let eval = Eval::new(source, false);
        assert_eq!(eval.source, source);
    }

    #[test]
    fn test_eval_terminal_number() {
        let eval = Eval::new("", false);
        let token = Token {
            lexeme: Lexeme::Number("42".to_string(), 42.0),
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&token).unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_terminal_string() {
        let eval = Eval::new("", false);
        let token = Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&token).unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_eval_terminal_boolean() {
        let eval = Eval::new("", false);

        let true_token = Token {
            lexeme: Lexeme::True,
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&true_token).unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let false_token = Token {
            lexeme: Lexeme::False,
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&false_token).unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_terminal_nil() {
        let eval = Eval::new("", false);
        let token = Token {
            lexeme: Lexeme::Nil,
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&token).unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_unary_bang() {
        let mut eval = Eval::new("", false);
        let bang_token = Token {
            lexeme: Lexeme::Bang,
            span: Span::new(0, 0, 1),
        };

        let true_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::True,
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &true_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(false));

        let false_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::False,
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &false_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let nil_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Nil,
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &nil_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("42".to_string(), 42.0),
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &number_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_unary_minus() {
        let mut eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus,
            span: Span::new(0, 0, 1),
        };

        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("42".to_string(), 42.0),
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&minus_token, &number_expr).unwrap();
        assert_eq!(result, EvalValue::Number(-42.0));
    }

    #[test]
    fn test_eval_binary_plus_numbers() {
        let mut eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus,
            span: Span::new(0, 0, 1),
        };

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval
            .eval_binary(&plus_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(15.0));
    }

    #[test]
    fn test_eval_binary_plus_strings() {
        let mut eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus,
            span: Span::new(0, 0, 1),
        };

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String(" world".to_string()),
            span: Span::new(0, 0, 1),
        });

        let result = eval
            .eval_binary(&plus_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::String("hello world".to_string()));
    }

    #[test]
    fn test_eval_binary_arithmetic() {
        let mut eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let minus_token = Token {
            lexeme: Lexeme::Minus,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&minus_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(5.0));

        let star_token = Token {
            lexeme: Lexeme::Star,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&star_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(50.0));

        let slash_token = Token {
            lexeme: Lexeme::Slash,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&slash_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(2.0));
    }

    #[test]
    fn test_eval_binary_equality() {
        let mut eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let eq_token = Token {
            lexeme: Lexeme::EqEq,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&eq_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let ne_token = Token {
            lexeme: Lexeme::BangEq,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&ne_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_binary_comparison() {
        let mut eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let greater_token = Token {
            lexeme: Lexeme::Greater,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&greater_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let greater_eq_token = Token {
            lexeme: Lexeme::GreaterEq,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&greater_eq_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let less_token = Token {
            lexeme: Lexeme::Less,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&less_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(false));

        let less_eq_token = Token {
            lexeme: Lexeme::LessEq,
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&less_eq_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_evaluate_simple_expression() {
        let mut eval = Eval::new("42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0)); // Returns default from eval method
    }

    #[test]
    fn test_evaluate_print_statement() {
        let mut eval = Eval::new("print 42;", false);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_evaluate_complex_expression() {
        let mut eval = Eval::new("1 + 2 * 3", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(7.0)); // Returns default from eval method
    }

    #[test]
    fn test_eval_errors_display() {
        let error1 = EvalErrors::InvalidUnaryOp {
            op: Lexeme::Bang,
            val: Box::new(EvalValue::String("test".to_string())),
            line: 1,
        };
        assert_eq!(format!("{error1}"), "invalid op ! for test\n[line 1]");

        let error2 = EvalErrors::InvalidBinaryOp {
            op: Lexeme::Plus,
            line: 2,
        };
        assert_eq!(format!("{error2}"), "invalid operation PLUS +\n[line 2]");

        let error3 = EvalErrors::StringsOrNumbers(3);
        assert_eq!(
            format!("{error3}"),
            "Operands must be two numbers or two strings.\n[line 3]"
        );
    }

    #[test]
    fn test_eval_unary_error_invalid_operator() {
        let mut eval = Eval::new("", false);
        let invalid_token = Token {
            lexeme: Lexeme::Plus,
            span: Span::new(0, 0, 1),
        };

        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("42".to_string(), 42.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_unary(&invalid_token, &number_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid op"));
    }

    #[test]
    fn test_eval_unary_error_invalid_type() {
        let mut eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus,
            span: Span::new(0, 0, 1),
        };

        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_unary(&minus_token, &string_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid op - for hello"));
    }

    #[test]
    fn test_eval_unary_bang_invalid_type() {
        let mut eval = Eval::new("", false);
        let bang_token = Token {
            lexeme: Lexeme::Bang,
            span: Span::new(0, 0, 1),
        };

        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_unary(&bang_token, &string_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid op ! for hello"));
    }

    #[test]
    fn test_eval_binary_error_invalid_operator() {
        let mut eval = Eval::new("", false);
        let invalid_token = Token {
            lexeme: Lexeme::Bang,
            span: Span::new(0, 0, 1),
        };

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_binary(&invalid_token, &left_expr, &right_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid operation"));
    }

    #[test]
    fn test_eval_binary_plus_mixed_types_error() {
        let mut eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus,
            span: Span::new(0, 0, 1),
        };
        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("42".to_string(), 42.0),
            span: Span::new(0, 0, 1),
        });
        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_binary(&plus_token, &number_expr, &string_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("Operands must be two numbers or two strings"));
    }

    #[test]
    fn test_eval_binary_arithmetic_invalid_types() {
        let mut eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus,
            span: Span::new(0, 0, 1),
        };

        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });
        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_binary(&minus_token, &string_expr, &number_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid operation"));
    }

    #[test]
    fn test_eval_binary_comparison_invalid_types() {
        let mut eval = Eval::new("", false);
        let greater_token = Token {
            lexeme: Lexeme::Greater,
            span: Span::new(0, 0, 1),
        };

        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });
        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_binary(&greater_token, &string_expr, &number_expr);
        assert!(result.is_err());

        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("invalid operation"));
    }

    #[test]
    fn test_eval_binary_equality_different_types() {
        let mut eval = Eval::new("", false);
        let eq_token = Token {
            lexeme: Lexeme::EqEq,
            span: Span::new(0, 0, 1),
        };

        let string_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::String("hello".to_string()),
            span: Span::new(0, 0, 1),
        });

        let number_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let result = eval.eval_binary(&eq_token, &string_expr, &number_expr);
        assert!(result.is_ok_and(|val| val == EvalValue::Boolean(false)));
    }

    #[test]
    fn test_eval_binary_division_by_zero() {
        let mut eval = Eval::new("", false);
        let slash_token = Token {
            lexeme: Lexeme::Slash,
            span: Span::new(0, 0, 1),
        };

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("0".to_string(), 0.0),
            span: Span::new(0, 0, 1),
        });

        // Division by zero should return infinity in Rust, not an error
        let result = eval
            .eval_binary(&slash_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(f64::INFINITY));
    }

    #[test]
    fn test_eval_variable_declaration_without_initializer() {
        let mut eval = Eval::new("var x;", false);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_variable_declaration_with_initializer() {
        let mut eval = Eval::new("var x = 42;", false);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_variable_usage() {
        let mut eval = Eval::new("var x = 42; print x;", false);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_assignment_expression() {
        let mut eval = Eval::new("var x = 10; x = 20;", false);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(20.0));
    }

    #[test]
    fn test_eval_assignment_undefined_variable() {
        let mut eval = Eval::new("x = 10;", false);
        let result = eval.evaluate();
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_logical_or_short_circuit() {
        let mut eval = Eval::new("true or false", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_and_short_circuit() {
        let mut eval = Eval::new("false and true", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_logical_or_no_short_circuit() {
        let mut eval = Eval::new("false or true", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_and_no_short_circuit() {
        let mut eval = Eval::new("true and false", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_logical_or_with_numbers() {
        let mut eval = Eval::new("0 or 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(0.0));
    }

    #[test]
    fn test_eval_logical_or_with_truthy_number() {
        let mut eval = Eval::new("42 or 0", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_or_with_strings() {
        let mut eval = Eval::new("\"\" or \"hello\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("".to_string()));
    }

    #[test]
    fn test_eval_logical_or_with_truthy_string() {
        let mut eval = Eval::new("\"hello\" or \"world\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_eval_logical_or_with_nil() {
        let mut eval = Eval::new("nil or 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_or_with_truthy_nil() {
        let mut eval = Eval::new("42 or nil", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_or_both_falsy() {
        let mut eval = Eval::new("nil or false", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_logical_and_with_numbers() {
        let mut eval = Eval::new("42 and 0", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(0.0));
    }

    #[test]
    fn test_eval_logical_and_with_falsy_number() {
        let mut eval = Eval::new("0 and 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_and_with_strings() {
        let mut eval = Eval::new("\"hello\" and \"world\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("world".to_string()));
    }

    #[test]
    fn test_eval_logical_and_with_empty_string() {
        let mut eval = Eval::new("\"\" and \"hello\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_eval_logical_and_with_nil() {
        let mut eval = Eval::new("nil and 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_logical_and_with_truthy_nil() {
        let mut eval = Eval::new("42 and nil", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_logical_and_with_false() {
        let mut eval = Eval::new("false and 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_logical_and_both_truthy() {
        let mut eval = Eval::new("42 and \"hello\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_is_truthy_function() {
        assert!(Eval::is_truthy(&EvalValue::Boolean(true)));
        assert!(!Eval::is_truthy(&EvalValue::Boolean(false)));
        assert!(!Eval::is_truthy(&EvalValue::Nil));
        assert!(Eval::is_truthy(&EvalValue::Number(42.0)));
        assert!(Eval::is_truthy(&EvalValue::Number(0.0)));
        assert!(Eval::is_truthy(&EvalValue::Number(-1.0)));
        assert!(Eval::is_truthy(&EvalValue::String("hello".to_string())));
        assert!(Eval::is_truthy(&EvalValue::String("".to_string())));
    }

    #[test]
    fn test_eval_logical_chained_or() {
        let mut eval = Eval::new("false or nil or \"hello\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_eval_logical_chained_and() {
        let mut eval = Eval::new("true and 42 and \"hello\"", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::String("hello".to_string()));
    }

    #[test]
    fn test_eval_logical_mixed_operations() {
        let mut eval = Eval::new("false or true and 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_mixed_operations_with_parentheses() {
        let mut eval = Eval::new("(false or true) and 42", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_complex_expression() {
        let mut eval = Eval::new("nil or false or (true and 42)", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(42.0));
    }

    #[test]
    fn test_eval_logical_with_equality() {
        let mut eval = Eval::new("5 == 5 or false", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_with_comparison() {
        let mut eval = Eval::new("5 > 3 and 10 < 20", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_or_short_circuit_no_evaluation() {
        let mut eval = Eval::new("var x = 5; true or (x = 10)", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(true));
    }

    #[test]
    fn test_eval_logical_and_short_circuit_no_evaluation() {
        let mut eval = Eval::new("var x = 5; false and (x = 10)", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_logical_or_evaluates_right_side() {
        let mut eval = Eval::new("var x = 5; nil or x", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(5.0));
    }

    #[test]
    fn test_eval_logical_and_evaluates_right_side() {
        let mut eval = Eval::new("var x = 5; true and x", true);
        let result = eval.evaluate().unwrap();
        assert_eq!(result, EvalValue::Number(5.0));
    }

    #[test]
    fn test_print_ast_simple() {
        let source = "var x = 42; print x;";
        let parser = Parser::new(source, false, false);
        let ast = parser.parse().unwrap();

        // This would print to stdout, so we just verify it doesn't panic
        print_ast(&ast);
    }

    #[test]
    fn test_print_ast_complex() {
        let source = r#"
        var a = 10;
        if (a > 5) {
            print "large";
        } else {
            print "small";
        }
        "#;
        let parser = Parser::new(source, false, false);
        let ast = parser.parse().unwrap();

        // This would print to stdout, so we just verify it doesn't panic
        print_ast(&ast);
    }
}
