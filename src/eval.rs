use anyhow::Result;
use log::trace;
use std::collections::HashMap;
use std::fmt::Display;
use thiserror::Error;

use crate::model::{Ast, AstExpr, AstStmt};
use crate::parser::Parser;
use crate::token::{Lexeme, Token};

#[derive(PartialEq, Debug, Clone)]
pub enum EvalValue {
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
        }
    }
}

#[derive(Error, Debug)]
pub enum EvalErrors {
    #[error("invalid op {} for {}", <&Lexeme as Into<String>>::into(op), val)]
    InvalidUnaryOp { op: Lexeme, val: EvalValue },
    #[error("invalid operation {} {}", op, <&Lexeme as Into<String>>::into(op))]
    InvalidBinaryOp { op: Lexeme },
    #[error("Operands must be two numbers or two strings.")]
    StringsOrNumbers,
}

pub type EvalResult = Result<EvalValue>;

struct EvalEnv {
    vars: HashMap<String, Option<EvalValue>>,
}

impl EvalEnv {
    fn insert_var(
        &mut self,
        id: String,
        initializer: Option<EvalValue>,
    ) -> Option<Option<EvalValue>> {
        self.vars.insert(id, initializer)
    }

    fn lookup_var(&self, id: &str) -> Option<EvalValue> {
        let var = self.vars.get(id);
        if let Some(Some(ev)) = var {
            Some(ev.clone())
        } else {
            None
        }
    }
}

pub struct Eval<'eval> {
    source: &'eval str,
    expression_mode: bool,
    state: EvalEnv,
}

impl<'eval> Eval<'_> {
    pub fn new(source: &str, expression_mode: bool) -> Eval {
        Eval {
            source,
            expression_mode,
            state: EvalEnv {
                vars: HashMap::new(),
            },
        }
    }

    pub fn evaluate(&mut self) -> EvalResult {
        let parser = Parser::new(self.source, self.expression_mode, false);
        let tree = parser.parse()?;

        self.eval(tree.iter())
    }

    fn eval(&mut self, tree: std::slice::Iter<'_, Ast>) -> EvalResult {
        let mut value = EvalValue::Nil;
        for ast in tree {
            value = self.eval_ast(ast)?;
            trace!("eval = {value}");
        }

        Ok(value)
    }

    fn eval_ast(&mut self, ast: &'eval Ast) -> EvalResult {
        trace!("eval_ast");
        match ast {
            Ast::Class => todo!("class decl"),
            Ast::Function => todo!("fun decl"),
            Ast::Variable(token, ast) => self.eval_var_decl(token, ast),
            Ast::Statement(stmt) => self.eval_stmt(stmt),
            Ast::Block(_b) => todo!("block decl"),
            Ast::Expression(e) => self.eval_expr(e),
        }
    }

    fn eval_stmt(&self, stmt: &'eval AstStmt) -> EvalResult {
        trace!("eval_stmt");
        match stmt {
            AstStmt::Expression(expr) => self.eval_expr(expr),
            AstStmt::Print(ast) => self.eval_print_stmt(ast),
            AstStmt::For => todo!(),
            AstStmt::If(_ast, _ast1, _ast2) => todo!(),
            AstStmt::Return(_ast) => todo!(),
            AstStmt::While(_ast, _ast1) => todo!(),
        }
    }

    fn eval_expr(&self, expr: &'eval AstExpr) -> EvalResult {
        trace!("eval_expr");
        match expr {
            AstExpr::Terminal(token) => self.eval_terminal(token),
            AstExpr::Group(expr) => self.eval_expr(expr),
            AstExpr::Unary { op, exp } => self.eval_unary(op, exp),
            AstExpr::Binary { op, left, right } => self.eval_binary(op, left, right),
            AstExpr::Assignment { id: _, expr: _ } => todo!("assignment"),
            AstExpr::Logical {
                op: _,
                left: _,
                right: _,
            } => todo!("logical expr"),
        }
    }

    fn eval_print_stmt(&self, expr: &'eval AstExpr) -> EvalResult {
        trace!("eval_print");
        let val = self.eval_expr(expr)?;
        trace!("print = {val}");
        println!("{val}");
        Ok(EvalValue::Nil)
    }

    fn eval_terminal(&self, token: &'eval Token) -> EvalResult {
        trace!("eval_terminal");
        let val = match &token.lexeme {
            Lexeme::Number(_, v) => EvalValue::Number(*v),
            Lexeme::String(s) => EvalValue::String(s.to_string()),
            Lexeme::Identifier(id) => self.eval_identifier(id),
            Lexeme::True(_) => EvalValue::Boolean(true),
            Lexeme::False(_) => EvalValue::Boolean(false),
            Lexeme::Nil(_) => EvalValue::Nil,
            _ => unimplemented!("{}", token.lexeme),
        };

        Ok(val)
    }

    fn eval_unary(&self, op: &Token, expr: &AstExpr) -> EvalResult {
        trace!("eval_unary");
        let val = self.eval_expr(expr)?;
        let result = match op.lexeme {
            Lexeme::Bang(_) => match val {
                EvalValue::Number(_) => EvalValue::Boolean(false),
                EvalValue::Boolean(v) => EvalValue::Boolean(!v),
                EvalValue::Nil => EvalValue::Boolean(true),
                _ => {
                    return Err(EvalErrors::InvalidUnaryOp {
                        op: op.lexeme.clone(),
                        val,
                    }
                    .into());
                }
            },
            Lexeme::Minus(_) => match val {
                EvalValue::Number(v) => EvalValue::Number(-v),
                _ => {
                    return Err(EvalErrors::InvalidUnaryOp {
                        op: op.lexeme.clone(),
                        val,
                    }
                    .into());
                }
            },
            _ => {
                return Err(EvalErrors::InvalidUnaryOp {
                    op: op.lexeme.clone(),
                    val,
                }
                .into());
            }
        };

        Ok(result)
    }

    fn eval_binary(&self, op: &Token, left: &AstExpr, right: &AstExpr) -> EvalResult {
        trace!("eval_binary");
        let left_expr = self.eval_expr(left)?;
        let right_expr = self.eval_expr(right)?;
        let result = match op.lexeme {
            Lexeme::Plus(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l + r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::String(l + &r),
                _ => return Err(EvalErrors::StringsOrNumbers.into()),
            },
            Lexeme::Minus(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l - r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::Star(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l * r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::Slash(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Number(l / r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::EqEq(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l == r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::Boolean(l == r),
                (EvalValue::Boolean(l), EvalValue::Boolean(r)) => EvalValue::Boolean(l == r),
                _ => EvalValue::Boolean(false),
            },
            Lexeme::BangEq(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l != r),
                (EvalValue::String(l), EvalValue::String(r)) => EvalValue::Boolean(l != r),
                (EvalValue::Boolean(l), EvalValue::Boolean(r)) => EvalValue::Boolean(l != r),
                _ => EvalValue::Boolean(true),
            },
            Lexeme::Less(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l < r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::LessEq(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l <= r),

                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::Greater(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l > r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            Lexeme::GreaterEq(_) => match (left_expr, right_expr) {
                (EvalValue::Number(l), EvalValue::Number(r)) => EvalValue::Boolean(l >= r),
                _ => {
                    return Err(EvalErrors::InvalidBinaryOp {
                        op: op.lexeme.clone(),
                    }
                    .into());
                }
            },
            _ => {
                return Err(EvalErrors::InvalidBinaryOp {
                    op: op.lexeme.clone(),
                }
                .into());
            }
        };

        Ok(result)
    }

    fn eval_var_decl(&mut self, token: &Token, ast: &Option<Box<AstExpr>>) -> EvalResult {
        let initializer = if let Some(expr) = ast {
            Some(self.eval_expr(expr)?)
        } else {
            None
        };

        if let Lexeme::Identifier(id) = &token.lexeme {
            self.state.insert_var(id.clone(), initializer);
        } else {
            panic!("invalid token {token}");
        }

        Ok(EvalValue::Nil)
    }

    fn eval_identifier(&self, id: &str) -> EvalValue {
        if let Some(value) = self.state.lookup_var(id) {
            value
        } else {
            EvalValue::Nil
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;

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
            lexeme: Lexeme::True("true".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&true_token).unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let false_token = Token {
            lexeme: Lexeme::False("false".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&false_token).unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_terminal_nil() {
        let eval = Eval::new("", false);
        let token = Token {
            lexeme: Lexeme::Nil("nil".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval.eval_terminal(&token).unwrap();
        assert_eq!(result, EvalValue::Nil);
    }

    #[test]
    fn test_eval_unary_bang() {
        let eval = Eval::new("", false);
        let bang_token = Token {
            lexeme: Lexeme::Bang('!'),
            span: Span::new(0, 0, 1),
        };

        let true_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::True("true".to_string()),
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &true_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(false));

        let false_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::False("false".to_string()),
            span: Span::new(0, 0, 1),
        });
        let result = eval.eval_unary(&bang_token, &false_expr).unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let nil_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Nil("nil".to_string()),
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
        let eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus('-'),
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
        let eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus('+'),
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
        let eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus('+'),
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
        let eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let minus_token = Token {
            lexeme: Lexeme::Minus('-'),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&minus_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(5.0));

        let star_token = Token {
            lexeme: Lexeme::Star('*'),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&star_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(50.0));

        let slash_token = Token {
            lexeme: Lexeme::Slash('/'),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&slash_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Number(2.0));
    }

    #[test]
    fn test_eval_binary_equality() {
        let eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let eq_token = Token {
            lexeme: Lexeme::EqEq("==".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&eq_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let ne_token = Token {
            lexeme: Lexeme::BangEq("!=".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&ne_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(false));
    }

    #[test]
    fn test_eval_binary_comparison() {
        let eval = Eval::new("", false);

        let left_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("10".to_string(), 10.0),
            span: Span::new(0, 0, 1),
        });
        let right_expr = AstExpr::Terminal(Token {
            lexeme: Lexeme::Number("5".to_string(), 5.0),
            span: Span::new(0, 0, 1),
        });

        let greater_token = Token {
            lexeme: Lexeme::Greater('>'),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&greater_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let greater_eq_token = Token {
            lexeme: Lexeme::GreaterEq(">=".to_string()),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&greater_eq_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(true));

        let less_token = Token {
            lexeme: Lexeme::Less('<'),
            span: Span::new(0, 0, 1),
        };
        let result = eval
            .eval_binary(&less_token, &left_expr, &right_expr)
            .unwrap();
        assert_eq!(result, EvalValue::Boolean(false));

        let less_eq_token = Token {
            lexeme: Lexeme::LessEq("<=".to_string()),
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
        let mut eval = Eval::new("print 42;", true);
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
            op: Lexeme::Bang('!'),
            val: EvalValue::String("test".to_string()),
        };
        assert_eq!(format!("{error1}"), "invalid op ! for test");

        let error2 = EvalErrors::InvalidBinaryOp {
            op: Lexeme::Plus('+'),
        };
        assert_eq!(format!("{error2}"), "invalid operation PLUS +");

        let error3 = EvalErrors::StringsOrNumbers;
        assert_eq!(
            format!("{error3}"),
            "Operands must be two numbers or two strings."
        );
    }

    #[test]
    fn test_eval_unary_error_invalid_operator() {
        let eval = Eval::new("", false);
        let invalid_token = Token {
            lexeme: Lexeme::Plus('+'),
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
        let eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus('-'),
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
        let eval = Eval::new("", false);
        let bang_token = Token {
            lexeme: Lexeme::Bang('!'),
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
        let eval = Eval::new("", false);
        let invalid_token = Token {
            lexeme: Lexeme::Bang('!'),
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
        let eval = Eval::new("", false);
        let plus_token = Token {
            lexeme: Lexeme::Plus('+'),
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
        let eval = Eval::new("", false);
        let minus_token = Token {
            lexeme: Lexeme::Minus('-'),
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
        let eval = Eval::new("", false);
        let greater_token = Token {
            lexeme: Lexeme::Greater('>'),
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
        let eval = Eval::new("", false);
        let eq_token = Token {
            lexeme: Lexeme::EqEq("==".to_string()),
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
        let eval = Eval::new("", false);
        let slash_token = Token {
            lexeme: Lexeme::Slash('/'),
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
}
