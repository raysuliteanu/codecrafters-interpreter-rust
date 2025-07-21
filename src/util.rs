use crate::model::{Ast, AstExpr, AstStmt};

pub fn print_ast(ast: &[Ast]) {
    const INDENT: &str = "    ";
    fn print_ast_node(ast: &Ast, level: usize) {
        let indent = INDENT.repeat(level);
        match ast {
            Ast::Class => println!("{indent}Class"),
            Ast::Function { name, params, body } => {
                let params = params
                    .iter()
                    .map(|s| format!("{indent}{s}"))
                    .collect::<Vec<_>>()
                    .join(", ");

                println!("{indent}Function {name}({params})");
                print_ast_node(body, level + 1);
            }
            Ast::Variable { name, initializer } => {
                println!("{indent}Variable({name:?}, {initializer:?}),");
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
        let indent = INDENT.repeat(level);
        match stmt {
            AstStmt::If {
                condition,
                then,
                or_else,
            } => {
                println!("{indent}If(");
                print_ast_expr(condition, level + 1);
                println!("{indent},");
                print_ast_node(then, level + 1);
                if let Some(else_block) = or_else {
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
        let indent = INDENT.repeat(level);
        match expr {
            AstExpr::Call {
                func,
                args,
                site: _,
            } => {
                println!("{indent}Call {{");
                println!("{indent}    func:");
                println!("{indent}{func}");
                println!("{indent}    args: [");
                for (i, arg) in args.iter().enumerate() {
                    print_ast_expr(arg, level + 2);
                    if i < args.len() - 1 {
                        println!("{indent}    ,");
                    }
                }
                println!("{indent}    ]");
                println!("{indent}}}");
            }
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
