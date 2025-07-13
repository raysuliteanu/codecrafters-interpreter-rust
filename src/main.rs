use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::trace;
use std::{
    fs,
    io::{BufRead as _, Read, Write as _, stdin, stdout},
    path::PathBuf,
    process::ExitCode,
};

use crate::{eval::EvalValue, parser::ParseError, token::Scanner};

mod eval;
mod model;
mod parser;
mod span;
mod token;

#[derive(Parser)]
struct Lox {
    #[command(subcommand)]
    commands: LoxCommands,
}

#[derive(Subcommand)]
enum LoxCommands {
    Tokenize { filename: String },
    Parse { filename: String },
    Evaluate { filename: String },
    Run { filename: Option<String> },
}

fn main() -> Result<ExitCode> {
    env_logger::init();

    let lox = Lox::parse();

    let mut rc = 0;
    match lox.commands {
        LoxCommands::Tokenize { filename } => {
            let source = get_source(filename)?;
            if let Err(e) = Scanner::new(&source, true).scan() {
                rc = e;
            }
        }
        LoxCommands::Parse { filename } => {
            let source = get_source(filename)?;
            if let Err(_e) = parser::Parser::new(&source, true, true).parse() {
                rc = 65;
            }
        }

        LoxCommands::Evaluate { filename } => {
            let source = get_source(filename)?;
            match eval::Eval::new(&source, true).evaluate() {
                Ok(r) => println!("{r}"),
                Err(e) => {
                    eprintln!("{e}");
                    rc = if e.downcast_ref::<ParseError>().is_some() {
                        65
                    } else {
                        70
                    };
                }
            }
        }

        LoxCommands::Run { filename } => {
            if let Some(file) = filename {
                let source = get_source(file)?;
                match eval::Eval::new(&source, false).evaluate() {
                    Ok(r) => {
                        if r != EvalValue::Nil {
                            println!("{r}");
                        }
                    }
                    Err(e) => {
                        eprintln!("{e}");
                        rc = if e.downcast_ref::<ParseError>().is_some() {
                            65
                        } else {
                            70
                        };
                    }
                }
            } else {
                repl()?;
            }
        }
    };

    Ok(ExitCode::from(rc))
}

pub fn repl() -> anyhow::Result<()> {
    let mut stdin = std::io::stdin().lock();
    loop {
        let mut expr = String::new();
        print!("> ");
        stdout().flush()?;
        let cnt = stdin.read_line(&mut expr)?;
        trace!("read {cnt} bytes");

        let source = expr.trim_end();

        if source == "q" || source == "quit" {
            break;
        }

        eval::Eval::new(source, false).evaluate()?;
    }

    Ok(())
}

fn get_source(filename: String) -> anyhow::Result<String> {
    trace!("get_source({filename})");
    let source = fs::read_to_string(PathBuf::from(&filename)).with_context(|| filename)?;

    Ok(source)
}
