use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::trace;
use std::{
    fs,
    io::{BufRead as _, Write as _, stdout},
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
    Tokenize {
        filename: String,
    },
    Parse {
        filename: String,
        #[arg(short, long)]
        no_expression_mode: bool,
        #[arg(short, long)]
        pretty_print: bool,
    },
    Evaluate {
        filename: String,
        #[arg(short, long)]
        no_expression_mode: bool,
    },
    Run {
        filename: Option<String>,
    },
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
        LoxCommands::Parse {
            filename,
            no_expression_mode,
            pretty_print,
        } => {
            let source = get_source(filename)?;
            let parser = parser::ParserBuilder::new(&source)
                .expression_mode(!no_expression_mode)
                .pretty_print(pretty_print)
                .build();
            if let Err(_e) = parser.parse() {
                rc = 65;
            }
        }
        LoxCommands::Evaluate {
            filename,
            no_expression_mode,
        } => {
            let source = get_source(filename)?;
            match eval::Eval::new(&source, !no_expression_mode).evaluate() {
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
