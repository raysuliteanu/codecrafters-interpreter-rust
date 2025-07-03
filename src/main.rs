use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::trace;
use parser::Ast;
use std::{
    fs,
    io::{BufRead as _, Read, Write as _, stdin, stdout},
    path::PathBuf,
    process::ExitCode,
};

use crate::token::Scanner;

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

    match lox.commands {
        LoxCommands::Tokenize { filename } => {
            let source = get_source(filename)?;
            Scanner::new(&source)
                .scan()
                .map_err(|e| eprintln!("{e}"))
                .iter()
                .flatten()
                .for_each(|t| println!("{t}"));
        }
        LoxCommands::Parse { filename } => {
            let source = get_source(filename)?;
            parser::Parser::new(&source)
                .parse()
                .map_err(|e| eprintln!("{e}"))
                .iter()
                .flatten()
                .for_each(|ast| println!("{ast}"));
        }

        LoxCommands::Evaluate { filename } => {
            let _source = get_source(filename)?;
        }

        LoxCommands::Run { filename } => {
            if let Some(file) = filename {
                let _source = get_source(file)?;
            } else {
                let _ = repl();
            }
        }
    }

    let rc = 0;

    Ok(ExitCode::from(rc))
}

pub fn repl() -> anyhow::Result<Vec<Ast>> {
    let mut stdin = std::io::stdin().lock();
    loop {
        let mut expr = String::new();
        print!("> ");
        let _ = stdout().flush();
        let _ = stdin.read_line(&mut expr)?;

        let source = expr.trim_end();

        if source == "q" || source == "quit" {
            break;
        }
    }

    Ok(vec![])
}

fn get_source(filename: String) -> anyhow::Result<String> {
    trace!("get_source({filename})");
    let source = fs::read_to_string(PathBuf::from(&filename)).with_context(|| filename)?;

    Ok(source)
}
