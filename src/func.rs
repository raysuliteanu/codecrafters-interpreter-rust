use std::time::{SystemTime, UNIX_EPOCH};

use crate::eval::{EvalErrors, EvalResult, EvalValue};

fn check_arity(expected: usize, args: &[EvalValue]) {
    assert_eq!(expected, args.len())
}

pub(crate) fn clock(args: &[EvalValue]) -> EvalResult<EvalValue> {
    check_arity(0, args);
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => Ok(EvalValue::Number(d.as_secs_f64())),
        // TODO: something better than panic, though shouldn't happen
        // since UNIX_EPOCH is always before "now"
        Err(e) => panic!("{e}"),
    }
}

// pub(crate) fn duration_since(args: &[EvalValue]) -> EvalResult<EvalValue> {
//     check_arity(1, args);
//
//     if args.len() == 1 {
//         match SystemTime::now().duration_since(args[0]) {
//             Ok(d) => Ok(EvalValue::Number(d.as_secs_f64())),
//             // TODO: something better than panic, though shouldn't happen
//             // since UNIX_EPOCH is always before "now"
//             Err(e) => panic!("{e}"),
//         }
//     } else {
//         Err(EvalErrors::FunctionCall)
//     }
// }
