use std::time::{SystemTime, UNIX_EPOCH};

use crate::eval::{EvalResult, EvalValue};

pub(crate) fn clock(_args: &[EvalValue]) -> EvalResult<EvalValue> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => Ok(EvalValue::Number(d.as_secs_f64())),
        Err(e) => panic!("{e}"),
    }
}
