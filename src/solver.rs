use std::marker::PhantomData;
use usiagent::error::PlayerError;

pub struct Solver<E> where E: PlayerError {
    error_type:PhantomData<E>
}
impl<E> Solver<E> where E: PlayerError {
    pub fn new() -> Solver<E> {
        Solver {
            error_type:PhantomData::<E>
        }
    }
}
