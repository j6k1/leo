use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Instant;
use usiagent::event::{UserEventDispatcher, UserEventQueue};
use usiagent::hash::KyokumenMap;
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{AppliedMove, State};
use usiagent::shogi::{MochigomaCollections, ObtainKind, Teban};
use crate::error::ApplicationError;
use crate::solver::Solver;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EvaluationResult(Score, Option<AppliedMove>);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Score {
    NEGINFINITE,
    Value(i32),
    INFINITE,
}
impl Neg for Score {
    type Output = Score;

    fn neg(self) -> Score {
        match self {
            Score::Value(v) => Score::Value(-v),
            Score::INFINITE => Score::NEGINFINITE,
            Score::NEGINFINITE => Score::INFINITE,
        }
    }
}
impl Add<i32> for Score {
    type Output = Self;

    fn add(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v + other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
        }
    }
}
impl Sub<i32> for Score {
    type Output = Self;

    fn sub(self, other:i32) -> Self::Output {
        match self {
            Score::Value(v) => Score::Value(v - other),
            Score::INFINITE => Score::INFINITE,
            Score::NEGINFINITE => Score::NEGINFINITE,
        }
    }
}

const BASE_DEPTH:u32 = 2;
const MAX_DEPTH:u32 = 6;
pub const TIMELIMIT_MARGIN:u64 = 50;
const NETWORK_DELAY:u32 = 1100;
const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
const DEFAULT_ADJUST_DEPTH:bool = true;
const MAX_THREADS:u32 = 1;
const MAX_PLY:u32 = 200;
const MAX_PLY_TIMELIMIT:u64 = 0;
const TURN_COUNT:u32 = 50;
const MIN_TURN_COUNT:u32 = 5;

pub struct Environment<L,S> where L: Logger, S: InfoSender {
    event_queue:Arc<Mutex<UserEventQueue>>,
    info_sender:S,
    on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    limit:Option<Instant>,
    current_limit:Option<Instant>,
    stop:Arc<AtomicBool>,
    quited:Arc<AtomicBool>,
    kyokumen_score_map:KyokumenMap<u64,(Score,u32)>,
    nodes:Arc<AtomicU64>,
    think_start_time:Instant
}
impl<L,S> Clone for Environment<L,S>
    where L: Logger,
          S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:self.event_queue.clone(),
            info_sender:self.info_sender.clone(),
            on_error_handler:self.on_error_handler.clone(),
            limit:self.limit.clone(),
            current_limit:self.current_limit.clone(),
            stop:self.stop.clone(),
            quited:self.quited.clone(),
            kyokumen_score_map:self.kyokumen_score_map.clone(),
            nodes:self.nodes.clone(),
            think_start_time:self.think_start_time.clone()
        }
    }
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               think_start_time:Instant,
               limit:Option<Instant>,
               current_limit:Option<Instant>) -> Environment<L,S> {
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            think_start_time:think_start_time,
            limit:limit,
            current_limit:current_limit,
            stop:stop,
            quited:quited,
            kyokumen_score_map:KyokumenMap::new(),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a,S,L> where L: Logger {
    event_dispatcher:&'a mut UserEventDispatcher<'a,S,ApplicationError,L>,
    solver_event_dispatcher:&'a mut UserEventDispatcher<'a,Solver,ApplicationError,L>,
    teban:Teban,
    state:&'a Arc<State>,
    alpha:Score,
    beta:Score,
    m:Option<AppliedMove>,
    mc:&'a Arc<MochigomaCollections>,
    pv:&'a Vec<AppliedMove>,
    obtained:Option<ObtainKind>,
    current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    self_already_oute_map:&'a mut Option<KyokumenMap<u64,bool>>,
    opponent_already_oute_map:&'a mut Option<KyokumenMap<u64,bool>>,
    oute_kyokumen_map:&'a KyokumenMap<u64,()>,
    mhash:u64,
    shash:u64,
    depth:u32,
    current_depth:u32,
    base_depth:u32,
    node_count:u64,
}
