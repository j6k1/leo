use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};
use usiagent::command::UsiInfoSubCommand;
use usiagent::event::{UserEventDispatcher, UserEventQueue};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{AppliedMove, LegalMove, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::{ApplicationError, SendSelDepthError};
use crate::nn::Evalutor;
use crate::solver::Solver;

pub const BASE_DEPTH:u32 = 2;
pub const MAX_DEPTH:u32 = 6;
pub const TIMELIMIT_MARGIN:u64 = 50;
pub const NETWORK_DELAY:u32 = 1100;
pub const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
pub const DEFAULT_ADJUST_DEPTH:bool = true;
pub const MAX_THREADS:u32 = 1;
pub const MAX_PLY:u32 = 200;
pub const MAX_PLY_TIMELIMIT:u64 = 0;
pub const TURN_COUNT:u32 = 50;
pub const MIN_TURN_COUNT:u32 = 5;

pub trait Search<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self,env:&mut Environment<L,S>, gs:GameState<'a,L,S>, evalutor:Evalutor) -> Result<EvaluationResult,ApplicationError>;

    fn timelimit_reached(&self,env:&mut Environment<L,S>,limit:&Option<Instant>) -> bool {
        let network_delay = env.network_delay;
        limit.map_or(false,|l| {
            l < Instant::now() || l - Instant::now() <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
        })
    }

    fn timeout_expected(&self,env:&mut Environment<L,S>,start_time:Instant,
                             current_depth:u32,nodes:u64,processed_nodes:u32) -> bool {
        const RATE:u64 = 8;

        if current_depth <= 1 {
            false
        } else {
            let nodes = nodes / RATE.pow(current_depth);

            (nodes > u32::MAX as u64) || (current_depth > 1 && env.adjust_depth &&
                env.current_limit.map(|l| {
                    env.think_start_time + ((Instant::now() - start_time) / processed_nodes) * nodes as u32 > l
                }).unwrap_or(false)
            ) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false)
        }
    }

    fn send_message(&self, env:&mut Environment<L,S>,
                           on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_seldepth(&self, env:&mut Environment<L,S>,
                            on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                            depth:u32, seldepth:u32) -> Result<(),ApplicationError> {

        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));
        commands.push(UsiInfoSubCommand::SelDepth(seldepth));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_info(&self, env:&mut Environment<L,S>,
                      depth:u32, seldepth:u32, pv:&Vec<AppliedMove>) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        if depth < seldepth {
            commands.push(UsiInfoSubCommand::Depth(depth));
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
        commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_score(&self,env:&mut Environment<L,S>,
                        on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                        teban:Teban,
                        s:Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        if env.display_evalute_score {
            let teban_str = match teban {
                Teban::Sente => "sente",
                Teban::Gote =>  "gote"
            };
            match &s {
                Score::INFINITE => {
                    self.send_message(env,on_error_handler, &format!("evalute score = inifinite. ({0})",teban_str))
                },
                Score::NEGINFINITE => {
                    self.send_message(env,on_error_handler, &format!("evalute score = neginifinite. ({0})",teban_str))
                },
                Score::Value(s) => {
                    self.send_message(env,on_error_handler, &format!("evalute score =  {0: >17} ({1})",s,teban_str))
                }
            }
        } else {
            Ok(())
        }
    }

    fn startup_strategy<'a>(&self,env:&'a mut Environment<L,S>, gs:&'a mut GameState<'a,L,S>, m:LegalMove, priority:u32)
                            -> Option<(u32,Option<ObtainKind>,u64,u64,KyokumenMap<u64,()>,KyokumenMap<u64,u32>,bool)> {

        let obtained = match m {
            LegalMove::To(ref m) => m.obtained(),
            _ => None,
        };

        let mut oute_kyokumen_map = gs.oute_kyokumen_map.clone();
        let mut current_kyokumen_map = gs.current_kyokumen_map.clone();

        let (mhash,shash) = {
            let o = gs.obtained.and_then(|o| MochigomaKind::try_from(o).ok());

            let mhash = env.hasher.calc_main_hash(gs.mhash,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);
            let shash = env.hasher.calc_sub_hash(gs.shash,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

            if priority == 10 {
                match oute_kyokumen_map.get(gs.teban,&mhash,&shash) {
                    Some(_) => {
                        return None;
                    },
                    None => {
                        oute_kyokumen_map.insert(gs.teban,mhash,shash,());
                    },
                }
            }

            (mhash,shash)
        };

        if priority < 10 {
            oute_kyokumen_map.clear(gs.teban);
        }

        let depth = match priority {
            5 | 10 => gs.depth + 1,
            _ => gs.depth,
        };

        let is_sennichite = match current_kyokumen_map.get(gs.teban,&mhash,&shash).unwrap_or(&0) {
            &c if c >= 3 => {
                return None;
            },
            &c if c > 0 => {
                current_kyokumen_map.insert(gs.teban,mhash,shash,c+1);

                true
            },
            _ => false,
        };

        Some((depth,obtained,mhash,shash,oute_kyokumen_map,current_kyokumen_map,is_sennichite))
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EvaluationResult {
    Immediate(Score, Option<AppliedMove>),
    Async,
    Timeout
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Score {
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

pub struct Environment<L,S> where L: Logger, S: InfoSender {
    event_queue:Arc<Mutex<UserEventQueue>>,
    info_sender:S,
    on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    hasher:Arc<KyokumenHash<u64>>,
    limit:Option<Instant>,
    current_limit:Option<Instant>,
    adjust_depth:bool,
    network_delay:u32,
    display_evalute_score:bool,
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
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            hasher:Arc::clone(&self.hasher),
            limit:self.limit.clone(),
            current_limit:self.current_limit.clone(),
            adjust_depth:self.adjust_depth,
            network_delay:self.network_delay,
            display_evalute_score:self.display_evalute_score,
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            kyokumen_score_map:self.kyokumen_score_map.clone(),
            nodes:Arc::clone(&self.nodes),
            think_start_time:self.think_start_time.clone()
        }
    }
}
impl<L,S> Environment<L,S> where L: Logger, S: InfoSender {
    pub fn new(event_queue:Arc<Mutex<UserEventQueue>>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               hasher:Arc<KyokumenHash<u64>>,
               think_start_time:Instant,
               limit:Option<Instant>,
               current_limit:Option<Instant>,
               adjust_depth:bool,
               network_delay:u32,
               display_evalute_score:bool
    ) -> Environment<L,S> {
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            hasher:hasher,
            think_start_time:think_start_time,
            limit:limit,
            current_limit:current_limit,
            adjust_depth:adjust_depth,
            network_delay:network_delay,
            display_evalute_score:display_evalute_score,
            stop:stop,
            quited:quited,
            kyokumen_score_map:KyokumenMap::new(),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a,L,S> where L: Logger + Send + 'static, S: InfoSender {
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
pub struct Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>
}
impl<L,S> Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Root<L,S> {
        Root {
            l:PhantomData::<L>,
            s:PhantomData::<S>
        }
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self, env: &mut Environment<L, S>, gs: GameState<'a, L, S>, evalutor: Evalutor)
        -> Result<EvaluationResult, ApplicationError> {
        todo!()
    }
}
pub struct Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
    sender:Sender<(AppliedMove,i32)>
}
impl<L,S> Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new(sender:Sender<(AppliedMove,i32)>) -> Recursive<L,S> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            sender:sender
        }
    }
}
impl<L,S> Search<L,S> for Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self, env: &mut Environment<L, S>, gs: GameState<'a, L, S>, evalutor: Evalutor) -> Result<EvaluationResult, ApplicationError> {
        todo!()
    }
}