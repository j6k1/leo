use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use concurrent_fixed_hashmap::ConcurrentFixedHashMap;
use usiagent::command::UsiInfoSubCommand;
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{AppliedMove, LegalMove, Rule, SquareToPoint, State};
use usiagent::shogi::{KomaKind, MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::{ApplicationError, SendSelDepthError};
use crate::nn::Evalutor;
use crate::search::Score::{INFINITE, NEGINFINITE};
use crate::solver::{GameStateForMate, MaybeMate, Solver};

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

pub trait Search<L,S>: Sized where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError>;

    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> bool {
        let network_delay = env.network_delay;
        env.limit.map_or(false,|l| {
            l < Instant::now() || l - Instant::now() <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
        })
    }

    fn timeout_expected(&self, env:&mut Environment<L,S>, start_time:Instant,
                        current_depth:u32, parent_nodes:u128, nodes:u32, processed_nodes:u32) -> bool {
        const SECOND_NANOS:u128 = 1000_000_000;
        const D:u128 = 8;

        if current_depth <= 1 || !env.adjust_depth {
            false
        } else {
            env.current_limit.map(|l| {
                let nanos = ((Instant::now() - start_time) / processed_nodes * nodes).as_nanos() * parent_nodes / D;
                env.think_start_time + Duration::new((nanos / SECOND_NANOS) as u64, (nanos % SECOND_NANOS) as u32) > l
            }).unwrap_or(false) || env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false)
        }
    }

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_seldepth(&self, env:&mut Environment<L,S>,
                            depth:u32, seldepth:u32) -> Result<(),ApplicationError> {

        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));
        commands.push(UsiInfoSubCommand::SelDepth(seldepth));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_info(&self, env:&mut Environment<L,S>,
                      depth:u32, seldepth:u32, pv:&VecDeque<AppliedMove>) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        if depth < seldepth {
            commands.push(UsiInfoSubCommand::Depth(depth));
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_score(&self,env:&mut Environment<L,S>,
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
                    self.send_message(env,&format!("evalute score = inifinite. ({0})",teban_str))
                },
                Score::NEGINFINITE => {
                    self.send_message(env,&format!("evalute score = neginifinite. ({0})",teban_str))
                },
                Score::Value(s) => {
                    self.send_message(env,&format!("evalute score =  {0: >17} ({1})",s,teban_str))
                }
            }
        } else {
            Ok(())
        }
    }

    fn startup_strategy<'a>(&self,env:&mut Environment<L,S>,
                            gs: &mut GameState<'a>,
                            m:LegalMove, priority:u32)
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

    fn before_search<'a,'b>(&self,
                         env: &mut Environment<L, S>,
                         gs: &mut GameState<'a>,
                         event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                         evalutor: &Evalutor)
        -> Result<BeforeSearchResult, ApplicationError> {

        if env.base_depth < gs.current_depth {
            self.send_seldepth(env,env.base_depth,gs.current_depth)?;
        }

        if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if !env.unique_kyokumen_map.contains_key(&(gs.teban,gs.mhash,gs.shash)) {
            env.nodes.fetch_add(1,atomic::Ordering::Release);
            env.unique_kyokumen_map.insert((gs.teban,gs.mhash,gs.shash),());
        }

        if let Some(ObtainKind::Ou) = gs.obtained {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Immediate(NEGINFINITE,gs.depth,gs.mhash,gs.shash,VecDeque::new())));
        }

        if let Some(m) = gs.m {
            if Rule::is_mate(gs.teban, &*gs.state) {
                let mut mvs = VecDeque::new();
                mvs.push_front(m);
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Immediate(INFINITE,gs.depth,gs.mhash,gs.shash,mvs)));
            }

            let r = env.kyokumen_score_map.get(&(gs.teban, gs.mhash, gs.shash)).map(|g| *g);

            if let Some((s,d)) = r {
                match s {
                    Score::INFINITE => {
                        if env.display_evalute_score {
                            self.send_message(env, "score corresponding to the hash was found in the map. value is infinite.")?;
                        }
                        let mut mvs = VecDeque::new();
                        mvs.push_front(m);

                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(Score::INFINITE, gs.depth, gs.mhash, gs.shash, mvs)
                        ));
                    },
                    Score::NEGINFINITE => {
                        if env.display_evalute_score {
                            self.send_message(env, "score corresponding to the hash was found in the map. value is neginfinite.")?;
                        }
                        let mut mvs = VecDeque::new();
                        mvs.push_front(m);

                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(Score::NEGINFINITE, gs.depth, gs.mhash, gs.shash, mvs)
                        ));
                    },
                    Score::Value(s) if d >= gs.depth => {
                        if env.display_evalute_score {
                            self.send_message(env, &format!("score corresponding to the hash was found in the map. value is {}.", s))?;
                        }
                        let mut mvs = VecDeque::new();
                        mvs.push_front(m);

                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(Score::Value(s), gs.depth, gs.mhash, gs.shash, mvs)
                        ));
                    },
                    _ => ()
                }
            }

            if (gs.depth <= 1 || gs.current_depth >= env.max_depth) && !Rule::is_mate(gs.teban.opposite(), &*gs.state) {
                let ms = GameStateForMate {
                    checkmate_state_map: Arc::clone(&gs.self_checkmate_state_map),
                    unique_kyokumen_map: Arc::clone(&env.unique_kyokumen_map),
                    current_depth: 0,
                    mhash: gs.mhash,
                    shash: gs.shash,
                    oute_kyokumen_map: gs.oute_kyokumen_map,
                    current_kyokumen_map: gs.current_kyokumen_map,
                    ignore_kyokumen_map: KyokumenMap::new(),
                    event_queue: env.event_queue.clone(),
                    teban: gs.teban,
                    state: gs.state,
                    mc: gs.mc
                };

                let solver = Solver::new();

                match solver.checkmate(
                    false,
                    env.limit.clone(),
                    env.max_ply_timelimit.map(|l| Instant::now() + l),
                    env.network_delay,
                    env.max_ply.clone(),
                    env.max_nodes.clone(),
                    Arc::clone(&env.nodes),
                    env.info_sender.clone(),
                    Arc::clone(&env.on_error_handler),
                    Arc::clone(&env.hasher),
                    env.base_depth,
                    Arc::clone(&env.stop),
                    Arc::clone(&env.quited),
                    ms
                )? {
                    MaybeMate::MateMoves(_, mvs) => {
                        let mut r  = mvs.into_iter().map(|m| {
                            AppliedMove::from(m)
                        }).collect::<VecDeque<AppliedMove>>();
                        r.push_front(m);

                        return Ok(BeforeSearchResult::Complete(EvaluationResult::Immediate(INFINITE, gs.depth, gs.mhash, gs.shash,r)));
                    },
                    _ => ()
                }
            }
        }
        event_dispatcher.dispatch_events(self,&*env.event_queue).map_err(|e| ApplicationError::from(e))?;

        if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if gs.depth == 0 || gs.current_depth >= env.max_depth {
            match gs.m {
                None => {
                    return Err(ApplicationError::LogicError(String::from("move is not set.")))?;
                },
                Some(m) => {
                    let (s,r) = mpsc::channel();
                    evalutor.submit(gs.teban,gs.state.get_banmen(),&gs.mc,m,s)?;
                    return Ok(BeforeSearchResult::Complete(EvaluationResult::Async(r)));
                }
            }
        }

        let mvs = if Rule::is_mate(gs.teban.opposite(),&*gs.state) {
            if gs.depth == 0 || gs.current_depth == env.max_depth {
                if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                    return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
                }
            }

            let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                let mut mvs = VecDeque::new();
                gs.m.map(|m| mvs.push_front(m));

                return Ok(BeforeSearchResult::Complete(
                    EvaluationResult::Immediate(NEGINFINITE, gs.depth,gs.mhash,gs.shash,mvs)
                ));
            } else {
                mvs
            }
        } else {
            if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
            }

            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            mvs
        };

        let mvs = mvs.into_iter().map(|m| {
            let ps = Rule::apply_move_to_partial_state_none_check(gs.state,gs.teban,gs.mc,m.to_applied_move());

            let (x,y,kind) = match m {
                LegalMove::To(ref mv) => {
                    let banmen = gs.state.get_banmen();
                    let (sx,sy) = mv.src().square_to_point();
                    let (x,y) = mv.dst().square_to_point();
                    let kind = banmen.0[sy as usize][sx as usize];

                    let kind = if mv.is_nari() {
                        kind.to_nari()
                    } else {
                        kind
                    };

                    (x,y,kind)
                },
                LegalMove::Put(ref mv) => {
                    let (x,y) = mv.dst().square_to_point();
                    let kind = mv.kind();

                    (x,y,KomaKind::from((gs.teban,kind)))
                }
            };
            if Rule::is_mate_with_partial_state_and_point_and_kind(gs.teban,&ps,x,y,kind) ||
                Rule::is_mate_with_partial_state_repeat_move_kinds(gs.teban,&ps) {
                (10,m)
            } else {
                match m {
                    LegalMove::To(ref mv) if mv.obtained().is_some() => {
                        (5,m)
                    },
                    _ => (1,m),
                }
            }
        }).collect::<Vec<(u32,LegalMove)>>();

        Ok(BeforeSearchResult::Mvs(mvs))
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Immediate(Score, u32, u64, u64, VecDeque<AppliedMove>),
    Async(Receiver<(AppliedMove,i32)>),
    Timeout
}
#[derive(Debug)]
pub enum BeforeSearchResult {
    Complete(EvaluationResult),
    Mvs(Vec<(u32,LegalMove)>)
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
    pub event_queue:Arc<Mutex<UserEventQueue>>,
    pub info_sender:S,
    pub on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    pub hasher:Arc<KyokumenHash<u64>>,
    pub unique_kyokumen_map:Arc<ConcurrentFixedHashMap<(Teban,u64,u64),()>>,
    pub limit:Option<Instant>,
    pub current_limit:Option<Instant>,
    pub turn_count:u32,
    pub min_turn_count:u32,
    pub base_depth:u32,
    pub max_depth:u32,
    pub max_nodes:Option<i64>,
    pub max_ply:Option<u32>,
    pub max_ply_mate:Option<u32>,
    pub max_ply_timelimit:Option<Duration>,
    pub adjust_depth:bool,
    pub network_delay:u32,
    pub display_evalute_score:bool,
    pub max_threads:u32,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub kyokumen_score_map:Arc<ConcurrentFixedHashMap<(Teban,u64,u64),(Score,u32)>>,
    pub nodes:Arc<AtomicU64>,
    pub think_start_time:Instant
}
impl<L,S> Clone for Environment<L,S> where L: Logger, S: InfoSender {
    fn clone(&self) -> Self {
        Environment {
            event_queue:Arc::clone(&self.event_queue),
            info_sender:self.info_sender.clone(),
            on_error_handler:Arc::clone(&self.on_error_handler),
            hasher:Arc::clone(&self.hasher),
            unique_kyokumen_map:Arc::clone(&self.unique_kyokumen_map),
            limit:self.limit.clone(),
            current_limit:self.current_limit.clone(),
            turn_count:self.turn_count,
            min_turn_count:self.min_turn_count,
            base_depth:self.base_depth,
            max_depth:self.max_depth,
            max_nodes:self.max_nodes.clone(),
            max_ply:self.max_ply.clone(),
            max_ply_mate:self.max_ply_mate.clone(),
            max_ply_timelimit:self.max_ply_timelimit.clone(),
            adjust_depth:self.adjust_depth,
            network_delay:self.network_delay,
            display_evalute_score:self.display_evalute_score,
            max_threads:self.max_threads,
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
               turn_count:u32,
               min_turn_count:u32,
               base_depth:u32,
               max_depth:u32,
               max_nodes:Option<i64>,
               max_ply:Option<u32>,
               max_ply_mate:Option<u32>,
               max_ply_timelimit:Option<Duration>,
               adjust_depth:bool,
               network_delay:u32,
               display_evalute_score:bool,
               max_threads:u32
    ) -> Environment<L,S> {
        let stop = Arc::new(AtomicBool::new(false));
        let quited = Arc::new(AtomicBool::new(false));

        Environment {
            event_queue:event_queue,
            info_sender:info_sender,
            on_error_handler:on_error_handler,
            hasher:hasher,
            unique_kyokumen_map:Arc::new(ConcurrentFixedHashMap::with_size(1 << 22)),
            think_start_time:think_start_time,
            limit:limit,
            current_limit:current_limit,
            turn_count:turn_count,
            min_turn_count:min_turn_count,
            base_depth:base_depth,
            max_depth:max_depth,
            max_nodes:max_nodes,
            max_ply:max_ply,
            max_ply_mate:max_ply_mate,
            max_ply_timelimit:max_ply_timelimit,
            adjust_depth:adjust_depth,
            network_delay:network_delay,
            display_evalute_score:display_evalute_score,
            max_threads:max_threads,
            stop:stop,
            quited:quited,
            kyokumen_score_map:Arc::new(ConcurrentFixedHashMap::with_size(1 << 22)),
            nodes:Arc::new(AtomicU64::new(0))
        }
    }
}
pub struct GameState<'a> {
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub alpha:Score,
    pub beta:Score,
    pub m:Option<AppliedMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub obtained:Option<ObtainKind>,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    pub self_checkmate_state_map:Arc<ConcurrentFixedHashMap<(Teban, u64, u64),bool>>,
    pub opponent_checkmate_state_map:Arc<ConcurrentFixedHashMap<(Teban, u64, u64),bool>>,
    pub oute_kyokumen_map:&'a KyokumenMap<u64,()>,
    pub mhash:u64,
    pub shash:u64,
    pub depth:u32,
    pub current_depth:u32,
    pub node_count:u128,
}
pub struct Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
    receiver:Receiver<Result<EvaluationResult, ApplicationError>>,
    sender:Sender<Result<EvaluationResult, ApplicationError>>
}
impl<L,S> Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Root<L,S> {
        let(s,r) = mpsc::channel();

        Root {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            receiver:r,
            sender:s
        }
    }

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
                                       -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler);

        {
            let stop = stop.clone();

            event_dispatcher.add_handler(UserEventKind::Stop, move |_,e| {
                match e {
                    &UserEvent::Stop => {
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        {
            let stop = stop.clone();
            let quited = quited.clone();

            event_dispatcher.add_handler(UserEventKind::Quit, move |_,e| {
                match e {
                    &UserEvent::Quit => {
                        quited.store(true,atomic::Ordering::Release);
                        stop.store(true,atomic::Ordering::Release);
                        Ok(())
                    },
                    e => Err(EventHandlerError::InvalidState(e.event_kind())),
                }
            });
        }

        event_dispatcher
    }

    pub fn termination<'a,'b>(&self,
                       await_mvs:Vec<Receiver<(AppliedMove,i32)>>,
                       env:&mut Environment<L,S>,
                       gs: &mut GameState<'a>,
                       evalutor: &Evalutor,
                       score:Score,
                       mut best_moves:VecDeque<AppliedMove>) -> Result<EvaluationResult,ApplicationError> {
        env.stop.store(true,atomic::Ordering::Release);

        let mut score = score;
        let mut opt_error = None;

        while evalutor.active_threads() > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r) {
                Ok(EvaluationResult::Immediate(s,_,_,_,mvs)) => {
                    if -s > score {
                        score = -s;
                        best_moves = mvs;
                        opt_error = opt_error.and(self.send_info(env, env.base_depth,gs.current_depth,&best_moves).err());
                    }
                },
                Err(e) => {
                    opt_error = Some(e);
                },
                _ => ()
            }
            opt_error = opt_error.and(evalutor.on_end_thread().map_err(|e| ApplicationError::from(e)).err());
        }

        if let Some(e) = opt_error {
            return Err(e);
        }

        for r in await_mvs {
            match r.recv().map_err(|e| ApplicationError::from(e)) {
                Ok((m,s)) => {
                    let s = Score::Value(s);

                    opt_error = opt_error.and(self.send_score(env,gs.teban,-s).err());

                    if -s > score {
                        score = -s;
                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        opt_error = opt_error.and(self.send_info(env, env.base_depth,gs.current_depth,&best_moves).err());
                    }
                },
                Err(e) => {
                    opt_error = opt_error.and(Some(e));
                }
            }
        }

        match opt_error {
            Some(e) => {
                Err(e)
            },
            None => {
                Ok(EvaluationResult::Immediate(score, gs.depth,gs.mhash,gs.shash,best_moves))
            }
        }
    }

    fn parallelized<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                           evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError>  {
        let mut gs = gs;

        let mvs = match self.before_search(env,&mut gs,event_dispatcher,evalutor)? {
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::Mvs(mvs) => {
                mvs
            }
        };

        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();
        let mut await_mvs = vec![];

        let mvs_count = mvs.len() as u64;

        let mut threads = env.max_threads.min(mvs_count as u32);

        let sender = self.sender.clone();

        let mut it = mvs.into_iter();
        let mut processed_nodes:u32 = 0;
        let start_time = Instant::now();

        loop {
            if threads == 0 {
                let r = self.receiver.recv();

                evalutor.on_end_thread()?;

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

                processed_nodes += 1;

                match r {
                    EvaluationResult::Immediate(s,depth,mhash,shash,mvs) => {
                        if !env.kyokumen_score_map.contains_key(&(gs.teban.opposite(),mhash,shash)) {
                            env.kyokumen_score_map.insert_new((gs.teban.opposite(), mhash, shash), (s, depth));
                        }

                        if let Some(mut g) = env.kyokumen_score_map.get_mut(&(gs.teban.opposite(),mhash,shash)) {
                            let (ref mut score,ref mut d) = *g;

                            if *d < depth {
                                *d = depth;
                                *score = s;
                            }
                        }

                        if !env.kyokumen_score_map.contains_key(&(gs.teban,mhash,shash)) {
                            env.kyokumen_score_map.insert_new((gs.teban, mhash, shash), (-s, depth));
                        }

                        if let Some(mut g) = env.kyokumen_score_map.get_mut(&(gs.teban,mhash,shash)) {
                            let (ref mut score,ref mut d) = *g;

                            if *d < depth {
                                *d = depth;
                                *score = -s;
                            }
                        }

                        self.send_info(env, env.base_depth,gs.current_depth,&mvs)?;

                        if -s > scoreval {
                            scoreval = -s;

                            best_moves = mvs;

                            if scoreval >= beta {
                                break;
                            }

                            if alpha < scoreval {
                                alpha = scoreval;
                            }
                        }

                        if self.timeout_expected(env,start_time,gs.current_depth,gs.node_count,mvs_count as u32,processed_nodes) {
                            break;
                        }
                    },
                    EvaluationResult::Async(r) => {
                        await_mvs.push(r);
                    },
                    EvaluationResult::Timeout => {
                        break;
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self,&*event_queue)?;

                if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                    break;
                }
            } else if let Some((priority,m)) = it.next() {
                match self.startup_strategy(env,
                                            gs,
                                            m,
                                            priority) {
                    Some((depth, obtained, mhash, shash,
                             oute_kyokumen_map,
                             current_kyokumen_map,
                             is_sennichite)) => {
                        let m = m.to_applied_move();
                        let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m);

                        match next {
                            (state, mc, _) => {
                                if is_sennichite {
                                    let s = if Rule::is_mate(gs.teban.opposite(), &state) {
                                        Score::NEGINFINITE
                                    } else {
                                        Score::Value(0)
                                    };
                                    if s > scoreval {
                                        scoreval = s;
                                        best_moves = VecDeque::new();
                                        best_moves.push_front(m);

                                        if alpha < scoreval {
                                            alpha = scoreval;
                                        }

                                        if scoreval >= beta {
                                            return self.termination(await_mvs, env, &mut gs, &evalutor, scoreval, best_moves);
                                        }
                                    }
                                    continue;
                                }

                                let teban = gs.teban;
                                let state = Arc::new(state);
                                let mc = Arc::new(mc);
                                let alpha = alpha;
                                let beta = beta;
                                let self_checkmate_state_map = Arc::clone(&gs.self_checkmate_state_map);
                                let opponent_checkmate_state_map = Arc::clone(&gs.opponent_checkmate_state_map);
                                let current_depth = gs.current_depth;

                                let mut env = env.clone();
                                let evalutor = evalutor.clone();

                                let sender = sender.clone();

                                let b = std::thread::Builder::new();

                                let sender = sender.clone();

                                evalutor.on_begin_thread();

                                let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
                                    let mut event_dispatcher = Self::create_event_dispatcher::<Recursive<L, S>>(&env.on_error_handler, &env.stop, &env.quited);

                                    let mut gs = GameState {
                                        teban: teban.opposite(),
                                        state: &state,
                                        alpha: -beta,
                                        beta: -alpha,
                                        m: Some(m),
                                        mc: &mc,
                                        obtained: obtained,
                                        current_kyokumen_map: &current_kyokumen_map,
                                        self_checkmate_state_map: opponent_checkmate_state_map,
                                        opponent_checkmate_state_map: self_checkmate_state_map,
                                        oute_kyokumen_map: &oute_kyokumen_map,
                                        mhash: mhash,
                                        shash: shash,
                                        depth: depth - 1,
                                        current_depth: current_depth + 1,
                                        node_count: mvs_count as u128 - processed_nodes as u128
                                    };

                                    let strategy = Recursive::new();

                                    let r = strategy.search(&mut env, &mut gs, &mut event_dispatcher, &evalutor);

                                    let _ = sender.send(r);
                                });

                                threads -= 1;
                            }
                        }
                    },
                    None => (),
                }
            } else if evalutor.active_threads() > 0 {
                threads -= 1;
            } else {
                break;
            }
        }

        self.termination(await_mvs,env, &mut gs,evalutor,scoreval,best_moves)
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let r = self.parallelized(env,gs,event_dispatcher,evalutor);
        let mut opt_err = None;

        env.stop.store(true,atomic::Ordering::Release);

        while evalutor.active_threads() > 0 {
            opt_err = opt_err.or(self.receiver.recv()?.map_err(|e| ApplicationError::from(e)).err());
            opt_err = opt_err.or(evalutor.on_end_thread().map(|_| ()).err());
        }

        if let Err(e) = r {
            let _ = env.on_error_handler.lock().map(|h| h.call(&e));
            let _ = self.send_message(env, format!("{}", &e).as_str());
            Err(e)
        } else if let Some(e) = opt_err {
            let _ = env.on_error_handler.lock().map(|h| h.call(&e));
            let _ = self.send_message(env,format!("{}",&e).as_str());
            Err(e)
        } else {
            r
        }
    }
}
pub struct Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
}
impl<L,S> Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new() -> Recursive<L,S> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
        }
    }
}
impl<L,S> Search<L,S> for Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Recursive<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let mut gs = gs;

        let mvs = match self.before_search(env,&mut gs,event_dispatcher,evalutor)? {
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::Mvs(mvs) => {
                mvs
            }
        };
        let prev_move = gs.m.ok_or(ApplicationError::LogicError(String::from(
            "move is not set."
        )))?;

        let mut alpha = gs.alpha;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        let mut processed_nodes:u32 = 0;
        let start_time = Instant::now();

        let mut await_mvs = vec![];
        let mvs_count = mvs.len();

        for &(priority,m) in &mvs {
            processed_nodes += 1;

            let parent_nodes = gs.node_count;

            match self.startup_strategy(env,gs,m,priority) {
                Some((depth,obtained,mhash,shash,
                         oute_kyokumen_map,
                         current_kyokumen_map,
                         is_sennichite)) => {
                    let next = Rule::apply_move_none_check(&gs.state,gs.teban,gs.mc,m.to_applied_move());

                    match next {
                        (state,mc,_) => {
                            if is_sennichite {
                                let s = if Rule::is_mate(gs.teban.opposite(),&state) {
                                    Score::NEGINFINITE
                                } else {
                                    Score::Value(0)
                                };
                                if s > scoreval {
                                    scoreval = s;
                                    best_moves = VecDeque::new();
                                    best_moves.push_front(m.to_applied_move());
                                    best_moves.push_front(prev_move);

                                    if scoreval >= gs.beta {
                                        return Ok(EvaluationResult::Immediate(scoreval,gs.depth,gs.mhash,gs.shash,best_moves));
                                    }
                                }

                                if alpha < scoreval {
                                    alpha = scoreval;
                                }
                                continue;
                            }

                            let state = Arc::new(state);
                            let mc = Arc::new(mc);

                            let mut gs = GameState {
                                teban: gs.teban.opposite(),
                                state: &state,
                                alpha: -gs.beta,
                                beta: -alpha,
                                m: Some(m.to_applied_move()),
                                mc: &mc,
                                obtained: obtained,
                                current_kyokumen_map: &current_kyokumen_map,
                                self_checkmate_state_map: Arc::clone(&gs.opponent_checkmate_state_map),
                                opponent_checkmate_state_map: Arc::clone(&gs.self_checkmate_state_map),
                                oute_kyokumen_map: &oute_kyokumen_map,
                                mhash: mhash,
                                shash: shash,
                                depth: depth - 1,
                                current_depth: gs.current_depth + 1,
                                node_count: mvs_count as u128 - processed_nodes as u128
                            };

                            let strategy = Recursive::new();

                            match strategy.search(env, &mut gs, event_dispatcher,evalutor)? {
                                EvaluationResult::Timeout => {
                                    return Ok(EvaluationResult::Timeout);
                                },
                                EvaluationResult::Immediate(s,depth,mhash,shash,mvs) => {
                                    if !env.kyokumen_score_map.contains_key(&(gs.teban.opposite(),mhash,shash)) {
                                        env.kyokumen_score_map.insert_new((gs.teban.opposite(), mhash, shash), (s, depth));
                                    }

                                    if let Some(mut g) = env.kyokumen_score_map.get_mut(&(gs.teban.opposite(), mhash, shash)) {
                                        let (ref mut score, ref mut d) = *g;

                                        if *d < depth {
                                            *d = depth;
                                            *score = s;
                                        }
                                    }

                                    if !env.kyokumen_score_map.contains_key(&(gs.teban,mhash,shash)) {
                                        env.kyokumen_score_map.insert_new((gs.teban, mhash, shash), (-s, depth));
                                    }

                                    if let Some(mut g) = env.kyokumen_score_map.get_mut(&(gs.teban, mhash, shash)) {
                                        let (ref mut score, ref mut d) = *g;

                                        if *d < depth {
                                            *d = depth;
                                            *score = -s;
                                        }
                                    }

                                    if -s > scoreval {
                                        scoreval = -s;
                                        best_moves = mvs;
                                        best_moves.push_front(prev_move);

                                        if scoreval >= gs.beta {
                                            return Ok(EvaluationResult::Immediate(scoreval, gs.depth,gs.mhash,gs.shash,best_moves));
                                        }
                                    }

                                    if alpha < -s {
                                        alpha = -s;
                                    }
                                },
                                EvaluationResult::Async(r) => {
                                    await_mvs.push(r);
                                }
                            }

                            event_dispatcher.dispatch_events(self,&*env.event_queue)?;

                            if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                                return Ok(EvaluationResult::Timeout);
                            } else if self.timeout_expected(env,start_time,gs.current_depth,parent_nodes,mvs_count as u32, processed_nodes) {
                                return Ok(EvaluationResult::Immediate(scoreval,gs.depth,gs.mhash,gs.shash,best_moves));
                            }
                        }
                    }
                },
                None => (),
            }
        }

        if !await_mvs.is_empty() {
            evalutor.begin_transaction()?;
        }

        let mut opt_error = None;

        for r in await_mvs {
            match r.recv().map_err(|e| ApplicationError::from(e)) {
                Ok((m,s)) => {
                    let s = Score::Value(s);

                    opt_error = opt_error.and(self.send_score(env,gs.teban,-s).err());

                    if -s > scoreval {
                        scoreval = -s;
                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        best_moves.push_front(prev_move);
                    }
                },
                Err(e) => {
                    opt_error = Some(e);
                }
            }
        }

        if let Some(e) = opt_error {
            Err(e)
        } else {
            Ok(EvaluationResult::Immediate(scoreval, gs.depth,gs.mhash,gs.shash,best_moves))
        }
    }
}