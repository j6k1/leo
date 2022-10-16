use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
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
use crate::error::ApplicationError::AllResultSendError;
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

pub trait Search<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self,env:&mut Environment<L,S>, gs:GameState<'a,L,S>, evalutor: &Evalutor, solver: &mut Solver) -> Result<EvaluationResult,ApplicationError>;

    fn timelimit_reached(&self,env:&mut Environment<L,S>) -> bool {
        let network_delay = env.network_delay;
        env.limit.map_or(false,|l| {
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

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
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
#[derive(Clone, PartialEq, Debug)]
pub enum EvaluationResult {
    Immediate(Score, VecDeque<AppliedMove>,AppliedMove),
    Async(Receiver<Result<EvaluationResult,ApplicationError>>),
    None,
    Timeout
}
#[derive(Clone, PartialEq, Debug)]
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
    event_queue:Arc<Mutex<UserEventQueue>>,
    info_sender:S,
    on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
    hasher:Arc<KyokumenHash<u64>>,
    limit:Option<Instant>,
    current_limit:Option<Instant>,
    max_depth:u32,
    adjust_depth:bool,
    network_delay:u32,
    display_evalute_score:bool,
    max_threads:u32,
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
            max_depth:self.max_depth,
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
               max_depth:u32,
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
            think_start_time:think_start_time,
            limit:limit,
            current_limit:current_limit,
            max_depth:max_depth,
            adjust_depth:adjust_depth,
            network_delay:network_delay,
            display_evalute_score:display_evalute_score,
            max_threads:max_threads,
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

    pub fn create_event_dispatcher<'a,T>(on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,stop:&Arc<AtomicBool>,quited:&Arc<AtomicBool>)
                                       -> UserEventDispatcher<'a,T,ApplicationError,L> {

        let mut event_dispatcher = USIEventDispatcher::new(&on_error_handler.clone());

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

    pub fn termination(&self,
                        r:&Receiver<Result<EvaluationResult,ApplicationError>>,
                        mut is_timeout:bool,
                        ignore_await:bool,
                        await_mvs:Vec<Receiver<(AppliedMove,i32)>>,
                        threads:u32,
                        env:&mut Environment<L,S>,
                        score:Score) -> Result<EvaluationResult,ApplicationError> {
        env.stop.store(true,atomic::Ordering::Release);

        let mut best_moves = VecDeque::new();

        let mut score = score;
        let mut opt_error = None;

        for _ in threads..env.max_threads {
            match r.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r) {
                Ok(EvaluationResult::Immediate(s,mut mvs,m)) if !is_timeout => {
                    if -s > score {
                        score = -s;
                        best_moves = mvs;
                        best_moves.push_front(m);
                    }
                },
                Ok(EvaluationResult::Timeout) => {
                    is_timeout = true;
                }
                Err(e) => {
                    opt_error = Some(Err(e));
                },
                _ => ()
            }
        }

        if is_timeout {
            return Ok(EvaluationResult::Timeout);
        }

        if !ignore_await {
            for r in await_mvs {
                match r.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r) {
                    Ok((m,s)) if !is_timeout => {
                        if -s > score {
                            score = -s;
                            best_moves = VecDeque::new();
                            best_moves.push_front(m);
                        }
                    },
                    Err(e) => {
                        opt_error = Some(Err(e));
                    },
                    _ => ()
                }
            }
        }

        match opt_error {
            Some(e) => {
                Err(e)
            },
            None if best_moves.len() == 0 => {
                Ok(EvaluationResult::Immediate(NEGINFINITE,VecDeque::new(),m))
            },
            None => {
                Ok(EvaluationResult::Immediate(score,best_moves,m))
            }
        }
    }

    pub fn before_search<'a>(&self,
                             env: &mut Environment<L, S>,
                             gs: &'a mut GameState<'a, L, S>,
                             evalutor: &Evalutor,
                             solver: &mut Solver)
                             -> Result<BeforeSearchResult, ApplicationError> {
        if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if let None = env.kyokumen_score_map.get(gs.teban,&gs.mhash,&gs.shash) {
            env.nodes.fetch_add(1,atomic::Ordering::Release);
        }

        if let Some(ObtainKind::Ou) = gs.obtained {
            return Ok(gs.m.map(|m| {
                BeforeSearchResult::Complete(EvaluationResult::Immediate(NEGINFINITE,VecDeque::new(),m))
            }).unwrap_or(BeforeSearchResult::Complete(EvaluationResult::None)));
        }

        if Rule::is_mate(gs.teban,&*gs.state) {
            return Ok(gs.m.map(|m| {
                BeforeSearchResult::Complete(EvaluationResult::Immediate(NEGINFINITE,VecDeque::new(),m))
            }).unwrap_or(BeforeSearchResult::Complete(EvaluationResult::None)));
        }

        if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
            self.send_message(env,"think timeout!");
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if let Some(m) = gs.m {
            if let Some(&(s, d)) = env.kyokumen_score_map.get(gs.teban, &gs.mhash, &gs.shash) {
                match s {
                    Score::INFINITE => {
                        self.send_message(env, "score corresponding to the hash was found in the map. value is infinite.");
                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(INFINITE, VecDeque::new(), m)
                        ));
                    },
                    Score::NEGINFINITE => {
                        self.send_message(env, "score corresponding to the hash was found in the map. value is neginfinite.");
                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(NEGINFINITE, VecDeque::new(), m)
                        ));
                    },
                    Score::Value(s) if d >= depth => {
                        self.send_message(env & format!("score corresponding to the hash was found in the map. value is {}.", s));
                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(s, VecDeque::new(), m)
                        ));
                    },
                    _ => ()
                }
            }

            if (gs.depth == 0 || gs.current_depth > env.max_depth) && !Rule::is_mate(gs.teban.opposite(), &*gs.state) {
                match env.solver.checkmate()? {
                    MaybeMate::MateMoves(_, mvs) => {
                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(INFINITE, mvs, m)
                        ));
                    },
                    MaybeMate::MateMoves(_, _) => {
                        return Ok(BeforeSearchResult::Complete(
                            EvaluationResult::Immediate(NEGINFINITE, mvs, m)
                        ));
                    },
                    _ => ()
                }
            }
        }

        let _ = event_dispatcher.dispatch_events(self,&*env.event_queue);

        if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
            self.send_message(env,"think timeout!");
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        let mvs = if Rule::is_mate(teban.opposite(),&*gs.state) {
            if gs.depth == 0 || gs.current_depth == env.max_depth {
                if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                    self.send_message(env,"think timeout!");
                    return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
                }
            }

            let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                return gs.m.map(|m| {
                    Ok(BeforeSearchResult::Complete(
                        EvaluationResult::Immediate(NEGINFINITE, VecDeque::new(), m)
                    ))
                }).unwrap_or(Ok(BeforeSearchResult::Complete(EvaluationResult::None)));
            } else if gs.depth == 0 || gs.current_depth == env.max_depth {
            } else {
                mvs
            }
        } else {
            if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                self.send_message(env, "think timeout!");
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
            }

            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            mvs
        };

        let mut mvs = gs.mvs.into_iter().map(|m| {
            let ps = Rule::apply_move_to_partial_state_none_check(gs.state,gs.teban,gs.mc,gs.m.to_applied_move());

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
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self,
                  env: &mut Environment<L, S>,
                  gs: GameState<'a, L, S>,
                  evalutor: &Evalutor,
                  solver: &mut Solver)
        -> Result<EvaluationResult, ApplicationError> {
        let mut gs = gs;

        let mvs = match self.before_search(env,&mut gs,evalutor,solver)? {
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::Mvs(mvs) => {
                mvs
            }
        };

        let mut alpha = gs.alpha;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();
        let mut await_mvs = vec![];
        let mut is_timeout = false;

        let (sender,receiver):(_,Receiver<Result<EvaluationResult, ApplicationError>>) = mpsc::channel();

        let mut threads = env.max_threads;s

        let mvs_count = mvs.len() as u64;

        let mut it = mvs.into_iter();
        let mut processed_nodes:u32 = 0;
        let start_time = Instant::now();

        loop {
            if threads == 0 {
                let r = receiver.recv()?.map_err(|e| ApplicationError::from(e));

                threads += 1;
                processed_nodes += 1;

                let nodes = gs.node_count * mvs_count - processed_nodes as u64;

                match r {
                    Ok(EvaluationResult::Immediate(s,mvs,m)) => {
                        if let Some(&(_,d)) = env.kyokumen_score_map.get(gs.teban.opposite(),&gs.mhash,&gs.shash) {
                            if d < gs.depth {
                                env.kyokumen_score_map.insert(gs.teban.opposite(), gs.mhash, gs.shash, (s,gs.depth));
                            }
                        } else {
                            env.kyokumen_score_map.insert(gs.teban.opposite(), gs.mhash, gs.shash, (s,gs.depth));
                        }

                        if let Some(&(_,d)) = env.kyokumen_score_map.get(gs.teban,&gs.mhash,&gs.shash) {
                            if d < gs.depth {
                                env.kyokumen_score_map.insert(gs.teban, gs.mhash, gs.shash, (-s,gs.depth));
                            }
                        } else {
                            env.kyokumen_score_map.insert(gs.teban, gs.mhash, gs.shash, (-s,gs.depth));
                        }

                        if -s > scoreval {
                            self.send_info(env, gs.base_depth,gs.current_depth,&mvs);
                            self.send_score(env,gs.teban,-s);

                            scoreval = -s;
                            mvs.push_front(m);

                            best_moves = mvs;

                            if scoreval >= gs.beta {
                                break;
                            }
                            if alpha < scoreval {
                                alpha = scoreval;
                            }
                        }

                        if self.timeout_expected(env,start_time,gs.current_depth,nodes,processed_nodes) {
                            self.send_message(env,"think timeout!");
                            break;
                        }
                    },
                    Ok(EvaluationResult::Async(r)) => {
                        await_mvs.push(r);
                    },
                    Ok(EvaluationResult::Timeout) => {
                        is_timeout = true;
                        break;
                    },
                    Err(e) => {
                        self.termination(&receiver, false,true,await_mvs,threads, env, scoreval)?;
                        return e;
                    }
                }

                if let Err(e) = gs.event_dispatcher.dispatch_events(gs,&*env.event_queue) {
                    self.termination(&receiver, false,true,await_mvs,threads, env, scoreval)?;
                    return e.map_err(|e| ApplicationError::from(e));
                }

                if self.timelimit_reached(env) || env.stop.load(atomic::Ordering::Acquire) {
                    break;
                }
            } else if let Some(&(priority,m)) = it.next() {
                match self.startup_strategy(env,
                                            &mut gs,
                                            m,
                                            priority) {
                    Some((depth,obtained,mhash,shash,
                         oute_kyokumen_map,
                         current_kyokumen_map,
                         is_sennichite)) => {

                        let m = m.to_applied_move();
                        let next = Rule::apply_move_none_check(&gs.state,gs.teban,gs.mc,m);

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
                                        best_moves.push_front(m);

                                        if alpha < scoreval {
                                            alpha = scoreval;
                                        }
                                        if scoreval >= gs.beta {
                                            return self.termination(&receiver, false,true,await_mvs,threads, env, scorevals);
                                        }
                                    }
                                    continue;
                                }

                                let teban = gs.teban;
                                let mut state = Arc::new(state);
                                let mut mc = Arc::new(mc);
                                let alpha = gs.alpha;
                                let beta = gs.beta;
                                let mut self_already_oute_map = gs.self_already_oute_map.clone();
                                let mut opponent_already_oute_map = gs.opponent_already_oute_map.clone();
                                let current_depth = gs.current_depth;
                                let base_depth = gs.base_depth;
                                let node_count = gs.node_count;

                                let mut env = env.clone();
                                let mut evalutor = evalutor.clone();

                                let sender = sender.clone();

                                let b = std::thread::Builder::new();

                                let _ = b.stack_size(1024 * 1024 * 200).spawn(move || {
                                    let repeat = match alpha {
                                        Score::NEGINFINITE | Score::INFINITE => 1,
                                        Score::Value(_) => 2,
                                    };

                                    let mut a = alpha;

                                    let mut res = Ok(EvaluationResult::Immediate(NEGINFINITE,VecDeque::new(),m));

                                    for i in 0..repeat {
                                        let b = match (i,repeat) {
                                            (0,2) => alpha + 1,
                                            (1,2) | (0,1) => gs.beta,
                                            _ => {
                                                let _ = sender.send(Err(ApplicationError::LogicError(String::from(
                                                    "The combination of i,repeat is invalid."
                                                ))))?;
                                                return;
                                            }
                                        };

                                        let mut event_dispatcher = Self::create_event_dispatcher(&env.on_error_handler,&env.stop,&env.quited);
                                        let mut solver_event_dispatcher = Self::create_event_dispatcher(&env.on_error_handler,&env.stop,&env.quited);

                                        let ms = GameStateForMate {
                                            already_oute_kyokumen_map: &mut self_already_oute_map,
                                            current_depth:current_depth,
                                            mhash:mhash,
                                            shash:shash,
                                            oute_kyokumen_map: &mut oute_kyokumen_map,
                                            current_kyokumen_map: &mut current_kyokumen_map,
                                            ignore_kyokumen_map: KykumenMap::new(),
                                            event_queue:env.event_queue.clone(),
                                            teban:teban,
                                            state:&state,
                                            mc:&mc
                                        };

                                        let mut solver = Solver::new(

                                        );

                                        let gs = GameState {
                                            event_dispatcher: &mut event_dispatcher,
                                            solver_event_dispatcher: &mut solver_event_dispatcher,
                                            teban: teban,
                                            state: &state,
                                            alpha: alpha,
                                            beta: beta,
                                            m:m,
                                            mc: &mc,
                                            obtained:obtained,
                                            current_kyokumen_map:&mut current_kyokumen_map,
                                            self_already_oute_map:&mut self_already_oute_map,
                                            opponent_already_oute_map:&mut opponent_already_oute_map,
                                            oute_kyokumen_map:&mut oute_kyokumen_map,
                                            mhash:mhash,
                                            shash:shash,
                                            depth:depth,
                                            current_depth:current_depth,
                                            base_depth:base_depth,
                                            node_count:node_count
                                        };

                                        let (s,r) = mpsc::channel();

                                        let mut strategy  = Recursive::new(s);

                                        let r = strategy.search(&mut env,gs,&evalutor,&mut solver);

                                        match r.map_err(|e| ApplicationError::from(e)) {
                                            Ok(EvaluationResult::Immediate(s,mut mvs,m)) => {
                                               res = Ok(EvaluationResult::Immediate(s,mvs, m));

                                                if -s <= alpha || -s  >= gs.beta {
                                                    break;
                                                } else {
                                                    a = -s;
                                                }
                                            },
                                            r @ Ok(EvaluationResult::Async(_)) => {
                                                let _ = sender.send(r);
                                                a = -s;
                                            },
                                            r @ Ok(EvaluationResult::Timepout) => {
                                                res = r;
                                                break;
                                            }
                                            Err(e) => {
                                                res = Err(e);
                                                break;
                                            }
                                        }
                                    }
                                    let _ = sender.send(res);
                                });

                                threads -= 1;
                            }
                        }
                    },
                    None => (),
                }
            } else {
                break;
            }
        }

        self.termination(&receiver, is_timeout,true,await_mvs,threads, env, scoreval)
    }
}
pub struct Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    l:PhantomData<L>,
    s:PhantomData<S>,
    sender:Sender<Result<EvaluationResult,ApplicationError>>
}
impl<L,S> Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    pub fn new(sender:Sender<Result<EvaluationResult,ApplicationError>>) -> Recursive<L,S> {
        Recursive {
            l:PhantomData::<L>,
            s:PhantomData::<S>,
            sender:sender
        }
    }
}
impl<L,S> Search<L,S> for Recursive<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a>(&self,
                  env: &mut Environment<L, S>, gs: GameState<'a, L, S>,
                  evalutor: &Evalutor, solver: &mut Solver) -> Result<EvaluationResult, ApplicationError> {
        todo!()
    }
}