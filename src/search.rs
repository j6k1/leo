use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use concurrent_fixed_hashmap::ConcurrentFixedHashMap;
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, Rule, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
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

    fn timeout_expected(&self, env:&mut Environment<L,S>) -> bool {
        env.current_limit.map(|l| Instant::now() >= l).unwrap_or(false)
    }

    fn send_message(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_message_immediate(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send_immediate(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_seldepth(&self, env:&mut Environment<L,S>,
                            depth:u32, seldepth:u32) -> Result<(),ApplicationError> {

        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));
        commands.push(UsiInfoSubCommand::SelDepth(seldepth));

        Ok(env.info_sender.send(commands).map_err(|e| SendSelDepthError::from(e))?)
    }

    fn send_info(&self, env:&mut Environment<L,S>,
                      depth:u32, seldepth:u32, pv:&VecDeque<LegalMove>, score:&Score) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

        let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

        match score {
            Score::INFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Plus)))
            },
            Score::NEGINFINITE => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Mate(UsiScoreMate::Minus)))
            },
            Score::Value(s) => {
                commands.push(UsiInfoSubCommand::Score(UsiScore::Cp(*s as i64)))
            }
        }
        if depth < seldepth {
            commands.push(UsiInfoSubCommand::Depth(depth));
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        Ok(env.info_sender.send_immediate(commands).map_err(|e| SendSelDepthError::from(e))?)
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
                            m:LegalMove, priority:u32,is_oute:bool)
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

            if is_oute {
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

        if !is_oute {
            oute_kyokumen_map.clear(gs.teban);
        }

        let depth = if priority > 1 {
            gs.depth + 1
        } else {
            gs.depth
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

        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
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
                    base_depth: env.base_depth,
                    current_depth: 0,
                    mhash: gs.mhash,
                    shash: gs.shash,
                    current_kyokumen_map: gs.current_kyokumen_map,
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
                    env.info_sender.clone(),
                    &env.on_error_handler,
                    Arc::clone(&env.hasher),
                    Arc::clone(&env.stop),
                    Arc::clone(&env.quited),
                    ms
                )? {
                    MaybeMate::MateMoves(mvs) => {
                        let mut r = mvs;
                        r.push_front(m);

                        return Ok(BeforeSearchResult::Complete(EvaluationResult::Immediate(INFINITE, gs.depth, gs.mhash, gs.shash,r)));
                    },
                    _ => ()
                }
            }
        }
        event_dispatcher.dispatch_events(self,&*env.event_queue).map_err(|e| ApplicationError::from(e))?;

        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
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
                if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
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
            if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
            }

            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            mvs
        };

        let mut mvs = mvs.into_iter().map(|m| {
            if let LegalMove::To(ref mv) = m {
                if let Some(&ObtainKind::Ou) = mv.obtained().as_ref() {
                    return (1000,false,m);
                }
            }

            if Rule::is_oute_move(gs.state,gs.teban,m) {
                (200,true,m)
            } else {
                match m {
                    LegalMove::To(ref mv) => {
                        match mv.obtained().as_ref() {
                            Some(&ObtainKind::Ou) => (1000,false,m),
                            Some(&ObtainKind::HishaN) => (100,false,m),
                            Some(&ObtainKind::Hisha) => (95,false,m),
                            Some(&ObtainKind::KakuN) => (80,false,m),
                            Some(&ObtainKind::Kaku) => (75,false,m),
                            Some(&ObtainKind::Kin) => (70,false,m),
                            Some(&ObtainKind::GinN) => (65,false,m),
                            Some(&ObtainKind::Gin) => (60,false,m),
                            Some(&ObtainKind::KeiN) => (55,false,m),
                            Some(&ObtainKind::Kei) => (50,false,m),
                            Some(&ObtainKind::KyouN) => (45,false,m),
                            Some(&ObtainKind::Kyou) => (40,false,m),
                            Some(&ObtainKind::FuN) => (35,false,m),
                            Some(&ObtainKind::Fu) => (30,false,m),
                            None => (1,false,m),
                        }
                    },
                    _ => (1,false,m),
                }
            }
        }).collect::<Vec<(u32,bool,LegalMove)>>();

        mvs.sort_by(|a,b| b.0.cmp(&a.0));

        Ok(BeforeSearchResult::Mvs(mvs))
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Immediate(Score, u32, u64, u64, VecDeque<LegalMove>),
    Async(Receiver<(LegalMove,i32)>),
    Timeout
}
#[derive(Debug)]
pub enum BeforeSearchResult {
    Complete(EvaluationResult),
    Mvs(Vec<(u32,bool,LegalMove)>)
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
            turn_count:turn_count,
            min_turn_count:min_turn_count,
            base_depth:base_depth,
            max_depth:max_depth,
            max_nodes:max_nodes,
            max_ply:max_ply,
            max_ply_mate:max_ply_mate,
            max_ply_timelimit:max_ply_timelimit,
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
    pub m:Option<LegalMove>,
    pub mc:&'a Arc<MochigomaCollections>,
    pub obtained:Option<ObtainKind>,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    pub oute_kyokumen_map:&'a KyokumenMap<u64,()>,
    pub mhash:u64,
    pub shash:u64,
    pub depth:u32,
    pub current_depth:u32
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
                       await_mvs:Vec<Receiver<(LegalMove,i32)>>,
                       env:&mut Environment<L,S>,
                       gs: &mut GameState<'a>,
                       evalutor: &Evalutor,
                       score:Score,
                       is_timeout:bool,
                       mut best_moves:VecDeque<LegalMove>) -> Result<EvaluationResult,ApplicationError> {
        env.stop.store(true,atomic::Ordering::Release);

        let mut score = score;

        while evalutor.active_threads() > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r)? {
                EvaluationResult::Immediate(s,_,_,_,mvs) => {
                    if -s > score {
                        score = -s;
                        best_moves = mvs;
                        self.send_info(env, env.base_depth,gs.current_depth,&best_moves,&score)?;
                    }
                },
                _ => ()
            }
            evalutor.on_end_thread().map_err(|e| ApplicationError::from(e))?;
        }

        for r in await_mvs {
            match r.recv_timeout(Duration::from_secs(1000)).map_err(|e| ApplicationError::from(e))? {
                (m,s) => {
                    env.nodes.fetch_add(1,atomic::Ordering::Release);

                    let s = Score::Value(s);

                    self.send_score(env,gs.teban,-s)?;

                    if -s > score {
                        score = -s;
                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        self.send_info(env, env.base_depth,gs.current_depth,&best_moves,&score)?;
                    }
                }
            }
        }

        if is_timeout && best_moves.is_empty() {
            Ok(EvaluationResult::Timeout)
        } else {
            Ok(EvaluationResult::Immediate(score, gs.depth, gs.mhash, gs.shash, best_moves))
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

        let mut is_timeout = false;

        loop {
            if threads == 0 {
                let r = self.receiver.recv();

                evalutor.on_end_thread()?;

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

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

                        if -s > scoreval {
                            scoreval = -s;

                            self.send_info(env, env.base_depth,gs.current_depth,&mvs, &scoreval)?;

                            best_moves = mvs;

                            if scoreval >= beta {
                                break;
                            }

                            if alpha < scoreval {
                                alpha = scoreval;
                            }
                        }

                        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                            is_timeout = true;
                            break;
                        }
                    },
                    EvaluationResult::Async(r) => {
                        await_mvs.push(r);
                    },
                    EvaluationResult::Timeout => {
                        is_timeout = true;
                        break;
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self,&*event_queue)?;

                if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                    is_timeout = true;
                    break;
                }
            } else if let Some((priority,is_oute,m)) = it.next() {
                match self.startup_strategy(env,
                                            gs,
                                            m,
                                            priority,
                                            is_oute) {
                    Some((depth, obtained, mhash, shash,
                             oute_kyokumen_map,
                             current_kyokumen_map,
                             is_sennichite)) => {
                        let next = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

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
                                            break;
                                        }
                                    }
                                    continue;
                                }

                                let teban = gs.teban;
                                let state = Arc::new(state);
                                let mc = Arc::new(mc);
                                let alpha = alpha;
                                let beta = beta;
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
                                        oute_kyokumen_map: &oute_kyokumen_map,
                                        mhash: mhash,
                                        shash: shash,
                                        depth: depth - 1,
                                        current_depth: current_depth + 1
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

        self.termination(await_mvs,env, &mut gs,evalutor,scoreval,is_timeout, best_moves)
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let r = self.parallelized(env,gs,event_dispatcher,evalutor);

        env.stop.store(true,atomic::Ordering::Release);

        match r {
            Ok(r) => {
                while evalutor.active_threads() > 0 {
                    self.receiver.recv()?.map_err(|e| ApplicationError::from(e))?;
                    evalutor.on_end_thread()?;
                }

                Ok(r)
            },
            Err(e) => {
                Err(e)
            }
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

        let teban = gs.teban;
        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();

        let mut await_mvs = vec![];

        for &(priority,is_oute,m) in &mvs {
            match self.startup_strategy(env,gs,m,priority,is_oute) {
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
                                    best_moves.push_front(m);
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
                                m: Some(m),
                                mc: &mc,
                                obtained: obtained,
                                current_kyokumen_map: &current_kyokumen_map,
                                oute_kyokumen_map: &oute_kyokumen_map,
                                mhash: mhash,
                                shash: shash,
                                depth: depth - 1,
                                current_depth: gs.current_depth + 1
                            };

                            let strategy = Recursive::new();

                            match strategy.search(env, &mut gs, event_dispatcher,evalutor)? {
                                EvaluationResult::Timeout if best_moves.is_empty() => {
                                    return Ok(EvaluationResult::Timeout);
                                },
                                EvaluationResult::Timeout => {
                                    return Ok(EvaluationResult::Immediate(scoreval, gs.depth,gs.mhash,gs.shash,best_moves));
                                },
                                EvaluationResult::Immediate(s,depth,mhash,shash,mvs) => {
                                    if !env.kyokumen_score_map.contains_key(&(teban.opposite(),mhash,shash)) {
                                        env.kyokumen_score_map.insert_new((teban.opposite(), mhash, shash), (s, depth));
                                    }

                                    if let Some(mut g) = env.kyokumen_score_map.get_mut(&(teban.opposite(), mhash, shash)) {
                                        let (ref mut score, ref mut d) = *g;

                                        if *d < depth {
                                            *d = depth;
                                            *score = s;
                                        }
                                    }

                                    if -s > scoreval {
                                        scoreval = -s;

                                        best_moves = mvs;
                                        best_moves.push_front(prev_move);

                                        if scoreval >= beta {
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

                            if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                                if best_moves.is_empty() {
                                    return Ok(EvaluationResult::Timeout);
                                } else {
                                    return Ok(EvaluationResult::Immediate(scoreval, gs.depth, gs.mhash, gs.shash, best_moves));
                                }
                            }
                        }
                    }
                }
                None => (),
            }
        }

        if !await_mvs.is_empty() {
            evalutor.begin_transaction()?;
        }

        for r in await_mvs {
            match r.recv_timeout(Duration::from_secs(1000)).map_err(|e| ApplicationError::from(e))? {
                (m,s) => {
                    env.nodes.fetch_add(1,atomic::Ordering::Release);

                    let s = Score::Value(s);

                    self.send_score(env,gs.teban,-s)?;

                    if -s > scoreval {
                        scoreval = -s;
                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        best_moves.push_front(prev_move);
                    }
                }
            }
        }

        Ok(EvaluationResult::Immediate(scoreval, gs.depth,gs.mhash,gs.shash,best_moves))
    }
}