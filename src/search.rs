use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Neg, Sub};
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use usiagent::command::{UsiInfoSubCommand, UsiScore, UsiScoreMate};
use usiagent::error::EventHandlerError;
use usiagent::event::{EventDispatcher, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, UserEventQueue, USIEventDispatcher};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, Rule, State};
use usiagent::shogi::{MochigomaCollections, MochigomaKind, ObtainKind, Teban};
use crate::error::{ApplicationError};
use crate::initial_estimation::{attack_priority, defense_priority};
use crate::nn::Evalutor;
use crate::solver::{GameStateForMate, MaybeMate, Solver};
use crate::transposition_table::{TT, TTPartialEntry, ZobristHash};

pub const BASE_DEPTH:u32 = 2;
pub const MAX_DEPTH:u32 = 6;
pub const TIMELIMIT_MARGIN:u64 = 50;
pub const NETWORK_DELAY:u32 = 1100;
pub const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;
pub const MAX_THREADS:u32 = 1;
pub const MAX_PLY:u32 = 200;
pub const MAX_PLY_TIMELIMIT:u64 = 0;
pub const TURN_COUNT:u32 = 50;
pub const MIN_TURN_COUNT:u32 = 5;
pub const DEFAULT_STRICT_MATE:bool = true;
pub const DEFAULT_MATE_HASH:usize = 8;

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

        Ok(env.info_sender.send(commands)?)
    }

    fn send_message_immediate(&self, env:&mut Environment<L,S>, message:&str) -> Result<(),ApplicationError>
        where Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Str(String::from(message)));

        Ok(env.info_sender.send_immediate(commands)?)
    }

    fn send_seldepth(&self, env:&mut Environment<L,S>,
                            depth:u32, seldepth:u32) -> Result<(),ApplicationError> {

        let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
        commands.push(UsiInfoSubCommand::Depth(depth));
        commands.push(UsiInfoSubCommand::SelDepth(seldepth));

        Ok(env.info_sender.send(commands)?)
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

        commands.push(UsiInfoSubCommand::Depth(depth));

        if depth < seldepth {
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));
        }

        if pv.len() > 0 {
            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));
        }
        commands.push(UsiInfoSubCommand::Time((Instant::now() - env.think_start_time).as_millis() as u64));

        env.info_sender.send(commands)?;
        Ok(env.info_sender.flush()?)
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
                            -> Option<(u32,Option<ObtainKind>,ZobristHash<u64>,KyokumenMap<u64,()>,KyokumenMap<u64,u32>,bool)> {

        let obtained = match m {
            LegalMove::To(ref m) => m.obtained(),
            _ => None,
        };

        let mut oute_kyokumen_map = gs.oute_kyokumen_map.clone();
        let mut current_kyokumen_map = gs.current_kyokumen_map.clone();

        let zh = {
            let o = match m {
                LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                _ => None
            };

            let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

            let (mhash,shash) = zh.keys();

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

            zh
        };

        let (mhash,shash) = zh.keys();

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

        Some((depth,obtained,zh,oute_kyokumen_map,current_kyokumen_map,is_sennichite))
    }

    fn before_search<'a,'b>(&self,
                         env: &mut Environment<L, S>,
                         gs: &mut GameState<'a>,
                         event_dispatcher:&mut UserEventDispatcher<'b,Self,ApplicationError,L>,
                         evalutor: &Evalutor)
        -> Result<BeforeSearchResult, ApplicationError> {
        if gs.depth == 0 || gs.current_depth >= gs.max_depth {
            return Err(ApplicationError::InvalidStateError(String::from(
                "Invalid search depth (before_search was called with 0 remaining search depth)"
            )));
        }

        if env.base_depth < gs.current_depth {
            self.send_seldepth(env,env.base_depth,gs.current_depth)?;
        }

        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if let Some(ObtainKind::Ou) = gs.obtained {
            let mut mvs = VecDeque::new();
            gs.m.map(|m| mvs.push_front(m));

            return Ok(BeforeSearchResult::CompleteForOpposite(Score::NEGINFINITE,mvs));
        }

        if let Some(m) = gs.m {
            if Rule::is_mate(gs.teban, &*gs.state) {
                let mut mvs = VecDeque::new();
                mvs.push_front(m);
                return Ok(BeforeSearchResult::CompleteForOpposite(Score::NEGINFINITE,mvs));
            }

            {
                let r = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone());

                if let Some(TTPartialEntry {
                                depth: d,
                                score: s,
                                best_move
                            }) = r {

                    match s {
                        Score::INFINITE => {
                            if env.display_evalute_score {
                                self.send_message(env, "score corresponding to the hash was found in the map. value is infinite.")?;
                            }

                            let mut mvs = VecDeque::new();
                            best_move.map(|m| mvs.push_front(m));
                            mvs.push_front(m);

                            return Ok(BeforeSearchResult::CompleteForOpposite(Score::NEGINFINITE, mvs))
                        },
                        Score::NEGINFINITE => {
                            if env.display_evalute_score {
                                self.send_message(env, "score corresponding to the hash was found in the map. value is neginfinite.")?;
                            }

                            let mut mvs = VecDeque::new();
                            best_move.map(|m| mvs.push_front(m));
                            mvs.push_front(m);

                            return Ok(BeforeSearchResult::CompleteForOpposite(Score::INFINITE, mvs));
                        },
                        Score::Value(s) if d as u32 >= gs.depth + 1 => {
                            if env.display_evalute_score {
                                self.send_message(env, &format!("score corresponding to the hash was found in the map. value is {}.", s))?;
                            }

                            let mut mvs = VecDeque::new();
                            best_move.map(|m| mvs.push_front(m));
                            mvs.push_front(m);

                            return Ok(BeforeSearchResult::CompleteForOpposite(-Score::Value(s), mvs));
                        },
                        _ => ()
                    }
                }
            }

            if (gs.depth == 1 || gs.current_depth + 1 == gs.max_depth) && !Rule::is_mate(gs.teban.opposite(), &*gs.state) {
                let (mhash,shash) = gs.zh.keys();

                let ms = GameStateForMate {
                    base_depth: env.base_depth,
                    current_depth: gs.current_depth,
                    mhash: mhash,
                    shash: shash,
                    current_kyokumen_map: gs.current_kyokumen_map,
                    event_queue: env.event_queue.clone(),
                    teban: gs.teban,
                    state: gs.state,
                    mc: gs.mc
                };

                let solver = Solver::new();

                match solver.checkmate(
                    false,
                    false,
                    false,
                    env.mate_hash,
                    env.limit.clone(),
                    env.max_ply_timelimit.map(|l| Instant::now() + l),
                    env.network_delay,
                    env.max_ply.clone(),
                    env.max_nodes.clone(),
                    env.info_sender.clone(),
                    &env.on_error_handler,
                    Arc::clone(&env.hasher),
                    Arc::clone(&env.stop),
                    Arc::clone(&env.nodes),
                    Arc::clone(&env.quited),
                    None,
                    ms
                )? {
                    MaybeMate::MateMoves(mvs) if mvs.len() > 0 => {
                        let next_move = mvs[0];
                        let mut r = mvs;

                        let o = match next_move {
                            LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                            _ => None
                        };

                        self.update_best_move(env,&gs.zh,gs.depth,Score::INFINITE,Some(next_move));

                        let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,next_move.to_applied_move(),&o);

                        r.push_front(m);

                        return Ok(BeforeSearchResult::Complete(EvaluationResult::Immediate(Score::INFINITE, r,zh.clone())));
                    },
                    MaybeMate::MateMoves(mvs) => {
                        let mut r = mvs;
                        r.push_front(m);

                        return Ok(BeforeSearchResult::CompleteForOpposite(Score::NEGINFINITE,r));
                    },
                    _ => ()
                }
            }
        }
        event_dispatcher.dispatch_events(self,&*env.event_queue).map_err(|e| ApplicationError::from(e))?;

        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
            return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
        }

        if gs.depth == 1 || gs.current_depth + 1 == gs.max_depth {
            let mvs = if Rule::is_mate(gs.teban.opposite(),&*gs.state) {
                let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

                if mvs.len() == 0 {
                    let mut mvs = VecDeque::new();
                    gs.m.map(|m| mvs.push_front(m));

                    return Ok(BeforeSearchResult::CompleteForOpposite(Score::INFINITE, mvs));
                } else {
                    mvs
                }
            } else {
                let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);
                mvs
            };

            let mut await_mvs = Vec::with_capacity(mvs.len());

            for m in mvs {
                let (s,r) = mpsc::channel();
                let (state,mc,_) = Rule::apply_move_none_check(&gs.state, gs.teban, gs.mc, m.to_applied_move());

                evalutor.submit(gs.teban,state.get_banmen(),&mc,m,s)?;

                await_mvs.push((r,m));
            }

            return Ok(BeforeSearchResult::AsyncMvs(await_mvs));
        }

        let mvs = if Rule::is_mate(gs.teban.opposite(),&*gs.state) {
            if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
            }

            let mvs = Rule::respond_oute_only_moves_all(gs.teban, &*gs.state, &*gs.mc);

            if mvs.len() == 0 {
                let mut mvs = VecDeque::new();
                gs.m.map(|m| mvs.push_front(m));

                return Ok(BeforeSearchResult::CompleteForOpposite(Score::INFINITE, mvs));
            } else {
                let mut mvs = mvs.into_iter().map(|m| {
                    let o = match m {
                        LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                        _ => None
                    };

                    let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

                    {
                        if let Some(TTPartialEntry { depth: _, score, best_move: _ }) = env.transposition_table.get(&zh).map(|g| g.deref().clone()) {
                            (m, score)
                        } else {
                            (m, Score::Value(0))
                        }
                    }
                }).collect::<Vec<(LegalMove,Score)>>();

                mvs.sort_by(|&a,&b| {
                    b.1.cmp(&a.1).then_with(|| defense_priority(gs.teban,&gs.state,a.0).cmp(&defense_priority(gs.teban,&gs.state,b.0)))
                });
                mvs
            }
        } else {
            if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                return Ok(BeforeSearchResult::Complete(EvaluationResult::Timeout));
            }

            let mvs:Vec<LegalMove> = Rule::legal_moves_all(gs.teban, &*gs.state, &*gs.mc);

            let mut mvs = mvs.into_iter().map(|m| {
                let o = match m {
                    LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                    _ => None
                };

                let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

                {
                    if let Some(TTPartialEntry { depth: _, score, best_move: _ }) = env.transposition_table.get(&zh).map(|g| g.deref().clone()) {
                        (m, score)
                    } else {
                        (m, Score::Value(0))
                    }
                }
            }).collect::<Vec<(LegalMove,Score)>>();

            mvs.sort_by(|&a,&b| {
                b.1.cmp(&a.1).then_with(|| attack_priority(gs.teban,&gs.state,a.0).cmp(&attack_priority(gs.teban,&gs.state,b.0)))
            });
            mvs
        };

        Ok(BeforeSearchResult::Mvs(mvs.into_iter().map(|(m,_)| m).collect::<Vec<LegalMove>>()))
    }

    fn update_best_move<'a>(&self, env: &mut Environment<L, S>,
                            zh: &'a ZobristHash<u64>,
                            depth: u32,
                            score:Score,
                            m: Option<LegalMove>) {
        let mut tte = env.transposition_table.entry(zh);
        let tte = tte.or_default();

        if tte.depth <= depth as i8 {
            tte.depth = depth as i8;
            tte.score = score;
            tte.best_move = m;
        }
    }
}
#[derive(Debug)]
pub enum EvaluationResult {
    Immediate(Score, VecDeque<LegalMove>, ZobristHash<u64>),
    Timeout
}
#[derive(Debug)]
pub enum BeforeSearchResult {
    Complete(EvaluationResult),
    CompleteForOpposite(Score, VecDeque<LegalMove>),
    Mvs(Vec<LegalMove>),
    AsyncMvs(Vec<(Receiver<(LegalMove,i32)>,LegalMove)>),
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
impl Default for Score {
    fn default() -> Self {
        Score::NEGINFINITE
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
    pub mate_hash:usize,
    pub stop:Arc<AtomicBool>,
    pub quited:Arc<AtomicBool>,
    pub transposition_table:Arc<TT<u64,Score,{1<<20},4>>,
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
            mate_hash:self.mate_hash,
            stop:Arc::clone(&self.stop),
            quited:Arc::clone(&self.quited),
            transposition_table:self.transposition_table.clone(),
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
               max_threads:u32,
               mate_hash:usize,
               transposition_table: &Arc<TT<u64,Score,{1 << 20},4>>
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
            mate_hash:mate_hash,
            stop:stop,
            quited:quited,
            transposition_table:Arc::clone(transposition_table),
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
    pub zh:ZobristHash<u64>,
    pub depth:u32,
    pub current_depth:u32,
    pub max_depth:u32
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
                       env:&mut Environment<L,S>,
                       gs: &mut GameState<'a>,
                       evalutor: &Evalutor,
                       score:Score,
                       is_timeout:bool,
                       mut best_moves:VecDeque<LegalMove>) -> Result<EvaluationResult,ApplicationError> {
        let mut score = score;
        let mut last_error = None;

        while evalutor.active_threads() > 0 {
            match self.receiver.recv().map_err(|e| ApplicationError::from(e)).and_then(|r| r) {
                Ok(EvaluationResult::Immediate(s,mvs,_)) => {
                    if -s > score {
                        score = -s;
                        best_moves = mvs;
                        if let Err(e) =  self.send_info(env, gs.max_depth,gs.current_depth,&best_moves,&score) {
                            last_error = Some(Err(e));
                        }
                    }
                },
                e @ Err(_) => {
                    last_error = Some(e);
                }
                _ => ()
            }

            if let Err(e) = evalutor.on_end_thread().map_err(|e| ApplicationError::from(e)) {
                last_error = Some(Err(e));
            }
        }

        if let Some(e) = last_error {
            e
        } else if is_timeout && best_moves.is_empty() {
            Ok(EvaluationResult::Timeout)
        } else {
            Ok(EvaluationResult::Immediate(score, best_moves,gs.zh.clone()))
        }
    }

    fn parallelized<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                           event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                           evalutor: &Evalutor, _: Option<Sender<(VecDeque<LegalMove>,Score,ZobristHash<u64>)>>,
                           best_moves:VecDeque<LegalMove>) -> Result<EvaluationResult,ApplicationError>  {
        let mut gs = gs;
        let mut best_moves = best_moves;

        let mut mvs = match self.before_search(env,&mut gs,event_dispatcher,evalutor)? {
            BeforeSearchResult::Complete(EvaluationResult::Immediate(score,mvs,zh)) => {
                {
                    let mut tte = env.transposition_table.entry(&zh);
                    let tte = tte.or_default();

                    if tte.depth < gs.depth as i8 - 1 {
                        tte.depth = gs.depth as i8 - 1;
                        tte.score = score;
                    }
                }

                self.update_best_move(env,&gs.zh,gs.depth,score,mvs.get(1).cloned());

                return Ok(EvaluationResult::Immediate(score,mvs,gs.zh.clone()));
            },
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::CompleteForOpposite(score,mvs) => {
                return Ok(EvaluationResult::Immediate(-score,mvs,gs.zh.clone()));
            },
            BeforeSearchResult::AsyncMvs(await_mvs) => {
                let mut scorevalue = Score::NEGINFINITE;

                evalutor.on_begin_thread();
                evalutor.begin_transaction()?;

                for r in await_mvs {
                    let (m, s) = r.0.recv()?;
                    env.nodes.fetch_add(1, atomic::Ordering::Release);

                    let s = Score::Value(s);

                    let o = match m {
                        LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                        _ => None
                    };

                    let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

                    {
                        let mut tte = env.transposition_table.entry(&zh);
                        let tte = tte.or_default();

                        if tte.depth < gs.depth as i8 - 1 {
                            tte.depth = gs.depth as i8 - 1;
                            tte.score = s;
                        }
                    }

                    if scorevalue < s {
                        scorevalue = s;

                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        gs.m.map(|m| best_moves.push_front(m));

                        self.send_info(env, gs.max_depth, gs.current_depth, &best_moves, &scorevalue)?;

                        self.update_best_move(env,&gs.zh,gs.depth,scorevalue,Some(m));
                    }
                }

                evalutor.on_end_thread()?;

                return Ok(EvaluationResult::Immediate(scorevalue, best_moves,gs.zh.clone()));
            },
            BeforeSearchResult::Mvs(mvs) => {
                mvs
            }
        };

        if let Some(TTPartialEntry {
                        depth: _,
                        score: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            m.map(|m| mvs.insert(0,m));
        }

        let mvs = mvs.into_iter().map(|m| {
            if let LegalMove::To(ref mv) = m {
                if let Some(&ObtainKind::Ou) = mv.obtained().as_ref() {
                    return (1000,false,m);
                }
            }

            if Rule::is_oute_move(gs.state,gs.teban,m) {
                (200,true,m)
            } else {
                (0,false,m)
            }
        }).collect::<Vec<(u32,bool,LegalMove)>>();

        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;

        let mvs_count = mvs.len() as u64;

        let mut threads = env.max_threads.min(mvs_count as u32);
        let mut force_recv = false;

        let sender = self.sender.clone();

        let mut it = mvs.into_iter();

        let mut is_timeout = false;

        loop {
            if threads == 0 || force_recv {
                let r = self.receiver.recv();

                evalutor.on_end_thread()?;

                if evalutor.active_threads() == 0 {
                    force_recv = false;
                }

                let r = r?.map_err(|e| ApplicationError::from(e))?;

                threads += 1;

                match r {
                    EvaluationResult::Immediate(s, mvs,zh) => {
                        {
                            let mut tte = env.transposition_table.entry(&zh);
                            let tte = tte.or_default();

                            if tte.depth < gs.depth as i8 - 1 {
                                tte.depth = gs.depth as i8 - 1;
                                tte.score = -s;
                            }
                        }

                        if -s > scoreval {
                            scoreval = -s;

                            best_moves = mvs;

                            self.send_info(env, gs.max_depth, gs.current_depth, &best_moves, &scoreval)?;

                            self.update_best_move(env,&gs.zh,gs.depth,scoreval,best_moves.front().cloned());

                            if scoreval >= beta {
                                break;
                            }

                            if alpha < scoreval {
                                alpha = scoreval;
                            }
                        }

                        if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                            if best_moves.is_empty() {
                                is_timeout = true;
                            }
                            break;
                        }
                    },
                    EvaluationResult::Timeout if best_moves.is_empty() => {
                        is_timeout = true;
                        break;
                    },
                    EvaluationResult::Timeout => {
                        break;
                    }
                }

                let event_queue = Arc::clone(&env.event_queue);
                event_dispatcher.dispatch_events(self, &*event_queue)?;

                if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                    is_timeout = true;
                    break;
                }
            } else if let Some((priority, is_oute, m)) = it.next() {
                match self.startup_strategy(env,
                                            gs,
                                            m,
                                            priority,
                                            is_oute) {
                    Some((depth, obtained, zh,
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

                                        self.update_best_move(env,&gs.zh,gs.depth,scoreval,Some(m));

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
                                let max_depth = gs.max_depth;

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
                                        zh: zh.clone(),
                                        depth: depth - 1,
                                        current_depth: current_depth + 1,
                                        max_depth:max_depth
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
                force_recv = true;
            } else {
                break;
            }
        }

        self.termination(env, &mut gs,evalutor,scoreval,is_timeout, best_moves)
    }
}
impl<L,S> Search<L,S> for Root<L,S> where L: Logger + Send + 'static, S: InfoSender {
    fn search<'a,'b>(&self,env:&mut Environment<L,S>, gs:&mut GameState<'a>,
                     event_dispatcher:&mut UserEventDispatcher<'b,Root<L,S>,ApplicationError,L>,
                     evalutor: &Evalutor) -> Result<EvaluationResult,ApplicationError> {
        let base_depth = gs.depth.min(env.max_depth);
        let mut depth = 1;
        let mut best_moves = VecDeque::new();
        let mut result = None;

        loop {
            gs.depth = depth;
            gs.max_depth = env.max_depth - (base_depth - depth);

            let current_result = self.parallelized(env, gs, event_dispatcher, evalutor, None, best_moves.clone())?;

            depth += 1;

            match current_result {
                r @ EvaluationResult::Immediate(_,_,_) if depth == base_depth + 1 => {
                    return Ok(r);
                },
                EvaluationResult::Immediate(s,mvs,zh) => {
                    best_moves = mvs.clone();
                    result = Some(EvaluationResult::Immediate(s,mvs,zh));
                },
                EvaluationResult::Timeout if best_moves.is_empty() => {
                    return Ok(EvaluationResult::Timeout)
                },
                EvaluationResult::Timeout => {
                    return Ok(result.unwrap_or(EvaluationResult::Timeout));
                }
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

        let mut mvs = match self.before_search(env,&mut gs,event_dispatcher,evalutor)? {
            BeforeSearchResult::Complete(EvaluationResult::Immediate(score,mvs,zh)) => {
                {
                    let mut tte = env.transposition_table.entry(&zh);
                    let tte = tte.or_default();

                    if tte.depth < gs.depth as i8 - 1 {
                        tte.depth = gs.depth as i8 - 1;
                        tte.score = score;
                    }
                }
                self.update_best_move(env,&gs.zh,gs.depth,score,mvs.get(1).cloned());

                return Ok(EvaluationResult::Immediate(score,mvs,gs.zh.clone()));
            },
            BeforeSearchResult::Complete(r) => {
                return Ok(r);
            },
            BeforeSearchResult::CompleteForOpposite(score,mvs) => {
                return Ok(EvaluationResult::Immediate(-score,mvs,gs.zh.clone()));
            },
            BeforeSearchResult::AsyncMvs(await_mvs) => {
                let prev_move = gs.m.ok_or(ApplicationError::LogicError(String::from(
                    "move is not set."
                )))?;

                evalutor.begin_transaction()?;

                let mut best_moves = VecDeque::new();
                let mut scorevalue = Score::NEGINFINITE;

                for r in await_mvs {
                    let (m, s) = r.0.recv()?;
                    env.nodes.fetch_add(1, atomic::Ordering::Release);

                    let s = Score::Value(s);

                    let o = match m {
                        LegalMove::To(m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
                        _ => None
                    };

                    let zh = gs.zh.updated(&env.hasher,gs.teban,gs.state.get_banmen(),gs.mc,m.to_applied_move(),&o);

                    {
                        let mut tte = env.transposition_table.entry(&zh);
                        let tte = tte.or_default();

                        if tte.depth < gs.depth as i8 - 1 {
                            tte.depth = gs.depth as i8 - 1;
                            tte.score = s;
                        }
                    }

                    if scorevalue < s {
                        scorevalue = s;

                        self.update_best_move(env,&gs.zh,gs.depth,scorevalue,Some(m));

                        best_moves = VecDeque::new();
                        best_moves.push_front(m);
                        best_moves.push_front(prev_move);
                    }
                }

                return Ok(EvaluationResult::Immediate(scorevalue, best_moves,gs.zh.clone()));
            },
            BeforeSearchResult::Mvs(mvs) => {
                mvs
            }
        };

        if let Some(TTPartialEntry {
                        depth: _,
                        score: _,
                        best_move: m
                    }) = env.transposition_table.get(&gs.zh).map(|tte| tte.deref().clone()) {
            m.map(|m| mvs.insert(0,m));
        }

        let mvs = mvs.into_iter().map(|m| {
            if let LegalMove::To(ref mv) = m {
                if let Some(&ObtainKind::Ou) = mv.obtained().as_ref() {
                    return (1000,false,m);
                }
            }

            if Rule::is_oute_move(gs.state,gs.teban,m) {
                (200,true,m)
            } else {
                (0,false,m)
            }
        }).collect::<Vec<(u32,bool,LegalMove)>>();

        let prev_move = gs.m.ok_or(ApplicationError::LogicError(String::from(
            "move is not set."
        )))?;

        let mut alpha = gs.alpha;
        let beta = gs.beta;
        let mut scoreval = Score::NEGINFINITE;
        let mut best_moves = VecDeque::new();
        let d = gs.depth;

        for &(priority,is_oute,m) in &mvs {
            match self.startup_strategy(env, gs, m, priority, is_oute) {
                Some((depth, obtained, zh,
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

                                    self.update_best_move(env,&gs.zh,d,scoreval,Some(m));

                                    if scoreval >= gs.beta {
                                        best_moves.push_front(prev_move);
                                        return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                    }
                                }

                                if alpha < scoreval {
                                    alpha = scoreval;
                                }
                                continue;
                            }

                            let state = Arc::new(state);
                            let mc = Arc::new(mc);
                            let prev_zh = gs.zh.clone();

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
                                zh: zh.clone(),
                                depth: depth - 1,
                                current_depth: gs.current_depth + 1,
                                max_depth:gs.max_depth
                            };

                            let strategy = Recursive::new();

                            match strategy.search(env, &mut gs, event_dispatcher, evalutor)? {
                                EvaluationResult::Timeout if best_moves.is_empty() => {
                                    return Ok(EvaluationResult::Timeout);
                                },
                                EvaluationResult::Timeout => {
                                    best_moves.push_front(prev_move);
                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves,zh.clone()));
                                },
                                EvaluationResult::Immediate(s, mvs, zh) => {
                                    {
                                        let mut tte = env.transposition_table.entry(&zh);
                                        let tte = tte.or_default();

                                        if tte.depth < gs.depth as i8 - 1 {
                                            tte.depth = gs.depth as i8 - 1;
                                            tte.score = -s;
                                        }
                                    }

                                    if -s > scoreval {
                                        scoreval = -s;

                                        best_moves = mvs;

                                        self.update_best_move(env,&prev_zh,d,scoreval,Some(m));

                                        if scoreval >= beta {
                                            best_moves.push_front(prev_move);
                                            return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                        }
                                    }

                                    if alpha < -s {
                                        alpha = -s;
                                    }
                                }
                            }

                            event_dispatcher.dispatch_events(self, &*env.event_queue)?;

                            if self.timelimit_reached(env) || self.timeout_expected(env) || env.stop.load(atomic::Ordering::Acquire) {
                                if best_moves.is_empty() {
                                    return Ok(EvaluationResult::Timeout);
                                } else {
                                    best_moves.push_front(prev_move);
                                    return Ok(EvaluationResult::Immediate(scoreval, best_moves, gs.zh.clone()));
                                }
                            }
                        }
                    }
                }
                None => (),
            }
        }

        best_moves.push_front(prev_move);

        Ok(EvaluationResult::Immediate(scoreval, best_moves,gs.zh.clone()))
    }
}