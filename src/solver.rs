use std::collections::VecDeque;
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Instant;
use concurrent_fixed_hashmap::ConcurrentFixedHashMap;

use usiagent::event::{EventQueue, UserEvent, UserEventKind};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, State};
use usiagent::shogi::*;
use crate::error::ApplicationError;
use crate::search;
use crate::solver::checkmate::{AscComparator, CheckmateStrategy, DescComparator};

#[derive(Debug,Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(u32,VecDeque<LegalMove>),
    MaxDepth,
    MaxNodes,
    Timeout,
    Unknown,
    Aborted
}

pub struct GameStateForMate<'a> {
    pub checkmate_state_map:Arc<ConcurrentFixedHashMap<(Teban, u64, u64),bool>>,
    pub unique_kyokumen_map:Arc<ConcurrentFixedHashMap<(Teban, u64, u64),()>>,
    pub current_depth:u32,
    pub mhash:u64,
    pub shash:u64,
    pub oute_kyokumen_map:&'a KyokumenMap<u64,()>,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    pub ignore_kyokumen_map:KyokumenMap<u64,()>,
    pub event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub mc:&'a Arc<MochigomaCollections>
}
pub struct Solver {
}
impl Solver {
    pub fn new() -> Solver {
        Solver {
        }
    }

    pub fn checkmate<'a,L,S>(&self,strict_moves:bool,
                     limit:Option<Instant>,
                     checkmate_limit:Option<Instant>,
                     network_delay:u32,
                     max_depth:Option<u32>,
                     max_nodes:Option<i64>,
                     nodes:Arc<AtomicU64>,
                     info_sender:S,
                     on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
                     hasher:Arc<KyokumenHash<u64>>,
                     base_depth:u32,
                     stop:Arc<AtomicBool>,
                     quited:Arc<AtomicBool>,
                     ms: GameStateForMate) -> Result<MaybeMate,ApplicationError> where L: Logger + Send + 'static, S: InfoSender {
        let (s,receiver) = mpsc::channel();
        let aborted = Arc::new(AtomicBool::new(false));

        {
            let s = s.clone();

            let limit = limit.clone();
            let checkmate_limit = checkmate_limit.clone();
            let network_delay = network_delay.clone();
            let max_depth = max_depth.clone();
            let max_nodes = max_nodes.clone();
            let nodes = Arc::clone(&nodes);
            let info_sender = info_sender.clone();
            let on_error_handler = Arc::clone(&on_error_handler);
            let hasher = Arc::clone(&hasher);
            let stop = Arc::clone(&stop);
            let quited = Arc::clone(&quited);
            let aborted = Arc::clone(&aborted);

            let checkmate_state_map = Arc::clone(&ms.checkmate_state_map);
            let unique_kyokumen_map = Arc::clone(&ms.unique_kyokumen_map);
            let current_depth = ms.current_depth;
            let mhash = ms.mhash;
            let shash = ms.shash;
            let mut ignore_kyokumen_map = ms.ignore_kyokumen_map.clone();
            let mut oute_kyokumen_map = ms.oute_kyokumen_map.clone();
            let mut current_kyokumen_map = ms.current_kyokumen_map.clone();
            let event_queue = Arc::clone(&ms.event_queue);
            let teban = ms.teban;
            let state = Arc::clone(ms.state);
            let mc = Arc::clone(ms.mc);

            std::thread::spawn(move || {
                let mut event_dispatcher = search::Root::<L,S>::create_event_dispatcher(&on_error_handler,&stop,&quited);

                let mut mate_strategy = CheckmateStrategy::new(
                    DescComparator,
                    AscComparator,
                    hasher,
                    strict_moves,
                    limit,
                    checkmate_limit,
                    network_delay,
                    max_depth,
                    max_nodes,
                    info_sender,
                    base_depth,
                    stop,
                    aborted,
                    current_depth);
                if let Err(ref e) = s.send(mate_strategy.oute_process(
                                                                      &checkmate_state_map,
                                                                      &unique_kyokumen_map,
                                                                      current_depth,
                                                                      &nodes,
                                                                      mhash,
                                                                      shash,
                                                                      &mut ignore_kyokumen_map,
                                                                      &mut oute_kyokumen_map,
                                                                      &mut current_kyokumen_map,
                                                                      &event_queue,
                                                                      &mut event_dispatcher,
                                                                      teban,
                                                                      &*state,
                                                                      &*mc)) {
                    let _ = on_error_handler.lock().map(|h| h.call(e));
                }
            });
        }

        {
            let s = s.clone();

            let limit = limit.clone();
            let checkmate_limit = checkmate_limit.clone();
            let network_delay = network_delay.clone();
            let max_depth = max_depth.clone();
            let max_nodes = max_nodes.clone();
            let nodes = Arc::clone(&nodes);
            let info_sender = info_sender.clone();
            let on_error_handler = Arc::clone(&on_error_handler);
            let hasher = Arc::clone(&hasher);
            let stop = Arc::clone(&stop);
            let quited = Arc::clone(&quited);
            let aborted = Arc::clone(&aborted);

            let checkmate_state_map = Arc::clone(&ms.checkmate_state_map);
            let unique_kyokumen_map = Arc::clone(&ms.unique_kyokumen_map);
            let current_depth = ms.current_depth;
            let mhash = ms.mhash;
            let shash = ms.shash;
            let mut ignore_kyokumen_map = ms.ignore_kyokumen_map.clone();
            let mut oute_kyokumen_map = ms.oute_kyokumen_map.clone();
            let mut current_kyokumen_map = ms.current_kyokumen_map.clone();
            let event_queue = Arc::clone(&ms.event_queue);
            let teban = ms.teban;
            let state = Arc::clone(ms.state);
            let mc = Arc::clone(ms.mc);

            std::thread::spawn(move || {
                let mut event_dispatcher = search::Root::<L,S>::create_event_dispatcher(&on_error_handler,&stop,&quited);

                let mut nomate_strategy = CheckmateStrategy::new(
                    AscComparator,
                    DescComparator,
                    hasher,
                    strict_moves,
                    limit,
                    checkmate_limit,
                    network_delay,
                    max_depth,
                    max_nodes,
                    info_sender,
                    base_depth,
                    stop,
                    aborted,
                    current_depth);
                if let Err(ref e) = s.send(nomate_strategy.oute_process(
                                                                        &checkmate_state_map,
                                                                        &unique_kyokumen_map,
                                                                        current_depth,
                                                                        &nodes,
                                                                        mhash,
                                                                        shash,
                                                                        &mut ignore_kyokumen_map,
                                                                        &mut oute_kyokumen_map,
                                                                        &mut current_kyokumen_map,
                                                                        &event_queue,
                                                                        &mut event_dispatcher,
                                                                        teban,
                                                                        &*state,
                                                                        &*mc)) {
                    let _ = on_error_handler.lock().map(|h| h.call(e));
                }
            });
        }

        let r = receiver.recv();

        aborted.store(true,atomic::Ordering::Release);

        let _ = receiver.recv();

        match r {
            Ok(Ok(MaybeMate::MateMoves(depth,mvs))) => {
                Ok(MaybeMate::MateMoves(depth,mvs))
            },
            Ok(Ok(MaybeMate::Nomate)) => {
                Ok(MaybeMate::Nomate)
            },
            Ok(r) => {
                r
            },
            Err(e) => {
                Err(ApplicationError::from(e))
            }
        }
    }
}

pub mod checkmate {
    use std::cmp::Ordering;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::{Arc, atomic, Mutex};
    use std::time::{Duration, Instant};
    use concurrent_fixed_hashmap::ConcurrentFixedHashMap;
    use usiagent::command::UsiInfoSubCommand;
    use usiagent::event::{EventDispatcher, EventQueue, UserEvent, UserEventKind, USIEventDispatcher};
    use usiagent::hash::{KyokumenHash, KyokumenMap};
    use usiagent::logger::Logger;
    use usiagent::player::InfoSender;
    use usiagent::rule::{LegalMove, Rule, State};
    use usiagent::shogi::{MochigomaCollections, MochigomaKind, Teban};
    use crate::error::{ApplicationError, SendSelDepthError};
    use crate::search::{TIMELIMIT_MARGIN};
    use crate::solver::{MaybeMate};

    pub trait Comparator<T>: Clone {
        fn cmp(&mut self,l:&T,r:&T) -> Ordering;
    }
    #[derive(Clone)]
    pub struct AscComparator;

    impl Comparator<(LegalMove,State,MochigomaCollections,usize)> for AscComparator {
        #[inline]
        fn cmp(&mut self,l:&(LegalMove,State,MochigomaCollections,usize),r:&(LegalMove,State,MochigomaCollections,usize)) -> Ordering {
            l.3.cmp(&r.3)
        }
    }

    #[derive(Clone)]
    pub struct DescComparator;

    impl Comparator<(LegalMove,State,MochigomaCollections,usize)> for DescComparator {
        #[inline]
        fn cmp(&mut self,l:&(LegalMove,State,MochigomaCollections,usize),r:&(LegalMove,State,MochigomaCollections,usize)) -> Ordering {
            r.3.cmp(&l.3)
        }
    }

    pub struct CheckmateStrategy<O,R,S>
        where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              S: InfoSender {
        oute_comparator: O,
        response_oute_comparator:R,
        hasher:Arc<KyokumenHash<u64>>,
        strict_moves:bool,
        limit:Option<Instant>,
        checkmate_limit:Option<Instant>,
        network_delay:u32,
        max_depth:Option<u32>,
        max_nodes:Option<i64>,
        info_sender:S,
        base_depth:u32,
        stop:Arc<AtomicBool>,
        aborted:Arc<AtomicBool>,
        current_depth:u32,
        nodes:i64,
    }

    pub type MateStrategy<S> = CheckmateStrategy<DescComparator,AscComparator,S>;
    pub type NomateStrategy<S> = CheckmateStrategy<AscComparator,DescComparator,S>;

    impl<O,R,S> CheckmateStrategy<O,R,S>
        where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              S: InfoSender {
        pub fn new(oute_comparator: O, response_oute_comparator: R,
               hasher:Arc<KyokumenHash<u64>>,
               strict_moves:bool,
               limit:Option<Instant>,
               checkmate_limit:Option<Instant>,
               network_delay:u32,
               max_depth:Option<u32>,
               max_nodes:Option<i64>,
               info_sender:S,
               base_depth:u32,
               stop:Arc<AtomicBool>,
               aborted:Arc<AtomicBool>,
               current_depth:u32,
       ) -> CheckmateStrategy<O,R,S> {
            CheckmateStrategy {
                oute_comparator: oute_comparator,
                response_oute_comparator: response_oute_comparator,
                hasher:hasher,
                strict_moves:strict_moves,
                limit:limit,
                checkmate_limit:checkmate_limit,
                network_delay:network_delay,
                max_depth:max_depth,
                max_nodes:max_nodes,
                info_sender:info_sender,
                base_depth:base_depth,
                stop:stop,
                aborted:aborted,
                current_depth:current_depth,
                nodes:0,
            }
        }

        pub fn oute_process<L: Logger>(&mut self,
                                       checkmate_state_map:&Arc<ConcurrentFixedHashMap<(Teban, u64, u64),bool>>,
                                       unique_kyokumen_map:&Arc<ConcurrentFixedHashMap<(Teban,u64,u64),()>>,
                                       current_depth:u32,
                                       nodes:&Arc<AtomicU64>,
                                       mhash:u64,
                                       shash:u64,
                                       ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                       oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                                       current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                       event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                       event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                       teban:Teban, state:&State, mc:&MochigomaCollections)
                                       -> Result<MaybeMate,ApplicationError>
                where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      S: InfoSender + Send {
            if self.aborted.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            if !unique_kyokumen_map.contains_key(&(teban,mhash,shash)) {
                nodes.fetch_add(1,atomic::Ordering::Release);
                unique_kyokumen_map.insert((teban,mhash,shash),());
            }

            self.nodes += 1;

            if self.max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(current_depth)?;

            let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
            let mut current_kyokumen_map = current_kyokumen_map.clone();
            let mut oute_kyokumen_map = oute_kyokumen_map.clone();

            let mvs = Rule::oute_only_moves_all(teban,state,mc);

            let mut mvs = mvs.into_iter().map(|m| {
                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());
                let len = Rule::respond_oute_only_moves_all(teban.opposite(),&next,&nmc).len();
                (m,next,nmc,len)
            }).collect::<Vec<(LegalMove,State,MochigomaCollections,usize)>>();

            let _ = event_dispatcher.dispatch_events(self,&*event_queue);

            let mut response_oute_comparator = self.response_oute_comparator.clone();

            mvs.sort_by(|a,b| response_oute_comparator.cmp(a,b));

            if mvs.len() == 0 {
                checkmate_state_map.insert((teban,mhash,shash),false);
                Ok(MaybeMate::Nomate)
            } else {
                for (m,next,nmc,_) in mvs {
                    if self.aborted.load(atomic::Ordering::Acquire) {
                        return Ok(MaybeMate::Aborted)
                    }

                    let o = match m {
                        LegalMove::To(ref m) => {
                            m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                        },
                        _ => None,
                    };

                    let mhash = self.hasher.calc_main_hash(mhash,teban,
                                                      state.get_banmen(),
                                                      &mc,m.to_applied_move(),&o);
                    let shash = self.hasher.calc_sub_hash(shash,teban,
                                                     state.get_banmen(),
                                                     &mc,m.to_applied_move(),&o);

                    let completed = checkmate_state_map.get(&(teban,mhash,shash));

                    if let Some(completed) = completed {
                        if *completed {
                            if !self.strict_moves {
                                let mut mvs = VecDeque::new();
                                mvs.push_front(m);
                                return Ok(MaybeMate::MateMoves(current_depth,mvs));
                            }
                        } else {
                            continue;
                        }
                    }

                    if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
                        continue;
                    }

                    if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
                        if c >= 3 {
                            continue;
                        }
                    }

                    if let Some(()) = oute_kyokumen_map.get(teban,&mhash,&shash) {
                        continue;
                    }

                    ignore_kyokumen_map.insert(teban,mhash,shash,());

                    match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                        &c => {
                            current_kyokumen_map.insert(teban, mhash, shash, c+1);
                        }
                    }

                    oute_kyokumen_map.insert(teban, mhash, shash, ());

                    match self.response_oute_process(checkmate_state_map,
                                                     unique_kyokumen_map,
                                                     current_depth+1,
                                                     nodes,
                                                     mhash,
                                                     shash,
                                                     &mut ignore_kyokumen_map,
                                                     &mut oute_kyokumen_map,
                                                     &mut current_kyokumen_map,
                                                     event_queue,
                                                     event_dispatcher,
                                                     teban,
                                                     &next,
                                                     &nmc)? {
                        MaybeMate::MateMoves(depth,mut mvs) => {
                            mvs.push_front(m);
                            return Ok(MaybeMate::MateMoves(depth,mvs))
                        },
                        r @ MaybeMate::MaxNodes | r @ MaybeMate::MaxDepth | r @ MaybeMate::Timeout | r @ MaybeMate::Aborted => {
                            return Ok(r);
                        },
                        MaybeMate::Nomate | MaybeMate::Unknown => ()
                    }
                }

                Ok(MaybeMate::Unknown)
            }
        }

        pub fn response_oute_process<L: Logger>(&mut self,
                                                checkmate_state_map:&Arc<ConcurrentFixedHashMap<(Teban, u64, u64),bool>>,
                                                unique_kyokumen_map:&Arc<ConcurrentFixedHashMap<(Teban,u64,u64),()>>,
                                                current_depth:u32,
                                                nodes:&Arc<AtomicU64>,
                                                mhash:u64,
                                                shash:u64,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError>
                where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      S: InfoSender + Send {
            if self.aborted.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            if !unique_kyokumen_map.contains_key(&(teban,mhash,shash)) {
                nodes.fetch_add(1,atomic::Ordering::Release);
                unique_kyokumen_map.insert((teban,mhash,shash),());
            }

            self.nodes += 1;

            if self.max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(current_depth)?;

            let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
            let mut current_kyokumen_map = current_kyokumen_map.clone();
            let mut oute_kyokumen_map = oute_kyokumen_map.clone();

            let mvs = Rule::respond_oute_only_moves_all(teban,state,mc);

            let mut mvs = mvs.into_iter().map(|m| {
                let (next,nmc,_) = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());
                let len = Rule::oute_only_moves_all(teban.opposite(),&next,&nmc).len();
                (m,next,nmc,len)
            }).collect::<Vec<(LegalMove,State,MochigomaCollections,usize)>>();

            let _ = event_dispatcher.dispatch_events(self,&*event_queue);

            let mut oute_comparator = self.oute_comparator.clone();

            mvs.sort_by(|a,b| oute_comparator.cmp(a,b));

            if mvs.len() == 0 {
                checkmate_state_map.insert((teban,mhash,shash),true);
                Ok(MaybeMate::MateMoves(current_depth,VecDeque::new()))
            } else {
                for (m,next,nmc,_) in mvs {
                    if self.aborted.load(atomic::Ordering::Acquire) {
                        return Ok(MaybeMate::Aborted)
                    }

                    let o = match m {
                        LegalMove::To(ref m) => {
                            m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                        },
                        _ => None,
                    };

                    let mhash = self.hasher.calc_main_hash(mhash,teban,
                                                      state.get_banmen(),
                                                      &mc,m.to_applied_move(),&o);
                    let shash = self.hasher.calc_sub_hash(shash,teban,
                                                     state.get_banmen(),
                                                     &mc,m.to_applied_move(),&o);

                    let completed = checkmate_state_map.get(&(teban,mhash,shash));

                    if let Some(completed) = completed {
                        if *completed {
                            continue;
                        } else {
                            return Ok(MaybeMate::Nomate);
                        }
                    }

                    if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
                        continue;
                    }

                    if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
                        if c >= 3 {
                            continue;
                        }
                    }

                    ignore_kyokumen_map.insert(teban,mhash,shash,());

                    match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                        &c => {
                            current_kyokumen_map.insert(teban, mhash, shash, c+1);
                        }
                    }

                    match self.oute_process(checkmate_state_map,
                                            unique_kyokumen_map,
                                            current_depth+1,
                                            nodes,
                                            mhash,
                                            shash,
                                            &mut ignore_kyokumen_map,
                                            &mut oute_kyokumen_map,
                                            &mut current_kyokumen_map,
                                            event_queue,
                                            event_dispatcher,
                                            teban,
                                            &next,
                                            &nmc)? {
                        MaybeMate::Nomate => {
                            return Ok(MaybeMate::Nomate)
                        },
                        r @ MaybeMate::MaxNodes | r @ MaybeMate::MaxDepth | r @ MaybeMate::Timeout | r @ MaybeMate::Aborted => {
                            return Ok(r)
                        },
                        MaybeMate::MateMoves(_,_) | MaybeMate::Unknown => ()
                    }
                }

                Ok(MaybeMate::Unknown)
            }
        }

        fn send_seldepth(&mut self, depth:u32) -> Result<(),SendSelDepthError>{
            let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
            commands.push(UsiInfoSubCommand::Depth(self.base_depth));
            commands.push(UsiInfoSubCommand::SelDepth(self.current_depth + depth));

            Ok(self.info_sender.send(commands)?)
        }

        fn check_timelimit(&self) -> bool {
            self.limit.map_or(false,|l| {
                let now = Instant::now();
                l < now ||
                    l - now <= Duration::from_millis(self.network_delay as u64 + TIMELIMIT_MARGIN) ||
                    self.checkmate_limit.map(|l| l < now).unwrap_or(false)
            })
        }
    }
}