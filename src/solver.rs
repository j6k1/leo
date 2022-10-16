use std::collections::VecDeque;
use std::sync::{Arc, atomic, mpsc, Mutex};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::time::Instant;
use usiagent::error::{EventHandlerError};
use usiagent::event::{EventDispatcher, EventQueue, MapEventKind, UserEvent, UserEventDispatcher, UserEventKind, USIEventDispatcher};
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
    pub already_oute_kyokumen_map:&'a mut Option<KyokumenMap<u64,bool>>,
    pub current_depth:u32,
    pub mhash:u64,
    pub shash:u64,
    pub oute_kyokumen_map:&'a mut KyokumenMap<u64,()>,
    pub current_kyokumen_map:&'a mut KyokumenMap<u64,u32>,
    pub ignore_kyokumen_map:KyokumenMap<u64,()>,
    pub event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub mc:&'a Arc<MochigomaCollections>
}
pub struct Solver {
    receiver:Receiver<Result<MaybeMate,ApplicationError>>,
    aborted:Arc<AtomicBool>,
}
impl Solver {
    pub fn new<'a,L,S>(strict_moves:bool,
                       limit:Option<Instant>,
                       checkmate_limit:Option<Instant>,
                       network_delay:u32,
                       max_depth:Option<u32>,
                       max_nodes:Option<u64>,
                       info_sender:S,
                       on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
                       hasher:Arc<KyokumenHash<u64>>,
                       base_depth:u32,
                       stop:Arc<AtomicBool>,
                       quited:Arc<AtomicBool>,
                       ms: GameStateForMate) -> Solver where L: Logger + Send + 'static, S: InfoSender {
        let (s,r) = mpsc::channel();
        let aborted = Arc::new(AtomicBool::new(false));

        {
            let s = s.clone();

            let limit = limit.clone();
            let checkmate_limit = checkmate_limit.clone();
            let network_delay = network_delay.clone();
            let max_depth = max_depth.clone();
            let max_nodes = max_nodes.clone();
            let info_sender = info_sender.clone();
            let on_error_handler = Arc::clone(&on_error_handler);
            let hasher = Arc::clone(&hasher);
            let stop = Arc::clone(&stop);
            let quited = Arc::clone(&quited);
            let aborted = Arc::clone(&aborted);

            let mut already_oute_kyokumen_map = ms.already_oute_kyokumen_map.clone();
            let current_depth = ms.current_depth;
            let mhash = ms.mhash;
            let shash = ms.shash;
            let mut ignore_kyokumen_map = ms.ignore_kyokumen_map.clone();
            let mut oute_kyokumen_map = ms.oute_kyokumen_map.clone();
            let mut current_kyokumen_map = ms.current_kyokumen_map.clone();
            let event_queue = Arc::clone(&ms.event_queue);
            let teban = ms.teban;
            let state = Arc.clone(ms.state);
            let mc = Arc::clone(&ms.mc);

            std::thread::spawn(move || {
                let mut event_dispatcher = search::Root::create_event_dispatcher(&on_error_handler,&stop,&quited);

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
                                                          on_error_handler.clone(),
                                                          base_depth,
                                                          stop,
                                                          aborted,
                                                          current_depth);
                if let Err(ref e) = s.send(mate_strategy.oute_process(&mut already_oute_kyokumen_map,
                                           current_depth,
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
            let info_sender = info_sender.clone();
            let on_error_handler = Arc::clone(&on_error_handler);
            let hasher = Arc::clone(&hasher);
            let stop = Arc::clone(&stop);
            let quited = Arc::clone(&quited);
            let aborted = Arc::clone(&aborted);

            let mut already_oute_kyokumen_map = ms.already_oute_kyokumen_map.clone();
            let current_depth = ms.current_depth;
            let mhash = ms.mhash;
            let shash = ms.shash;
            let mut ignore_kyokumen_map = ms.ignore_kyokumen_map.clone();
            let mut oute_kyokumen_map = ms.oute_kyokumen_map.clone();
            let mut current_kyokumen_map = ms.current_kyokumen_map.clone();
            let event_queue = Arc::clone(&ms.event_queue);
            let teban = ms.teban;
            let state = Arc.clone(ms.state);
            let mc = Arc::clone(&ms.mc);

            std::thread::spawn(move || {
                let mut event_dispatcher = search::Root::create_event_dispatcher(&on_error_handler,&stop,&quited);

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
                    on_error_handler.clone(),
                    base_depth,
                    stop,
                    aborted,
                    current_depth);
                if let Err(ref e) = s.send(nomate_strategy.oute_process(&mut already_oute_kyokumen_map,
                                                                      current_depth,
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

        Solver {
            receiver:r,
            aborted:aborted
        }
    }

    pub fn checkmate(&self) -> Result<MaybeMate,ApplicationError> {
        let r = self.receiver.recv();

        let m;

        match r {
            Ok(Ok(MaybeMate::MateMoves(depth,mvs))) => {
                m = MaybeMate::MateMoves(depth,mvs);
            },
            Ok(Ok(MaybeMate::Nomate)) => {
                m = MaybeMate::Nomate;
            },
            e => {
                self.aborted.store(true,atomic::Ordering::Release);

                let _ = self.receiver.recv();

                return e?;
            }
        }

        self.aborted.store(true,atomic::Ordering::Release);

        let _ = self.receiver.recv()?;

        Ok(m)
    }
}

pub mod checkmate {
    use std::cmp::Ordering;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, atomic, Mutex};
    use std::time::{Duration, Instant};
    use usiagent::command::UsiInfoSubCommand;
    use usiagent::event::{EventDispatcher, EventQueue, UserEvent, UserEventKind, USIEventDispatcher};
    use usiagent::hash::{KyokumenHash, KyokumenMap};
    use usiagent::logger::Logger;
    use usiagent::OnErrorHandler;
    use usiagent::player::InfoSender;
    use usiagent::rule::{LegalMove, Rule, State};
    use usiagent::shogi::{MochigomaCollections, MochigomaKind, Teban};
    use crate::error::{ApplicationError, SendSelDepthError};
    use crate::search::TIMELIMIT_MARGIN;
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

    pub struct CheckmateStrategy<O,R,L,S>
        where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              L: Logger + Send,
              S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        oute_comparator: O,
        response_oute_comparator:R,
        hasher:Arc<KyokumenHash<u64>>,
        strict_moves:bool,
        limit:Option<Instant>,
        checkmate_limit:Option<Instant>,
        network_delay:u32,
        max_depth:Option<u32>,
        max_nodes:Option<u64>,
        info_sender:S,
        on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
        base_depth:u32,
        stop:Arc<AtomicBool>,
        aborted:Arc<AtomicBool>,
        current_depth:u32,
        nodes:u64,
    }

    pub type MateStrategy<L,S> = CheckmateStrategy<DescComparator,AscComparator,L,S>;
    pub type NomateStrategy<L,S> = CheckmateStrategy<AscComparator,DescComparator,L,S>;

    impl<O,R,L,S> CheckmateStrategy<O,R,L,S>
        where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
              L: Logger + Send, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
        pub fn new(oute_comparator: O, response_oute_comparator: R,
               hasher:Arc<KyokumenHash<u64>>,
               strict_moves:bool,
               limit:Option<Instant>,
               checkmate_limit:Option<Instant>,
               network_delay:u32,
               max_depth:Option<u32>,
               max_nodes:Option<u64>,
               info_sender:S,
               on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
               base_depth:u32,
               stop:Arc<AtomicBool>,
               aborted:Arc<AtomicBool>,
               current_depth:u32,
       ) -> CheckmateStrategy<O,R,L,S> {
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
                on_error_handler:on_error_handler,
                base_depth:base_depth,
                stop:stop,
                aborted:aborted,
                current_depth:current_depth,
                nodes:0,
            }
        }

        pub fn oute_process(&mut self,
                             already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
                             current_depth:u32,
                             mhash:u64,
                             shash:u64,
                             ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                             oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                             current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                             event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                             event_dispatcher:&mut USIEventDispatcher<UserEventKind,
                                 UserEvent,Self,L,ApplicationError>,
                             teban:Teban,state:&State,mc:&MochigomaCollections)
            -> Result<MaybeMate,ApplicationError>
                where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      S: InfoSender + Send, L: Logger + Send + 'static {
            if self.aborted.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.nodes += 1;

            self.send_seldepth(current_depth)?;

            if self.max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

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
                already_oute_kyokumen_map.as_mut().map(|m| m.insert(teban,mhash,shash,false));
                Ok(MaybeMate::Nomate)
            } else {
                for (m,next,nmc,_) in mvs {
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

                    let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
                        m.get(teban,&mhash,&shash)
                    });

                    if let Some(true) = completed {
                        if !self.strict_moves {
                            let mut mvs = VecDeque::new();
                            mvs.push_front(m);
                            return Ok(MaybeMate::MateMoves(current_depth,mvs));
                        }
                    } else if let Some(false) = completed {
                        continue;
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

                    match self.response_oute_process(already_oute_kyokumen_map,
                                                     current_depth,
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
                        r @ _ => {
                            return Ok(r);
                        }
                    }
                }

                Ok(MaybeMate::Unknown)
            }
        }

        pub fn response_oute_process(&mut self,
                                      already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
                                      current_depth:u32,
                                      mhash:u64,
                                      shash:u64,
                                      ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                      oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                                      current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                      event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                      event_dispatcher:&mut USIEventDispatcher<UserEventKind,
                                          UserEvent,Self,L,ApplicationError>,
                                      teban:Teban,state:&State,mc:&MochigomaCollections)
            -> Result<MaybeMate,ApplicationError>
                where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                      S: InfoSender + Send, L: Logger + Send + 'static {
            if self.aborted.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.nodes += 1;

            self.send_seldepth(current_depth)?;

            if self.max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

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
                Ok(MaybeMate::MateMoves(current_depth,VecDeque::new()))
            } else {
                for (m,next,nmc,_) in mvs {
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

                    let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
                        m.get(teban,&mhash,&shash)
                    });

                    if let Some(true) = completed {
                        continue;
                    } else if let Some(false) = completed {
                        return Ok(MaybeMate::Nomate);
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

                    match self.oute_process(already_oute_kyokumen_map,
                                            current_depth,
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
                            return Ok(MaybeMate::MateMoves(depth,mvs));
                        },
                        r @ _ => {
                            return Ok(r)
                        }
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