use std::cmp::Ordering;
use std::collections::VecDeque;
use std::marker::PhantomData;
use usiagent::error::PlayerError;
use usiagent::rule::LegalMove;
use usiagent::shogi::*;

#[derive(Debug)]
pub enum MaybeMate {
    Nomate,
    MateMoves(u32,VecDeque<LegalMove>),
    MaxDepth,
    MaxNodes,
    Timeout,
    Unknown
}

pub struct Solver {
}
impl Solver {
    pub fn new() -> Solver {
        Solver {
        }
    }
}

mod checkmate {
    use std::cmp::Ordering;
    use std::collections::VecDeque;
    use std::marker::PhantomData;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, atomic, mpsc, Mutex};
    use std::sync::mpsc::Receiver;
    use std::time::{Duration, Instant};
    use usiagent::command::UsiInfoSubCommand;
    use usiagent::error::PlayerError;
    use usiagent::event::{EventDispatcher, EventQueue, UserEvent, UserEventKind, USIEventDispatcher};
    use usiagent::hash::{KyokumenHash, KyokumenMap};
    use usiagent::logger::Logger;
    use usiagent::OnErrorHandler;
    use usiagent::player::InfoSender;
    use usiagent::rule::{LegalMove, Rule, State};
    use usiagent::shogi::{MochigomaCollections, MochigomaKind, Teban};
    use crate::error::ApplicationError;
    use crate::player::TIMELIMIT_MARGIN;
    use crate::solver::{MaybeMate, Solver};
    use crate::solver::MaybeMate::MateMoves;

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

    pub struct CheckmateStrategy<O,R> where O: Comparator<(LegalMove,usize)>,
                                            R: Comparator<(LegalMove,usize)> {
        oute_comparator:O,
        response_oute_comparator:R,
        current_depth:u32,
        nodes:u64,
        limit:Option<Instant>,
        checkmate_limit:Option<Instant>,
        network_delay:u32
    }

    pub type MateStrategy = CheckmateStrategy<DescComparator,AscComparator>;
    pub type NomateStrategy= CheckmateStrategy<AscComparator,DescComparator>;

    impl<O,R> CheckmateStrategy<O,R>
        where O: Comparator<(LegalMove,usize)>,
              R: Comparator<(LegalMove,usize)> {
        fn new(oute_comparator: O, response_oute_comparator: R,
               current_depth:u32,
               limit:Option<Instant>,
               checkmate_limit:Option<Instant>,
               network_delay:u32
        ) -> CheckmateStrategy<O, R> {
            CheckmateStrategy {
                oute_comparator: oute_comparator,
                response_oute_comparator: response_oute_comparator,
                current_depth:current_depth,
                nodes:0,
                limit:limit,
                checkmate_limit:checkmate_limit,
                network_delay:network_delay
            }
        }

        fn mate(&self,current_depth:u32,teban:Teban,state:State,mc:MochigomaCollections) -> Receiver<MaybeMate> {
            let (s,r) = mpsc::channel();

            let mut oute_comparator = self.oute_comparator.clone();
            let mut response_oute_comparator = self.response_oute_comparator.clone();

            std::thread::spawn(move || {

            });

            r
        }

        fn oute_process<S,L>(&mut self,
                             strict_moves:bool,
                             max_depth:Option<u32>,
                             max_nodes:Option<u64>,
                             limit:&Option<Instant>,
                             network_delay:u32,
                             already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
                             hasher:&KyokumenHash<u64>,
                             info_sender:&mut S,
                             on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                             base_depth:u32,
                             current_depth:u32,
                             stop:&Arc<AtomicBool>,
                             mhash:u64,
                             shash:u64,
                             ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                             oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                             current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                             event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                             event_dispatcher:&mut USIEventDispatcher<UserEventKind,
                                 UserEvent,Self,L,ApplicationError>,
                             teban:Teban,state:&State,mc:&MochigomaCollections)
            -> MaybeMate where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                               R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                               S: InfoSender + Send, L: Logger + Send + 'static {
            self.nodes += 1;

            self.send_seldepth(info_sender, &on_error_handler, base_depth, self.current_depth + current_depth);

            if max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return MaybeMate::MaxDepth;
            }

            if max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return MaybeMate::MaxNodes;
            }

            if self.check_timelimit() || stop.load(atomic::Ordering::Acquire) {
                return MaybeMate::Timeout;
            }

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
                MaybeMate::Nomate
            } else {
                for (m,next,nmc,_) in mvs {
                    let o = match m {
                        LegalMove::To(ref m) => {
                            m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                        },
                        _ => None,
                    };

                    let mhash = hasher.calc_main_hash(mhash,teban,
                                                      state.get_banmen(),
                                                      &mc,m.to_applied_move(),&o);
                    let shash = hasher.calc_sub_hash(shash,teban,
                                                     state.get_banmen(),
                                                     &mc,m.to_applied_move(),&o);

                    let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
                        m.get(teban,&mhash,&shash)
                    });

                    if let Some(true) = completed {
                        if !strict_moves {
                            let mut mvs = VecDeque::new();
                            mvs.push_front(m);
                            return MaybeMate::MateMoves(current_depth,mvs);
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

                    match self.response_oute_process(strict_moves,
                                                     max_depth,
                                                     max_nodes,
                                                     limit,
                                                     network_delay,
                                                     already_oute_kyokumen_map,
                                                     hasher,
                                                     info_sender,
                                                     on_error_handler,
                                                     base_depth,
                                                     current_depth,
                                                     stop,
                                                     mhash,
                                                     shash,
                                                     ignore_kyokumen_map,
                                                     oute_kyokumen_map,
                                                     current_kyokumen_map,
                                                     event_queue,
                                                     event_dispatcher,
                                                     teban,
                                                     &next,
                                                     &nmc) {
                        MaybeMate::MateMoves(depth,mut mvs) => {
                            mvs.push_front(m);
                            return MaybeMate::MateMoves(depth,mvs)
                        },
                        MaybeMate::Nomate => {

                        },
                        r @ _ => {
                            return r
                        }
                    }
                }

                MaybeMate::Unknown
            }
        }

        fn response_oute_process<S,L>(&mut self,
                                      strict_moves:bool,
                                      max_depth:Option<u32>,
                                      max_nodes:Option<u64>,
                                      limit:&Option<Instant>,
                                      network_delay:u32,
                                      already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
                                      hasher:&KyokumenHash<u64>,
                                      info_sender:&mut S,
                                      on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                                      base_depth:u32,
                                      current_depth:u32,
                                      stop:&Arc<AtomicBool>,
                                      mhash:u64,
                                      shash:u64,
                                      ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                      oute_kyokumen_map:&mut KyokumenMap<u64,()>,
                                      current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                      event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                      event_dispatcher:&mut USIEventDispatcher<UserEventKind,
                                          UserEvent,Self,L,ApplicationError>,
                                      teban:Teban,state:&State,mc:&MochigomaCollections)
            -> MaybeMate where O: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                               R: Comparator<(LegalMove,State,MochigomaCollections,usize)>,
                               S: InfoSender + Send, L: Logger + Send + 'static {
            self.nodes += 1;

            self.send_seldepth(info_sender, &on_error_handler, base_depth, self.current_depth + current_depth);

            if max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return MaybeMate::MaxDepth;
            }

            if max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
                return MaybeMate::MaxNodes;
            }

            if self.check_timelimit() || stop.load(atomic::Ordering::Acquire) {
                return MaybeMate::Timeout;
            }

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
                MaybeMate::MateMoves(current_depth,VecDeque::new())
            } else {
                for (m,next,nmc,_) in mvs {
                    let o = match m {
                        LegalMove::To(ref m) => {
                            m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                        },
                        _ => None,
                    };

                    let mhash = hasher.calc_main_hash(mhash,teban,
                                                      state.get_banmen(),
                                                      &mc,m.to_applied_move(),&o);
                    let shash = hasher.calc_sub_hash(shash,teban,
                                                     state.get_banmen(),
                                                     &mc,m.to_applied_move(),&o);

                    let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
                        m.get(teban,&mhash,&shash)
                    });

                    if let Some(true) = completed {
                        continue;
                    } else if let Some(false) = completed {
                        return MaybeMate::Nomate;
                    }

                    if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
                        continue;
                    }

                    if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
                        if c >= 3 {
                            continue;
                        }
                    }

                    match self.oute_process(strict_moves,
                                            max_depth,
                                            max_nodes,
                                            limit,
                                            network_delay,
                                            already_oute_kyokumen_map,
                                            hasher,
                                            info_sender,
                                            on_error_handler,
                                            base_depth,
                                            current_depth,
                                            stop,
                                            mhash,
                                            shash,
                                            ignore_kyokumen_map,
                                            oute_kyokumen_map,
                                            current_kyokumen_map,
                                            event_queue,
                                            event_dispatcher,
                                            teban,
                                            &next,
                                            &nmc) {
                        MaybeMate::MateMoves(depth,mut mvs) => {
                            mvs.push_front(m);
                            return MaybeMate::MateMoves(depth,mvs)
                        },
                        MaybeMate::Nomate => {

                        },
                        r @ _ => {
                            return r
                        }
                    }
                }

                MaybeMate::Unknown
            }
        }

        fn send_seldepth<L,S>(&self, info_sender:&mut S,
                              on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32, seldepth:u32)
            where L: Logger + Send, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

            let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
            commands.push(UsiInfoSubCommand::Depth(depth));
            commands.push(UsiInfoSubCommand::SelDepth(seldepth));


            match info_sender.send(commands) {
                Ok(_) => (),
                Err(ref e) => {
                    let _ = on_error_handler.lock().map(|h| h.call(e));
                }
            }
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