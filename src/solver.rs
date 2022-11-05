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
use crate::solver::checkmate::{OrNodeComparator, CheckmateStrategy, AndNodeComparator};

#[derive(Debug,Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(u32),
    MaxDepth,
    Skip,
    MaxNodes,
    Timeout,
    Aborted
}

pub struct GameStateForMate<'a> {
    pub base_deoth:u32,
    pub current_depth:u32,
    pub mhash:u64,
    pub shash:u64,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
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
                     node_count:Arc<AtomicU64>,
                     info_sender:S,
                     on_error_handler:Arc<Mutex<OnErrorHandler<L>>>,
                     hasher:Arc<KyokumenHash<u64>>,
                     stop:Arc<AtomicBool>,
                     quited:Arc<AtomicBool>,
                     ms: GameStateForMate) -> Result<MaybeMate,ApplicationError> where L: Logger + Send + 'static, S: InfoSender {
    }
}

pub mod checkmate {
    use std::borrow::Borrow;
    use std::cell::RefCell;
    use std::cmp::Ordering;
    use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
    use std::rc::Rc;
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

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub enum Number {
        Value(u64),
        INFINITE
    }

    pub struct Node {
        id:u64,
        pn:Number,
        dn:Number,
        max_depth:u32,
        ignore:bool,
        ref_nodes:HashSet<u64>,
        update_nodes:HashSet<u64>,
        m:LegalMove,
        children:BTreeSet<Rc<RefCell<Node>>>,
        comparator:Box<dyn Comparator<Node>>
    }

    impl Node {
        pub fn new_or_node(last_id:&mut u64,depth:u32,m:LegalMove) -> Node {
            *last_id += 1;

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                max_depth: depth,
                ignore:false,
                ref_nodes:HashSet::new(),
                update_nodes:HashSet::new(),
                m:m,
                children:BTreeSet::new(),
                comparator:Box::new(OrNodeComparator)
            }
        }

        pub fn new_and_node(last_id:&mut u64,depth:u32,m:LegalMove) -> Node {
            *last_id += 1;

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                max_depth: depth,
                ignore:false,
                ref_nodes:HashSet::new(),
                update_nodes:HashSet::new(),
                m:m,
                children:BTreeSet::new(),
                comparator:Box::new(AndNodeComparator)
            }
        }
    }

    pub trait Comparator<T> {
        fn cmp(&mut self,l:&T,r:&T) -> Ordering;
    }

    #[derive(Clone)]
    pub struct OrNodeComparator;

    impl Comparator<Node> for OrNodeComparator {
        #[inline]
        fn cmp(&mut self,l:&Node,r:&Node) -> Ordering {
            l.borrow().pn.cmp(&r.borrow().pn).then(l.borrow().max_depth.cmp(&r.borrow().max_depth))
        }
    }

    #[derive(Clone)]
    pub struct AndNodeComparator;

    impl Comparator<(LegalMove,State,MochigomaCollections,usize)> for AndNodeComparator {
        #[inline]
        fn cmp(&mut self,l:&Node,r:&Node) -> Ordering {
            l.borrow().dn.cmp(&r.borrow().dn).then(l.borrow().max_depth.cmp(&r.borrow().max_depth))
        }
    }

    pub struct CheckmateStrategy<S> where S: InfoSender {
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
        current_depth:u32,
        node_count:i64,
    }

    impl<S> CheckmateStrategy<S> where S: InfoSender {
        pub fn new(hasher:Arc<KyokumenHash<u64>>,
               strict_moves:bool,
               limit:Option<Instant>,
               checkmate_limit:Option<Instant>,
               network_delay:u32,
               max_depth:Option<u32>,
               max_nodes:Option<i64>,
               info_sender:S,
               base_depth:u32,
               stop:Arc<AtomicBool>,
               current_depth:u32,
       ) -> CheckmateStrategy<S> {
            CheckmateStrategy {
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
                current_depth:current_depth,
                node_count:0,
            }
        }

        pub fn update_nodes(&mut self, depth:u32, current_nodes:&VecDeque<Rc<RefCell<Node>>>,) -> Result<(),ApplicationError>{
            for (i,n) in (0..=depth).rev().zip(current_nodes.iter().rev()) {
                if i % 2 == 0 {
                    let mut pn = Number::INFINITE;
                    let mut dn = 0;

                    for child in n.try_borrow()?.children.iter() {
                        pn = child.try_borrow()?.pn.min(pn);
                        dn += child.try_borrow()?.dn;
                    }

                    n.try_borrow_mut()?.pn = pn;
                    n.try_borrow_mut()?.dn = Number::Value(dn);
                } else {
                    let mut pn = 0;
                    let mut dn = Number::INFINITE;

                    for child in n.try_borrow()?.children.iter() {
                        pn += child.try_borrow()?.pn;
                        dn = child.try_borrow()?.dn.min(dn);
                    }

                    n.try_borrow_mut()?.pn = Number::Value(pn);
                    n.try_borrow_mut()?.dn = dn;
                }
            }

            Ok(())
        }

        pub fn oute_process<L: Logger>(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                       current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                       last_id:&mut u64,
                                       current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                       node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                       current_moves:&mut VecDeque<LegalMove>,
                                       event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                       event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                       teban:Teban, state:&State, mc:&MochigomaCollections)
                                       -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            if self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.node_count += 1;

            if self.max_depth.map(|d| depth >= d).unwrap_or(false) {
                let mut n = current_nodes.back().ok_or(ApplicationError::LogicError(String::from(
                    "Current node is not set."
                )))?.try_borrow_mut()?;

                n.ignore = true;

                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let mvs = Rule::oute_only_moves_all(teban,state,mc);

            let mut nodes = mvs.into_iter().map(|m| {
                Rc::new(RefCell::new(Node::new_or_node(last_id,depth,m)))
            }).collect::<VecDeque<Rc<RefCell<Node>>>>();

            let _ = event_dispatcher.dispatch_events(self,&*event_queue);

            if nodes.len() == 0 {
                {
                    let mut n = current_nodes.back().ok_or(ApplicationError::LogicError(String::from(
                        "Current node is not set."
                    )))?.try_borrow_mut()?;

                    n.pn = Number::INFINITE;
                    n.dn = Number::Value(0);
                }

                current_nodes.pop_back();
                current_moves.pop_back();

                self.update_nodes(depth-1, current_nodes)?;

                let mut d = 0;

                for (n,&m) in current_nodes.iter().zip(current_moves.iter()) {
                    if n.try_borrow()?.m != m {
                        break;
                    }
                    d += 1;
                }

                Ok(MaybeMate::Continuation(depth - d))
            } else {
                {
                    let mut n = current_nodes.back().ok_or(ApplicationError::LogicError(String::from(
                        "Current node is not set."
                    )))?.try_borrow_mut()?;

                    for child in nodes.iter() {
                        n.children.insert(Rc::clone(child));
                    }
                }

                self.update_nodes(depth,current_nodes)?;

                let mut d = 0;

                for (n,&m) in current_nodes.iter().zip(current_moves.iter()) {
                    if n.try_borrow()?.m != m {
                        break;
                    }
                    d += 1;
                }

                if d == depth {
                    loop {
                        let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                        let mut current_kyokumen_map = current_kyokumen_map.clone();

                        let mut update = false;

                        for n in nodes.iter() {
                            if n.try_borrow()?.ignore {
                                continue;
                            }

                            let m = n.try_borrow()?.m;

                            if self.stop.load(atomic::Ordering::Acquire) {
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

                            if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
                                n.try_borrow()?.ignore = true;
                                continue;
                            }

                            if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
                                if c >= 3 {
                                    n.try_borrow_mut()?.pn = Number::INFINITE;
                                    n.try_borrow_mut()?.dn = Number::Value(0);

                                    continue;
                                }
                            }

                            ignore_kyokumen_map.insert(teban,mhash,shash,());

                            match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                                &c => {
                                    current_kyokumen_map.insert(teban, mhash, shash, c+1);
                                }
                            }

                            current_moves.push_back(m);
                            current_nodes.push_back(Rc::clone(n));

                            let next = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

                            match next {
                                (state,mc,_) => {
                                    match self.response_oute_process(depth + 1,
                                                                     mhash,
                                                                     shash,
                                                                     &mut ignore_kyokumen_map,
                                                                     &mut current_kyokumen_map,
                                                                     last_id,
                                                                     current_nodes,
                                                                     node_map,
                                                                     current_moves,
                                                                     event_queue,
                                                                     event_dispatcher,
                                                                     teban.opposite(),
                                                                     &state,
                                                                     &mc
                                    )? {
                                        MaybeMate::Continuation(0) => {
                                            update = true;
                                            break;
                                        },
                                        MaybeMate::Continuation(depth) => {
                                            Ok(MaybeMate::Continuation(depth - 1))
                                        },
                                        r @ MaybeMate::MateMoves(_) | MaybeMate::Nomate | MaybeMate::Unknown |
                                        MaybeMate::MaxNodes | MaybeMate::Timeout | MaybeMate::Aborted => {
                                            Ok(r)
                                        },
                                        MaybeMate::Skip | MaybeMate::MaxDepth => {

                                        }
                                    }
                                }
                            }
                        }

                        if !update {
                            break;s
                        }
                    }

                    Ok(MaybeMate::Skip)
                } else {
                    Ok(MaybeMate::Continuation(depth - d))
                }
            }
        }

        pub fn response_oute_process<L: Logger>(&mut self,
                                                depth:u32,
                                                mhash:u64,
                                                shash:u64,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                last_id:&mut u64,
                                                current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                                current_moves:&mut VecDeque<LegalMove>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            if self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.node_count += 1;

            if self.max_depth.map(|d| current_depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
            let mut current_kyokumen_map = current_kyokumen_map.clone();

            let mvs = Rule::respond_oute_only_moves_all(teban,state,mc);

            let mut nodes = mvs.into_iter().map(|m| {
                Rc::new(RefCell::new(Node::new_or_node(last_id,depth,m)))
            }).collect::<VecDeque<Rc<RefCell<Node>>>>();

            let _ = event_dispatcher.dispatch_events(self,&*event_queue);

            if nodes.len() == 0 {
                {
                    let mut n = current_nodes.back().ok_or(ApplicationError::LogicError(String::from(
                        "Current node is not set."
                    )))?.try_borrow_mut()?;

                    n.pn = Number::Value(0);
                    n.dn = Number::INFINITE;
                }

                current_nodes.pop_back();
                current_moves.pop_back();

                self.update_nodes(depth-1, current_nodes)?;

                let mut d = 0;

                for (n,&m) in current_nodes.iter().zip(current_moves.iter()) {
                    if n.try_borrow()?.m != m {
                        break;
                    }
                    d += 1;
                }

                Ok(MaybeMate::Continuation(depth - d))
            } else {
                {
                    let mut n = current_nodes.back().ok_or(ApplicationError::LogicError(String::from(
                        "Current node is not set."
                    )))?.try_borrow_mut()?;

                    for child in nodes.iter() {
                        n.children.insert(Rc::clone(child));
                    }
                }

                self.update_nodes(depth,current_nodes)?;

                let mut d = 0;

                for (n,&m) in current_nodes.iter().zip(current_moves.iter()) {
                    if n.try_borrow()?.m != m {
                        break;
                    }
                    d += 1;
                }

                if d == depth {
                    loop {
                        let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                        let mut current_kyokumen_map = current_kyokumen_map.clone();

                        let mut update = false;

                        for n in nodes.iter() {
                            if n.try_borrow()?.ignore {
                                continue;
                            }

                            let m = n.try_borrow()?.m;

                            let o = match m {
                                LegalMove::To(ref m) => {
                                    m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
                                },
                                _ => None,
                            };

                            let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                                   state.get_banmen(),
                                                                   &mc, m.to_applied_move(), &o);
                            let shash = self.hasher.calc_sub_hash(shash, teban,
                                                                  state.get_banmen(),
                                                                  &mc, m.to_applied_move(), &o);

                            if let Some(()) = ignore_kyokumen_map.get(teban, &mhash, &shash) {
                                n.try_borrow_mut()?.ignore = true;

                                continue;
                            }

                            if let Some(&c) = current_kyokumen_map.get(teban, &mhash, &shash) {
                                if c >= 3 {
                                    n.try_borrow_mut()?.pn = Number::INFINITE;
                                    n.try_borrow_mut()?.dn = Number::Value(0);

                                    continue;
                                }
                            }

                            ignore_kyokumen_map.insert(teban, mhash, shash, ());

                            match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                                &c => {
                                    current_kyokumen_map.insert(teban, mhash, shash, c + 1);
                                }
                            }

                            let next = Rule::apply_move_none_check(state,mc,m.to_applied_move());

                            match next {
                                (state,mc,_) => {
                                    match self.oute_process(depth + 1,
                                                            mhash,
                                                            shash,
                                                            &mut ignore_kyokumen_map,
                                                            &mut current_kyokumen_map,
                                                            last_id,
                                                            current_nodes,
                                                            node_map,
                                                            current_moves,
                                                            event_queue,
                                                            event_dispatcher,
                                                            teban.opposite(),
                                                            &state,
                                                            &mc
                                    )? {
                                        MaybeMate::Continuation(0) => {
                                            update = true;

                                            break;
                                        },
                                        MaybeMate::Continuation(depth) => {
                                            Ok(MaybeMate::Continuation(depth-1))
                                        },
                                        r @ MaybeMate::MateMoves(_) | MaybeMate::Nomate | MaybeMate::Unknown |
                                        MaybeMate::MaxNodes | MaybeMate::Timeout | MaybeMate::Aborted => {
                                            Ok(r)
                                        },
                                        MaybeMate::Skip | MaybeMate::MaxDepth => {

                                        }
                                    }
                                }
                            }
                        }

                        if !update {
                            break;
                        }
                    }
                    Ok(MaybeMate::Skip)
                }
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