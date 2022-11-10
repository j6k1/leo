use std::cell::RefCell;
use std::collections::{VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool};
use std::time::Instant;

use usiagent::event::{EventQueue, UserEvent, UserEventKind};
use usiagent::hash::{KyokumenHash, KyokumenMap};
use usiagent::logger::Logger;
use usiagent::OnErrorHandler;
use usiagent::player::InfoSender;
use usiagent::rule::{LegalMove, State};
use usiagent::shogi::*;
use crate::error::ApplicationError;
use crate::search::Root;
use crate::solver::checkmate::{CheckmateStrategy,Node};

#[derive(Debug,Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(Node),
    MaxDepth,
    Skip,
    MaxNodes,
    Timeout,
    Aborted
}

pub struct GameStateForMate<'a> {
    pub base_depth:u32,
    pub current_depth:u32,
    pub mhash:u64,
    pub shash:u64,
    pub current_kyokumen_map:&'a KyokumenMap<u64,u32>,
    pub event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
    pub teban:Teban,
    pub state:&'a Arc<State>,
    pub mc:&'a Arc<MochigomaCollections>,
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
                     info_sender:S,
                     on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                     hasher:Arc<KyokumenHash<u64>>,
                     stop:Arc<AtomicBool>,
                     quited:Arc<AtomicBool>,
                     ms: GameStateForMate) -> Result<MaybeMate,ApplicationError> where L: Logger + Send + 'static, S: InfoSender {
        let mut strategy = CheckmateStrategy::<S>::new(hasher,
                                                  strict_moves,
                                                  limit,
                                                  checkmate_limit,
                                                  network_delay,
                                                  max_depth,
                                                  max_nodes,
                                                  info_sender,
                                                  Arc::clone(&stop),
                                                  ms.base_depth,
                                                  ms.current_depth);
        let mut last_id = 0;

        let mut event_dispatcher = Root::<L,S>::create_event_dispatcher::<CheckmateStrategy<S>>(on_error_handler,&stop,&quited);

        let root_childrren = strategy.expand_root_nodes(0, &mut last_id, ms.teban, &ms.state, &ms.mc)?;

        strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              &KyokumenMap::new(),
                              &KyokumenMap::new(),
                              &mut last_id,
                              &root_childrren,
                              None,
                              &mut VecDeque::new(),
                              &mut KyokumenMap::new(),
                              &mut VecDeque::new(),
                              &mut None,
                              &ms.event_queue,
                              &mut event_dispatcher,
                              ms.teban,
                              &ms.state,
                              &ms.mc)
    }
}

pub mod checkmate {
    use std::borrow::Borrow;
    use std::cell::{RefCell};
    use std::cmp::Ordering;
    use std::collections::{BTreeSet, HashSet, VecDeque};
    use std::ops::{Add, AddAssign};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool};
    use std::sync::{Arc, atomic, Mutex};
    use std::time::{Duration, Instant};
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
    use crate::solver::MaybeMate::{Continuation, MateMoves};

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub enum Number {
        Value(u64),
        INFINITE
    }

    impl Add for Number {
        type Output = Number;

        fn add(self, rhs: Self) -> Self::Output {
            match (self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => Number::INFINITE,
                (Number::Value(l),Number::Value(r)) => Number::Value(l+r)
            }
        }
    }
  
    impl AddAssign for Number {
        fn add_assign(&mut self, rhs: Self) {
            let v = match (&self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => {
                    Number::INFINITE
                },
                (Number::Value(l),Number::Value(r)) => {
                    Number::Value(*l + r)
                }
            };

            *self = v;
        }    
    }

    pub struct Node {
        id:u64,
        pn:Number,
        dn:Number,
        mate_depth:u32,
        ref_nodes:HashSet<u64>,
        update_nodes:HashSet<u64>,
        m:LegalMove,
        children:Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
        comparator:Box<dyn Comparator<Node>>
    }

    impl Node {
        pub fn new_or_node(last_id:&mut u64,m:LegalMove,parent_id:u64) -> Node {
            *last_id += 1;
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                mate_depth: 0,
                ref_nodes:ref_nodes,
                update_nodes:HashSet::new(),
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(OrNodeComparator)
            }
        }

        pub fn new_and_node(last_id:&mut u64,m:LegalMove,parent_id:u64) -> Node {
            *last_id += 1;
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                mate_depth: 0,
                ref_nodes:ref_nodes,
                update_nodes:HashSet::new(),
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(AndNodeComparator)
            }
        }
    }

    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for Node {}

    impl Ord for Node {
        fn cmp(&self, other: &Self) -> Ordering {
            self.comparator.cmp(self,other)
        }
    }

    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    pub trait Comparator<T> {
        fn cmp(&self,l:&T,r:&T) -> Ordering;
    }

    #[derive(Clone)]
    pub struct OrNodeComparator;

    impl Comparator<Node> for OrNodeComparator {
        #[inline]
        fn cmp(&self,l:&Node,r:&Node) -> Ordering {
            l.borrow().pn.cmp(&r.borrow().pn)
                .then(l.borrow().mate_depth.cmp(&r.borrow().mate_depth))
                .then(l.borrow().id.cmp(&r.borrow().id))
        }
    }

    #[derive(Clone)]
    pub struct AndNodeComparator;

    impl Comparator<Node> for AndNodeComparator {
        #[inline]
        fn cmp(&self,l:&Node,r:&Node) -> Ordering {
            l.borrow().dn.cmp(&r.borrow().dn)
                .then(l.borrow().mate_depth.cmp(&r.borrow().mate_depth))
                .then(l.borrow().id.cmp(&r.borrow().id))
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
        stop:Arc<AtomicBool>,
        base_depth:u32,
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
               stop:Arc<AtomicBool>,
               base_depth:u32,
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
                stop:stop,
                base_depth:base_depth,
                current_depth:current_depth,
                node_count:0,
            }
        }

        pub fn normalize_node(&mut self,
                             n:&Rc<RefCell<Node>>,
                             mhash:u64,
                             shash:u64,
                             teban:Teban,
                             node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>)
            -> Result<Rc<RefCell<Node>>,ApplicationError> {

            if let Some(n) = node_map.get(teban,&mhash,&shash) {
                Ok(Rc::clone(n))
            } else {
                Ok(Rc::clone(n))
            }
        }

        pub fn preprocess<L: Logger>(&mut self,
                                     depth:u32,
                                     mhash:u64,
                                     shash:u64,
                                     last_id:&mut u64,
                                     current_node:&Rc<RefCell<Node>>,
                                     node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>)
            -> Result<(bool,Rc<RefCell<Node>>),ApplicationError> {
            let n = self.normalize_node(current_node,mhash,shash,teban,node_map)?;
            let parent_id = n.try_borrow()?.id;
            let mut expanded = false;

            {
                let mut children = n.try_borrow()?.children.try_borrow_mut();

                if children.len() == 0 {
                    expanded = true;

                    if depth % 2 == 0 {
                        let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                        let nodes = mvs.into_iter().map(|m| {
                            Rc::new(RefCell::new(Node::new_and_node(last_id, m, parent_id)))
                        }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                        for child in nodes.iter() {
                            childrens.insert(Rc::clone(child));
                        }
                    } else {
                        let mvs = Rule::oute_only_moves_all(teban, state, mc);

                        let nodes = mvs.into_iter().map(|m| {
                            Rc::new(RefCell::new(Node::new_or_node(last_id, m, parent_id)))
                        }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                        for child in nodes.iter() {
                            childrens.insert(Rc::clone(child));
                        }
                    }
                }
            }

            Ok((expanded,n))
        }

        pub fn expand_root_nodes(&mut self,
                                 depth:u32,
                                 last_id:&mut u64,
                                 teban:Teban,
                                 state:&State,
                                 mc:&MochigomaCollections) -> Result<Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,ApplicationError> {
            let mvs = Rule::oute_only_moves_all(teban, state, mc);

            let nodes = mvs.into_iter().map(|m| {
                Rc::new(RefCell::new(Node::new_or_node(last_id, depth + 1, m, 0)))
            }).collect::<VecDeque<Rc<RefCell<Node>>>>();

            let children = Rc::new(RefCell::new(BTreeSet::new()));

            for child in nodes.iter() {
                children.try_borrow_mut()?.insert(Rc::clone(child));
            }

            Ok(children)
        }

        pub fn inter_process<L: Logger>(&mut self,
                                                depth:u32,
                                                mhash:u64,
                                                shash:u64,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                last_id:&mut u64,
                                                root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                                current_moves:&mut VecDeque<LegalMove>,
                                                mate_depth:&mut Option<u32>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            let r = if depth % 2 == 0 {
                match self.oute_process(depth,
                                        mhash,
                                        shash,
                                        ignore_kyokumen_map,
                                        current_kyokumen_map,
                                        last_id,
                                        root_children,
                                        current_node,
                                        current_nodes,
                                        node_map,
                                        current_moves,
                                        mate_depth,
                                        event_queue,
                                        event_dispatcher,
                                        teban,
                                        state,
                                        mc) {
                    r => {
                        r
                    }
                }
            } else {
                match self.response_oute_process(depth,
                                        mhash,
                                        shash,
                                        ignore_kyokumen_map,
                                        current_kyokumen_map,
                                        last_id,
                                        root_children,
                                        current_node,
                                        current_nodes,
                                        node_map,
                                        current_moves,
                                        mate_depth,
                                        event_queue,
                                        event_dispatcher,
                                        teban,
                                        state,
                                        mc) {
                    r => {
                        r
                    }
                }
            };

            current_moves.pop_back();
            current_moves.pop_back();

            r
        }

        pub fn build_moves(&mut self,n:&Rc<RefCell<Node>>) -> Result<VecDeque<LegalMove>,ApplicationError> {
            let mut mvs = moves.clone();

            let mut n = Some(Rc::clone(n));

            while let Some(c) = n {
                mvs.push_back(c.try_borrow()?.m);
                n = c.try_borrow()?.children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
            }

            Ok(mvs)
        }

        pub fn update_node(&mut self, depth:u32, last_id:&mut u64, n:&Rc<RefCell<Node>>) -> Result<Node,ApplicationError> {
            let parent_id = n.try_borrow()?.id;
            let m = n.try_borrow()?.m;

            if depth % 2 == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in n.try_borrow()?.children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += dn;
                }

                let mut n = Node::new_or_node(last_id,m,parent_id);

                n.pn = pn;
                n.dn = dn;

                Ok(n)
            } else {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in n.try_borrow()?.children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += dn;
                }

                let mut n = Node::new_and_node(last_id,m,parent_id);

                n.pn = pn;
                n.dn = dn;

                Ok(n)
            }
        }

        pub fn oute_process<L: Logger>(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       ignore_kyokumen_map:&KyokumenMap<u64,()>,
                                       current_kyokumen_map:&KyokumenMap<u64,u32>,
                                       last_id:&mut u64,
                                       root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                                       current_node:Option<Rc<RefCell<Node>>>,
                                       current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                       node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                       current_moves:&mut VecDeque<LegalMove>,
                                       mate_depth:&mut Option<u32>,
                                       event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                       event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                       teban:Teban, state:&State, mc:&MochigomaCollections)
                                       -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            if self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.node_count += 1;

            if self.max_depth.map(|d| depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let children = if let Some(n) = current_node {
                let mut n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                if children.try_borrow()?.len() == 0 {
                    println!("info string no_mate.");

                    let mut n = self.update_node(depth, last_id, &n);

                    Ok(n)
                } else {
                    Rc::clone(&n.try_borrow()?.children)
                }
            } else {
                let mvs = Rule::oute_only_moves_all(teban, state, mc);

                let nodes = mvs.into_iter().map(|m| {
                    Rc::new(RefCell::new(Node::new_or_node(last_id, m, 0)))
                }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                let children = Rc::new(RefCell::new(BTreeSet::new()));

                for child in nodes.iter() {
                    children.try_borrow_mut()?.insert(Rc::clone(child));
                }

                children
            };

            loop {
                let mut update_nodes = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    for n in children.try_borrow()?.iter() {
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

                        let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                               state.get_banmen(),
                                                               &mc, m.to_applied_move(), &o);
                        let shash = self.hasher.calc_sub_hash(shash, teban,
                                                              state.get_banmen(),
                                                              &mc, m.to_applied_move(), &o);

                        if let Some(()) = ignore_kyokumen_map.get(teban, &mhash, &shash) {
                            continue;
                        }

                        if let Some(&c) = current_kyokumen_map.get(teban, &mhash, &shash) {
                            if c >= 3 {
                                continue;
                            }
                        }

                        ignore_kyokumen_map.insert(teban, mhash, shash, ());

                        match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                            &c => {
                                current_kyokumen_map.insert(teban, mhash, shash, c + 1);
                            }
                        }

                        current_moves.push_back(m);

                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         &mut ignore_kyokumen_map,
                                                         &mut current_kyokumen_map,
                                                         last_id,
                                                         root_children,
                                                         Some(Rc::clone(n)),
                                                         current_nodes,
                                                         node_map,
                                                         current_moves,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
                                        println!("info string {},{},{:?},{:?}", 0, depth, n.try_borrow()?.pn, n.try_borrow()?.dn);
                                        update_nodes = Some((Rc::clone(n), u));

                                        break;
                                    },
                                    r @ MaybeMate::MaxNodes => {
                                        return Ok(r);
                                    },
                                    r @ MaybeMate::Timeout => {
                                        return Ok(r);
                                    },
                                    r @ MaybeMate::Aborted => {
                                        return Ok(r);
                                    },
                                    MaybeMate::Skip | MaybeMate::MaxDepth => {},
                                    MaybeMate::MateMoves(_) => {
                                        return Err(ApplicationError::LogicError(String::from(
                                            "It is an unexpected type MaybeMate::MateMovess"
                                        )));
                                    },
                                    r => {
                                        return Err(ApplicationError::LogicError(format!("It is an unexpected type {:?}", r)));
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((n, u)) = update_nodes {
                    let pn = u.pn;
                    let dn = u.dn;

                    let u = Rc::new(RefCell::new(u));

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    let mut u = self.update_node(depth, last_id, &u)?;

                    if u.pn == Number::Value(0) && u.dn == Number::INFINITE {
                        u.mate_depth = n.try_borrow()?.mate_depth + 1;
                        return Ok(MaybeMate::Continuation(u));
                    } else if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(u));
                    }
                } else {
                    break;
                }
            }

            if depth == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                if pn == Number::Value(0) && dn == Number::INFINITE {
                    return Ok(MaybeMate::MateMoves(self.build_moves(children
                                                                    .try_borrow()?
                                                                    .iter()
                                                                    .next()
                                                                    .ok_or(ApplicationError::LogicError(String::from(
                                                                        "The legal move has not yet been set."
                                                                    )))?)?));
                } else if pn == Number::INFINITE && dn == Number::Value(0) {
                    return Ok(MaybeMate::Nomate);
                }
            }

            Ok(MaybeMate::Skip)
        }

        pub fn response_oute_process<L: Logger>(&mut self,
                                                depth:u32,
                                                mhash:u64,
                                                shash:u64,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                last_id:&mut u64,
                                                root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                                current_moves:&mut VecDeque<LegalMove>,
                                                mate_depth:&mut Option<u32>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            if self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            self.node_count += 1;

            if mate_depth.map(|d|  depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::Skip);
            }

            if self.max_depth.map(|d| depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;


            let children = if let Some(n) = current_node {
                let mut n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                if children.try_borrow()?.len() == 0 {
                    println!("info string mate.");

                    let mut n = self.update_node(depth, last_id, &n);

                    Ok(n)
                } else {
                    Rc::clone(&n.try_borrow()?.children)
                }
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "current move is not set."
                )));
            };

            loop {
                let mut update_nodes = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    for n in children.try_borrow()?.iter() {
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

                        let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                               state.get_banmen(),
                                                               &mc, m.to_applied_move(), &o);
                        let shash = self.hasher.calc_sub_hash(shash, teban,
                                                              state.get_banmen(),
                                                              &mc, m.to_applied_move(), &o);

                        if let Some(()) = ignore_kyokumen_map.get(teban, &mhash, &shash) {
                            continue;
                        }

                        if let Some(&c) = current_kyokumen_map.get(teban, &mhash, &shash) {
                            if c >= 3 {
                                continue;
                            }
                        }

                        ignore_kyokumen_map.insert(teban, mhash, shash, ());

                        match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
                            &c => {
                                current_kyokumen_map.insert(teban, mhash, shash, c + 1);
                            }
                        }

                        current_moves.push_back(m);

                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         &mut ignore_kyokumen_map,
                                                         &mut current_kyokumen_map,
                                                         last_id,
                                                         root_children,
                                                         Some(Rc::clone(n)),
                                                         current_nodes,
                                                         node_map,
                                                         current_moves,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
                                        println!("info string respond {},{},{:?},{:?}", 0, depth, n.try_borrow()?.pn, n.try_borrow()?.dn);
                                        update_nodes = Some((Rc::clone(n), u));

                                        break;
                                    },
                                    r @ MaybeMate::MaxNodes => {
                                        return Ok(r);
                                    },
                                    r @ MaybeMate::Timeout => {
                                        return Ok(r);
                                    },
                                    r @ MaybeMate::Aborted => {
                                        return Ok(r);
                                    },
                                    MaybeMate::Skip | MaybeMate::MaxDepth => {},
                                    MaybeMate::MateMoves(_) => {
                                        return Err(ApplicationError::LogicError(String::from(
                                            "It is an unexpected type MaybeMate::MateMovess"
                                        )));
                                    },
                                    r => {
                                        return Err(ApplicationError::LogicError(format!("It is an unexpected type {:?}", r)));
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some((n, u)) = update_nodes {
                    let pn = u.pn;
                    let dn = u.dn;

                    let u = Rc::new(RefCell::new(u));

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    let mut u = self.update_node(depth, last_id, &u)?;

                    if u.pn == Number::Value(0) && u.dn == Number::INFINITE {
                        u.mate_depth = n.try_borrow()?.mate_depth + 1;
                        return Ok(MaybeMate::Continuation(u));
                    } else if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(u));
                    }
                } else {
                    break;
                }
            }

            if depth == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                if pn == Number::Value(0) && dn == Number::INFINITE {
                    return Ok(MaybeMate::MateMoves(self.build_moves(children
                        .try_borrow()?
                        .iter()
                        .next()
                        .ok_or(ApplicationError::LogicError(String::from(
                            "The legal move has not yet been set."
                        )))?)?));
                } else if pn == Number::INFINITE && dn == Number::Value(0) {
                    return Ok(MaybeMate::Nomate);
                }
            }

            Ok(MaybeMate::Skip)
        }

        fn send_seldepth(&mut self, depth:u32) -> Result<(),SendSelDepthError>{
            let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
            commands.push(UsiInfoSubCommand::Depth(self.base_depth));
            commands.push(UsiInfoSubCommand::SelDepth(self.current_depth + depth));

//            Ok(self.info_sender.send(commands)?)
            Ok(self.info_sender.send_immediate(commands)?)
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