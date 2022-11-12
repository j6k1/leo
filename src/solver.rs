use std::cell::RefCell;
use std::collections::{VecDeque};
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
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

#[derive(Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(Rc<RefCell<Node>>),
    MaxDepth,
    Skip,
    MaxNodes,
    Timeout,
    Aborted
}
impl Debug for MaybeMate {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            &MaybeMate::Nomate => {
                write!(f,"MaybeMate::Nomate")
            },
            &MaybeMate::MateMoves(ref mvs) => {
                write!(f,"MaybeMate::MateMoves({:?})",mvs)
            },
            &MaybeMate::Unknown => {
                write!(f,"MaybeMate::Unknown")
            },
            &MaybeMate::Continuation(_) => {
                write!(f,"MaybeMate::Continuation")
            },
            &MaybeMate::MaxDepth => {
                write!(f,"MaybeMate::MaxDepth")
            },
            &MaybeMate::Skip => {
                write!(f,"MaybeMate::Skip")
            },
            &MaybeMate::MaxNodes => {
                write!(f,"MaybeMate::MaxNodes")
            },
            &MaybeMate::Timeout => {
                write!(f,"MaybeMate::Timeout")
            },
            &MaybeMate::Aborted => {
                write!(f,"MaybeMate::Aborted")
            }
        }
    }
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

        strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              &KyokumenMap::new(),
                              &KyokumenMap::new(),
                              &mut last_id,
                              0,
                              None,
                              &mut KyokumenMap::new(),

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
        expanded:bool,
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
                expanded: false,
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(AndNodeComparator)
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
                expanded: false,
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(OrNodeComparator)
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
                             parent_id:u64,
                             teban:Teban,
                             node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>)
            -> Result<(Rc<RefCell<Node>>,bool),ApplicationError> {

            if let Some(n) = node_map.get(teban,&mhash,&shash) {
                Ok((Rc::clone(n),n.try_borrow()?.update_nodes.contains(&parent_id)))
            } else {
                Ok((Rc::clone(n),false))
            }
        }

        pub fn expand_nodes(&mut self,
                                       depth:u32,
                                       last_id:&mut u64,
                                       n:&Rc<RefCell<Node>>,
                                       teban:Teban,
                                       state:&State,
                                       mc:&MochigomaCollections)
                                       -> Result<(),ApplicationError> {
            let parent_id = n.try_borrow()?.id;

            {
                let n = n.try_borrow()?;
                let mut children = n.children.try_borrow_mut()?;

                if depth % 2 == 0 {
                    let mvs = Rule::oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        Rc::new(RefCell::new(Node::new_and_node(last_id, m, parent_id)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.insert(Rc::clone(child));
                    }
                } else {
                    let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        Rc::new(RefCell::new(Node::new_or_node(last_id, m, parent_id)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.insert(Rc::clone(child));
                    }
                }
            }

            n.try_borrow_mut()?.expanded = true;

            let len = n.try_borrow()?.children.try_borrow()?.len();

            if depth % 2 == 0 {
                n.try_borrow_mut()?.pn = Number::Value(1);
                n.try_borrow_mut()?.dn = Number::Value(len as u64);
            } else {
                n.try_borrow_mut()?.pn = Number::Value(len as u64);
                n.try_borrow_mut()?.dn = Number::Value(1);
            }

            let update_nodes = n.try_borrow()?.ref_nodes.clone();
            n.try_borrow_mut()?.update_nodes = update_nodes;

            Ok(())
        }

        pub fn expand_root_nodes(&mut self,
                                 last_id:&mut u64,
                                 teban:Teban,
                                 state:&State,
                                 mc:&MochigomaCollections) -> Result<Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,ApplicationError> {
            let mvs = Rule::oute_only_moves_all(teban, state, mc);

            let nodes = mvs.into_iter().map(|m| {
                Rc::new(RefCell::new(Node::new_and_node(last_id, m, 0)))
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
                                                parent_id:u64,
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
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
                                        parent_id,
                                        current_node,
                                        node_map,
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
                                        parent_id,
                                        current_node,
                                        node_map,
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

            r
        }

        pub fn build_moves(&mut self,n:&Rc<RefCell<Node>>) -> Result<VecDeque<LegalMove>,ApplicationError> {
            let mut mvs = VecDeque::new();

            let mut n = Some(Rc::clone(n));

            while let Some(c) = n {
                mvs.push_back(c.try_borrow()?.m);
                n = c.try_borrow()?.children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
            }

            Ok(mvs)
        }

        pub fn update_node(&mut self, depth:u32, last_id:&mut u64, parent_id:u64, n:&Rc<RefCell<Node>>) -> Result<Node,ApplicationError> {
            let m = n.try_borrow()?.m;
            let children = Rc::clone(&n.try_borrow()?.children);
            let expanded = n.try_borrow()?.expanded;
            let mate_depth = n.try_borrow()?.mate_depth;

            if depth % 2 == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in n.try_borrow()?.children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                let ref_nodes = n.try_borrow()?.ref_nodes.clone();

                let mut n = Node::new_or_node(last_id,m,parent_id);

                n.pn = pn;
                n.dn = dn;
                n.children = children;
                n.ref_nodes = ref_nodes.clone();
                n.update_nodes = ref_nodes;
                n.expanded = expanded;
                n.mate_depth = mate_depth;

                Ok(n)
            } else {
                let mut pn = Number::Value(0);
                let mut dn = Number::INFINITE;

                for n in n.try_borrow()?.children.try_borrow()?.iter() {
                    pn += n.try_borrow()?.pn;
                    dn = dn.min(n.try_borrow()?.dn);
                }

                let ref_nodes = n.try_borrow()?.ref_nodes.clone();

                let mut n = Node::new_and_node(last_id,m,parent_id);

                n.pn = pn;
                n.dn = dn;
                n.children = children;
                n.ref_nodes = ref_nodes.clone();
                n.update_nodes = ref_nodes;
                n.expanded = expanded;
                n.mate_depth = mate_depth;

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
                                       parent_id:u64,
                                       current_node:Option<Rc<RefCell<Node>>>,
                                       node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
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

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let children = if let Some(n) = current_node.as_ref() {
                let (mut n,need_update) = self.normalize_node(&n,mhash,shash,parent_id,teban,node_map)?;

                let expanded = n.try_borrow()?.expanded;

                if need_update {
                    return Ok(MaybeMate::Continuation(n));
                } else if !expanded {
                    self.expand_nodes(depth, last_id, &n, teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let mut u = Node::new_or_node(last_id,n.try_borrow()?.m,parent_id);

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(0);
                        u.ref_nodes = n.try_borrow()?.ref_nodes.clone();
                        u.update_nodes = n.try_borrow()?.ref_nodes.clone();
                        u.expanded =  true;

                        n = Rc::new(RefCell::new(u));
                    }

                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let children = &n.try_borrow()?.children;

                    Rc::clone(children)
                }
            } else {
                self.expand_root_nodes(last_id,teban,state,mc)?
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

                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         &mut ignore_kyokumen_map,
                                                         &mut current_kyokumen_map,
                                                         last_id,
                                                         parent_id,
                                                         Some(Rc::clone(n)),
                                                         node_map,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
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
                                            "It is an unexpected type MaybeMate::MateMoves"
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

                if let Some((n, u)) = update_nodes.take() {
                    let mate_depth = u.try_borrow()?.mate_depth;

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    if let Some(n) = current_node.as_ref() {
                        u.try_borrow_mut()?.update_nodes.remove(&n.try_borrow()?.id);

                        let pn = n.try_borrow()?.pn;
                        let dn = n.try_borrow()?.dn;
                        let mut u = self.update_node(depth, last_id, parent_id, n)?;

                        if u.pn == Number::Value(0) && u.dn == Number::INFINITE {
                            u.mate_depth = mate_depth + 1;
                            return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                        } else if u.pn != pn || u.dn != dn {
                            return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                        }
                    } else if !self.strict_moves && u.try_borrow()?.pn == Number::Value(0) && u.try_borrow()?.dn == Number::INFINITE {
                        return Ok(MaybeMate::MateMoves(self.build_moves(&u)?));
                    } else {
                        let mut pn = Number::INFINITE;
                        let mut dn = Number::Value(0);

                        for n in children.try_borrow()?.iter() {
                            pn = pn.min(n.try_borrow()?.pn);
                            dn += n.try_borrow()?.dn;
                        }

                        if pn == Number::Value(0) && dn == Number::INFINITE {
                            return Ok(MaybeMate::MateMoves(self.build_moves(&u)?));
                        }
                    }
                } else if depth == 0 {
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
                    } else {
                        return Ok(MaybeMate::Unknown);
                    }
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
                                                parent_id:u64,
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
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

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let children = if let Some(n) = current_node.as_ref() {
                let (mut n,need_update) = self.normalize_node(&n,mhash,shash,parent_id,teban,node_map)?;

                let expanded = n.try_borrow()?.expanded;

                if need_update {
                    return Ok(MaybeMate::Continuation(n));
                } else if !expanded {
                    self.expand_nodes(depth, last_id, &n, teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let mut u = Node::new_and_node(last_id,n.try_borrow()?.m,parent_id);

                        u.pn = Number::Value(0);
                        u.dn = Number::INFINITE;
                        u.ref_nodes = n.try_borrow()?.ref_nodes.clone();
                        u.update_nodes = n.try_borrow()?.ref_nodes.clone();
                        u.expanded =  true;

                        n = Rc::new(RefCell::new(u));
                    }

                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let children = &n.try_borrow()?.children;

                    Rc::clone(children)
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

                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         &mut ignore_kyokumen_map,
                                                         &mut current_kyokumen_map,
                                                         last_id,
                                                         parent_id,
                                                         Some(Rc::clone(n)),
                                                         node_map,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
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
                                    MaybeMate::Skip | MaybeMate::MaxDepth => {
                                    },
                                    MaybeMate::MateMoves(_) => {
                                        return Err(ApplicationError::LogicError(String::from(
                                            "It is an unexpected type MaybeMate::MateMoves"
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

                if let Some((n, u)) = update_nodes.take() {
                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    let n = current_node.as_ref().map(|n| Rc::clone(n))
                                                         .ok_or(ApplicationError::LogicError(String::from(
                                                            "current node is not set."
                                                         )))?;
                    u.try_borrow_mut()?.update_nodes.remove(&n.try_borrow()?.id);

                    let pn = n.try_borrow()?.pn;
                    let dn = n.try_borrow()?.dn;
                    let u = self.update_node(depth, last_id, parent_id, &n)?;

                    if u.pn == Number::INFINITE && u.dn == Number::Value(0) {
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                    } else if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                    }
                } else {
                    break;
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