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
use crate::solver::checkmate::{CheckmateStrategy};

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

        let root_children = strategy.expand_nodes(0,&mut last_id,ms.teban,&ms.state,&ms.mc)?;

        strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              &KyokumenMap::new(),
                              &KyokumenMap::new(),
                              &mut last_id,
                              &root_children,
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
    use crate::solver::MaybeMate::Continuation;

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
        max_depth:u32,
        ref_nodes:HashSet<u64>,
        update_nodes:HashSet<u64>,
        m:LegalMove,
        children:Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
        comparator:Box<dyn Comparator<Node>>
    }

    impl Node {
        pub fn new_or_node(last_id:&mut u64,depth:u32,m:LegalMove,parent_id:u64) -> Node {
            *last_id += 1;
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                max_depth: depth,
                ref_nodes:ref_nodes,
                update_nodes:HashSet::new(),
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(OrNodeComparator)
            }
        }

        pub fn new_and_node(last_id:&mut u64,depth:u32,m:LegalMove,parent_id:u64) -> Node {
            *last_id += 1;
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: *last_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                max_depth: depth,
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
                .then(l.borrow().max_depth.cmp(&r.borrow().max_depth))
                .then(l.borrow().id.cmp(&r.borrow().id))
        }
    }

    #[derive(Clone)]
    pub struct AndNodeComparator;

    impl Comparator<Node> for AndNodeComparator {
        #[inline]
        fn cmp(&self,l:&Node,r:&Node) -> Ordering {
            l.borrow().dn.cmp(&r.borrow().dn)
                .then(l.borrow().max_depth.cmp(&r.borrow().max_depth))
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

        pub fn update_nodes(&mut self, depth:u32, current_nodes:&VecDeque<Rc<RefCell<Node>>>,) -> Result<u32,ApplicationError> {
            let mut is_mate = false;

            let mut d = depth;
            let mut update_pn_dn = true;

            for (i,n) in (0..=depth).rev().zip(current_nodes.iter().rev()) {
                if !is_mate && !update_pn_dn {
                    break;
                }

                println!("info string update nodes depth {}",i);
                if i % 2 == 0 {
                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        n.try_borrow_mut()?.pn = Number::INFINITE;
                        n.try_borrow_mut()?.dn = Number::Value(0);
                        d -= 1;
                    } else {
                        if update_pn_dn {
                            let mut pn = Number::INFINITE;
                            let mut dn = Number::Value(0);

                            for child in n.try_borrow()?.children.try_borrow()?.iter() {
                                pn = child.try_borrow()?.pn.min(pn);
                                dn += child.try_borrow()?.dn;
                            }

                            if n.try_borrow()?.pn != pn || n.try_borrow()?.dn != dn {
                                n.try_borrow_mut()?.pn = pn;
                                n.try_borrow_mut()?.dn = dn;
                                d -= 1;
                            } else {
                                update_pn_dn = false;
                            }
                        }
                    }
                } else {
                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        n.try_borrow_mut()?.pn = Number::Value(0);
                        n.try_borrow_mut()?.dn = Number::INFINITE;
                        n.try_borrow_mut()?.max_depth = depth;
                        d -= 1;
                        is_mate = true;
                    } else {
                        if update_pn_dn {
                            let mut pn = Number::Value(0);
                            let mut dn = Number::INFINITE;

                            for child in n.try_borrow()?.children.try_borrow()?.iter() {
                                pn += child.try_borrow()?.pn;
                                dn = child.try_borrow()?.dn.min(dn);
                            }

                            if n.try_borrow()?.pn != pn || n.try_borrow()?.dn != dn {
                                n.try_borrow_mut()?.pn = pn;
                                n.try_borrow_mut()?.dn = dn;
                                d -= 1;
                            } else {
                                update_pn_dn = false;
                            }
                        }

                        if is_mate {
                            n.try_borrow_mut()?.max_depth = depth;
                        }
                    }
                }
            }

            let mut parent_id = None;

            for n in current_nodes.iter().rev().skip(1) {
                let ref_nodes = n.try_borrow()?.ref_nodes.clone();
                n.try_borrow_mut()?.update_nodes = ref_nodes;
            }

            for n in current_nodes.iter() {
                if let Some(id) = parent_id {
                    n.try_borrow_mut()?.update_nodes.remove(&id);
                }
                parent_id = Some(n.try_borrow()?.id);
            }

            Ok(d)
        }

        pub fn next_depth(&mut self,
                          depth:u32,
                          root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                          current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                          current_moves:&mut VecDeque<LegalMove>) -> Result<u32,ApplicationError> {

            let mut d = depth;

            if d == 0 {
                let mut n = root_children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
                let mut it = current_moves.iter();

                while let (Some(c),Some(m)) = (n,it.next()) {
                    if c.try_borrow()?.m != *m {
                        break;
                    }
                    n = c.try_borrow()?.children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
                    d += 1;
                }
            } else {
                let mut n = current_nodes.get(d as usize - 1).map(|n| Rc::clone(n));
                let mut it = current_moves.iter().skip(d as usize - 1);

                while let (Some(c),Some(m)) = (n,it.next()) {
                    if c.try_borrow()?.m != *m {
                        break;
                    }
                    n = c.try_borrow()?.children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
                    d += 1;
                }
            }

            Ok(d)
        }
        pub fn preprocess<L: Logger>(&mut self,
                                     depth:u32,
                                     mhash:u64,
                                     shash:u64,
                                     last_id:&mut u64,
                                     root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                                     current_node:Option<Rc<RefCell<Node>>>,
                                     current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                                     node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                     current_moves:&mut VecDeque<LegalMove>,
                                     event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                     event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                     teban:Teban,
                                     state:&State,
                                     mc:&MochigomaCollections) -> Result<(u32,Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>),ApplicationError> {
            let mut d = depth;

            match current_node {
                Some(current_node) => {
                    let len = current_node.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        Ok(if let Some(n) = node_map.get(teban,&mhash,&shash) {
                            if depth == 0 {
                                return Err(ApplicationError::LogicError(String::from(
                                    "Root node is registered in node map."
                                )));
                            }

                            let parent = current_nodes.back();

                            let opt_id = match parent {
                                Some(n) => Some(n.try_borrow()?.id),
                                None => None
                            };

                            current_nodes.push_back(Rc::clone(n));

                            if let Some(id) = opt_id {
                                let update = n.try_borrow()?.update_nodes.contains(&id);

                                if update {
                                    let dep = self.update_nodes(depth, current_nodes)?;

                                    d = self.next_depth(dep,root_children,current_nodes,current_moves)?;
                                }

                                let _ = event_dispatcher.dispatch_events(self, &*event_queue);
                            }

                            (d, Rc::clone(&n.try_borrow()?.children))
                        } else {
                            let n = current_node;

                            current_nodes.push_back(Rc::clone(&n));

                            let nodes = if depth % 2 == 0 {
                                let mvs = Rule::oute_only_moves_all(teban, state, mc);

                                mvs.into_iter().map(|m| {
                                    Ok(Rc::new(RefCell::new(Node::new_or_node(last_id, depth + 1, m, n.try_borrow()?.id))))
                                }).collect::<Result<VecDeque<Rc<RefCell<Node>>>, ApplicationError>>()?
                            } else {
                                let mvs = Rule::respond_oute_only_moves_all(teban,state,mc);

                                mvs.into_iter().map(|m| {
                                    Ok(Rc::new(RefCell::new(Node::new_and_node(last_id, depth + 1, m, n.try_borrow()?.id))))
                                }).collect::<Result<VecDeque<Rc<RefCell<Node>>>, ApplicationError>>()?
                            };

                            if nodes.len() > 0 {
                                {
                                    let n = n.try_borrow_mut()?;

                                    for child in nodes.iter() {
                                        n.children.try_borrow_mut()?.insert(Rc::clone(child));
                                    }
                                }

                                let dep = self.update_nodes(depth, current_nodes)?;

                                d = self.next_depth(dep,root_children,current_nodes,current_moves)?;

                                let _ = event_dispatcher.dispatch_events(self, &*event_queue);
                            }

                            if depth > 0 {
                                node_map.insert(teban, mhash, shash, Rc::clone(&n));
                            }

                            let children = Rc::clone(&n.try_borrow()?.children);

                            (d, children)
                        })
                    } else {
                        let n = current_node;

                        current_nodes.push_back(Rc::clone(&n));

                        let children = Rc::clone(&n.try_borrow()?.children);

                        Ok((d, children))
                    }
                },
                None => {
                    Ok((d,Rc::clone(root_children)))
                }
            }
        }

        pub fn expand_nodes(&mut self,
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

        pub fn on_mate(&mut self,
                       depth:u32,
                       root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                       current_node:&Option<Rc<RefCell<Node>>>,
                       current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                       current_moves:&mut VecDeque<LegalMove>) -> Result<MaybeMate,ApplicationError> {

            current_nodes.push_back(Rc::clone(current_node.as_ref().ok_or(ApplicationError::LogicError(String::from(
                "Current node is not set."
            )))?));

            let d = self.update_nodes(depth, current_nodes)?;

            if let Some(n) = current_node.as_ref() {
                println!("info string mate depth {}, pn {:?}, dn {:?}",depth,n.try_borrow()?.pn,n.try_borrow()?.dn);
            }

            let d = self.next_depth(d,root_children,current_nodes,current_moves)?;

            if depth == d {
                Ok(MaybeMate::Continuation(0))
            } else {
                println!("info string mate {},{}",depth,d);
                Ok(MaybeMate::Continuation(depth - d - 1))
            }
        }

        pub fn on_nomate(&mut self,
                       depth:u32,
                       root_children:&Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
                       current_node:&Option<Rc<RefCell<Node>>>,
                       current_nodes:&mut VecDeque<Rc<RefCell<Node>>>,
                       current_moves:&mut VecDeque<LegalMove>) -> Result<MaybeMate,ApplicationError> {

            current_nodes.push_back(Rc::clone(current_node.as_ref().ok_or(ApplicationError::LogicError(String::from(
                "Current node is not set."
            )))?));

            let d = self.update_nodes(depth, current_nodes)?;

            let d = self.next_depth(d,root_children,current_nodes,current_moves)?;

            println!("info string no_mate Continuation {} {}",depth,d);

            if depth == d {
                Ok(MaybeMate::Continuation(1))
            } else {
                Ok(MaybeMate::Continuation(depth - 1 - d))
            }
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

        pub fn build_moves(&mut self, moves:&VecDeque<LegalMove>,n:&Rc<RefCell<Node>>) -> Result<VecDeque<LegalMove>,ApplicationError> {
            let mut mvs = moves.clone();

            let mut n = Some(Rc::clone(n));

            while let Some(c) = n {
                mvs.push_back(c.try_borrow()?.m);
                n = c.try_borrow()?.children.try_borrow()?.iter().next().map(|n| Rc::clone(n));
            }

            Ok(mvs)
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

            println!("info string send_seldepth {}",depth);

            self.send_seldepth(depth)?;

            let (d,children) = self.preprocess(depth,
                                          mhash,
                                          shash,
                                          last_id,
                                          root_children,
                                          current_node.as_ref().map(|n| Rc::clone(n)),
                                          current_nodes,
                                          node_map,
                                          current_moves,
                                          event_queue,
                                          event_dispatcher,
                                          teban,
                                          state,
                                          mc)?;

            if children.try_borrow()?.len() == 0 {
                if depth == 0 {
                    Ok(MaybeMate::Nomate)
                } else {
                    println!("info string on_nomate.");
                    self.on_nomate(depth,root_children,&current_node,current_nodes,current_moves)
                }
            } else {
                if d == depth {
                    'outer: loop {
                        let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                        let mut current_kyokumen_map = current_kyokumen_map.clone();

                        {
                            println!("info string {}, {}",depth,children.try_borrow()?.len());

                            for n in children.try_borrow()?.iter() {
                                let m = n.try_borrow()?.m;

                                println!("info string {:?}",m.to_move());

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
                                            MaybeMate::Continuation(0) => {
                                                println!("info string Continuation 0");

                                                println!("info string {},{:?},{:?}",depth,n.try_borrow()?.pn,n.try_borrow()?.dn);

                                                if depth == 0 && !self.strict_moves &&
                                                    n.try_borrow()?.pn == Number::Value(0) && n.try_borrow()?.dn == Number::INFINITE {
                                                    let mvs = self.build_moves(current_moves,n)?;
                                                    return Ok(MaybeMate::MateMoves(mvs));
                                                } else if n.try_borrow()?.pn == Number::Value(0) && n.try_borrow()?.dn == Number::INFINITE {
                                                    let mvs = self.build_moves(current_moves,n)?;

                                                    if depth == 0 && mvs.len() == 1 {
                                                        return Ok(MaybeMate::MateMoves(mvs));
                                                    } else if !self.strict_moves {
                                                        return Ok(MaybeMate::Continuation(0));
                                                    } else if depth == 0 {
                                                        *mate_depth = Some(mvs.len() as u32);
                                                    }
                                                } else if depth == 0 && n.try_borrow()?.pn == Number::INFINITE && n.try_borrow()?.dn == Number::Value(0) {
                                                    return Ok(MaybeMate::Nomate);
                                                } else if n.try_borrow()?.pn != Number::INFINITE || n.try_borrow()?.dn != Number::Value(0) {
                                                    println!("info string continue 'outer");
                                                    continue 'outer;
                                                }
                                            },
                                            MaybeMate::Continuation(depth) => {
                                                println!("info string Continuation {}",depth);
                                                return Ok(MaybeMate::Continuation(depth - 1));
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
                                            r => {
                                                return Err(ApplicationError::LogicError(format!("It is an unexpected type  {:?}",r)));
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }

                    if depth == 0 {
                        let n = children.try_borrow()?.iter().next().map(|n| Rc::clone(n));

                        if let Some(n) = n {
                            if n.try_borrow()?.pn == Number::Value(0) &&
                               n.try_borrow()?.dn == Number::INFINITE {
                                Ok(MaybeMate::MateMoves(self.build_moves(current_moves,&n)?))
                            } else {
                                Ok(MaybeMate::Unknown)
                            }
                        } else {
                            Ok(MaybeMate::Nomate)
                        }
                    } else {
                        Ok(MaybeMate::Skip)
                    }
                } else {
                    Ok(MaybeMate::Continuation(depth - 1 - d))
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

            println!("info string send_seldepth {}",depth);

            self.send_seldepth(depth)?;

            let (d,children) = self.preprocess(depth,
                                          mhash,
                                          shash,
                                          last_id,
                                          root_children,
                                          current_node.as_ref().map(|n| Rc::clone(n)),
                                          current_nodes,
                                          node_map,
                                          current_moves,
                                          event_queue,
                                          event_dispatcher,
                                          teban,
                                          state,
                                          mc)?;
            if children.try_borrow()?.len() == 0 {
                println!("info string on_mate.");
                self.on_mate(depth,root_children,&current_node,current_nodes,current_moves)
            } else {
                if d == depth {
                    'outer: loop {
                        let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                        let mut current_kyokumen_map = current_kyokumen_map.clone();

                        {
                            for n in children.try_borrow()?.iter() {
                                let m = n.try_borrow()?.m;

                                println!("info string {:?} respond.",m.to_move());

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
                                            MaybeMate::Continuation(0) => {
                                                if n.try_borrow()?.pn == Number::INFINITE &&
                                                   n.try_borrow()?.dn == Number::Value(0) {
                                                    return Ok(Continuation(0));
                                                } else if n.try_borrow()?.pn != Number::Value(0) ||
                                                          n.try_borrow()?.dn != Number::INFINITE {
                                                    continue 'outer;
                                                }
                                            },
                                            MaybeMate::Continuation(depth) => {
                                                println!("info string respond Continuation {}",depth);
                                                return Ok(MaybeMate::Continuation(depth - 1));
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
                                            r => {
                                                return Err(ApplicationError::LogicError(format!("It is an unexpected type  {:?}",r)));
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }

                    let n = current_node.ok_or(ApplicationError::LogicError(String::from(
                        "Current node is none."
                    )))?;

                    if n.try_borrow()?.pn == Number::Value(0) &&
                       n.try_borrow()?.dn == Number::INFINITE {
                        Ok(MaybeMate::Continuation(0))
                    } else {
                        Ok(MaybeMate::Skip)
                    }
                } else {
                    println!("info string nomate Continuation {}",depth - d);
                    Ok(MaybeMate::Continuation(depth - 1 - d))
                }
            }
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