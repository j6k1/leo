use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign};
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
use crate::solver::checkmate::{CheckmateStrategy, Node, ReverseNode, UniqID};

#[derive(Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(Rc<RefCell<Node>>),
    MaxDepth,
    Skip,
    Pending,
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
            &MaybeMate::Pending => {
                write!(f,"MaybeMate::Pending")
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
#[derive(Clone,Copy,PartialEq,Eq)]
pub struct Fraction {
    n:u64,
    d:u64
}
fn gcd(a:u64,b:u64) -> u64 {
    if b > a {
        gcd(a,b)
    } else if b == 0 {
        a
    } else {
        gcd(b,a%b)
    }
}
impl Fraction {
    pub fn new(n:u64) -> Fraction {
        Fraction {
            n:n,
            d:1
        }
    }
}

impl Add for Fraction {
    type Output = Fraction;
    fn add(self, rhs: Self) -> Self::Output {
        let ad = self.d;
        let bd = rhs.d;
        let an = self.n * bd;
        let bn = rhs.n * ad;
        let d = ad * bd;
        let n = an + bn;

        if n >= d {
            let g = gcd(n,d);

            Fraction {
                n:n / g,
                d:d / g
            }
        } else {
            Fraction {
                n:n,
                d:d
            }
        }
    }
}
impl AddAssign for Fraction {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Div for Fraction {
    type Output = Fraction;
    fn div(self, rhs: Self) -> Self::Output {
        let ad = self.d;
        let bd = rhs.d;
        let an = self.n;
        let bn = rhs.n;

        let n = an * bd;
        let d = bn * ad;

        if n >= d {
            let g = gcd(n,d);

            Fraction {
                n:n / g,
                d:d / g
            }
        } else {
            Fraction {
                n:n,
                d:d
            }
        }
    }
}
impl DivAssign for Fraction {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl Ord for Fraction {
    fn cmp(&self, other: &Self) -> Ordering {
        let ad = self.d;
        let bd = other.d;
        let an = self.n;
        let bn = other.n;

        let an = an * bd;
        let bn = bn * ad;

        an.cmp(&bn)
    }
}
impl PartialOrd for Fraction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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
        let mut event_dispatcher = Root::<L,S>::create_event_dispatcher::<CheckmateStrategy<S>>(on_error_handler,&stop,&quited);

        let mut uniq_id = UniqID::new();
        let id = uniq_id.gen();

        strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              &KyokumenMap::new(),
                              &KyokumenMap::new(),
                              &mut uniq_id,
                              id,
                              &Rc::new(ReverseNode::new(id,None)),
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
    use std::rc::{Rc, Weak};
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
        parent_id:u64,
        pn:Number,
        dn:Number,
        depth:u32,
        mate_depth:u32,
        reverse_node:Rc<ReverseNode>,
        skip_set:HashSet<u64>,
        expanded:bool,
        m:LegalMove,
        children:Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
        comparator:Box<dyn Comparator<Node>>
    }

    impl Node {
        pub fn new_or_node(id:u64,depth:u32,m:LegalMove,parent_id:u64,parent:&Rc<ReverseNode>) -> Node {
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: id,
                parent_id:parent_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                depth: depth,
                mate_depth: 0,
                reverse_node:Rc::new(ReverseNode::new(id,Some(parent))),
                skip_set:HashSet::new(),
                expanded: false,
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(AndNodeComparator)
            }
        }

        pub fn new_and_node(id:u64,depth:u32,m:LegalMove,parent_id:u64,parent:&Rc<ReverseNode>) -> Node {
            let mut ref_nodes = HashSet::new();
            ref_nodes.insert(parent_id);

            Node {
                id: id,
                parent_id:parent_id,
                pn: Number::Value(1),
                dn: Number::Value(1),
                depth: depth,
                mate_depth: 0,
                reverse_node:Rc::new(ReverseNode::new(id,Some(parent))),
                skip_set:HashSet::new(),
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

    pub struct ReverseNode {
        id:u64,
        parent:Option<Weak<ReverseNode>>
    }

    impl ReverseNode {
        pub fn new<'a>(id:u64,parent:Option<&'a Rc<ReverseNode>>) -> ReverseNode {
            ReverseNode {
                id:id,
                parent:parent.map(|n| Rc::downgrade(n))
            }
        }

        pub fn find_root(&self) -> u64 {
            let mut id = self.id;
            let mut parent = self.parent.as_ref().and_then(|n| n.upgrade());

            while let Some(n) = parent {
                id = n.id;
                parent = n.parent.as_ref().and_then(|n| n.upgrade());
            }

            id
        }

        pub fn is_active(&self) -> bool {
            self.find_root() == 0
        }
    }

    pub struct UniqID {
        last_id:u64
    }

    impl UniqID {
        pub fn new() -> UniqID {
            UniqID {
                last_id: 0
            }
        }

        pub fn gen(&mut self) -> u64 {
            let id = self.last_id;

            self.last_id += 1;

            id
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
                             depth:u32,
                             teban:Teban,
                             node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>)
            -> Result<(Rc<RefCell<Node>>,bool,bool),ApplicationError> {

            if let Some(c) = node_map.get(teban,&mhash,&shash) {
                if c.try_borrow()?.parent_id == parent_id {
                    Ok((Rc::clone(n),false,false))
                } else if c.try_borrow()?.depth > depth {
                    Ok((Rc::clone(c), true, false))
                } else if !c.try_borrow()?.reverse_node.is_active() {
                    Ok((Rc::clone(c),true,false))
                } else {
                    Ok((Rc::clone(n),false,true))
                }
            } else {
                Ok((Rc::clone(n),false,false))
            }
        }

        pub fn expand_nodes(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       uniq_id:&mut UniqID,
                                       n:&Rc<RefCell<Node>>,
                                       node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>,
                                       teban:Teban,
                                       state:&State,
                                       mc:&MochigomaCollections)
                                       -> Result<(),ApplicationError> {
            let parent_id = n.try_borrow()?.id;

            {
                let n = n.try_borrow()?;
                let reverse_node = &n.reverse_node;

                let mut children = n.children.try_borrow_mut()?;

                if depth % 2 == 0 {
                    let mvs = Rule::oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_and_node(id, depth,m, parent_id,reverse_node)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.insert(Rc::clone(child));
                    }
                } else {
                    let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_or_node(id, depth, m, parent_id,reverse_node)))
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

            node_map.insert(teban,mhash,shash,Rc::clone(n));

            Ok(())
        }

        pub fn expand_root_nodes(&mut self,
                                 parent_id:u64,
                                 reverse_node:&Rc<ReverseNode>,
                                 uniq_id:&mut UniqID,
                                 teban:Teban,
                                 state:&State,
                                 mc:&MochigomaCollections) -> Result<Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,ApplicationError> {
            let mvs = Rule::oute_only_moves_all(teban, state, mc);

            let nodes = mvs.into_iter().map(|m| {
                let id = uniq_id.gen();
                Rc::new(RefCell::new(Node::new_and_node(id, 0, m, parent_id, reverse_node)))
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
                                                uniq_id:&mut UniqID,
                                                parent_id:u64,
                                                parent:&Rc<ReverseNode>,
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
                                        uniq_id,
                                        parent_id,
                                        parent,
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
                                        uniq_id,
                                        parent_id,
                                        parent,
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

        pub fn update_node(&mut self, depth:u32, id:u64, parent_id:u64, n:&Rc<RefCell<Node>>) -> Result<Node,ApplicationError> {
            let m = n.try_borrow()?.m;
            let children = Rc::clone(&n.try_borrow()?.children);
            let skip_set = n.try_borrow()?.skip_set.clone();
            let expanded = n.try_borrow()?.expanded;
            let mate_depth = n.try_borrow()?.mate_depth;

            if depth % 2 == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(0);

                for n in n.try_borrow()?.children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                let reverse_node = &n.try_borrow()?.reverse_node;
                let mut n = Node::new_or_node(id,depth, m,parent_id, reverse_node);

                n.pn = pn;
                n.dn = dn;
                n.parent_id = parent_id;
                n.children = children;
                n.skip_set = skip_set;
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

                let reverse_node = &n.try_borrow()?.reverse_node;
                let mut n = Node::new_and_node(id,depth, m,parent_id,reverse_node);

                n.pn = pn;
                n.dn = dn;
                n.children = children;
                n.parent_id = parent_id;
                n.skip_set = skip_set;
                n.expanded = expanded;
                n.mate_depth = mate_depth;

                Ok(n)
            }
        }

        pub fn oute_process<'a,L: Logger>(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       ignore_kyokumen_map:&KyokumenMap<u64,()>,
                                       current_kyokumen_map:&KyokumenMap<u64,u32>,
                                       uniq_id:&mut UniqID,
                                       parent_id:u64,
                                       parent:&Rc<ReverseNode>,
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

            if self.max_depth.map(|d| depth > d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            self.node_count += 1;

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let children = if let Some(n) = current_node.as_ref() {
                let (mut n,need_update,pending) = self.normalize_node(&n,mhash,shash,parent_id,depth,teban,node_map)?;

                if pending {
                    return Ok(MaybeMate::Pending);
                }

                let expanded = n.try_borrow()?.expanded;

                if need_update {
                    n.try_borrow_mut()?.parent_id = parent_id;
                    return Ok(MaybeMate::Continuation(n));
                } else if !expanded {
                    self.expand_nodes(depth, mhash,shash,uniq_id, &n,node_map, teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let id = uniq_id.gen();
                        let mut u = Node::new_or_node(id,depth, n.try_borrow()?.m,parent_id,parent);

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(0);
                        u.expanded =  true;

                        n = Rc::new(RefCell::new(u));
                    }

                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let children = &n.try_borrow()?.children;

                    Rc::clone(children)
                }
            } else {
                let children = self.expand_root_nodes(parent_id, parent, uniq_id,teban,state,mc)?;

                if children.try_borrow()?.len() == 0 {
                    return Ok(MaybeMate::Nomate);
                } else {
                    children
                }
            };

            let mut pending = false;

            loop {
                let parent_id = current_node.as_ref().map(|n| {
                    n.try_borrow().map(|n| n.id)
                }).unwrap_or(Ok(parent_id))?;

                let parent_reverse = current_node.as_ref().map(|n| {
                    n.try_borrow().map(|n| Rc::clone(&n.reverse_node))
                }).unwrap_or(Ok(Rc::clone(parent)))?;

                let mut update_nodes = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    for n in children.try_borrow()?.iter() {
                        if n.try_borrow()?.skip_set.contains(&parent_id) {
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

                        let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                               state.get_banmen(),
                                                               &mc, m.to_applied_move(), &o);
                        let shash = self.hasher.calc_sub_hash(shash, teban,
                                                              state.get_banmen(),
                                                              &mc, m.to_applied_move(), &o);

                        if let Some(()) = ignore_kyokumen_map.get(teban, &mhash, &shash) {
                            n.try_borrow_mut()?.skip_set.insert(parent_id);
                            continue;
                        }

                        if let Some(&c) = current_kyokumen_map.get(teban, &mhash, &shash) {
                            if c >= 3 {
                                let id = n.try_borrow()?.id;
                                let mut u = Node::new_or_node(id,depth, n.try_borrow()?.m,parent_id,&parent_reverse);

                                u.pn = Number::INFINITE;
                                u.dn = Number::Value(0);
                                u.expanded =  true;

                                let u = Rc::new(RefCell::new(u));

                                return Ok(MaybeMate::Continuation(u));
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
                                                         uniq_id,
                                                         parent_id,
                                                         &parent_reverse,
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
                                        n.try_borrow_mut()?.skip_set.insert(parent_id);
                                    },
                                    MaybeMate::Pending => {
                                        pending = true;
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

                        event_dispatcher.dispatch_events(self,event_queue)?;

                        if self.stop.load(atomic::Ordering::Acquire) {
                            return Ok(MaybeMate::Aborted)
                        }
                    }
                }

                if let Some((n, u)) = update_nodes.take() {
                    {
                        let id = u.try_borrow()?.id;
                        let pn = u.try_borrow()?.pn;
                        let dn = u.try_borrow()?.dn;

                        if (pn == Number::Value(0) && dn == Number::INFINITE) ||
                            (pn == Number::INFINITE && dn == Number::Value(0)) {
                            u.try_borrow_mut()?.reverse_node = Rc::new(ReverseNode::new(id,None));
                        }
                    }

                    let mate_depth = u.try_borrow()?.mate_depth;

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    if let Some(n) = current_node.as_ref() {
                        let pn = n.try_borrow()?.pn;
                        let dn = n.try_borrow()?.dn;
                        let id = n.try_borrow()?.id;
                        let mut u = self.update_node(depth, id, parent_id, n)?;

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
                } else {
                    break;
                }
            }

            if pending {
                Ok(MaybeMate::Pending)
            } else {
                Ok(MaybeMate::Skip)
            }
        }

        pub fn response_oute_process<L: Logger>(&mut self,
                                                depth:u32,
                                                mhash:u64,
                                                shash:u64,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                uniq_id:&mut UniqID,
                                                parent_id:u64,
                                                parent:&Rc<ReverseNode>,
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

            if mate_depth.map(|d|  depth >= d).unwrap_or(false) {
                return Ok(MaybeMate::Skip);
            }

            if self.max_depth.map(|d| depth > d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            self.node_count += 1;

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            self.send_seldepth(depth)?;

            let children = if let Some(n) = current_node.as_ref() {
                let (mut n,need_update, pending) = self.normalize_node(&n,mhash,shash,parent_id,depth, teban,node_map)?;

                if pending {
                    return Ok(MaybeMate::Pending);
                }

                let expanded = n.try_borrow()?.expanded;

                if need_update {
                    return Ok(MaybeMate::Continuation(n));
                } else if !expanded {
                    self.expand_nodes(depth, mhash,shash,uniq_id, &n, node_map,teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let id = n.try_borrow()?.id;

                        let mut u = Node::new_and_node(id,depth, n.try_borrow()?.m,parent_id,parent);

                        u.pn = Number::Value(0);
                        u.dn = Number::INFINITE;
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

            let mut pending = false;

            loop {
                let parent_id = current_node.as_ref().map(|n| {
                    n.try_borrow().map(|n| n.id)
                }).unwrap_or(Ok(parent_id))?;

                let parent_reverse = current_node.as_ref().map(|n| {
                    n.try_borrow().map(|n| Rc::clone(&n.reverse_node))
                }).unwrap_or(Ok(Rc::clone(parent)))?;

                let mut update_nodes = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    for n in children.try_borrow()?.iter() {
                        if n.try_borrow()?.skip_set.contains(&parent_id) {
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

                        let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                               state.get_banmen(),
                                                               &mc, m.to_applied_move(), &o);
                        let shash = self.hasher.calc_sub_hash(shash, teban,
                                                              state.get_banmen(),
                                                              &mc, m.to_applied_move(), &o);

                        if let Some(()) = ignore_kyokumen_map.get(teban, &mhash, &shash) {
                            n.try_borrow_mut()?.skip_set.insert(parent_id);
                            continue;
                        }

                        if let Some(&c) = current_kyokumen_map.get(teban, &mhash, &shash) {
                            if c >= 3 {
                                let id = n.try_borrow()?.id;
                                let mut u = Node::new_and_node(id,depth, n.try_borrow()?.m,parent_id,&parent_reverse);

                                u.pn = Number::Value(0);
                                u.dn = Number::INFINITE;
                                u.expanded =  true;

                                let u = Rc::new(RefCell::new(u));

                                return Ok(MaybeMate::Continuation(u));
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
                                                         uniq_id,
                                                         parent_id,
                                                         &parent_reverse,
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
                                        n.try_borrow_mut()?.skip_set.insert(parent_id);
                                    },
                                    MaybeMate::Pending => {
                                        pending = true;
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

                        event_dispatcher.dispatch_events(self,event_queue)?;

                        if self.stop.load(atomic::Ordering::Acquire) {
                            return Ok(MaybeMate::Aborted)
                        }
                    }
                }

                if let Some((n, u)) = update_nodes.take() {
                    {
                        let id = u.try_borrow()?.id;
                        let pn = u.try_borrow()?.pn;
                        let dn = u.try_borrow()?.dn;

                        if (pn == Number::Value(0) && dn == Number::INFINITE) ||
                            (pn == Number::INFINITE && dn == Number::Value(0)) {
                            u.try_borrow_mut()?.reverse_node = Rc::new(ReverseNode::new(id,None));
                        }
                    }

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    let n = current_node.as_ref().map(|n| Rc::clone(n))
                                                         .ok_or(ApplicationError::LogicError(String::from(
                                                            "current node is not set."
                                                         )))?;
                    let pn = n.try_borrow()?.pn;
                    let dn = n.try_borrow()?.dn;
                    let id = n.try_borrow()?.id;
                    let u = self.update_node(depth, id, parent_id, &n)?;

                    if u.pn == Number::INFINITE && u.dn == Number::Value(0) {
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                    } else if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u))));
                    }
                } else {
                    break;
                }
            }

            if pending {
                Ok(MaybeMate::Pending)
            } else {
                Ok(MaybeMate::Skip)
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