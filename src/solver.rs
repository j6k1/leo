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
use crate::solver::checkmate::{CheckmateStrategy, Node, UniqID};

#[derive(Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(Rc<RefCell<Node>>,u64,u64),
    Mate,
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
            &MaybeMate::Continuation(_,_,_) => {
                write!(f,"MaybeMate::Continuation")
            },
            &MaybeMate::Mate => {
                write!(f,"MaybeMate::Mate")
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
#[derive(Clone,Copy,PartialEq,Eq)]
pub struct Fraction {
    n:u64,
    d:u64
}
fn gcd(a:u64,b:u64) -> u64 {
    if b > a {
        gcd(b,a)
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

    pub fn is_zero(&self) -> bool {
        self.n == 0
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
impl Div<u64> for Fraction {
    type Output = Fraction;
    fn div(self, rhs: u64) -> Self::Output {
        let n = self.n;
        let d = self.d * rhs;

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
impl DivAssign<u64> for Fraction {
    fn div_assign(&mut self, rhs: u64) {
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
impl Debug for Fraction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,"{} / {}",self.n,self.d)
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
    use std::collections::{BTreeSet, VecDeque};
    use std::ops::{Add, AddAssign, Div};
    use std::rc::{Rc};
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
    use crate::solver::{Fraction, MaybeMate};

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub enum Number {
        Value(Fraction),
        INFINITE
    }

    impl Number {
        pub fn is_zero(&self) -> bool {
            match self {
                &Number::INFINITE => false,
                &Number::Value(v) => v.is_zero()
            }
        }
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

    impl Div<u64> for Number {
        type Output = Number;
        fn div(self, rhs: u64) -> Self::Output {
            match self {
                Number::INFINITE => Number::INFINITE,
                Number::Value(f) => Number::Value(f / rhs)
            }
        }
    }

    pub struct Node {
        id:u64,
        pn:Number,
        dn:Number,
        mate_depth:u32,
        ref_count:u64,
        skip_depth:Option<u32>,
        sennichite:bool,
        expanded:bool,
        m:LegalMove,
        children:Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,
        comparator:Box<dyn Comparator<Node>>
    }

    impl Node {
        pub fn new_or_node(id:u64,m:LegalMove) -> Node {
            Node {
                id: id,
                pn: Number::Value(Fraction::new(1)),
                dn: Number::Value(Fraction::new(1)),
                mate_depth: 0,
                ref_count:1,
                skip_depth:None,
                sennichite: false,
                expanded: false,
                m:m,
                children:Rc::new(RefCell::new(BTreeSet::new())),
                comparator:Box::new(AndNodeComparator)
            }
        }

        pub fn new_and_node(id:u64,m:LegalMove) -> Node {
            Node {
                id: id,
                pn: Number::Value(Fraction::new(1)),
                dn: Number::Value(Fraction::new(1)),
                mate_depth: 0,
                ref_count:1,
                skip_depth:None,
                sennichite: false,
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
                             teban:Teban,
                             node_map:&mut KyokumenMap<u64,Rc<RefCell<Node>>>)
            -> Result<Rc<RefCell<Node>>,ApplicationError> {

            let expanded = n.try_borrow()?.expanded;

            if n.try_borrow()?.sennichite {
                return Ok(Rc::clone(n))
            }

            if let Some(c) = node_map.get(teban,&mhash,&shash) {
                if !expanded {
                    c.try_borrow_mut()?.ref_count += 1;
                }
                Ok(Rc::clone(c))
            } else {
                Ok(Rc::clone(n))
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
            {
                let n = n.try_borrow()?;

                let mut children = n.children.try_borrow_mut()?;

                if depth % 2 == 0 {
                    let mvs = Rule::oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_and_node(id, m)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.insert(Rc::clone(child));
                    }
                } else {
                    let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_or_node(id, m)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.insert(Rc::clone(child));
                    }
                }
            }

            n.try_borrow_mut()?.expanded = true;

            let len = n.try_borrow()?.children.try_borrow()?.len();

            let parent_count = n.try_borrow()?.ref_count;

            if depth % 2 == 0 {
                n.try_borrow_mut()?.pn = Number::Value(Fraction::new(1) / parent_count);
                n.try_borrow_mut()?.dn = Number::Value(Fraction::new(len as u64) / parent_count);
            } else {
                n.try_borrow_mut()?.pn = Number::Value(Fraction::new(len as u64) / parent_count);
                n.try_borrow_mut()?.dn = Number::Value(Fraction::new(1) / parent_count);
            }

            node_map.insert(teban,mhash,shash,Rc::clone(n));

            Ok(())
        }

        pub fn expand_root_nodes(&mut self,
                                 uniq_id:&mut UniqID,
                                 teban:Teban,
                                 state:&State,
                                 mc:&MochigomaCollections) -> Result<Rc<RefCell<BTreeSet<Rc<RefCell<Node>>>>>,ApplicationError> {
            let mvs = Rule::oute_only_moves_all(teban, state, mc);

            let nodes = mvs.into_iter().map(|m| {
                let id = uniq_id.gen();
                Rc::new(RefCell::new(Node::new_and_node(id, m)))
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

        pub fn update_node(&mut self, depth:u32, n:&Rc<RefCell<Node>>) -> Result<Node,ApplicationError> {
            let (id,m,children,ref_count,skip_depth,expanded,mate_depth) = {
                let mut n =  n.try_borrow_mut()?;

                let mut children = Rc::new(RefCell::new(BTreeSet::new()));

                std::mem::swap(&mut n.children, &mut children);

                (n.id,n.m,children,n.ref_count,n.skip_depth,n.expanded,n.mate_depth)
            };

            if depth % 2 == 0 {
                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(Fraction::new(0));

                for n in children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                let mut n = Node::new_or_node(id,m);

                n.pn = pn / ref_count;
                n.dn = dn / ref_count;
                n.children = children;
                n.ref_count = ref_count;
                n.skip_depth = skip_depth;
                n.expanded = expanded;
                n.mate_depth = mate_depth;

                Ok(n)
            } else {
                let mut pn = Number::Value(Fraction::new(0));
                let mut dn = Number::INFINITE;

                for n in children.try_borrow()?.iter() {
                    pn += n.try_borrow()?.pn;
                    dn = dn.min(n.try_borrow()?.dn);
                }

                let mut n = Node::new_and_node(id,m);

                n.pn = pn / ref_count;
                n.dn = dn / ref_count;
                n.children = children;
                n.ref_count = ref_count;
                n.skip_depth = skip_depth;
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
                let mut n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                if n.try_borrow()?.pn == Number::INFINITE && n.try_borrow()?.dn.is_zero() {
                    return Ok(MaybeMate::Nomate);
                }

                let expanded = n.try_borrow()?.expanded;

                println!("info string expanded {}",expanded);

                if !expanded {
                    self.expand_nodes(depth, mhash,shash,uniq_id, &n,node_map, teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let id = uniq_id.gen();
                        let mut u = Node::new_or_node(id,n.try_borrow()?.m);

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(Fraction::new(0));
                        u.ref_count = n.try_borrow()?.ref_count;
                        u.skip_depth = n.try_borrow()?.skip_depth.clone();
                        u.expanded =  true;

                        n = Rc::new(RefCell::new(u));

                        println!("info string nomate.");
                    }

                    return Ok(MaybeMate::Continuation(n,mhash,shash));
                } else {
                    let children = &n.try_borrow()?.children;

                    Rc::clone(children)
                }
            } else {
                let children = self.expand_root_nodes(uniq_id,teban,state,mc)?;

                if children.try_borrow()?.len() == 0 {
                    return Ok(MaybeMate::Nomate);
                } else {
                    children
                }
            };

            println!("info string len {}",children.try_borrow()?.len());

            loop {
                let parent_id = if let Some(n) = current_node.as_ref() {
                    self.normalize_node(n,mhash,shash,teban,node_map)?.try_borrow()?.id
                } else {
                    parent_id
                };

                let mut update_info = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    println!("info string len {}",children.try_borrow()?.len());

                    for n in children.try_borrow()?.iter() {
                        if n.try_borrow()?.sennichite {
                            println!("info string skip sennichite.");
                            continue;
                        }

                        if n.try_borrow()?.skip_depth.map(|d| d <= depth).unwrap_or(false) {
                            println!("info string skip {}",depth);
                            continue;
                        }

                        let m = n.try_borrow()?.m;

                        println!("info string {:?}",m.to_move());
                        println!("info string pn {:?}, dn {:?}",n.try_borrow()?.pn,n.try_borrow()?.dn);

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

                        {
                            let s = ignore_kyokumen_map.get(teban, &mhash, &shash).is_some();
                            let sc = current_kyokumen_map.get(teban, &mhash, &shash).map(|&c| c >= 3).unwrap_or(false);

                            if s || sc {
                                println!("info string sennichite.");
                                let mut u = self.update_node(depth + 1, n)?;

                                u.pn = Number::INFINITE;
                                u.dn = Number::Value(Fraction::new(0));
                                u.expanded = true;
                                u.sennichite = true;

                                let u = Rc::new(RefCell::new(u));

                                update_info = Some((Rc::clone(n), u,mhash,shash));
                                break;
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
                                                         Some(Rc::clone(n)),
                                                         node_map,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u,mhash,shash) => {
                                        update_info = Some((Rc::clone(n), u,mhash,shash));
                                        break;
                                    },
                                    MaybeMate::Nomate => {

                                    },
                                    MaybeMate::Mate if depth == 0 && !self.strict_moves => {
                                        return Ok(MaybeMate::MateMoves(self.build_moves(n)?));
                                    },
                                    MaybeMate::Mate if depth == 0 => {
                                    },
                                    MaybeMate::Mate => {
                                        return Ok(MaybeMate::Mate);
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
                                        let skip_depth = n.try_borrow()?.skip_depth.clone().map(|d| d.max(depth));

                                        n.try_borrow_mut()?.skip_depth = skip_depth;
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

                if let Some((n, u,mh,sh)) = update_info.take() {
                    let mate_depth = u.try_borrow()?.mate_depth;

                    let sennichite = u.try_borrow()?.sennichite;

                    if !sennichite {
                        node_map.insert(teban.opposite(), mh, sh, Rc::clone(&u));
                    }

                    println!("info string borrow children.");
                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));
                    println!("info string borrowed children.");

                    if let Some(n) = current_node.as_ref() {
                        let n = self.normalize_node(n,mhash,shash,teban,node_map)?;
                        let pn = n.try_borrow()?.pn;
                        let dn = n.try_borrow()?.dn;
                        let mut u = self.update_node(depth, &n)?;

                        println!("info string {}, pn {:?}, u.pn {:?}, dn {:?}, u.dn {:?}",depth,pn,u.pn,dn,u.dn);

                        if u.pn.is_zero() && u.dn == Number::INFINITE {
                            u.mate_depth = mate_depth + 1;
                            return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u)),mhash,shash));
                        } else if u.pn != pn || u.dn != dn {
                            return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u)), mhash, shash));
                        } else {
                            n.try_borrow_mut()?.children = u.children;
                        }
                    } else if !self.strict_moves && u.try_borrow()?.pn.is_zero() && u.try_borrow()?.dn == Number::INFINITE {
                        return Ok(MaybeMate::MateMoves(self.build_moves(&u)?));
                    } else {
                        let mut pn = Number::INFINITE;
                        let mut dn = Number::Value(Fraction::new(0));

                        for n in children.try_borrow()?.iter() {
                            pn = pn.min(n.try_borrow()?.pn);
                            dn += n.try_borrow()?.dn;
                        }

                        if pn.is_zero() && dn == Number::INFINITE {
                            return Ok(MaybeMate::MateMoves(self.build_moves(&u)?));
                        }
                    }
                } else if depth == 0 {
                    let mut pn = Number::INFINITE;
                    let mut dn = Number::Value(Fraction::new(0));

                    for n in children.try_borrow()?.iter() {
                        pn = pn.min(n.try_borrow()?.pn);
                        dn += n.try_borrow()?.dn;
                    }

                    if pn.is_zero() && dn == Number::INFINITE {
                        return Ok(MaybeMate::MateMoves(self.build_moves(children
                            .try_borrow()?
                            .iter()
                            .next()
                            .ok_or(ApplicationError::LogicError(String::from(
                                "The legal move has not yet been set."
                            )))?)?));
                    } else if pn == Number::INFINITE && dn.is_zero() {
                        return Ok(MaybeMate::Nomate);
                    } else {
                        return Ok(MaybeMate::Unknown);
                    }
                } else {
                    break;
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
                                                uniq_id:&mut UniqID,
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
                let mut n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                if n.try_borrow()?.pn.is_zero() && n.try_borrow()?.dn == Number::INFINITE {
                    return Ok(MaybeMate::Mate);
                } else if n.try_borrow()?.pn == Number::INFINITE && n.try_borrow()?.dn.is_zero() {
                    return Ok(MaybeMate::Nomate);
                }

                let expanded = n.try_borrow()?.expanded;

                if !expanded {
                    self.expand_nodes(depth, mhash,shash,uniq_id, &n, node_map,teban, state, mc)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let id = n.try_borrow()?.id;

                        let mut u = Node::new_and_node(id,n.try_borrow()?.m);

                        u.pn = Number::Value(Fraction::new(0));
                        u.dn = Number::INFINITE;
                        u.ref_count = n.try_borrow()?.ref_count;
                        u.skip_depth = n.try_borrow()?.skip_depth.clone();
                        u.expanded =  true;

                        n = Rc::new(RefCell::new(u));
                    }

                    return Ok(MaybeMate::Continuation(n,mhash,shash));
                } else {
                    let children = &n.try_borrow()?.children;

                    Rc::clone(children)
                }
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "current move is not set."
                )));
            };

            println!("info string response len {}",children.try_borrow()?.len());

            loop {
                let parent_id = if let Some(n) = current_node.as_ref() {
                    self.normalize_node(n,mhash,shash,teban,node_map)?.try_borrow()?.id
                } else {
                    parent_id
                };

                let mut update_info = None;
                let mut ignore_kyokumen_map = ignore_kyokumen_map.clone();
                let mut current_kyokumen_map = current_kyokumen_map.clone();

                {
                    for n in children.try_borrow()?.iter() {
                        if n.try_borrow()?.sennichite {
                            continue;
                        }

                        if n.try_borrow()?.skip_depth.map(|d| d <= depth).unwrap_or(false) {
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
                        {
                            let s = ignore_kyokumen_map.get(teban, &mhash, &shash).is_some();
                            let sc = current_kyokumen_map.get(teban, &mhash, &shash).map(|&c| c >= 3).unwrap_or(false);

                            if sc {
                                println!("info string response sennichite.");
                                let mut u = self.update_node(depth + 1, n)?;

                                u.pn = Number::Value(Fraction::new(0));
                                u.dn = Number::INFINITE;
                                u.expanded = true;
                                u.sennichite = true;

                                let u = Rc::new(RefCell::new(u));

                                update_info = Some((Rc::clone(n), u, mhash, shash));
                                break;
                            }

                            if s {
                                let skip_depth = n.try_borrow()?.skip_depth.clone().map(|d| d.max(depth));

                                n.try_borrow_mut()?.skip_depth = skip_depth;

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
                                                         uniq_id,
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
                                    MaybeMate::Continuation(u,mhash,shash) => {
                                        update_info = Some((Rc::clone(n), u, mhash, shash));
                                        break;
                                    },
                                    MaybeMate::Nomate => {
                                        return Ok(MaybeMate::Nomate);
                                    },
                                    MaybeMate::Mate => {

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
                                        let skip_depth = n.try_borrow()?.skip_depth.clone().map(|d| d.max(depth));

                                        n.try_borrow_mut()?.skip_depth = skip_depth;
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

                if let Some((n, u,mh,sh)) = update_info.take() {
                    let sennichite = u.try_borrow()?.sennichite;

                    if !sennichite {
                        node_map.insert(teban.opposite(), mh, sh, Rc::clone(&u));
                    }

                    children.try_borrow_mut()?.remove(&n);
                    children.try_borrow_mut()?.insert(Rc::clone(&u));

                    let n = current_node.as_ref().map(|n| Rc::clone(n))
                                                         .ok_or(ApplicationError::LogicError(String::from(
                                                            "current node is not set."
                                                         )))?;
                    let n = self.normalize_node(&n,mhash,shash,teban,node_map)?;
                    let pn = n.try_borrow()?.pn;
                    let dn = n.try_borrow()?.dn;
                    let u = self.update_node(depth, &n)?;

                    println!("info string update response pn {:?}, dn {:?}",u.pn,u.dn);
                    println!("info string is sennichite {}",u.sennichite);

                    if u.pn == Number::INFINITE && u.dn.is_zero() {
                        println!("info string return nomate.");
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u)),mhash,shash));
                    } else if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(Rc::new(RefCell::new(u)),mhash,shash));
                    } else {
                        n.try_borrow_mut()?.children = u.children;
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