use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub, SubAssign, Div, DivAssign, Mul, MulAssign};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool};
use std::time::Instant;
use usiagent::error::InfoSendError;

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
        if self.d == 1 && rhs.d == 1 {
            Fraction {
                n: self.n + rhs.n,
                d: 1
            }
        } else if self.d == rhs.d {
            let d = self.d;
            let n = self.n + rhs.n;

            let g = gcd(n,d);

            if g == 1 {
                Fraction {
                    n: n,
                    d: d
                }
            } else {
                Fraction {
                    n: n / g,
                    d: d / g
                }
            }
        } else {
            let ad = self.d;
            let bd = rhs.d;
            let an = self.n * bd;
            let bn = rhs.n * ad;
            let d = ad * bd;
            let n = an + bn;

            let g = gcd(n, d);

            if g == 1 {
                Fraction {
                    n: n,
                    d: d
                }
            } else {
                Fraction {
                    n: n / g,
                    d: d / g
                }
            }
        }
    }
}
impl AddAssign for Fraction {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Fraction {
    type Output = Fraction;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.d == 1 && rhs.d == 1 {
            Fraction {
                n: self.n - rhs.n,
                d: 1
            }
        } else if self.d == rhs.d {
            let d = self.d;
            let n = self.n - rhs.n;

            let g = gcd(n,d);

            if g == 1 {
                Fraction {
                    n: n,
                    d: d
                }
            } else {
                Fraction {
                    n: n / g,
                    d: d / g
                }
            }
        } else {
            let ad = self.d;
            let bd = rhs.d;
            let an = self.n * bd;
            let bn = rhs.n * ad;
            let d = ad * bd;
            let n = an - bn;

            let g = gcd(n, d);

            if g == 1 {
                Fraction {
                    n: n,
                    d: d
                }
            } else {
                Fraction {
                    n: n / g,
                    d: d / g
                }
            }
        }
    }
}
impl SubAssign for Fraction {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Div<u64> for Fraction {
    type Output = Fraction;
    fn div(self, rhs: u64) -> Self::Output {
        let n = self.n;
        let d = self.d * rhs;

        let g = gcd(n,d);

        if g == 1 {
            Fraction {
                n: n,
                d: d
            }
        } else {
            Fraction {
                n: n / g,
                d: d / g
            }
        }
    }
}
impl DivAssign<u64> for Fraction {
    fn div_assign(&mut self, rhs: u64) {
        *self = *self / rhs;
    }
}
impl Mul<u64> for Fraction {
    type Output = Fraction;
    fn mul(self, rhs: u64) -> Self::Output {
        let n = self.n * rhs;
        let d = self.d;

        let g = gcd(n,d);

        if g == 1 {
            Fraction {
                n: n,
                d: d
            }
        } else {
            Fraction {
                n: n / g,
                d: d / g
            }
        }
    }
}
impl MulAssign<u64> for Fraction {
    fn mul_assign(&mut self, rhs: u64) {
        *self = *self * rhs;
    }
}
impl Ord for Fraction {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.d == other.d {
            self.n.cmp(&other.n)
        } else {
            let ad = self.d;
            let bd = other.d;
            let an = self.n;
            let bn = other.n;

            let an = an * bd;
            let bn = bn * ad;

            an.cmp(&bn)
        }
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

    pub fn checkmate<L,S>(&self,strict_moves:bool,
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
                     on_complete:Option<Box<dyn FnOnce(u64) -> Result<(),InfoSendError>>>,
                     ms: GameStateForMate) -> Result<MaybeMate,ApplicationError>
        where L: Logger + Send + 'static, S: InfoSender {
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

        let r = strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              &mut KyokumenMap::new(),
                              &mut KyokumenMap::new(),
                              &mut uniq_id,
                              None,
                              &mut KyokumenMap::new(),
                              &mut None,
                              &ms.event_queue,
                              &mut event_dispatcher,
                              ms.teban,
                              &ms.state,
                              &ms.mc);


        if let Some(on_complete) = on_complete {
            on_complete(strategy.node_count() as u64)?;
        }

        r
    }
}

pub mod checkmate {
    use std::cell::{RefCell};
    use std::cmp::Ordering;
    use std::collections::{BinaryHeap, VecDeque};
    use std::ops::{Add, AddAssign, Sub, SubAssign, Deref, Div, Mul};
    use std::rc::{Rc};
    use std::sync::atomic::{AtomicBool};
    use std::sync::{Arc, atomic, Mutex};
    use std::time::{Duration, Instant};
    use usiagent::command::UsiInfoSubCommand;
    use usiagent::error::InfoSendError;
    use usiagent::event::{EventDispatcher, EventQueue, UserEvent, UserEventKind, USIEventDispatcher};
    use usiagent::hash::{KyokumenHash, KyokumenMap};
    use usiagent::logger::Logger;
    use usiagent::player::InfoSender;
    use usiagent::rule::{LegalMove, Rule, State};
    use usiagent::shogi::{MochigomaCollections, MochigomaKind, Teban};
    use crate::error::{ApplicationError};
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

    impl Sub for Number {
        type Output = Number;

        fn sub(self, rhs: Self) -> Self::Output {
            match (self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => Number::INFINITE,
                (Number::Value(l),Number::Value(r)) => Number::Value(l-r)
            }
        }
    }
  
    impl SubAssign for Number {
        fn sub_assign(&mut self, rhs: Self) {
            let v = match (&self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => {
                    Number::INFINITE
                },
                (Number::Value(l),Number::Value(r)) => {
                    Number::Value(*l - r)
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

    impl Mul<u64> for Number {
        type Output = Number;
        fn mul(self, rhs: u64) -> Self::Output {
            match self {
                Number::INFINITE => Number::INFINITE,
                Number::Value(f) => Number::Value(f * rhs)
            }
        }
    }

    pub struct Node {
        id:u64,
        pn_base:Number,
        dn_base:Number,
        pn:Number,
        dn:Number,
        priority:usize,
        mate_depth:u32,
        ref_count:u64,
        sennichite:bool,
        expanded:bool,
        decided:bool,
        m:LegalMove,
        children:Rc<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,
        comparator:Comparator
    }

    impl Node {
        pub fn new_or_node(id:u64,m:LegalMove) -> Node {
            let priority = match m {
                LegalMove::Put(m) => {
                    MochigomaKind::Hisha as usize - m.kind() as usize
                },
                _ => {
                    0
                }
            };

            Node {
                id: id,
                pn_base: Number::Value(Fraction::new(1)),
                dn_base: Number::Value(Fraction::new(1)),
                pn: Number::Value(Fraction::new(1)),
                dn: Number::Value(Fraction::new(1)),
                priority: priority,
                mate_depth: 0,
                ref_count:1,
                sennichite: false,
                expanded: false,
                decided:false,
                m:m,
                children:Rc::new(RefCell::new(BinaryHeap::new())),
                comparator:Comparator::AndNodeComparator
            }
        }

        pub fn new_and_node(id:u64,m:LegalMove) -> Node {
            let priority = match m {
                LegalMove::Put(_) => 0,
                LegalMove::To(m) => {
                    let nari = m.is_nari();

                    if let Some(o) = m.obtained() {
                        if let Ok(k) = MochigomaKind::try_from(o) {
                            k as usize * 2 + if nari {
                                1
                            } else {
                                0
                            } + 2
                        } else {
                            0
                        }
                    } else if nari {
                        1
                    } else {
                        0
                    }
                }
            };

            Node {
                id: id,
                pn_base: Number::Value(Fraction::new(1)),
                dn_base: Number::Value(Fraction::new(1)),
                pn: Number::Value(Fraction::new(1)),
                dn: Number::Value(Fraction::new(1)),
                priority: priority,
                mate_depth: 0,
                ref_count:1,
                sennichite: false,
                expanded: false,
                decided:false,
                m:m,
                children:Rc::new(RefCell::new(BinaryHeap::new())),
                comparator:Comparator::OrNodeComparator
            }
        }

        pub fn to_decided_node(&self,id:u64) -> Node {
            match self.comparator {
                Comparator::OrNodeComparator | Comparator::DecidedOrNodeComparator => {
                    Node {
                        id: id,
                        pn_base: self.pn_base,
                        dn_base: self.dn_base,
                        pn: self.pn,
                        dn: self.dn,
                        priority: self.priority,
                        mate_depth: self.mate_depth,
                        ref_count: self.ref_count,
                        sennichite: self.sennichite,
                        expanded: self.expanded,
                        decided: true,
                        m: self.m,
                        children: Rc::clone(&self.children),
                        comparator: Comparator::DecidedOrNodeComparator
                    }
                },
                Comparator::AndNodeComparator | Comparator::DecidedAndNodeComparator => {
                    Node {
                        id: id,
                        pn_base: self.pn_base,
                        dn_base: self.dn_base,
                        pn: self.pn,
                        dn: self.dn,
                        priority: self.priority,
                        mate_depth: self.mate_depth,
                        ref_count: self.ref_count,
                        sennichite: self.sennichite,
                        expanded: self.expanded,
                        decided: true,
                        m: self.m,
                        children: Rc::clone(&self.children),
                        comparator: Comparator::DecidedAndNodeComparator
                    }
                }
            }
        }

        pub fn update(&mut self,u:&Rc<RefCell<Node>>) -> Result<(),ApplicationError> {
            let (pn,dn) = if let Some(mut p) = self.children.try_borrow_mut()?.peek_mut() {
                let pn = p.try_borrow()?.pn;
                let dn = p.try_borrow()?.dn;

                *p = Rc::clone(u);
                (pn,dn)
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "Node to be updated could not be found."
                )));
            };

            match self.comparator {
                Comparator::AndNodeComparator | Comparator::DecidedAndNodeComparator => {
                    let mut pn = Number::INFINITE;

                    for n in self.children.try_borrow()?.iter() {
                        pn = pn.min(n.try_borrow()?.pn);
                    }
                    self.pn = pn;
                    self.dn_base = self.dn_base - dn + u.try_borrow()?.dn;
                    self.dn = self.dn_base / self.ref_count;
                },
                Comparator::OrNodeComparator | Comparator::DecidedOrNodeComparator => {
                    let mut dn = Number::INFINITE;

                    for n in self.children.try_borrow()?.iter() {
                        dn = dn.min(n.try_borrow()?.dn);
                    }
                    self.pn_base = self.pn_base - pn + u.try_borrow()?.pn;
                    self.pn = self.pn_base / self.ref_count;
                    self.dn = dn;
                }
            }

            Ok(())
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

    impl Clone for Node {
        fn clone(&self) -> Self {
            Node {
                id: self.id,
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                priority: self.priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                sennichite: self.sennichite,
                expanded: self.expanded,
                decided: self.decided,
                m: self.m,
                children: Rc::clone(&self.children),
                comparator: self.comparator.clone()
            }
        }
    }

    pub struct MapNode {
        pn_base:Number,
        dn_base:Number,
        pn:Number,
        dn:Number,
        priority:usize,
        mate_depth:u32,
        ref_count:u64,
        expanded:bool,
        decided:bool,
        children:Rc<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,
    }

    impl MapNode {
        pub fn reflect_to(&self,n:&Node) -> Node {
            Node {
                id: n.id,
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                priority: self.priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                sennichite: n.sennichite,
                expanded: self.expanded,
                decided: self.decided,
                m: n.m,
                children: Rc::clone(&self.children),
                comparator: n.comparator
            }
        }
    }

    impl Clone for MapNode {
        fn clone(&self) -> Self {
            MapNode {
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                priority: self.priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                expanded: self.expanded,
                decided: self.decided,
                children: Rc::clone(&self.children)
            }
        }
    }

    impl<'a> From<&'a Node> for MapNode {
        fn from(n: &'a Node) -> Self {
            MapNode {
                pn_base: n.pn_base,
                dn_base: n.dn_base,
                pn: n.pn,
                dn: n.dn,
                priority: n.priority,
                mate_depth: n.mate_depth,
                ref_count: n.ref_count,
                expanded: n.expanded,
                decided: n.decided,
                children: Rc::clone(&n.children)
            }
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
    #[derive(Clone,Copy)]
    pub enum Comparator {
        OrNodeComparator,
        AndNodeComparator,
        DecidedOrNodeComparator,
        DecidedAndNodeComparator
    }

    impl Comparator {
        pub fn cmp(&self,l:&Node,r:&Node) -> Ordering {
            match self {
                &Comparator::OrNodeComparator => {
                    if r.decided && l.pn == Number::INFINITE {
                        Ordering::Greater.reverse()
                    } else if r.decided {
                        Ordering::Less.reverse()
                    } else {
                        l.pn.cmp(&r.pn)
                            .then(l.mate_depth.cmp(&r.mate_depth))
                            .then(r.priority.cmp(&l.priority))
                            .then(l.id.cmp(&r.id)).reverse()
                    }
                },
                &Comparator::AndNodeComparator => {
                    if r.decided && l.pn.is_zero() && l.dn == Number::INFINITE {
                        Ordering::Less.reverse()
                    } else if r.decided && l.dn == Number::INFINITE {
                        Ordering::Greater.reverse()
                    } else {
                        l.dn.cmp(&r.dn)
                            .then(r.mate_depth.cmp(&l.mate_depth))
                            .then(r.priority.cmp(&l.priority))
                            .then(l.id.cmp(&r.id)).reverse()
                    }
                },
                &Comparator::DecidedOrNodeComparator => {
                    if r.decided {
                        l.pn.cmp(&r.pn)
                            .then(r.dn.cmp(&l.dn))
                            .then(l.mate_depth.cmp(&r.mate_depth))
                            .then(r.priority.cmp(&l.priority))
                            .then(l.id.cmp(&r.id)).reverse()
                    } else if r.pn == Number::INFINITE {
                        Ordering::Less.reverse()
                    } else {
                        Ordering::Greater.reverse()
                    }
                },
                &Comparator::DecidedAndNodeComparator => {
                    if r.decided {
                        r.dn.cmp(&l.dn)
                            .then(l.pn.cmp(&r.pn))
                            .then(r.mate_depth.cmp(&l.mate_depth))
                            .then(r.priority.cmp(&l.priority))
                            .then(l.id.cmp(&r.id)).reverse()
                    } else if r.pn.is_zero() && r.dn == Number::INFINITE {
                        Ordering::Greater.reverse()
                    } else if r.dn == Number::INFINITE {
                        Ordering::Less.reverse()
                    } else {
                        Ordering::Greater.reverse()
                    }
                }
            }
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

        pub fn node_count(&self) -> i64 {
            self.node_count
        }

        pub fn normalize_node(&mut self,
                             n:&Rc<RefCell<Node>>,
                             mhash:u64,
                             shash:u64,
                             teban:Teban,
                             node_map:&mut KyokumenMap<u64,MapNode>)
            -> Result<Rc<RefCell<Node>>,ApplicationError> {
            if n.try_borrow()?.sennichite {
                return Ok(Rc::clone(n))
            }

            if let Some(c) = node_map.get_mut(teban,&mhash,&shash) {
                let expanded = n.try_borrow()?.expanded;

                if !expanded {
                    c.ref_count += 1;
                }
                Ok(Rc::new(RefCell::new(c.reflect_to(n.try_borrow()?.deref()))))
            } else {
                Ok(Rc::clone(n))
            }
        }

        pub fn expand_nodes(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       uniq_id:&mut UniqID,
                                       mut n:Node,
                                       node_map:&mut KyokumenMap<u64,MapNode>,
                                       teban:Teban,
                                       state:&State,
                                       mc:&MochigomaCollections)
                                       -> Result<Rc<RefCell<Node>>,ApplicationError> {
            {
                let mut children = n.children.try_borrow_mut()?;

                if depth % 2 == 0 {
                    let mvs = Rule::oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_and_node(id, m)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.push(Rc::clone(child));
                    }
                } else {
                    let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        Rc::new(RefCell::new(Node::new_or_node(id, m)))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.push(Rc::clone(child));
                    }
                }
            }

            n.expanded = true;

            let len = n.children.try_borrow()?.len();

            let parent_count = n.ref_count;

            if depth % 2 == 0 {
                n.pn = Number::Value(Fraction::new(1));
                n.dn_base = Number::Value(Fraction::new(len as u64));
                n.dn = Number::Value(Fraction::new(len as u64) / parent_count);
            } else {
                n.pn_base = Number::Value(Fraction::new(len as u64));
                n.pn = Number::Value(Fraction::new(len as u64) / parent_count);
                n.dn = Number::Value(Fraction::new(1));
            }

            let n = Rc::new(RefCell::new(n));

            node_map.insert(teban,mhash,shash,n.try_borrow()?.deref().into());

            Ok(n)
        }

        pub fn expand_root_nodes(&mut self,
                                 uniq_id:&mut UniqID,
                                 teban:Teban,
                                 state:&State,
                                 mc:&MochigomaCollections)
            -> Result<Rc<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,ApplicationError> {

            let mvs = Rule::oute_only_moves_all(teban, state, mc);

            let nodes = mvs.into_iter().map(|m| {
                let id = uniq_id.gen();
                Rc::new(RefCell::new(Node::new_and_node(id, m)))
            }).collect::<VecDeque<Rc<RefCell<Node>>>>();

            let children = Rc::new(RefCell::new(BinaryHeap::new()));

            for child in nodes.iter() {
                children.try_borrow_mut()?.push(Rc::clone(child));
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
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,MapNode>,
                                                mate_depth:&mut Option<u32>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {

            ignore_kyokumen_map.insert(teban.opposite(), mhash, shash, ());

            match current_kyokumen_map.get(teban.opposite(), &mhash, &shash).unwrap_or(&0) {
                &c => {
                    current_kyokumen_map.insert(teban.opposite(), mhash, shash, c + 1);
                }
            }

            let r = if depth % 2 == 0 {
                match self.oute_process(depth,
                                        mhash,
                                        shash,
                                        ignore_kyokumen_map,
                                        current_kyokumen_map,
                                        uniq_id,
                                        current_node,
                                        node_map,
                                        mate_depth,
                                        event_queue,
                                        event_dispatcher,
                                        teban,
                                        state,
                                        mc) {
                    r => {
                        ignore_kyokumen_map.remove(teban.opposite(),&mhash,&shash);

                        if let Some(&c) = current_kyokumen_map.get(teban.opposite(),&mhash,&shash) {
                            if c <= 1 {
                                current_kyokumen_map.remove(teban.opposite(), &mhash, &shash);
                            } else {
                                current_kyokumen_map.insert(teban.opposite(),mhash,shash,c-1);
                            }
                        }
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
                                                 current_node,
                                                 node_map,
                                                 mate_depth,
                                                 event_queue,
                                                 event_dispatcher,
                                                 teban,
                                                 state,
                                                 mc) {
                    r => {
                        ignore_kyokumen_map.remove(teban.opposite(),&mhash,&shash);

                        if let Some(&c) = current_kyokumen_map.get(teban.opposite(),&mhash,&shash) {
                            if c <= 1 {
                                current_kyokumen_map.remove(teban.opposite(), &mhash, &shash);
                            } else {
                                current_kyokumen_map.insert(teban.opposite(),mhash,shash,c-1);
                            }
                        }
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
                n = c.try_borrow()?.children.try_borrow()?.peek().map(|n| Rc::clone(n));
            }

            Ok(mvs)
        }

        pub fn oute_process<'a,L: Logger>(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                       current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                       uniq_id:&mut UniqID,
                                       current_node:Option<Rc<RefCell<Node>>>,
                                       node_map:&mut KyokumenMap<u64,MapNode>,
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

            let (current_node,children) = if let Some(n) = current_node.as_ref() {
                let pn = n.try_borrow()?.pn;
                let dn = n.try_borrow()?.dn;

                let n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                if mate_depth.map(|d|  depth >= d).unwrap_or(false) {
                    let u = n.try_borrow()?.to_decided_node(uniq_id.gen());

                    let u = Rc::new(RefCell::new(u));

                    if !u.try_borrow()?.sennichite {
                        node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                if n.try_borrow()?.decided {
                    let n = n.try_borrow()?.to_decided_node(uniq_id.gen());
                    let n = Rc::new(RefCell::new(n));

                    return Ok(MaybeMate::Continuation(n));
                } else if pn != n.try_borrow()?.pn || dn != n.try_borrow()?.dn {
                    return Ok(MaybeMate::Continuation(n));
                }

                let expanded = n.try_borrow()?.expanded;

                if !expanded {
                    let n = Node::clone(n.try_borrow()?.deref());

                    let n = self.expand_nodes(depth, mhash,shash,uniq_id, n,node_map, teban, state, mc)?;

                    self.send_seldepth(depth)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        n.try_borrow_mut()?.pn = Number::INFINITE;
                        n.try_borrow_mut()?.dn = Number::Value(Fraction::new(0));
                    }

                    node_map.insert(teban,mhash,shash,n.try_borrow()?.deref().into());

                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let n = Node::clone(n.try_borrow()?.deref());

                    let n = Rc::new(RefCell::new(n));

                    let children = Rc::clone(&n.try_borrow()?.children);

                    (Some(n),children)
                }
            } else {
                let children = self.expand_root_nodes(uniq_id,teban,state,mc)?;

                self.send_seldepth(depth)?;

                if children.try_borrow()?.len() == 0 {
                    return Ok(MaybeMate::Nomate);
                }

                (None,children)
            };

            if self.max_depth.map(|d| depth > d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            loop {
                let n = children.try_borrow_mut()?.peek().map(|n| {
                    Rc::clone(n)
                }).ok_or(ApplicationError::LogicError(String::from(
                    "None of the child nodes exist."
                )))?;

                if n.try_borrow()?.decided || n.try_borrow()?.pn == Number::INFINITE {
                    if let Some(u) = current_node.as_ref() {
                        let u = u.try_borrow()?.to_decided_node(uniq_id.gen());

                        let u = Rc::new(RefCell::new(u));

                        if !u.try_borrow()?.sennichite {
                            node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                        }

                        return Ok(MaybeMate::Continuation(u));
                    } else {
                        break;
                    }
                }

                let update_node;

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

                {
                    let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                           state.get_banmen(),
                                                           &mc, m.to_applied_move(), &o);
                    let shash = self.hasher.calc_sub_hash(shash, teban,
                                                          state.get_banmen(),
                                                          &mc, m.to_applied_move(), &o);

                    let s = ignore_kyokumen_map.get(teban, &mhash, &shash).is_some();
                    let sc = current_kyokumen_map.get(teban, &mhash, &shash).map(|&c| c >= 3).unwrap_or(false);

                    if s || sc {
                        let mut u = Node::clone(n.try_borrow()?.deref());

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(Fraction::new(0));
                        u.sennichite = true;

                        let u = Rc::new(RefCell::new(u));

                        update_node = u;
                    } else {
                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         ignore_kyokumen_map,
                                                         current_kyokumen_map,
                                                         uniq_id,
                                                         Some(Rc::clone(&n)),
                                                         node_map,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
                                        update_node = u;
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
                                        let u = Node::clone(n.try_borrow()?.deref());

                                        let u = Rc::new(RefCell::new(u));

                                        u.try_borrow_mut()?.pn = Number::INFINITE;

                                        update_node = u;
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

                event_dispatcher.dispatch_events(self, event_queue)?;

                if self.stop.load(atomic::Ordering::Acquire) {
                    return Ok(MaybeMate::Aborted)
                }

                let u = update_node;

                let md = u.try_borrow()?.mate_depth;

                if let Some(c) = current_node.as_ref() {
                    let pn = c.try_borrow()?.pn;
                    let dn = c.try_borrow()?.dn;

                    c.try_borrow_mut()?.update(&u)?;

                    let u = c;

                    let update_mate_depth = u.try_borrow()?.pn.is_zero() && u.try_borrow()?.dn == Number::INFINITE &&
                        (u.try_borrow()?.mate_depth == 0 || md + 1 < u.try_borrow()?.mate_depth);

                    if update_mate_depth {
                        u.try_borrow_mut()?.mate_depth = md + 1;
                    }

                    if !u.try_borrow()?.sennichite {
                        node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                    }

                    if u.try_borrow()?.pn != pn || u.try_borrow()?.dn != dn {
                        return Ok(MaybeMate::Continuation(Rc::clone(u)));
                    }
                } else {
                    if let Some(mut p) = children.try_borrow_mut()?.peek_mut() {
                        *p = Rc::clone(&u);
                    } else {
                        return Err(ApplicationError::LogicError(String::from(
                            "Node to be updated could not be found."
                        )));
                    }

                    if !self.strict_moves && u.try_borrow()?.pn.is_zero() && u.try_borrow()?.dn == Number::INFINITE {
                        *mate_depth = Some(md + 1);
                        return Ok(MaybeMate::MateMoves(self.build_moves(&u)?));
                    } else if u.try_borrow()?.pn.is_zero() && u.try_borrow()?.dn == Number::INFINITE && mate_depth.map(|d| {
                        md + 1 < d
                    }).unwrap_or(true) {
                        *mate_depth = Some(md + 1);
                    }
                }
            }

            if depth == 0 {
                if let Some(n) = children
                    .try_borrow()?
                    .peek() {

                    if n.try_borrow()?.pn.is_zero() && n.try_borrow()?.dn == Number::INFINITE {
                        return Ok(MaybeMate::MateMoves(self.build_moves(n)?));
                    }
                } else {
                    return Err(ApplicationError::LogicError(String::from(
                        "The legal move has not yet been set."
                    )));
                }

                let mut pn = Number::INFINITE;
                let mut dn = Number::Value(Fraction::new(0));

                for n in children.try_borrow()?.iter() {
                    pn = pn.min(n.try_borrow()?.pn);
                    dn += n.try_borrow()?.dn;
                }

                if pn == Number::INFINITE && dn.is_zero() {
                    Ok(MaybeMate::Nomate)
                } else {
                    Ok(MaybeMate::Unknown)
                }
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
                                                current_node:Option<Rc<RefCell<Node>>>,
                                                node_map:&mut KyokumenMap<u64,MapNode>,
                                                mate_depth:&mut Option<u32>,
                                                event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
                                                event_dispatcher:&mut USIEventDispatcher<UserEventKind, UserEvent,Self,L,ApplicationError>,
                                                teban:Teban, state:&State, mc:&MochigomaCollections)
                                                -> Result<MaybeMate,ApplicationError> where S: InfoSender + Send {
            if self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Aborted)
            }

            if self.max_nodes.map(|n| self.node_count >= n).unwrap_or(false) {
                return Ok(MaybeMate::MaxNodes);
            }

            self.node_count += 1;

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            let current_node = if let Some(n) = current_node.as_ref() {
                let pn = n.try_borrow()?.pn;
                let dn = n.try_borrow()?.dn;

                let n = self.normalize_node(&n,mhash,shash,teban,node_map)?;

                let max_mate_depth = n.try_borrow()?.mate_depth + depth;

                if mate_depth.map(|d|  max_mate_depth >= d).unwrap_or(false) {
                    let u = n.try_borrow()?.to_decided_node(uniq_id.gen());

                    let u = Rc::new(RefCell::new(u));

                    if !u.try_borrow()?.sennichite {
                        node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                if n.try_borrow()?.decided {
                    let n = n.try_borrow()?.to_decided_node(uniq_id.gen());
                    let n = Rc::new(RefCell::new(n));

                    return Ok(MaybeMate::Continuation(n));
                } else if pn != n.try_borrow()?.pn || dn != n.try_borrow()?.dn {
                    return Ok(MaybeMate::Continuation(n));
                }

                let expanded = n.try_borrow()?.expanded;

                if !expanded {
                    let n = Node::clone(n.try_borrow()?.deref());

                    let n = self.expand_nodes(depth, mhash, shash, uniq_id, n, node_map, teban, state, mc)?;

                    self.send_seldepth(depth)?;

                    let len = n.try_borrow()?.children.try_borrow()?.len();

                    if len == 0 {
                        let mut u = n.try_borrow()?.to_decided_node(uniq_id.gen());

                        u.pn = Number::Value(Fraction::new(0));
                        u.dn = Number::INFINITE;

                        let u = Rc::new(RefCell::new(u));

                        node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());

                        return Ok(MaybeMate::Continuation(u));
                    } else {
                        node_map.insert(teban, mhash, shash, n.try_borrow()?.deref().into());

                        return Ok(MaybeMate::Continuation(n));
                    }
                } else {
                    let n = Node::clone(n.try_borrow()?.deref());

                    Rc::new(RefCell::new(n))
                }
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "current move is not set."
                )));
            };

            if self.max_depth.map(|d| depth > d).unwrap_or(false) {
                return Ok(MaybeMate::MaxDepth);
            }

            let children = Rc::clone(&current_node.try_borrow()?.children);

            loop {
                let n = children.try_borrow_mut()?.peek().map(|n| {
                    Rc::clone(n)
                }).ok_or(ApplicationError::LogicError(String::from(
                    "None of the child nodes exist."
                )))?;

                if n.try_borrow()?.decided || n.try_borrow()?.dn == Number::INFINITE {
                    let u = Rc::clone(&current_node);
                    let u = u.try_borrow()?.to_decided_node(uniq_id.gen());

                    let u = Rc::new(RefCell::new(u));

                    if !u.try_borrow()?.sennichite {
                        node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                    }
                    
                    return Ok(MaybeMate::Continuation(u));
                }

                let update_node;

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

                {
                    let mhash = self.hasher.calc_main_hash(mhash, teban,
                                                           state.get_banmen(),
                                                           &mc, m.to_applied_move(), &o);
                    let shash = self.hasher.calc_sub_hash(shash, teban,
                                                          state.get_banmen(),
                                                          &mc, m.to_applied_move(), &o);
                    let s = ignore_kyokumen_map.get(teban, &mhash, &shash).is_some();
                    let sc = current_kyokumen_map.get(teban, &mhash, &shash).map(|&c| c >= 3).unwrap_or(false);

                    if sc {
                        let mut u = Node::clone(n.try_borrow()?.deref());

                        u.pn = Number::Value(Fraction::new(0));
                        u.dn = Number::INFINITE;
                        u.sennichite = true;

                        let u = Rc::new(RefCell::new(u));

                        update_node = u;
                    } else if s {
                        let mut u = Node::clone(n.try_borrow()?.deref());

                        u.dn = Number::INFINITE;
                        u.sennichite = true;

                        let u = Rc::new(RefCell::new(u));

                        update_node = u;
                    } else {
                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         ignore_kyokumen_map,
                                                         current_kyokumen_map,
                                                         uniq_id,
                                                         Some(Rc::clone(&n)),
                                                         node_map,
                                                         mate_depth,
                                                         event_queue,
                                                         event_dispatcher,
                                                         teban.opposite(),
                                                         &state,
                                                         &mc
                                )? {
                                    MaybeMate::Continuation(u) => {
                                        update_node = u;
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
                                        let mut u = Node::clone(n.try_borrow()?.deref());

                                        u.dn = Number::INFINITE;

                                        let u = Rc::new(RefCell::new(u));

                                        update_node = u;
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

                event_dispatcher.dispatch_events(self, event_queue)?;

                if self.stop.load(atomic::Ordering::Acquire) {
                    return Ok(MaybeMate::Aborted)
                }

                let u = update_node;

                let c = Rc::clone(&current_node);

                let pn = c.try_borrow()?.pn;
                let dn = c.try_borrow()?.dn;

                c.try_borrow_mut()?.update(&u)?;

                if c.try_borrow()?.pn.is_zero() && c.try_borrow()?.dn == Number::INFINITE {
                    if c.try_borrow()?.mate_depth == 0 || u.try_borrow()?.mate_depth != n.try_borrow()?.mate_depth {
                        let mut mate_depth = 0;

                        for n in c.try_borrow()?.children.try_borrow()?.iter() {
                            mate_depth = mate_depth.max(n.try_borrow()?.mate_depth + 1);
                        }

                        c.try_borrow_mut()?.mate_depth = mate_depth;
                    }
                }

                let u = c;

                if !u.try_borrow()?.sennichite {
                    node_map.insert(teban, mhash, shash, u.try_borrow()?.deref().into());
                }

                if u.try_borrow()?.pn != pn || u.try_borrow()?.dn != dn {
                    return Ok(MaybeMate::Continuation(Rc::clone(&u)));
                }
            }
        }

        fn send_seldepth(&mut self, depth:u32) -> Result<(),InfoSendError>{
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