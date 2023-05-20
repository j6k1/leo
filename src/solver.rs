use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Sub, SubAssign, Div, DivAssign, Mul, MulAssign};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
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
use crate::solver::checkmate::{CheckmateStrategy, NodeRepository, NormalizedNode, UniqID};

#[derive(Clone)]
pub enum MaybeMate {
    Nomate,
    MateMoves(VecDeque<LegalMove>),
    Unknown,
    Continuation(NormalizedNode),
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
#[inline]
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
    #[inline]
    pub fn new(n:u64) -> Fraction {
        Fraction {
            n:n,
            d:1
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.n == 0
    }
}

impl Add for Fraction {
    type Output = Fraction;
    #[inline]
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
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Fraction {
    type Output = Fraction;
    #[inline]
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
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Div<u64> for Fraction {
    type Output = Fraction;
    #[inline]
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
    #[inline]
    fn div_assign(&mut self, rhs: u64) {
        *self = *self / rhs;
    }
}
impl Mul<u64> for Fraction {
    type Output = Fraction;
    #[inline]
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
    #[inline]
    fn mul_assign(&mut self, rhs: u64) {
        *self = *self * rhs;
    }
}
impl Ord for Fraction {
    #[inline]
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
    #[inline]
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
                     hash_size:usize,
                     limit:Option<Instant>,
                     checkmate_limit:Option<Instant>,
                     network_delay:u32,
                     max_depth:Option<u32>,
                     max_nodes:Option<i64>,
                     info_sender:S,
                     on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
                     hasher:Arc<KyokumenHash<u64>>,
                     stop:Arc<AtomicBool>,
                     nodes:Arc<AtomicU64>,
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
                                                  Arc::clone(&nodes),
                                                  ms.base_depth,
                                                  ms.current_depth);
        let mut event_dispatcher = Root::<L,S>::create_event_dispatcher::<CheckmateStrategy<S>>(on_error_handler,&stop,&quited);

        let mut uniq_id = UniqID::new();

        let r = strategy.oute_process(0,
                              ms.mhash,
                              ms.shash,
                              None,
                              &HashSet::new(),
                              &mut KyokumenMap::new(),
                              &mut KyokumenMap::new(),
                              &mut uniq_id,
                              None,
                              &mut NodeRepository::new(hash_size * 1024 * 1024),
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
    use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
    use std::ops::{Add, AddAssign, Sub, SubAssign, Deref, Div, Mul};
    use std::rc::{Rc, Weak};
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::{Arc, atomic, Mutex};
    use std::time::{Duration, Instant};
    use usiagent::command::UsiInfoSubCommand;
    use usiagent::error::InfoSendError;
    use usiagent::event::{EventDispatcher, EventQueue, UserEvent, UserEventKind, USIEventDispatcher};
    use usiagent::hash::{KyokumenHash, KyokumenMap};
    use usiagent::logger::Logger;
    use usiagent::player::InfoSender;
    use usiagent::rule::{LegalMove, Rule, State};
    use usiagent::shogi::{KomaKind, MochigomaCollections, MochigomaKind, ObtainKind, Teban};
    use crate::error::{ApplicationError};
    use crate::initial_estimation::{calc_asc_priority, initial_pn_dn_plus_and_node, initial_pn_dn_plus_or_node};
    use crate::search::{TIMELIMIT_MARGIN};
    use crate::solver::{Fraction, MaybeMate};

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub enum Number {
        Value(Fraction),
        INFINITE
    }

    impl Number {
        #[inline]
        pub fn is_zero(&self) -> bool {
            match self {
                &Number::INFINITE => false,
                &Number::Value(v) => v.is_zero()
            }
        }
    }

    impl Add for Number {
        type Output = Number;

        #[inline]
        fn add(self, rhs: Self) -> Self::Output {
            match (self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => Number::INFINITE,
                (Number::Value(l),Number::Value(r)) => Number::Value(l+r)
            }
        }
    }
  
    impl AddAssign for Number {
        #[inline]
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

        #[inline]
        fn sub(self, rhs: Self) -> Self::Output {
            match (self,rhs) {
                (Number::INFINITE,_) | (_,Number::INFINITE) => Number::INFINITE,
                (Number::Value(l),Number::Value(r)) => Number::Value(l-r)
            }
        }
    }
  
    impl SubAssign for Number {
        #[inline]
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
        #[inline]
        fn div(self, rhs: u64) -> Self::Output {
            match self {
                Number::INFINITE => Number::INFINITE,
                Number::Value(f) => Number::Value(f / rhs)
            }
        }
    }

    impl Mul<u64> for Number {
        type Output = Number;
        #[inline]
        fn mul(self, rhs: u64) -> Self::Output {
            match self {
                Number::INFINITE => Number::INFINITE,
                Number::Value(f) => Number::Value(f * rhs)
            }
        }
    }

    pub struct NodeRepository {
        map:KyokumenMap<u64,NodeRepositoryItem>,
        list:Vec<Rc<RefCell<GCEntry>>>,
        max_size:usize,
        current_size:usize,
        referenced_count:u32,
        generation:u64,
    }

    const GENERATION_BOUND:u32 = 1024u32;
    const GC_PERCENTAGE:usize = 80usize;

    impl NodeRepository {
        pub fn new(max_size:usize) -> NodeRepository {
            NodeRepository {
                map:KyokumenMap::new(),
                list:Vec::new(),
                max_size:max_size,
                current_size:0,
                referenced_count:0,
                generation:0

            }
        }

        #[inline]
        pub fn insert_gc_entry(&mut self,e:Rc<RefCell<GCEntry>>) {
            self.list.push(e);
        }

        #[inline]
        pub fn contains(&self,teban:Teban,mhash:u64,shash:u64) -> bool {
            self.map.get(teban,&mhash,&shash).is_some()
        }

        pub fn get_or_add(&mut self, teban:Teban, mhash:u64, shash:u64, depth:u32, path:&HashSet<(Teban,u64,u64)>, n:&Node) -> Result<NormalizedNode,ApplicationError> {
            if let Some(&mut NodeRepositoryItem {
                    ref node, ref mut gc_entry
                }) = self.map.get_mut(teban,&mhash,&shash) {

                gc_entry.try_borrow_mut()?.frequency += 1;

                self.referenced_count += 1;

                if self.referenced_count == GENERATION_BOUND {
                    self.generation += 1;
                    self.referenced_count = 0;
                }

                gc_entry.try_borrow_mut()?.generation = self.generation;

                Ok(node.reflect_to(n,depth,path)?.into())
            } else {
                let node = NormalizedNode::from(n);

                self.add(teban,mhash,shash,&node)?;

                Ok(node)
            }
        }

        pub fn add(&mut self,teban:Teban,mhash:u64,shash:u64,n:&NormalizedNode) -> Result<(),ApplicationError> {
            let gc_entry = GCEntry {
                teban:teban,
                mhash:mhash,
                shash:shash,
                frequency:1,
                generation:self.generation,
                mate:n.pn.is_zero() && n.dn == Number::INFINITE
            };

            let gc_entry = Rc::new(RefCell::new(gc_entry));

            let item = NodeRepositoryItem {
                node: n.into(),
                gc_entry:Rc::clone(&gc_entry)
            };

            if self.current_size + std::mem::size_of::<NodeRepositoryItem>() > self.max_size {
                self.gc()?;
            }

            self.insert_gc_entry(gc_entry);

            self.map.insert(teban,mhash,shash,item);

            self.current_size += std::mem::size_of::<NodeRepositoryItem>();

            Ok(())
        }

        pub fn update(&mut self,teban:Teban,mhash:u64,shash:u64,n:&NormalizedNode) -> Result<(),ApplicationError> {
            if let Some(&mut NodeRepositoryItem { ref mut node, ref mut gc_entry }) = self.map.get_mut(teban,&mhash,&shash) {
                self.referenced_count += 1;

                if self.referenced_count == GENERATION_BOUND {
                    self.generation += 1;
                    self.referenced_count = 0;
                }

                gc_entry.try_borrow_mut()?.mate = node.pn.is_zero() && node.dn == Number::INFINITE;
                gc_entry.try_borrow_mut()?.generation = self.generation;

                *node = n.into();

                Ok(())
            } else {
                self.add(teban,mhash,shash,n)?;

                Ok(())
            }
        }

        pub fn gc(&mut self) -> Result<(),ApplicationError> {
            let size = std::mem::size_of::<NodeRepositoryItem>();
            let rs = self.max_size * GC_PERCENTAGE / 100;

            self.list.sort();

            while self.current_size > rs {
                if let Some(gc_entry) = self.list.last() {
                    if gc_entry.try_borrow()?.mate {
                        break;
                    }
                }

                if let Some(gc_entry) = self.list.pop() {
                    self.map.remove(gc_entry.try_borrow()?.teban,&gc_entry.try_borrow()?.mhash,&gc_entry.try_borrow()?.shash);
                    self.current_size -= size;
                } else {
                    break;
                }
            }

            Ok(())
        }
    }

    pub struct NodeRepositoryItem {
        node:MapNode,
        gc_entry:Rc<RefCell<GCEntry>>
    }

    impl Clone for NodeRepositoryItem {
        #[inline]
        fn clone(&self) -> Self {
            NodeRepositoryItem {
                node: self.node.clone(),
                gc_entry: self.gc_entry.clone()
            }
        }
    }

    #[derive(Clone,PartialEq,Eq)]
    pub struct GCEntry {
        teban:Teban,
        mhash:u64,
        shash:u64,
        generation:u64,
        frequency:u64,
        mate:bool
    }

    impl Ord for GCEntry {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            if !self.mate && other.mate {
                Ordering::Greater
            } else if self.mate && !other.mate {
                Ordering::Less
            } else {
                other.generation.cmp(&self.generation).then(other.frequency.cmp(&self.frequency))
            }
        }
    }
    
    impl PartialOrd for GCEntry {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(&other))
        }
    }

    #[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
    pub enum NodeState {
        UnDecided,
        Decided,
        Unknown(u32)
    }

    impl NodeState {
        pub fn is_unknown(&self,depth:u32) -> bool {
            if let &NodeState::Unknown(d) = self {
                d >= depth
            } else {
                false
            }
        }
    }
    pub struct Node {
        id:u64,
        pn_base:Number,
        dn_base:Number,
        pn:Number,
        dn:Number,
        asc_priority:i32,
        mate_depth:u32,
        ref_count:u64,
        sennichite:bool,
        expanded:bool,
        state:NodeState,
        m:LegalMove,
        children:Weak<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,
        parent_refs:Rc<HashMap<(Teban,u64,u64),Rc<ParentRef>>>,
        mate_node:Option<Rc<RefCell<Node>>>,
        comparator:Comparator,
        generation:u32
    }

    impl Node {
        #[inline]
        pub fn new_and_node(id:u64, teban:Teban, state: &State, m:LegalMove, parent:Option<Rc<ParentRef>>) -> Node {
            let (pn,dn) = initial_pn_dn_plus_and_node(teban,state,m);

            let mut parent_refs = HashMap::new();

            parent.as_ref().map(|p| parent_refs.insert((p.teban,p.mhash,p.shash),Rc::clone(p)));

            let asc_priority = match m {
                LegalMove::Put(m) => {
                    m.kind() as i32 + KomaKind::SHishaN as i32 + ObtainKind::HishaN as i32 + 4
                },
                LegalMove::To(m) if m.obtained() == Some(ObtainKind::Ou) => {
                    0
                },
                LegalMove::To(m)=> {
                    let src = m.src();
                    let x = src / 9;
                    let y = src - x * 9;
                    let kind = state.get_banmen().0[y as usize][x as usize];

                    match m.obtained() {
                        Some(o) => {
                            o as i32 - ObtainKind::HishaN as i32 + 1
                        },
                        None if kind == KomaKind::SOu || kind == KomaKind::GOu => {
                            ObtainKind::HishaN as i32 + 2
                        },
                        None => {
                            ObtainKind::HishaN as i32 + 3
                        }
                    }
                }
            };

            Node {
                id: id,
                pn_base: Number::Value(pn),
                dn_base: Number::Value(dn),
                pn: Number::Value(pn),
                dn: Number::Value(dn),
                asc_priority: asc_priority,
                mate_depth: 0,
                ref_count:1,
                sennichite: false,
                expanded: false,
                state:NodeState::UnDecided,
                m:m,
                children:Weak::new(),
                parent_refs:Rc::new(parent_refs),
                mate_node:None,
                comparator:Comparator::AndNodeComparator,
                generation:0
            }
        }

        #[inline]
        pub fn new_or_node(id:u64, teban:Teban,state:&State,m:LegalMove, parent:Option<Rc<ParentRef>>) -> Node {
            let (pn,dn) = initial_pn_dn_plus_or_node(teban,state,m);

            let mut parent_refs = HashMap::new();

            parent.as_ref().map(|p| parent_refs.insert((p.teban,p.mhash,p.shash),Rc::clone(p)));

            Node {
                id: id,
                pn_base: Number::Value(pn),
                dn_base: Number::Value(dn),
                pn: Number::Value(pn),
                dn: Number::Value(dn),
                asc_priority: calc_asc_priority(teban,state,m),
                mate_depth: 0,
                ref_count:1,
                sennichite: false,
                expanded: false,
                state: NodeState::UnDecided,
                m:m,
                children:Weak::new(),
                parent_refs:Rc::new(parent_refs),
                mate_node:None,
                comparator:Comparator::OrNodeComparator,
                generation:0
            }
        }

        pub fn merge(&self,other:&MapNode,path:&HashSet<(Teban,u64,u64)>) -> HashMap<(Teban,u64,u64),Rc<ParentRef>> {
            let mut parent_refs = self.parent_refs.deref().clone();

            for (&(teban,mhash,shash),parent) in other.parent_refs.iter() {
                if self.parent_refs.contains_key(&(teban,mhash,shash)) || parent.contains(path) {
                    parent_refs.insert((teban,mhash,shash),Rc::clone(parent));
                }
            }

            parent_refs
        }
    }

    impl PartialEq for Node {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for Node {}

    impl Ord for Node {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.comparator.cmp(self,other)
        }
    }

    impl PartialOrd for Node {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Clone for Node {
        #[inline]
        fn clone(&self) -> Self {
            Node {
                id: self.id,
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                asc_priority: self.asc_priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                sennichite: self.sennichite,
                expanded: self.expanded,
                state: self.state,
                m: self.m,
                children: self.children.clone(),
                parent_refs: Rc::clone(&self.parent_refs),
                mate_node: self.mate_node.clone(),
                comparator: self.comparator.clone(),
                generation: self.generation
            }
        }
    }

    impl<'a> From<&'a NormalizedNode> for Node {
        #[inline]
        fn from(n: &'a NormalizedNode) -> Node {
            Node {
                id: n.id,
                pn_base: n.pn_base,
                dn_base: n.dn_base,
                pn: n.pn,
                dn: n.dn,
                asc_priority: n.asc_priority,
                mate_depth: n.mate_depth,
                ref_count: n.ref_count,
                sennichite: n.sennichite,
                expanded: n.expanded,
                state: n.state,
                m: n.m,
                children: Rc::downgrade(&n.children),
                parent_refs: Rc::clone(&n.parent_refs),
                mate_node: n.mate_node.clone(),
                comparator: n.comparator.clone(),
                generation: n.generation
            }
        }
    }

    impl From<NormalizedNode> for Node {
        #[inline]
        fn from(n: NormalizedNode) -> Node {
            Node::from(&n)
        }
    }

    pub struct ParentRef {
        parent:Option<Rc<ParentRef>>,
        teban:Teban,
        mhash:u64,
        shash:u64
    }

    impl ParentRef {
        pub fn new(parent:Option<Rc<ParentRef>>, teban:Teban, mhash:u64, shash:u64) -> ParentRef {
            ParentRef {
                parent:parent,
                teban:teban,
                mhash:mhash,
                shash:shash
            }
        }

        pub fn contains(&self, path:&HashSet<(Teban, u64, u64)>) -> bool {
            let mut n = self;

            while let Some(p) = n.parent.as_ref() {
                if path.contains(&(p.teban,p.mhash,p.shash)) {
                    return true;
                }
                n = p;
            }

            false
        }
    }

    pub struct MapNode {
        pn_base:Number,
        dn_base:Number,
        pn:Number,
        dn:Number,
        asc_priority:i32,
        mate_depth:u32,
        ref_count:u64,
        expanded:bool,
        state:NodeState,
        children:Rc<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,
        parent_refs:Rc<HashMap<(Teban,u64,u64),Rc<ParentRef>>>,
        mate_node:Option<Rc<RefCell<Node>>>,
        generation:u32
    }

    impl MapNode {
        #[inline]
        pub fn reflect_to(&self,n:&Node,depth:u32,path:&HashSet<(Teban,u64,u64)>) -> Result<Node,ApplicationError> {
            let parent_refs = n.merge(self, path);
            let ref_count = (parent_refs.len() as u64).max(1);

            let (pn_base, dn_base, pn, dn, state) = if let NodeState::Unknown(d) = self.state {
                if depth < d {
                    if !self.expanded {
                        match n.comparator {
                            Comparator::AndNodeComparator => {
                                (
                                    Number::Value(Fraction::new(1)),
                                    Number::Value(Fraction::new(1)),
                                    Number::Value(Fraction::new(1)),
                                    Number::Value(Fraction::new(1)) / ref_count,
                                    NodeState::UnDecided
                                )
                            },
                            Comparator::OrNodeComparator => {
                                (
                                    Number::Value(Fraction::new(1)),
                                    Number::Value(Fraction::new(1)),
                                    Number::Value(Fraction::new(1)) / ref_count,
                                    Number::Value(Fraction::new(1)),
                                    NodeState::UnDecided
                                )
                            }
                        }
                    } else {
                        match n.comparator {
                            Comparator::AndNodeComparator => {
                                let mut pn = Number::INFINITE;
                                let mut dn = Number::Value(Fraction::new(0));

                                for n in self.children.try_borrow()?.iter() {
                                    pn = pn.min(n.try_borrow()?.pn);
                                    dn += n.try_borrow()?.dn;
                                }

                                (pn, dn, pn, dn / ref_count, NodeState::UnDecided)
                            },
                            Comparator::OrNodeComparator => {
                                let mut dn = Number::INFINITE;
                                let mut pn = Number::Value(Fraction::new(0));

                                for n in self.children.try_borrow()?.iter() {
                                    dn = dn.min(n.try_borrow()?.dn);
                                    pn += n.try_borrow()?.pn;
                                }

                                (pn, dn, pn / ref_count, dn, NodeState::UnDecided)
                            }
                        }
                    }
                } else {
                    (self.pn_base, self.dn_base, self.pn, self.dn, self.state)
                }
            } else {
                (self.pn_base, self.dn_base, self.pn, self.dn, self.state)
            };

            Ok(Node {
                id: n.id,
                pn_base: pn_base,
                dn_base: dn_base,
                pn: pn,
                dn: dn,
                asc_priority: self.asc_priority,
                mate_depth: self.mate_depth,
                ref_count: ref_count,
                sennichite: n.sennichite,
                expanded: self.expanded,
                state: state,
                m: n.m,
                children: Rc::downgrade(&self.children),
                parent_refs: Rc::new(parent_refs),
                mate_node: self.mate_node.clone(),
                comparator: n.comparator,
                generation: self.generation
            })
        }
    }

    impl Clone for MapNode {
        #[inline]
        fn clone(&self) -> Self {
            MapNode {
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                asc_priority: self.asc_priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                expanded: self.expanded,
                state: self.state,
                children: Rc::clone(&self.children),
                parent_refs: Rc::clone(&self.parent_refs),
                mate_node: self.mate_node.clone(),
                generation: self.generation
            }
        }
    }

    impl<'a> From<&'a NormalizedNode> for MapNode {
        fn from(n: &'a NormalizedNode) -> Self {
            MapNode {
                pn_base: n.pn_base,
                dn_base: n.dn_base,
                pn: n.pn,
                dn: n.dn,
                asc_priority: n.asc_priority,
                mate_depth: n.mate_depth,
                ref_count: n.ref_count,
                expanded: n.expanded,
                state: n.state,
                children: Rc::clone(&n.children),
                parent_refs: Rc::clone(&n.parent_refs),
                mate_node: n.mate_node.clone(),
                generation: n.generation
            }
        }
    }
    pub struct NormalizedNode {
        id:u64,
        pn_base:Number,
        dn_base:Number,
        pn:Number,
        dn:Number,
        asc_priority:i32,
        mate_depth:u32,
        ref_count:u64,
        sennichite:bool,
        expanded:bool,
        state:NodeState,
        m:LegalMove,
        children:Rc<RefCell<BinaryHeap<Rc<RefCell<Node>>>>>,
        parent_refs:Rc<HashMap<(Teban,u64,u64),Rc<ParentRef>>>,
        mate_node: Option<Rc<RefCell<Node>>>,
        comparator:Comparator,
        generation:u32
    }

    impl NormalizedNode {
        #[inline]
        pub fn to_decided_node(&self,id:u64) -> NormalizedNode {
            match self.comparator {
                Comparator::OrNodeComparator => {
                    NormalizedNode {
                        id: id,
                        pn_base: self.pn_base,
                        dn_base: self.dn_base,
                        pn: self.pn,
                        dn: self.dn,
                        asc_priority: self.asc_priority,
                        mate_depth: self.mate_depth,
                        ref_count: self.ref_count,
                        sennichite: self.sennichite,
                        expanded: self.expanded,
                        state: NodeState::Decided,
                        m: self.m,
                        children: Rc::clone(&self.children),
                        parent_refs: Rc::clone(&self.parent_refs),
                        mate_node: self.mate_node.clone(),
                        comparator: self.comparator.clone(),
                        generation: self.generation
                    }
                },
                Comparator::AndNodeComparator => {
                    NormalizedNode {
                        id: id,
                        pn_base: self.pn_base,
                        dn_base: self.dn_base,
                        pn: self.pn,
                        dn: self.dn,
                        asc_priority: self.asc_priority,
                        mate_depth: self.mate_depth,
                        ref_count: self.ref_count,
                        sennichite: self.sennichite,
                        expanded: self.expanded,
                        state: NodeState::Decided,
                        m: self.m,
                        children: Rc::clone(&self.children),
                        parent_refs: Rc::clone(&self.parent_refs),
                        mate_node: self.mate_node.clone(),
                        comparator: self.comparator.clone(),
                        generation: self.generation
                    }
                }
            }
        }

        #[inline]
        pub fn to_unknown_node(&self,depth:u32) -> NormalizedNode {
            NormalizedNode {
                id: self.id,
                pn_base: Number::INFINITE,
                dn_base: Number::Value(Fraction::new(0)),
                pn: Number::INFINITE,
                dn: Number::Value(Fraction::new(0)),
                asc_priority: self.asc_priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                sennichite: self.sennichite,
                expanded: self.expanded,
                state: NodeState::Unknown(depth),
                m: self.m,
                children: Rc::clone(&self.children),
                parent_refs: Rc::clone(&self.parent_refs),
                mate_node: self.mate_node.clone(),
                comparator: self.comparator.clone(),
                generation: self.generation
            }
        }

        pub fn update(&mut self,u:&NormalizedNode) -> Result<(),ApplicationError> {
            let u = Rc::new(RefCell::new(u.into()));

            let (pn,dn) = if let Some(mut p) = self.children.try_borrow_mut()?.peek_mut() {
                let pn = p.try_borrow()?.pn;
                let dn = p.try_borrow()?.dn;

                *p = Rc::clone(&u);
                (pn,dn)
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "Node to be updated could not be found."
                )));
            };

            match self.comparator {
                Comparator::AndNodeComparator => {
                    let mut pn = Number::INFINITE;

                    for n in self.children.try_borrow()?.iter() {
                        pn = pn.min(n.try_borrow()?.pn);
                    }

                    self.pn = pn;
                    self.dn_base = self.dn_base - dn + u.try_borrow()?.dn;
                    self.dn = self.dn_base / self.ref_count;
                },
                Comparator::OrNodeComparator => {
                    self.pn_base = self.pn_base - pn + u.try_borrow()?.pn;
                    self.pn = self.pn_base / self.ref_count;
                    let mut dn = Number::INFINITE;

                    for n in self.children.try_borrow()?.iter() {
                        dn = dn.min(n.try_borrow()?.dn);
                    }

                    self.dn = dn;
                }
            }

            Ok(())
        }
    }

    impl Clone for NormalizedNode {
        #[inline]
        fn clone(&self) -> Self {
            NormalizedNode {
                id: self.id,
                pn_base: self.pn_base,
                dn_base: self.dn_base,
                pn: self.pn,
                dn: self.dn,
                asc_priority: self.asc_priority,
                mate_depth: self.mate_depth,
                ref_count: self.ref_count,
                sennichite: self.sennichite,
                expanded: self.expanded,
                state: self.state,
                m: self.m,
                children: Rc::clone(&self.children),
                parent_refs: Rc::clone(&self.parent_refs),
                mate_node: self.mate_node.clone(),
                comparator: self.comparator.clone(),
                generation: self.generation
            }
        }
    }

    impl<'a> From<&'a Node> for NormalizedNode {
        #[inline]
        fn from(n: &'a Node) -> Self {
            NormalizedNode {
                id: n.id,
                pn_base: n.pn_base,
                dn_base: n.dn_base,
                pn: n.pn,
                dn: n.dn,
                asc_priority: n.asc_priority,
                mate_depth: n.mate_depth,
                ref_count: n.ref_count,
                sennichite: n.sennichite,
                expanded: n.expanded,
                state: n.state,
                m: n.m,
                children: n.children.upgrade().unwrap_or(Rc::new(RefCell::new(BinaryHeap::new()))),
                parent_refs: Rc::clone(&n.parent_refs),
                mate_node: n.mate_node.clone(),
                comparator: n.comparator,
                generation: n.generation
            }
        }
    }

    impl From<Node> for NormalizedNode {
        #[inline]
        fn from(n: Node) -> Self {
            NormalizedNode::from(&n)
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

        #[inline]
        pub fn gen(&mut self) -> u64 {
            let id = self.last_id;

            self.last_id += 1;

            id
        }
    }
    #[derive(Clone,Copy)]
    pub enum Comparator {
        OrNodeComparator,
        AndNodeComparator
    }

    impl Comparator {
        #[inline]
        pub fn cmp(&self,l:&Node,r:&Node) -> Ordering {
            match self {
                &Comparator::OrNodeComparator => {
                    l.state.cmp(&r.state)
                        .then_with(|| l.pn.cmp(&r.pn))
                        .then_with(|| r.mate_depth.cmp(&l.mate_depth))
                        .then_with(|| l.asc_priority.cmp(&r.asc_priority))
                        .then_with(|| l.id.cmp(&r.id)).reverse()
                },
                &Comparator::AndNodeComparator => {
                    l.state.cmp(&r.state)
                        .then_with(|| l.dn.cmp(&r.dn))
                        .then_with(|| r.mate_depth.cmp(&l.mate_depth))
                        .then_with(|| l.asc_priority.cmp(&r.asc_priority))
                        .then_with(|| l.id.cmp(&r.id)).reverse()
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
        nodes:Arc<AtomicU64>,
        base_depth:u32,
        current_depth:u32,
        node_count:i64,
        attempt_check_timelimit:u32
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
               nodes:Arc<AtomicU64>,
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
                nodes:nodes,
                base_depth:base_depth,
                current_depth:current_depth,
                node_count:0,
                attempt_check_timelimit:0
            }
        }

        pub fn node_count(&self) -> i64 {
            self.node_count
        }

        pub fn normalize_node(&mut self,
                             n:&Node,
                             mhash:u64,
                             shash:u64,
                             teban:Teban,
                             depth:u32,
                             path:&HashSet<(Teban,u64,u64)>,
                             node_repo:&mut NodeRepository)
            -> Result<NormalizedNode,ApplicationError> {
            if n.sennichite {
                return Ok(n.into())
            }

            if n.children.strong_count() > 0 || node_repo.contains(teban,mhash,shash) {
                let n = node_repo.get_or_add(teban, mhash, shash, depth,path,n)?;

                Ok(n)
            } else {
                let mut n = node_repo.get_or_add(teban, mhash, shash, depth,path,n)?;

                n.expanded = false;

                Ok(n)
            }
        }

        pub fn expand_nodes(&mut self,
                            depth:u32,
                            mhash:u64,
                            shash:u64,
                            parent_ref:Option<&Rc<ParentRef>>,
                            uniq_id:&mut UniqID,
                            mut n:NormalizedNode,
                            node_repo:&mut NodeRepository,
                            teban:Teban,
                            state:&State,
                            mc:&MochigomaCollections)
                            -> Result<NormalizedNode,ApplicationError> {
            let (pn,dn) = {
                let mut children = n.children.try_borrow_mut()?;

                if depth % 2 == 0 {
                    let mut pn = Number::INFINITE;
                    let mut dn = Number::Value(Fraction::new(0));

                    let mvs = Rule::oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        let n = Node::new_or_node(id, teban, state, m, parent_ref.map(|p| Rc::clone(p)));

                        pn = pn.min(n.pn);
                        dn += n.dn;

                        Rc::new(RefCell::new(n))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.push(Rc::clone(child));
                    }

                    (pn,dn)
                } else {
                    let mut pn = Number::Value(Fraction::new(0));
                    let mut dn = Number::INFINITE;

                    let mvs = Rule::respond_oute_only_moves_all(teban, state, mc);

                    let nodes = mvs.into_iter().map(|m| {
                        let id = uniq_id.gen();
                        let n = Node::new_and_node(id, teban, state, m, parent_ref.map(|p| Rc::clone(p)));

                        pn += n.pn;
                        dn = dn.min(n.dn);

                        Rc::new(RefCell::new(n))
                    }).collect::<VecDeque<Rc<RefCell<Node>>>>();

                    for child in nodes.iter() {
                        children.push(Rc::clone(child));
                    }

                    (pn,dn)
                }
            };

            n.expanded = true;

            let parent_count = n.ref_count;

            if depth % 2 == 0 {
                n.pn = pn;
                n.dn_base = dn;
                n.dn = n.dn_base / parent_count;
            } else {
                n.pn_base = pn;
                n.pn = pn / parent_count;
                n.dn = dn;
            }

            node_repo.update(teban,mhash,shash,&n)?;

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
                Rc::new(RefCell::new(Node::new_or_node(id, teban, state, m, None)))
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
                                        parent_ref:Option<&Rc<ParentRef>>,
                                        path:&HashSet<(Teban,u64,u64)>,
                                        ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                        current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                        uniq_id:&mut UniqID,
                                        current_node:Option<Node>,
                                        node_repo:&mut NodeRepository,
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

            let mut path = path.clone();

            path.insert((teban,mhash,shash));

            let r = if depth % 2 == 0 {
                match self.oute_process(depth,
                                        mhash,
                                        shash,
                                        Some(Rc::new(ParentRef::new(parent_ref.map(|p| {
                                            Rc::clone(p)
                                        }), teban, mhash, shash))),
                                        &path,
                                        ignore_kyokumen_map,
                                        current_kyokumen_map,
                                        uniq_id,
                                        current_node,
                                        node_repo,
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
                                                 Some(Rc::new(ParentRef::new(parent_ref.map(|p| {
                                                     Rc::clone(p)
                                                 }), teban, mhash, shash))),
                                                 &path,
                                                 ignore_kyokumen_map,
                                                 current_kyokumen_map,
                                                 uniq_id,
                                                 current_node,
                                                 node_repo,
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

            mvs.push_back(n.try_borrow()?.m);

            let mut n = n.try_borrow()?.mate_node.as_ref().map(|n| Rc::clone(n));

            while let Some(c) = n {
                mvs.push_back(c.try_borrow()?.m);
                n = c.try_borrow()?.mate_node.as_ref().map(|n| Rc::clone(n))
            }

            Ok(mvs)
        }

        pub fn oute_process<L: Logger>(&mut self,
                                       depth:u32,
                                       mhash:u64,
                                       shash:u64,
                                       parent_ref:Option<Rc<ParentRef>>,
                                       path:&HashSet<(Teban,u64,u64)>,
                                       ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                       current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                       uniq_id:&mut UniqID,
                                       current_node:Option<Node>,
                                       node_repo:&mut NodeRepository,
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

            self.nodes.fetch_add(1,atomic::Ordering::Release);
            self.node_count += 1;

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            let (mut current_node,children) = if let Some(n) = current_node.as_ref() {
                let pn = n.pn;
                let dn = n.dn;

                let mut n = self.normalize_node(&n,mhash,shash,teban,depth,path,node_repo)?;

                if mate_depth.map(|d|  depth >= d).unwrap_or(false) {
                    let u = if n.pn.is_zero() && n.dn == Number::INFINITE {
                        n.to_decided_node(uniq_id.gen())
                    } else {
                        n.to_unknown_node(depth)
                    };

                    if !u.sennichite {
                        node_repo.update(teban,mhash,shash,&u)?;
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                if n.state == NodeState::Decided || n.state.is_unknown(depth) || pn != n.pn || dn != n.dn {
                    return Ok(MaybeMate::Continuation(n));
                }

                match n.m {
                    LegalMove::To(m) => {
                        if m.obtained() == Some(ObtainKind::Ou) {
                            n.pn = Number::INFINITE;
                            n.dn = Number::Value(Fraction::new(0));

                            node_repo.update(teban,mhash,shash,&n)?;
                            return Ok(MaybeMate::Continuation(n));
                        }
                    },
                    _ => ()
                }

                let expanded = n.expanded;

                if !expanded {
                    let mut n = self.expand_nodes(depth, mhash,shash,
                                                       parent_ref.as_ref(),
                                                                uniq_id,
                                                                n,
                                                                node_repo,
                                                                teban, state, mc)?;

                    self.send_seldepth(depth)?;

                    let len = n.children.try_borrow()?.len();

                    if len == 0 {
                        n.pn = Number::INFINITE;
                        n.dn = Number::Value(Fraction::new(0));
                    }

                    node_repo.update(teban,mhash,shash,&n)?;
                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let children = Rc::clone(&n.children);

                    if children.try_borrow()?.len() == 0 {
                        return Ok(MaybeMate::Continuation(n));
                    }

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

            let mut mate_mvs = None;

            loop {
                let n = children.try_borrow_mut()?.peek().map(|n| {
                    Rc::clone(n)
                }).ok_or(ApplicationError::LogicError(String::from(
                    "None of the child nodes exist."
                )))?;

                if n.try_borrow()?.state == NodeState::Decided {
                    if let Some(u) = current_node.as_ref() {
                        let u = u.to_decided_node(uniq_id.gen());

                        if !u.sennichite {
                            node_repo.update(teban, mhash, shash, &u)?;
                        }

                        return Ok(MaybeMate::Continuation(u));
                    } else {
                        break;
                    }
                } else if n.try_borrow()?.state.is_unknown(depth+1) {
                    if let Some(u) = current_node.as_ref() {
                        let u = u.to_unknown_node(depth);

                        if !u.sennichite {
                            node_repo.update(teban, mhash, shash, &u)?;
                        }

                        return Ok(MaybeMate::Continuation(u));
                    } else {
                        break;
                    }
                } else if n.try_borrow()?.pn == Number::INFINITE && n.try_borrow()?.dn.is_zero() {
                    if let Some(u) = current_node.as_ref() {
                        if u.pn.is_zero() && u.dn == Number::INFINITE {
                            let u = u.to_decided_node(uniq_id.gen());

                            node_repo.update(teban, mhash, shash, &u)?;

                            return Ok(MaybeMate::Continuation(u));
                        } else {
                            let mut u = u.clone();

                            u.pn = Number::INFINITE;
                            u.dn = Number::Value(Fraction::new(0));

                            if !u.sennichite {
                                node_repo.update(teban, mhash, shash, &u)?;
                            }

                            return Ok(MaybeMate::Continuation(u));
                        }
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
                    let sc = current_kyokumen_map.get(teban, &mhash, &shash).is_some();

                    if s || sc {
                        let mut u = NormalizedNode::from(n.try_borrow()?.deref());

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(Fraction::new(0));
                        u.sennichite = true;

                        update_node = u;
                    } else {
                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         parent_ref.as_ref(),
                                                         path,
                                                         ignore_kyokumen_map,
                                                         current_kyokumen_map,
                                                         uniq_id,
                                                         Some(n.try_borrow()?.deref().clone()),
                                                         node_repo,
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
                                        let u = NormalizedNode::from(n.try_borrow()?.deref());

                                        update_node = u.to_unknown_node(depth+1);
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
                let is_mate = u.pn.is_zero() && u.dn == Number::INFINITE;

                let md = u.mate_depth;

                if let Some(c) = current_node.as_mut() {
                    let pn = c.pn;
                    let dn = c.dn;

                    c.update(&u)?;

                    let md = u.mate_depth + depth + 1;

                    if c.pn.is_zero() && c.dn == Number::INFINITE {
                        if !self.strict_moves || c.mate_node.is_none() {
                            c.mate_depth = u.mate_depth + 1;
                            c.mate_node = Some(Rc::new(RefCell::new(u.into())));
                        } else if is_mate && mate_depth.map(|d| d > md).unwrap_or(true) {
                            c.mate_depth = u.mate_depth + 1;
                            c.mate_node = Some(Rc::new(RefCell::new(u.into())));
                        }
                    }

                    if !c.sennichite {
                        node_repo.update(teban, mhash, shash, &c)?;
                    }

                    if c.pn != pn || c.dn != dn {
                        return Ok(MaybeMate::Continuation(c.clone()));
                    } else if is_mate && mate_depth.map(|d| {
                        d > md
                    }).unwrap_or(false) {
                        return Ok(MaybeMate::Continuation(c.clone()));
                    }
                } else {
                    if let Some(mut p) = children.try_borrow_mut()?.peek_mut() {
                        let u = u.clone();
                        *p = Rc::new(RefCell::new(u.into()));
                    } else {
                        return Err(ApplicationError::LogicError(String::from(
                            "Node to be updated could not be found."
                        )));
                    }

                    if !self.strict_moves && u.pn.is_zero() && u.dn == Number::INFINITE {
                        *mate_depth = Some(md + 1);
                        let mate_mvs = self.build_moves(&Rc::new(RefCell::new(u.into())))?;
                        self.send_pv(&mate_mvs)?;
                        return Ok(MaybeMate::MateMoves(mate_mvs));
                    } else if u.pn.is_zero() && u.dn == Number::INFINITE {
                        *mate_depth = Some(md + 1);
                        let mvs = self.build_moves(&Rc::new(RefCell::new(u.into())))?;
                        self.send_pv(&mvs)?;
                        mate_mvs = Some(mvs);
                    }
                }
            }

            if depth == 0 {
                if let Some(mvs) = mate_mvs {
                   Ok(MaybeMate::MateMoves(mvs))
                } else {
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
                }
            } else {
                Ok(MaybeMate::Skip)
            }
        }

        pub fn response_oute_process<L: Logger>(&mut self,
                                                depth:u32,
                                                mhash:u64,
                                                shash:u64,
                                                parent_ref:Option<Rc<ParentRef>>,
                                                path:&HashSet<(Teban,u64,u64)>,
                                                ignore_kyokumen_map:&mut KyokumenMap<u64,()>,
                                                current_kyokumen_map:&mut KyokumenMap<u64,u32>,
                                                uniq_id:&mut UniqID,
                                                current_node:Option<Node>,
                                                node_repo:&mut NodeRepository,
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

            self.nodes.fetch_add(1,atomic::Ordering::Release);
            self.node_count += 1;

            event_dispatcher.dispatch_events(self,event_queue)?;

            if self.check_timelimit() || self.stop.load(atomic::Ordering::Acquire) {
                return Ok(MaybeMate::Timeout);
            }

            let mut current_node = if let Some(n) = current_node.as_ref() {
                let pn = n.pn;
                let dn = n.dn;

                let mut n = self.normalize_node(n,mhash,shash,teban,depth,path,node_repo)?;

                if mate_depth.map(|d|  depth >= d).unwrap_or(false) {
                    let u = if n.pn.is_zero() && n.dn == Number::INFINITE {
                        n.to_decided_node(uniq_id.gen())
                    } else {
                        n.to_unknown_node(depth)
                    };

                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, &u)?;
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                if n.state == NodeState::Decided || n.state.is_unknown(depth) || pn != n.pn || dn != n.dn {
                    return Ok(MaybeMate::Continuation(n));
                }

                match n.m {
                    LegalMove::To(m) => {
                        if m.obtained() == Some(ObtainKind::Ou) {
                            n.pn = Number::Value(Fraction::new(0));
                            n.dn = Number::INFINITE;

                            n = n.to_decided_node(uniq_id.gen());

                            node_repo.update(teban, mhash, shash, &n)?;

                            return Ok(MaybeMate::Continuation(n));
                        }
                    },
                    _ => ()
                }
                let expanded = n.expanded;

                if !expanded {
                    let mut n = self.expand_nodes(depth, mhash, shash,
                                                        parent_ref.as_ref(),
                                                                 uniq_id,
                                                                 n,
                                                                 node_repo, teban, state, mc)?;

                    self.send_seldepth(depth)?;

                    let len = n.children.try_borrow()?.len();

                    if len == 0 {
                        n.pn = Number::Value(Fraction::new(0));
                        n.dn = Number::INFINITE;
                        n = n.to_decided_node(uniq_id.gen());
                    }

                    node_repo.update(teban, mhash, shash, &n)?;

                    return Ok(MaybeMate::Continuation(n));
                } else {
                    let len = n.children.try_borrow()?.len();

                    if len == 0 {
                        return Ok(MaybeMate::Continuation(n));
                    }

                    n
                }
            } else {
                return Err(ApplicationError::LogicError(String::from(
                    "current move is not set."
                )));
            };

            let children = Rc::clone(&current_node.children);

            if current_node.pn.is_zero() && current_node.dn == Number::INFINITE {
                let n = children.try_borrow()?.peek().map(|n| {
                    Rc::clone(n)
                }).ok_or(ApplicationError::LogicError(String::from(
                    "None of the child nodes exist."
                )))?;

                if n.try_borrow()?.state == NodeState::Decided {
                    let n = NormalizedNode::from(n.try_borrow()?.deref());

                    let u = n.to_decided_node(uniq_id.gen());

                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, &u)?;
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                let update_node;

                let m = n.try_borrow()?.m;

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
                    let sc = current_kyokumen_map.get(teban, &mhash, &shash).is_some();

                    if s || sc {
                        let mut u = NormalizedNode::from(n.try_borrow()?.deref());

                        u.pn = Number::Value(Fraction::new(0));
                        u.dn = Number::INFINITE;
                        u.sennichite = true;

                        update_node = u;
                    } else {
                        let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                        match next {
                            (state, mc, _) => {
                                match self.inter_process(depth + 1,
                                                         mhash,
                                                         shash,
                                                         parent_ref.as_ref(),
                                                         path,
                                                         ignore_kyokumen_map,
                                                         current_kyokumen_map,
                                                         uniq_id,
                                                         Some(n.try_borrow()?.deref().clone()),
                                                         node_repo,
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
                                        let u = NormalizedNode::from(n.try_borrow()?.deref());

                                        update_node = u.to_unknown_node(depth+1);
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

                let c = &mut current_node;

                if u.state.is_unknown(depth+1) {
                    let u = c.to_unknown_node(depth);
                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, &u)?;
                    }

                    return Ok(MaybeMate::Continuation(u));
                }

                c.update(&u)?;

                if mate_depth.map(|d| d > u.mate_depth + depth + 1).unwrap_or(true) {
                    let n = Node::from(&u);
                    let mut n = Rc::new(RefCell::new(n));

                    for c in children.try_borrow()?.iter() {
                        if c.try_borrow()?.pn.is_zero() && c.try_borrow()?.dn == Number::INFINITE &&
                           c.try_borrow()?.mate_depth > n.try_borrow()?.mate_depth {
                           n = Rc::clone(c);
                        }
                    }

                    c.mate_depth = n.try_borrow()?.mate_depth + 1;
                    c.mate_node = Some(n);

                    let u = c;

                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, u)?;
                    }

                    return Ok(MaybeMate::Continuation(u.clone()));
                } else {
                    let u = c.to_decided_node(uniq_id.gen());

                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, &u)?;
                    }

                    return Ok(MaybeMate::Continuation(u));
                }
            } else {
                loop {
                    let n = children.try_borrow()?.peek().map(|n| {
                        Rc::clone(n)
                    }).ok_or(ApplicationError::LogicError(String::from(
                        "None of the child nodes exist."
                    )))?;

                    if n.try_borrow()?.state == NodeState::Decided || n.try_borrow()?.state.is_unknown(depth+1) ||
                        (n.try_borrow()?.pn.is_zero() && n.try_borrow()?.dn == Number::INFINITE) {
                        let u = current_node;

                        let u = if n.try_borrow()?.state == NodeState::Decided {
                            u.to_decided_node(uniq_id.gen())
                        } else {
                            u.to_unknown_node(depth)
                        };

                        if !u.sennichite {
                            node_repo.update(teban, mhash, shash, &u)?;
                        }

                        return Ok(MaybeMate::Continuation(u));
                    } else if n.try_borrow()?.pn == Number::INFINITE && n.try_borrow()?.dn.is_zero() {
                        let mut u = current_node;

                        u.pn = Number::INFINITE;
                        u.dn = Number::Value(Fraction::new(0));

                        if !u.sennichite {
                            node_repo.update(teban, mhash, shash, &u)?;
                        }

                        return Ok(MaybeMate::Continuation(u.into()));
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
                        let sc = current_kyokumen_map.get(teban, &mhash, &shash).is_some();

                        if s || sc {
                            let mut u = NormalizedNode::from(n.try_borrow()?.deref());

                            u.pn = Number::INFINITE;
                            u.dn = Number::Value(Fraction::new(0));
                            u.sennichite = true;

                            update_node = u;
                        } else {
                            let next = Rule::apply_move_none_check(state, teban, mc, m.to_applied_move());

                            match next {
                                (state, mc, _) => {
                                    match self.inter_process(depth + 1,
                                                             mhash,
                                                             shash,
                                                             parent_ref.as_ref(),
                                                             path,
                                                             ignore_kyokumen_map,
                                                             current_kyokumen_map,
                                                             uniq_id,
                                                             Some(n.try_borrow()?.deref().clone()),
                                                             node_repo,
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
                                            let u = NormalizedNode::from(n.try_borrow()?.deref());

                                            update_node = u.to_unknown_node(depth+1);
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

                    let c = &mut current_node;

                    if u.state.is_unknown(depth+1) {
                        let u = c.to_unknown_node(depth);
                        if !u.sennichite {
                            node_repo.update(teban, mhash, shash, &u)?;
                        }

                        return Ok(MaybeMate::Continuation(u));
                    }

                    let pn = c.pn;
                    let dn = c.dn;

                    c.update(&u)?;

                    if c.pn.is_zero() && c.dn == Number::INFINITE {
                        let n = Node::from(&u);
                        let mut n = Rc::new(RefCell::new(n));

                        for c in children.try_borrow()?.iter() {
                            if c.try_borrow()?.pn.is_zero() && c.try_borrow()?.dn == Number::INFINITE &&
                                c.try_borrow()?.mate_depth > n.try_borrow()?.mate_depth {
                                n = Rc::clone(c);
                            }
                        }

                        c.mate_depth = n.try_borrow()?.mate_depth + 1;
                        c.mate_node = Some(n);
                    }

                    let u = c;

                    if !u.sennichite {
                        node_repo.update(teban, mhash, shash, u)?;
                    }

                    if u.pn != pn || u.dn != dn {
                        return Ok(MaybeMate::Continuation(u.clone()));
                    }
                }
            }
        }

        fn send_pv(&mut self, pv:&VecDeque<LegalMove>) -> Result<(),ApplicationError> {
            let mut commands: Vec<UsiInfoSubCommand> = Vec::new();

            commands.push(UsiInfoSubCommand::CurrMove(pv[0].to_move()));
            commands.push(UsiInfoSubCommand::Pv(pv.clone().into_iter().map(|m| m.to_move()).collect()));

            Ok(self.info_sender.send_immediate(commands)?)
        }

        fn send_seldepth(&mut self, depth:u32) -> Result<(),InfoSendError>{
            let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
            commands.push(UsiInfoSubCommand::Depth(self.base_depth));
            commands.push(UsiInfoSubCommand::SelDepth(self.current_depth + depth));

            Ok(self.info_sender.send(commands)?)
        }

        fn check_timelimit(&mut self) -> bool {
            const CHECK_TIMELIMIT_BOUNDARY:u32 = 4096;

            self.attempt_check_timelimit = self.attempt_check_timelimit % CHECK_TIMELIMIT_BOUNDARY;

            let r;

            if self.attempt_check_timelimit == 0 {
                r = self.limit.map_or(false, |l| {
                    let now = Instant::now();
                    l < now ||
                        l - now <= Duration::from_millis(self.network_delay as u64 + TIMELIMIT_MARGIN) ||
                        self.checkmate_limit.map(|l| l < now).unwrap_or(false)
                });
            } else {
                r = false;
            }

            self.attempt_check_timelimit += 1;

            r
        }
    }
}