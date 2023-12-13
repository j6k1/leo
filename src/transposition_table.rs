use std::mem;
use std::num::Wrapping;
use std::ops::{Add, BitXor, Deref, DerefMut, Index, IndexMut, Sub};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use usiagent::hash::{InitialHash, KyokumenHash};
use usiagent::rule::{LegalMove,AppliedMove};
use usiagent::shogi::{Banmen, Mochigoma, MochigomaCollections, MochigomaKind, Teban};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub trait ToBucketIndex {
    fn to_bucket_index(self) -> usize;
}
impl ToBucketIndex for u128 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u64 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u32 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u16 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
impl ToBucketIndex for u8 {
    fn to_bucket_index(self) -> usize {
        self as usize
    }
}
#[derive(Debug,Clone)]
pub struct ZobristHash<T>
    where T: Add + Sub + BitXor<Output = T> + Copy + InitialHash,
             Wrapping<T>: Add<Output = Wrapping<T>> + Sub<Output = Wrapping<T>> + BitXor<Output = Wrapping<T>> + Copy,
             Standard: Distribution<T> {
    mhash:T,
    shash:T,
    teban:Teban
}
impl<T> ZobristHash<T>
    where T: Add + Sub + BitXor<Output = T> + Copy + InitialHash,
             Wrapping<T>: Add<Output = Wrapping<T>> + Sub<Output = Wrapping<T>> + BitXor<Output = Wrapping<T>> + Copy,
             Standard: Distribution<T> {
    pub fn new(hasher:&KyokumenHash<T>,teban:Teban,banmen:&Banmen,ms:&Mochigoma,mg:&Mochigoma) -> ZobristHash<T> {
        let (mhash,shash) = hasher.calc_initial_hash(&banmen, &ms, &mg);

        ZobristHash {
            mhash:mhash,
            shash:shash,
            teban:teban
        }
    }

    pub fn updated(&self,hasher:&KyokumenHash<T>,teban:Teban,banmen:&Banmen,mc:&MochigomaCollections,m:AppliedMove,obtained:&Option<MochigomaKind>)
        -> ZobristHash<T> {
        let mhash = hasher.calc_main_hash(self.mhash,teban,banmen,mc,m,obtained);
        let shash = hasher.calc_sub_hash(self.shash,teban,banmen,mc,m,obtained);

        ZobristHash {
            mhash:mhash,
            shash:shash,
            teban:teban
        }
    }

    pub fn keys(&self) -> (T,T) {
        (self.mhash,self.shash)
    }

    pub fn teban(&self) -> Teban {
        self.teban
    }
}
#[derive(Debug,Clone)]
pub struct TTPartialEntry<T> where T: Default {
    pub depth:i8,
    pub score:T,
    pub best_move:Option<LegalMove>
}
impl<T> Default for TTPartialEntry<T> where T: Default {
    fn default() -> Self {
        TTPartialEntry {
            depth:-1,
            score:T::default(),
            best_move: None
        }
    }
}
#[derive(Debug,Clone)]
pub struct TTEntry<T,K> where K: Eq, T: Default {
    used:bool,
    mhash:K,
    shash:K,
    teban:Teban,
    entry:TTPartialEntry<T>
}
impl<T,K> Default for TTEntry<T,K> where K: Eq + Default, T: Default {
    fn default() -> Self {
        TTEntry {
            used:false,
            mhash:K::default(),
            shash:K::default(),
            teban:Teban::Sente,
            entry: TTPartialEntry::default()
        }
    }
}
pub struct ReadGuard<'a,T,K,const N:usize> where K: Eq, T: Default {
    locked_bucket:RwLockReadGuard<'a, [TTEntry<T,K>;N]>,
    index:usize
}
impl<'a,T,K,const N:usize> ReadGuard<'a,T,K,N> where K: Eq, T: Default {
    fn new(locked_bucket:RwLockReadGuard<'a,[TTEntry<T,K>;N]>,index:usize) -> ReadGuard<'a,T,K,N> {
        ReadGuard {
            locked_bucket:locked_bucket,
            index:index
        }
    }
}
impl<'a,T,K,const N:usize> Deref for ReadGuard<'a,T,K,N> where K: Eq, T: Default {
    type Target = TTPartialEntry<T>;

    fn deref(&self) -> &Self::Target {
        &self.locked_bucket.deref().index(self.index).entry
    }
}
pub struct WriteGuard<'a,T,K,const N:usize> where K: Eq, T: Default {
    locked_bucket:RwLockWriteGuard<'a, [TTEntry<T,K>;N]>,
    index:usize
}
impl<'a,T,K,const N:usize> WriteGuard<'a,T,K,N> where K: Eq, T: Default {
    fn new(locked_bucket:RwLockWriteGuard<'a,[TTEntry<T,K>;N]>,index:usize) -> WriteGuard<'a,T,K,N> {
        WriteGuard {
            locked_bucket:locked_bucket,
            index:index
        }
    }

    fn remove(&mut self) -> TTPartialEntry<T> {
        let mut e = self.locked_bucket.deref_mut().index_mut(self.index);

        e.used = false;

        mem::replace(&mut e.entry,TTPartialEntry::default())
    }

    fn insert(&mut self,entry:TTEntry<T,K>) {
        let e = self.locked_bucket.deref_mut().index_mut(self.index);

        *e = entry
    }
}
impl<'a,T,K,const N:usize> Deref for WriteGuard<'a,T,K,N> where K: Eq, T: Default {
    type Target = TTPartialEntry<T>;

    fn deref(&self) -> &Self::Target {
        &self.locked_bucket.deref().index(self.index).entry
    }
}

impl<'a,T,K,const N:usize> DerefMut for WriteGuard<'a,T,K,N> where K: Eq, T: Default {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.locked_bucket.deref_mut().index_mut(self.index).entry
    }
}
pub struct OccupiedTTEntry<'a,T,K,const N:usize> where K: Eq, T: Default {
    write_guard:WriteGuard<'a,T,K,N>
}
impl<'a,T,K,const N:usize> OccupiedTTEntry<'a,T,K,N> where K: Eq, T: Default {
    fn new(write_guard:WriteGuard<'a,T,K,N>) -> OccupiedTTEntry<'a,T,K,N> {
        OccupiedTTEntry {
            write_guard:write_guard
        }
    }

    pub fn get(&self) -> &TTPartialEntry<T> {
        self.write_guard.deref()
    }

    pub fn get_mut(&mut self) -> &mut TTPartialEntry<T> {
        self.write_guard.deref_mut()
    }

    pub fn remove(&mut self) -> TTPartialEntry<T> {
        self.write_guard.remove()
    }

    pub fn insert(&mut self,entry:TTPartialEntry<T>) -> &mut TTPartialEntry<T> {
        *self.write_guard.deref_mut() = entry;
        self.write_guard.deref_mut()
    }
}
pub struct VacantTTEntry<'a,T,K,const N:usize> where K: Eq + Copy, T: Default {
    mhash:K,
    shash:K,
    teban:Teban,
    write_guard:WriteGuard<'a,T,K,N>
}
impl<'a,T,K,const N:usize> VacantTTEntry<'a,T,K,N> where K: Eq + Copy, T: Default {
    fn new(write_guard:WriteGuard<'a,T,K,N>,mhash:K,shash:K,teban:Teban) -> VacantTTEntry<'a,T,K,N> {
        VacantTTEntry {
            mhash:mhash,
            shash:shash,
            teban:teban,
            write_guard:write_guard
        }
    }

    pub fn insert(&mut self,entry:TTPartialEntry<T>) -> &mut TTPartialEntry<T> {
        self.write_guard.insert(TTEntry {
            used:true,
            mhash:self.mhash,
            shash:self.shash,
            teban:self.teban,
            entry:entry
        });

        self.write_guard.deref_mut()
    }
}
pub enum Entry<'a,T,K,const N:usize> where K: Eq + Copy, T: Default {
    OccupiedTTEntry(OccupiedTTEntry<'a,T,K,N>),
    VacantTTEntry(VacantTTEntry<'a,T,K,N>)
}
impl<'a,T,K,const N:usize> Entry<'a,T,K,N> where K: Eq + Copy, T: Default {
    pub fn or_insert(&mut self,entry:TTPartialEntry<T>) -> &mut TTPartialEntry<T> where K: Copy {
        match self {
            Entry::OccupiedTTEntry(ref mut e) => {
                e.get_mut()
            },
            Entry::VacantTTEntry(ref mut e) => {
                e.insert(entry)
            }
        }
    }

    pub fn or_default(&mut self) -> &mut TTPartialEntry<T> where K: Copy {
        self.or_insert(TTPartialEntry::default())
    }
}
const fn support_fast_mod(v:usize) -> bool {
    v != 0 && v & (v - 1) == 0
}
pub struct TT<K,T,const S:usize,const N:usize>
    where K: Eq + Default + Add + Sub + BitXor<Output = K> + Copy + InitialHash + ToBucketIndex,
             Wrapping<K>: Add<Output = Wrapping<K>> + Sub<Output = Wrapping<K>> + BitXor<Output = Wrapping<K>> + Copy,
             Standard: Distribution<K>,
             [TTEntry<T,K>;N]: Default,
             T: Default {
    buckets:Vec<RwLock<[TTEntry<T,K>;N]>>
}
impl<K,T,const S:usize,const N:usize> TT<K,T,S,N>
    where K: Eq + Default + Add + Sub + BitXor<Output = K> + Copy + InitialHash + ToBucketIndex,
             Wrapping<K>: Add<Output = Wrapping<K>> + Sub<Output = Wrapping<K>> + BitXor<Output = Wrapping<K>> + Copy,
             Standard: Distribution<K>,
             [TTEntry<T,K>;N]: Default,
             T: Default {
    pub fn new() -> TT<K,T,S,N> {
        let mut buckets = Vec::with_capacity(S);
        buckets.resize_with(S,RwLock::default);

        TT {
            buckets:buckets
        }
    }

    pub fn clear(&mut self) {
        self.buckets.fill_with(RwLock::default);
    }

    fn bucket_index(&self,zh:&ZobristHash<K>) -> usize {
        if support_fast_mod(S) {
            zh.mhash.to_bucket_index() & (S - 1)
        } else {
            zh.mhash.to_bucket_index() % S
        }
    }

    pub fn get(&self,zh: &ZobristHash<K>) -> Option<ReadGuard<'_,T,K,N>> {
        let index = self.bucket_index(zh);

        match self.buckets[index].read() {
            bucket => {
                for i in 0..bucket.len() {
                    if bucket[i].used && bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Some(ReadGuard::new(bucket, i));
                    }
                }

                None
            }
        }
    }

    pub fn get_mut(&self,zh: &ZobristHash<K>) -> Option<WriteGuard<'_,T,K,N>> {
        let index = self.bucket_index(zh);

        match self.buckets[index].write() {
            bucket => {
                for i in 0..bucket.len() {
                    if bucket[i].used && bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Some(WriteGuard::new(bucket, i));
                    }
                }

                None
            }
        }
    }

    pub fn insert(&self,zh: &ZobristHash<K>, entry:TTPartialEntry<T>) -> Option<TTPartialEntry<T>> {
        let index = self.bucket_index(zh);

        let tte = TTEntry {
            used:true,
            mhash:zh.mhash,
            shash:zh.shash,
            teban:zh.teban,
            entry: entry
        };

        match self.buckets[index].write() {
            mut bucket => {
                for i in 0..bucket.len() {
                    if bucket[i].used && bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        let tte = mem::replace(&mut bucket[i],tte);

                        return Some(tte.entry);
                    }
                }

                let mut index = 0;
                let mut priority = i8::MAX;

                for i in 0..bucket.len() {
                    if !bucket[i].used {
                        index = i;
                        break;
                    }

                    let pri = bucket[i].entry.depth;

                    if pri <= priority {
                        priority = pri;
                        index = i;
                    }
                }

                let tte = mem::replace(&mut bucket[index],tte);

                if tte.used {
                    Some(tte.entry)
                } else {
                    None
                }
            }
        }
    }

    pub fn entry(&self,zh: &ZobristHash<K>) -> Entry<'_,T,K,N> {
        let index = self.bucket_index(zh);

        match self.buckets[index].write() {
            bucket => {
                for i in 0..bucket.len() {
                    if bucket[i].used && bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Entry::OccupiedTTEntry(
                            OccupiedTTEntry::new(WriteGuard::new(bucket, i))
                        );
                    }
                }

                let mut index = 0;
                let mut priority = i8::MAX;

                for i in 0..bucket.len() {
                    if !bucket[i].used {
                        index = i;
                        break;
                    }

                    let pri = bucket[i].entry.depth;

                    if pri <= priority {
                        priority = pri;
                        index = i;
                    }
                }

                Entry::VacantTTEntry(
                    VacantTTEntry::new(
              WriteGuard::new(bucket, index),
                        zh.mhash,
                        zh.shash,
                        zh.teban
                    )
                )
            }
        }
    }

    pub fn contains_key(&self,zh:&ZobristHash<K>) -> bool where K: Eq {
        let index = self.bucket_index(zh);

        match self.buckets[index].read() {
            bucket => {
                for i in 0..bucket.len() {
                    if bucket[i].used && bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return true;
                    }
                }

                return false;
            }
        }
    }
}