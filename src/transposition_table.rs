use std::mem;
use std::num::Wrapping;
use std::ops::{Add, BitXor, Deref, DerefMut, Index, IndexMut, Sub};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use usiagent::hash::{InitialHash, KyokumenHash};
use usiagent::rule::AppliedMove;
use usiagent::shogi::{Banmen, Mochigoma, MochigomaCollections, MochigomaKind, Teban};

pub trait ToTTIndex {
    fn to_tt_index(self) -> usize;
}
impl ToTTIndex for u128 {
    fn to_tt_index(self) -> usize {
        self as usize
    }
}
impl ToTTIndex for u64 {
    fn to_tt_index(self) -> usize {
        self as usize
    }
}
impl ToTTIndex for u32 {
    fn to_tt_index(self) -> usize {
        self as usize
    }
}
impl ToTTIndex for u16 {
    fn to_tt_index(self) -> usize {
        self as usize
    }
}
impl ToTTIndex for u8 {
    fn to_tt_index(self) -> usize {
        self as usize
    }
}
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
pub struct TTPartialEntry {
    pub depth:u8,
    pub static_eval:u32,
    pub search_eval:u32
}
impl Default for TTPartialEntry {
    fn default() -> Self {
        TTPartialEntry {
            depth:0,
            static_eval:0,
            search_eval:0
        }
    }
}
pub struct TTEntry<K> where K: Eq {
    used:bool,
    mhash:K,
    shash:K,
    teban:Teban,
    generation:u16,
    entry:TTPartialEntry
}
impl<K> TTEntry<K> where K: Eq + Default {
    pub fn priority(&self,generation:u16) -> u16 {
        i16::MAX as u16 + self.entry.depth as u16 - (generation - self.generation)
    }
}
impl<K> Default for TTEntry<K> where K: Eq + Default {
    fn default() -> Self {
        TTEntry {
            used:false,
            mhash:K::default(),
            shash:K::default(),
            teban:Teban::Sente,
            generation:0,
            entry: TTPartialEntry::default()
        }
    }
}
pub struct ReadGuard<'a,K,const N:usize> where K: Eq {
    locked_bucket:RwLockReadGuard<'a, [TTEntry<K>;N]>,
    index:usize
}
impl<'a,K,const N:usize> ReadGuard<'a,K,N> where K: Eq {
    fn new(locked_bucket:RwLockReadGuard<'a,[TTEntry<K>;N]>,index:usize) -> ReadGuard<'a,K,N> {
        ReadGuard {
            locked_bucket:locked_bucket,
            index:index
        }
    }
}
impl<'a,K,const N:usize> Deref for ReadGuard<'a,K,N> where K: Eq {
    type Target = TTPartialEntry;

    fn deref(&self) -> &Self::Target {
        &self.locked_bucket.deref().index(self.index).entry
    }
}
pub struct WriteGuard<'a,K,const N:usize> where K: Eq {
    locked_bucket:RwLockWriteGuard<'a, [TTEntry<K>;N]>,
    index:usize
}
impl<'a,K,const N:usize> WriteGuard<'a,K,N> where K: Eq {
    fn new(locked_bucket:RwLockWriteGuard<'a,[TTEntry<K>;N]>,index:usize) -> WriteGuard<'a,K,N> {
        WriteGuard {
            locked_bucket:locked_bucket,
            index:index
        }
    }

    fn remove(&mut self) -> TTPartialEntry {
        let mut e = self.locked_bucket.deref_mut().index_mut(self.index);

        e.used = false;

        mem::replace(&mut e.entry,TTPartialEntry::default())
    }

    fn insert(&mut self,entry:TTEntry<K>) {
        let e = self.locked_bucket.deref_mut().index_mut(self.index);

        *e = entry
    }
}
impl<'a,K,const N:usize> Deref for WriteGuard<'a,K,N> where K: Eq {
    type Target = TTPartialEntry;

    fn deref(&self) -> &Self::Target {
        &self.locked_bucket.deref().index(self.index).entry
    }
}

impl<'a,K,const N:usize> DerefMut for WriteGuard<'a,K,N> where K: Eq {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.locked_bucket.deref_mut().index_mut(self.index).entry
    }
}
pub struct OccupiedTTEntry<'a,K,const N:usize> where K: Eq {
    write_guard:WriteGuard<'a,K,N>
}
impl<'a,K,const N:usize> OccupiedTTEntry<'a,K,N> where K: Eq {
    fn new(write_guard:WriteGuard<'a,K,N>) -> OccupiedTTEntry<'a,K,N> {
        OccupiedTTEntry {
            write_guard:write_guard
        }
    }

    pub fn get(&self) -> &TTPartialEntry {
        self.write_guard.deref()
    }

    pub fn get_mut(&mut self) -> &mut TTPartialEntry {
        self.write_guard.deref_mut()
    }

    pub fn remove(&mut self) -> TTPartialEntry {
        self.write_guard.remove()
    }

    pub fn insert(&mut self,entry:TTPartialEntry) -> &mut TTPartialEntry {
        *self.write_guard.deref_mut() = entry;
        self.write_guard.deref_mut()
    }
}
pub struct VacantTTEntry<'a,K,const N:usize> where K: Eq + Copy {
    mhash:K,
    shash:K,
    teban:Teban,
    generation:u16,
    write_guard:WriteGuard<'a,K,N>
}
impl<'a,K,const N:usize> VacantTTEntry<'a,K,N> where K: Eq + Copy {
    fn new(write_guard:WriteGuard<'a,K,N>,mhash:K,shash:K,teban:Teban,generation:u16) -> VacantTTEntry<'a,K,N> {
        VacantTTEntry {
            mhash:mhash,
            shash:shash,
            teban:teban,
            generation:generation,
            write_guard:write_guard
        }
    }

    pub fn insert(&mut self,entry:TTPartialEntry) -> &mut TTPartialEntry {
        self.write_guard.insert(TTEntry {
            used:true,
            mhash:self.mhash,
            shash:self.shash,
            teban:self.teban,
            generation:self.generation,
            entry:entry
        });

        self.write_guard.deref_mut()
    }
}
pub enum Entry<'a,K,const N:usize> where K: Eq + Copy {
    OccupiedTTEntry(OccupiedTTEntry<'a,K,N>),
    VacantTTEntry(VacantTTEntry<'a,K,N>)
}
impl<'a,K,const N:usize> Entry<'a,K,N> where K: Eq + Copy {
    pub fn or_insert(&mut self,entry:TTPartialEntry) -> &mut TTPartialEntry where K: Copy {
        match self {
            Entry::OccupiedTTEntry(e) => {
                e.get_mut()
            },
            Entry::VacantTTEntry(e) => {
                e.insert(entry)
            }
        }
    }
}
const fn support_fast_mod(v:usize) -> bool {
    v != 0 && v & (v - 1) == 0
}
pub struct TT<K,const S:usize,const N:usize>
    where K: Eq + Default + Add + Sub + BitXor<Output = K> + Copy + InitialHash + ToTTIndex,
             Wrapping<K>: Add<Output = Wrapping<K>> + Sub<Output = Wrapping<K>> + BitXor<Output = Wrapping<K>> + Copy,
             Standard: Distribution<K>,
             [TTEntry<K>;N]: Default {
    buckets:Vec<RwLock<[TTEntry<K>;N]>>,
    generation:u16
}
impl<K,const S:usize,const N:usize> TT<K,S,N>
    where K: Eq + Default + Add + Sub + BitXor<Output = K> + Copy + InitialHash + ToTTIndex,
             Wrapping<K>: Add<Output = Wrapping<K>> + Sub<Output = Wrapping<K>> + BitXor<Output = Wrapping<K>> + Copy,
             Standard: Distribution<K>,
             [TTEntry<K>;N]: Default  {
    pub fn with_size() -> TT<K,S,N> {
        let mut buckets = Vec::with_capacity(S);
        buckets.resize_with(S,RwLock::default);

        TT {
            buckets:buckets,
            generation:0
        }
    }

    fn bucket_index(&self,zh:&ZobristHash<K>) -> usize {
        if support_fast_mod(S) {
            zh.mhash.to_tt_index() & (S - 1)
        } else {
            zh.mhash.to_tt_index() % S
        }
    }

    pub fn generation_to_next(&mut self) {
        self.generation += 8;
    }

    pub fn get(&self,zh: &ZobristHash<K>) -> Option<ReadGuard<'_,K,N>> {
        let index = self.bucket_index(zh);

        match self.buckets[index].read() {
            Ok(bucket) => {
                for i in 0..bucket.len() {
                    if bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Some(ReadGuard::new(bucket, i));
                    }
                }

                None
            },
            Err(e) => {
                panic!("{}", e);
            }
        }
    }

    pub fn get_mut(&self,zh: &ZobristHash<K>) -> Option<WriteGuard<'_,K,N>> {
        let index = self.bucket_index(zh);

        match self.buckets[index].write() {
            Ok(bucket) => {
                for i in 0..bucket.len() {
                    if bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Some(WriteGuard::new(bucket, i));
                    }
                }

                None
            },
            Err(e) => {
                panic!("{}", e);
            }
        }
    }

    pub fn insert(&self,zh: &ZobristHash<K>, entry:TTPartialEntry) -> Option<TTPartialEntry> {
        let index = self.bucket_index(zh);

        let tte = TTEntry {
            used:true,
            mhash:zh.mhash,
            shash:zh.shash,
            teban:zh.teban,
            generation:self.generation,
            entry: entry
        };

        match self.buckets[index].write() {
            Ok(mut bucket) => {
                for i in 0..bucket.len() {
                    if bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        let tte = mem::replace(&mut bucket[i],tte);

                        return Some(tte.entry);
                    }
                }

                let mut index = 0;
                let mut priority = u16::MAX;

                for i in 0..bucket.len() {
                    if !bucket[i].used {
                        index = i;
                        break;
                    }

                    let pri = bucket[i].priority(self.generation);

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
            },
            Err(e) => {
                panic!("{}",e);
            }
        }
    }

    pub fn entry(&self,zh: &ZobristHash<K>) -> Entry<'_,K,N> {
        let index = self.bucket_index(zh);

        match self.buckets[index].write() {
            Ok(bucket) => {
                for i in 0..bucket.len() {
                    if bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return Entry::OccupiedTTEntry(
                            OccupiedTTEntry::new(WriteGuard::new(bucket, i))
                        );
                    }
                }

                let mut index = 0;
                let mut priority = u16::MAX;

                for i in 0..bucket.len() {
                    if !bucket[i].used {
                        index = i;
                        break;
                    }

                    let pri = bucket[i].priority(self.generation);

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
                        zh.teban,
                        self.generation
                    )
                )
            },
            Err(e) => {
                panic!("{}", e);
            }
        }
    }

    pub fn contains_key(&self,zh:&ZobristHash<K>) -> bool where K: Eq {
        let index = self.bucket_index(zh);

        match self.buckets[index].read() {
            Ok(bucket) => {
                for i in 0..bucket.len() {
                    if bucket[i].mhash == zh.mhash && bucket[i].shash == zh.shash && bucket[i].teban == zh.teban {
                        return true;
                    }
                }

                return false;
            },
            Err(e) => {
                panic!("{}", e);
            }
        }
    }
}