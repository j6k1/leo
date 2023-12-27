use std::cell::RefCell;
use std::ops::DerefMut;
use std::path::{Path};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{fs};
use rand::{prelude, Rng, SeedableRng};
use rand::prelude::{Distribution};
use rand_distr::Normal;
use rand_xorshift::XorShiftRng;
use nncombinator::activation::{ReLu, Tanh};
use nncombinator::arr::{Arr, SerializedVec};
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::device::{Device, DeviceCpu, DeviceGpu};
use nncombinator::layer::{AddLayer, AddLayerTrain, BatchForwardBase, BatchTrain, ForwardAll, PreTrain, TryAddLayer};
use nncombinator::layer::input::InputLayer;
use nncombinator::layer::output::LinearOutputLayer;
use nncombinator::layer::linear::{LinearLayerBuilder};
use nncombinator::layer::activation::ActivationLayer;
use nncombinator::layer::batchnormalization::BatchNormalizationLayerBuilder;
use nncombinator::lossfunction::{Mse};
use nncombinator::ope::UnitValue;
use nncombinator::optimizer::{SGD};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence, PersistenceType, SaveToFile};
use packedsfen::hcpe::reader::HcpeReader;
use packedsfen::traits::Reader;
use packedsfen::{hcpe, yaneuraou};
use packedsfen::hcpe::haffman_code::GameResult;
use packedsfen::yaneuraou::reader::PackedSfenReader;
use usiagent::event::{EventQueue, GameEndState, UserEvent, UserEventKind};
use usiagent::shogi::{Banmen, KomaKind, Mochigoma, MOCHIGOMA_KINDS, MochigomaCollections, Teban};
use crate::error::{ApplicationError};

const BANMEN_SIZE:usize = 81;

const SELF_TEBAN_INDEX:usize = 0;
const OPPONENT_TEBAN_INDEX:usize = SELF_TEBAN_INDEX + 1;

const OU_INDEX:usize = OPPONENT_TEBAN_INDEX + 1;
const FU_INDEX:usize = OU_INDEX + BANMEN_SIZE;
const KYOU_INDEX:usize = FU_INDEX + BANMEN_SIZE;
const KEI_INDEX:usize = KYOU_INDEX + BANMEN_SIZE;
const GIN_INDEX:usize = KEI_INDEX + BANMEN_SIZE;
const KIN_INDEX:usize = GIN_INDEX + BANMEN_SIZE;
const KAKU_INDEX:usize = KIN_INDEX + BANMEN_SIZE;
const HISHA_INDEX:usize = KAKU_INDEX + BANMEN_SIZE;
const NARIFU_INDEX:usize = HISHA_INDEX + BANMEN_SIZE;
const NARIKYOU_INDEX:usize = NARIFU_INDEX + BANMEN_SIZE;
const NARIKEI_INDEX:usize = NARIKYOU_INDEX + BANMEN_SIZE;
const NARIGIN_INDEX:usize = NARIKEI_INDEX + BANMEN_SIZE;
const NARIKAKU_INDEX:usize = NARIGIN_INDEX + BANMEN_SIZE;
const NARIHISHA_INDEX:usize = NARIKAKU_INDEX + BANMEN_SIZE;
const OPPONENT_FU_INDEX:usize = NARIHISHA_INDEX + BANMEN_SIZE;
const OPPONENT_KYOU_INDEX:usize = OPPONENT_FU_INDEX + BANMEN_SIZE;
const OPPONENT_KEI_INDEX:usize = OPPONENT_KYOU_INDEX + BANMEN_SIZE;
const OPPONENT_GIN_INDEX:usize = OPPONENT_KEI_INDEX + BANMEN_SIZE;
const OPPONENT_KIN_INDEX:usize = OPPONENT_GIN_INDEX + BANMEN_SIZE;
const OPPONENT_KAKU_INDEX:usize = OPPONENT_KIN_INDEX + BANMEN_SIZE;
const OPPONENT_HISHA_INDEX:usize = OPPONENT_KAKU_INDEX + BANMEN_SIZE;
const OPPONENT_OU_INDEX:usize = OPPONENT_HISHA_INDEX + BANMEN_SIZE;
const OPPONENT_NARIFU_INDEX:usize = OPPONENT_OU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKYOU_INDEX:usize = OPPONENT_NARIFU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKEI_INDEX:usize = OPPONENT_NARIKYOU_INDEX + BANMEN_SIZE;
const OPPONENT_NARIGIN_INDEX:usize = OPPONENT_NARIKEI_INDEX + BANMEN_SIZE;
const OPPONENT_NARIKAKU_INDEX:usize = OPPONENT_NARIGIN_INDEX + BANMEN_SIZE;
const OPPONENT_NARIHISHA_INDEX:usize = OPPONENT_NARIKAKU_INDEX + BANMEN_SIZE;

const MOCHIGOMA_FU_INDEX:usize = OPPONENT_NARIHISHA_INDEX + BANMEN_SIZE;
const MOCHIGOMA_KYOU_INDEX:usize = MOCHIGOMA_FU_INDEX + 19;
const MOCHIGOMA_KEI_INDEX:usize = MOCHIGOMA_KYOU_INDEX + 5;
const MOCHIGOMA_GIN_INDEX:usize = MOCHIGOMA_KEI_INDEX + 5;
const MOCHIGOMA_KIN_INDEX:usize = MOCHIGOMA_GIN_INDEX + 5;
const MOCHIGOMA_KAKU_INDEX:usize = MOCHIGOMA_KIN_INDEX + 5;
const MOCHIGOMA_HISHA_INDEX:usize = MOCHIGOMA_KAKU_INDEX + 3;
const OPPONENT_MOCHIGOMA_FU_INDEX:usize = MOCHIGOMA_HISHA_INDEX + 3;
const OPPONENT_MOCHIGOMA_KYOU_INDEX:usize = OPPONENT_MOCHIGOMA_FU_INDEX + 19;
const OPPONENT_MOCHIGOMA_KEI_INDEX:usize = OPPONENT_MOCHIGOMA_KYOU_INDEX + 5;
const OPPONENT_MOCHIGOMA_GIN_INDEX:usize = OPPONENT_MOCHIGOMA_KEI_INDEX + 5;
const OPPONENT_MOCHIGOMA_KIN_INDEX:usize = OPPONENT_MOCHIGOMA_GIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_KAKU_INDEX:usize = OPPONENT_MOCHIGOMA_KIN_INDEX + 5;
const OPPONENT_MOCHIGOMA_HISHA_INDEX:usize = OPPONENT_MOCHIGOMA_KAKU_INDEX + 3;

const SELF_INDEX_MAP:[usize; 7] = [
    MOCHIGOMA_FU_INDEX,
    MOCHIGOMA_KYOU_INDEX,
    MOCHIGOMA_KEI_INDEX,
    MOCHIGOMA_GIN_INDEX,
    MOCHIGOMA_KIN_INDEX,
    MOCHIGOMA_KAKU_INDEX,
    MOCHIGOMA_HISHA_INDEX
];

const OPPONENT_INDEX_MAP:[usize; 7] = [
    OPPONENT_MOCHIGOMA_FU_INDEX,
    OPPONENT_MOCHIGOMA_KYOU_INDEX,
    OPPONENT_MOCHIGOMA_KEI_INDEX,
    OPPONENT_MOCHIGOMA_GIN_INDEX,
    OPPONENT_MOCHIGOMA_KIN_INDEX,
    OPPONENT_MOCHIGOMA_KAKU_INDEX,
    OPPONENT_MOCHIGOMA_HISHA_INDEX
];
const SCALE:f32 = 1.;
pub trait BatchNeuralNetwork<U,D,P,PT,I,O>: ForwardAll<Input=I,Output=O> +
                                 BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,O>> +
                                 BatchTrain<U,D> + Persistence<U,P,PT>
                                 where U: UnitValue<U>,
                                       D: Device<U>,
                                       PT: PersistenceType {}
impl<T,U,D,P,PT,I,O> BatchNeuralNetwork<U,D,P,PT,I,O> for T
    where T: ForwardAll<Input=I,Output=O> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,O>> +
             BatchTrain<U,D> + Persistence<U,P,PT>,
             U: UnitValue<U>,
             D: Device<U>,
             PT: PersistenceType {}
pub struct EvalutorCreator {
}
impl EvalutorCreator {
    pub fn create(save_dir:String, nna_path:String, nnb_path:String,enable_shake_shake:bool)
        -> Result<Evalutor<impl ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
                         PreTrain<f32,OutStack = impl Send + Sync + 'static>>,ApplicationError> {

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nna = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2517,256>::new().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256,32>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32,1>::new().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,Tanh::new(&device),&device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l,&device)
        });

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

        let device = DeviceCpu::new()?;

        let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nnb = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2517,256>::new().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256,32>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32,1>::new().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,Tanh::new(&device),&device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l,&device)
        });

        {
            let save_dir = Path::new(&save_dir);

            let nna_path = Path::new(&nna_path);

            if save_dir.join(nna_path).exists() {
                let mut pa = BinFilePersistence::new(save_dir
                    .join(nna_path)
                )?;

                nna.load(&mut pa)?;
            }

            let nnb_path = Path::new(&nnb_path);

            if save_dir.join(nnb_path).exists() {
                let mut pb = BinFilePersistence::new(save_dir
                    .join(nnb_path)
                )?;

                nnb.load(&mut pb)?;
            }
        }

        Ok(Evalutor {
            nna:Arc::new(nna),
            nnb:Arc::new(nnb),
            enable_shake_shake:enable_shake_shake
        })
    }
}
pub struct Evalutor<M>
    where M: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
             PreTrain<f32> + Send + Sync + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    nna:Arc<M>,
    nnb:Arc<M>,
    enable_shake_shake:bool
}
impl<M> Evalutor<M>
    where M: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
             PreTrain<f32> + Send + Sync + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {

    pub fn evaluation(&self,teban:Teban,banmen:&Banmen,mc:&MochigomaCollections) -> Result<i32,ApplicationError> {
        let input = InputCreator::make_input(true, teban, &banmen, &mc);

        let ra = self.nna.forward_all(input.clone())?;
        let rb = self.nnb.forward_all(input)?;

        let (a,b) = calc_alpha_beta(self.enable_shake_shake);

        return Ok((ra[0] * a + rb[0] * b * (1 << 23) as f32) as i32)
    }
}
impl<M> Clone for Evalutor<M>
    where M: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
             PreTrain<f32> + Send + Sync + 'static,
          <M as PreTrain<f32>>::OutStack: Send + Sync + 'static {
    fn clone(&self) -> Self {
        Evalutor {
            nna:Arc::clone(&self.nna),
            nnb:Arc::clone(&self.nnb),
            enable_shake_shake:self.enable_shake_shake
        }
    }
}
pub struct Trainer<M>
    where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2517>,Arr<f32,1>> {

    nna:M,
    nnb:M,
    optimizer:SGD<f32>,
    nna_path:String,
    nnb_path:String,
    nnsavedir:String,
    packed_sfen_reader:PackedSfenReader,
    hcpe_reader:HcpeReader,
    bias_shake_shake:bool,
    similar:bool
}
pub struct TrainerCreator {
}

impl TrainerCreator {
    pub fn create(save_dir:String, nna_path:String, nnb_path:String, enable_shake_shake:bool, similar:bool)
                  -> Result<Trainer<impl BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2517>,Arr<f32,1>>>,ApplicationError> {

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

        let memory_pool = Arc::new(Mutex::new(MemoryPool::new(Alloctype::Device)?));

        let device = DeviceGpu::new(&memory_pool)?;

        let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nna = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2517,256>::new().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256,32>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32,1>::new().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,Tanh::new(&device),&device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l,&device)
        });

        let mut rnd = prelude::thread_rng();
        let rnd_base = Rc::new(RefCell::new(XorShiftRng::from_seed(rnd.gen())));

        let n1 = Normal::<f32>::new(0.0, (2f32/2517f32).sqrt()).unwrap();
        let n2 = Normal::<f32>::new(0.0, (2f32/256f32).sqrt()).unwrap();
        let n3 = Normal::<f32>::new(0.0, 1f32/32f32.sqrt()).unwrap();

        let device = DeviceGpu::new(&memory_pool)?;

        let net:InputLayer<f32,Arr<f32,2517>,_> = InputLayer::new();

        let rnd = rnd_base.clone();

        let mut nnb = net.try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<2517,256>::new().build(l,&device, move || n1.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<256,32>::new().build(l,&device, move || n2.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).try_add_layer(|l| {
            let rnd = rnd.clone();
            LinearLayerBuilder::<32,1>::new().build(l,&device, move || n3.sample(&mut rnd.borrow_mut().deref_mut()), || 0.)
        })?.try_add_layer(|l| {
            BatchNormalizationLayerBuilder::new().build(l,&device)
        })?.add_layer(|l| {
            ActivationLayer::new(l,Tanh::new(&device),&device)
        }).add_layer_train(|l| {
            LinearOutputLayer::new(l,&device)
        });

        {
            let save_dir = Path::new(&save_dir);

            let nna_path = Path::new(&nna_path);

            if save_dir.join(nna_path).exists() {
                let mut pa = BinFilePersistence::new(save_dir
                    .join(nna_path)
                )?;

                nna.load(&mut pa)?;
            }

            let nnb_path = Path::new(&nnb_path);

            if save_dir.join(nnb_path).exists() {
                let mut pb = BinFilePersistence::new(save_dir
                    .join(nnb_path)
                )?;

                nnb.load(&mut pb)?;
            }
        }

        Ok(Trainer {
            nna:nna,
            nnb:nnb,
            optimizer:SGD::new(0.005),
            nna_path: nna_path,
            nnb_path: nnb_path,
            nnsavedir: save_dir,
            packed_sfen_reader:PackedSfenReader::new(),
            hcpe_reader:HcpeReader::new(),
            bias_shake_shake:enable_shake_shake,
            similar:similar
        })
    }
}
pub fn calc_alpha_beta(bias_shake_shake:bool) -> (f32,f32) {
    if bias_shake_shake {
        let mut rnd = rand::thread_rng();
        let mut rnd = XorShiftRng::from_seed(rnd.gen());

        let a = rnd.gen();
        let b = 1f32 - a ;

        (a,b)
    } else {
        (0.5f32,0.5f32)
    }
}
impl<M> Trainer<M> where M: BatchNeuralNetwork<f32,DeviceGpu<f32>,BinFilePersistence<f32>,Linear,Arr<f32,2517>,Arr<f32,1>> {

    pub fn make_learn_input(mut acc:((Vec<Arr<f32,1>>,Vec<Arr<f32,2517>>),(Vec<Arr<f32,1>>,Vec<Arr<f32,2517>>)),
                            (t,input,a,b):(f32,Arr<f32,2517>,f32,f32))
        -> ((Vec<Arr<f32,1>>,Vec<Arr<f32,2517>>),(Vec<Arr<f32,1>>,Vec<Arr<f32,2517>>)) {
        let mut ans = Arr::<f32, 1>::new();
        ans[0] = t * a;

        (acc.0).0.push(ans);
        (acc.0).1.push(input.clone() * SCALE);

        let mut ans = Arr::<f32, 1>::new();
        ans[0] = t * b;

        (acc.1).0.push(ans);
        (acc.1).1.push(input * SCALE);

        acc
    }

    pub fn learning_by_training_csa<'a>(&mut self,
                                        last_teban:Teban,
                                        history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
                                        s:&GameEndState,
                                        _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
                                        sente_rate: f32,
                                        gote_rate: f32)
                                        -> Result<(f32,f32),ApplicationError> {

        let lossf = Mse::new();

        let mut teban = last_teban;
        let bias_shake_shake = self.bias_shake_shake;
        let similar = self.similar;

        let batch = history.iter().rev().map(move |(banmen,mc,_,_)| {
            let (a, b) = calc_alpha_beta(bias_shake_shake);

            let input = InputCreator::make_input(true, teban, banmen, mc);

            let t = if similar {
                1f32
            } else {
                match s {
                    GameEndState::Win if teban == last_teban  && teban == Teban::Sente => {
                        sente_rate
                    }
                    GameEndState::Win if teban == last_teban => {
                        gote_rate
                    },
                    GameEndState::Lose if teban == Teban::Gote => {
                        -gote_rate
                    },
                    GameEndState::Lose => {
                        -sente_rate
                    },
                    _ => 0f32
                }
            };

            teban = teban.opposite();

            (t,input,a,b)
        }).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())), Self::make_learn_input);

        let ma = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
        let mb = self.nna.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

        self.save()?;

        Ok((ma,mb))
    }

    pub fn test_by_csa(&mut self,
                       teban:Teban,
                       kyokumen:&(Banmen,MochigomaCollections,u64,u64))
                       -> Result<f32,ApplicationError> {
        let (banmen,mc,_,_) = kyokumen;

        let input = InputCreator::make_input(true, teban, &banmen, &mc);

        let ra = self.nna.forward_all(input.clone() * SCALE)?;
        let rb = self.nnb.forward_all(input * SCALE)?;

        Ok(ra[0] * 0.5 + rb[0] * 0.5)
    }

    pub fn learning_by_packed_sfens<'a>(&mut self,
                                        packed_sfens:Vec<Vec<u8>>,
                                        _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
                                        -> Result<(f32,f32),ApplicationError> {

        let lossf = Mse::new();
        let bias_shake_shake = self.bias_shake_shake;
        let similar = self.similar;

        let mut sfens_with_extended = Vec::with_capacity(packed_sfens.len());

        for entry in packed_sfens.into_iter() {
            let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
                value: _,
                best_move: _,
                end_ply: _,
                game_result
            }) = self.packed_sfen_reader.read_sfen_with_extended(entry)?;

            sfens_with_extended.push((teban,banmen,mc,game_result));
        }

        let (sente_win_count,gote_win_count) = sfens_with_extended.iter()
            .map(|(teban,_,_,es)| {
                let (s,g) = match (es,teban) {
                    (&GameEndState::Draw,_) => {
                        (0,0)
                    },
                    (&GameEndState::Win,&Teban::Sente) | (&GameEndState::Lose,&Teban::Gote) => {
                        (1,0)
                    },
                    _ => {
                        (0,1)
                    }
                };

                (s,g)
            }).fold((0,0), |acc,(s,g)| {
                (acc.0 + s, acc.1 + g)
            });

        let (sente_rate,gote_rate) = if sente_win_count >= gote_win_count {
            (gote_win_count as f32 / sente_win_count as f32,1.)
        } else {
            (1.,sente_win_count as f32 / gote_win_count as f32)
        };

        let batch = sfens_with_extended.iter()
            .map(|(teban,banmen,mc,es)| {
                let (a,b) = calc_alpha_beta(bias_shake_shake);

                let teban = *teban;

                let input = InputCreator::make_input(true, teban, banmen, mc);

                let t = if similar {
                    1f32
                } else {
                    match es {
                        GameEndState::Win if teban == Teban::Sente => {
                            sente_rate
                        },
                        GameEndState::Win => {
                            gote_rate
                        },
                        GameEndState::Lose if teban == Teban::Sente => {
                            -gote_rate
                        },
                        GameEndState::Lose => {
                            -sente_rate
                        },
                        _ => 0f32
                    }
                };

                (t,input,a,b)
        }).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())),  Self::make_learn_input);

        let ma = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
        let mb = self.nnb.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

        Ok((ma,mb))
    }

    pub fn test_by_packed_sfens(&mut self,
                                packed_sfen:Vec<u8>)
                                -> Result<(GameEndState,f32),ApplicationError> {
        let ((teban,banmen,mc),yaneuraou::haffman_code::ExtendFields {
            value: _,
            best_move: _,
            end_ply: _,
            game_result
        }) = self.packed_sfen_reader.read_sfen_with_extended(packed_sfen)?;

        let input = InputCreator::make_input(true, teban, &banmen, &mc);

        let ra = self.nna.forward_all(input.clone() * SCALE)?;
        let rb = self.nnb.forward_all(input * SCALE)?;

        Ok((game_result,ra[0] * 0.5 + rb[0] * 0.5))
    }

    pub fn learning_by_hcpe<'a>(&mut self,
                                hcpes:Vec<Vec<u8>>,
                                _:&'a Mutex<EventQueue<UserEvent,UserEventKind>>)
                                -> Result<(f32,f32),ApplicationError> {

        let lossf = Mse::new();
        let bias_shake_shake = self.bias_shake_shake;
        let similar = self.similar;

        let mut sfens_with_extended = Vec::with_capacity(hcpes.len());

        for entry in hcpes.into_iter() {
            let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
                eval: _,
                best_move: _,
                game_result
            }) = self.hcpe_reader.read_sfen_with_extended(entry)?;

            sfens_with_extended.push((teban, banmen, mc, game_result));
        }

        let (sente_win_count,gote_win_count) = sfens_with_extended.iter().map(|(_,_,_,es)| {
            match es {
                GameResult::Draw => (0,0),
                GameResult::SenteWin => (1,0),
                _ => (0,1)
            }
        }).fold((0,0), |acc,(s,g)| {
            (acc.0 + s, acc.1 + g)
        });

        let (sente_rate,gote_rate) = if sente_win_count >= gote_win_count {
            (gote_win_count as f32 / sente_win_count as f32,1.)
        } else {
            (1.,sente_win_count as f32 / gote_win_count as f32)
        };

        let batch = sfens_with_extended.iter()
            .map(|(teban,banmen,mc,es)| {
                let (a,b) = calc_alpha_beta(bias_shake_shake);

                let teban = *teban;

                let input = InputCreator::make_input(true,teban, banmen, mc);

                let (rate,es) = match (es,teban) {
                    (GameResult::Draw,_) => {
                        (1.,GameEndState::Draw)
                    },
                    (GameResult::SenteWin,Teban::Sente) => {
                        (sente_rate,GameEndState::Win)
                    },
                    (GameResult::GoteWin,Teban::Gote) => {
                        (gote_rate,GameEndState::Win)
                    },
                    (GameResult::SenteWin,Teban::Gote) => {
                        (sente_rate,GameEndState::Lose)
                    },
                    (GameResult::GoteWin,Teban::Sente) => {
                        (gote_rate,GameEndState::Lose)
                    }
                };

                let t = if similar {
                    1f32
                } else {
                    match es {
                        GameEndState::Win => {
                            rate
                        }
                        GameEndState::Lose => {
                            -rate
                        },
                        _ => 0f32
                    }
                };

                (t,input,a,b)
            }).fold(((Vec::new(),Vec::new()),(Vec::new(),Vec::new())), Self::make_learn_input);

        let ma = self.nna.batch_train((batch.0).0.into(),(batch.0).1.into(),&mut self.optimizer,&lossf)?;
        let mb = self.nnb.batch_train((batch.1).0.into(),(batch.1).1.into(),&mut self.optimizer,&lossf)?;

        Ok((ma,mb))
    }

    pub fn test_by_packed_hcpe(&mut self,
                               hcpe:Vec<u8>)
                               -> Result<(GameEndState,f32),ApplicationError> {
        let ((teban,banmen,mc),hcpe::haffman_code::ExtendFields {
            eval: _,
            best_move: _,
            game_result
        }) = self.hcpe_reader.read_sfen_with_extended(hcpe)?;

        let input = InputCreator::make_input(true, teban, &banmen, &mc);

        let ra = self.nna.forward_all(input.clone() * SCALE)?;
        let rb = self.nnb.forward_all(input * SCALE)?;

        let s = match game_result {
            GameResult::SenteWin if teban == Teban::Sente => {
                GameEndState::Win
            },
            GameResult::SenteWin => {
                GameEndState::Lose
            },
            GameResult::GoteWin if teban == Teban::Gote => {
                GameEndState::Win
            },
            GameResult::GoteWin => {
                GameEndState::Lose
            },
            _ => GameEndState::Draw
        };

        Ok((s,ra[0] * 0.5 + rb[0] * 0.5))
    }

    pub fn save(&mut self) -> Result<(),ApplicationError> {
        let tmp_nna_path = Path::new(&self.nnsavedir).join(&format!("{}.{}", &self.nna_path, "tmp"));
        let tmp_nnb_path = Path::new(&self.nnsavedir).join(&format!("{}.{}", &self.nnb_path, "tmp"));

        let mut pa = BinFilePersistence::new(tmp_nna_path.as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        let mut pb = BinFilePersistence::new(tmp_nnb_path.as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        self.nna.save(&mut pa)?;
        self.nnb.save(&mut pb)?;

        pa.save(&tmp_nna_path)?;
        pb.save(&tmp_nnb_path)?;

        fs::rename(Path::new(&tmp_nna_path),Path::new(&self.nnsavedir).join(&self.nna_path).as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        fs::rename(Path::new(&tmp_nnb_path),Path::new(&self.nnsavedir).join(&self.nnb_path).as_os_str()
            .to_str().ok_or(ApplicationError::InvalidSettingError(
            String::from("ニューラルネットワークのモデルのパスの処理時にエラーが発生しました。")
        ))?)?;

        Ok(())
    }
}
pub struct InputCreator;

impl InputCreator {
    pub fn make_input(is_self:bool,t:Teban,b:&Banmen,mc:&MochigomaCollections) -> Arr<f32,2517> {
        let mut inputs = Arr::new();

        let index = if is_self {
            SELF_TEBAN_INDEX
        } else {
            OPPONENT_TEBAN_INDEX
        };

        inputs[index] = 1f32;

        match b {
            &Banmen(ref kinds) => {
                for y in 0..9 {
                    for x in 0..9 {
                        let kind = kinds[y][x];

                        if kind != KomaKind::Blank {
                            let index = InputCreator::input_index_of_banmen(t,kind,x as u32,y as u32).unwrap();

                            inputs[index] = 1f32;
                        }
                    }
                }
            }
        }

        let ms = Mochigoma::new();
        let mg = Mochigoma::new();
        let (ms,mg) = match mc {
            &MochigomaCollections::Pair(ref ms,ref mg) => (ms,mg),
            &MochigomaCollections::Empty => (&ms,&mg),
        };

        let (ms,mg) = match t {
            Teban::Sente => (ms,mg),
            Teban::Gote => (mg,ms),
        };

        for &k in &MOCHIGOMA_KINDS {
            let c = ms.get(k);

            for i in 0..c {
                let offset = SELF_INDEX_MAP[k as usize];

                let offset = offset as usize;

                inputs[offset + i as usize] = 1f32;
            }

            let c = mg.get(k);

            for i in 0..c {
                let offset = OPPONENT_INDEX_MAP[k as usize];

                let offset = offset as usize;

                inputs[offset + i as usize] = 1f32;
            }
        }
        inputs
    }

    #[inline]
    fn input_index_of_banmen(teban:Teban,kind:KomaKind,x:u32,y:u32) -> Result<usize,ApplicationError> {
        const SENTE_INDEX_MAP:[usize; 28] = [
            FU_INDEX,
            KYOU_INDEX,
            KEI_INDEX,
            GIN_INDEX,
            KIN_INDEX,
            KAKU_INDEX,
            HISHA_INDEX,
            OU_INDEX,
            NARIFU_INDEX,
            NARIKYOU_INDEX,
            NARIKEI_INDEX,
            NARIGIN_INDEX,
            NARIKAKU_INDEX,
            NARIHISHA_INDEX,
            OPPONENT_FU_INDEX,
            OPPONENT_KYOU_INDEX,
            OPPONENT_KEI_INDEX,
            OPPONENT_GIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KAKU_INDEX,
            OPPONENT_HISHA_INDEX,
            OPPONENT_OU_INDEX,
            OPPONENT_NARIFU_INDEX,
            OPPONENT_NARIKYOU_INDEX,
            OPPONENT_NARIKEI_INDEX,
            OPPONENT_NARIGIN_INDEX,
            OPPONENT_NARIKAKU_INDEX,
            OPPONENT_NARIHISHA_INDEX
        ];

        const GOTE_INDEX_MAP:[usize; 28] = [
            OPPONENT_FU_INDEX,
            OPPONENT_KYOU_INDEX,
            OPPONENT_KEI_INDEX,
            OPPONENT_GIN_INDEX,
            OPPONENT_KIN_INDEX,
            OPPONENT_KAKU_INDEX,
            OPPONENT_HISHA_INDEX,
            OPPONENT_OU_INDEX,
            OPPONENT_NARIFU_INDEX,
            OPPONENT_NARIKYOU_INDEX,
            OPPONENT_NARIKEI_INDEX,
            OPPONENT_NARIGIN_INDEX,
            OPPONENT_NARIKAKU_INDEX,
            OPPONENT_NARIHISHA_INDEX,
            FU_INDEX,
            KYOU_INDEX,
            KEI_INDEX,
            GIN_INDEX,
            KIN_INDEX,
            KAKU_INDEX,
            HISHA_INDEX,
            OU_INDEX,
            NARIFU_INDEX,
            NARIKYOU_INDEX,
            NARIKEI_INDEX,
            NARIGIN_INDEX,
            NARIKAKU_INDEX,
            NARIHISHA_INDEX
        ];

        let index = match teban {
            Teban::Sente | Teban::Gote if kind == KomaKind::Blank => {
                return Err(ApplicationError::LogicError(
                    String::from(
                        "Calculation of index of difference input data of neural network failed. (KomaKind is 'Blank')"
                    )));
            },
            Teban::Sente => {
                SENTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
            },
            Teban::Gote => {
                let (x,y) = (8-x,8-y);

                GOTE_INDEX_MAP[kind as usize] + y as usize * 9 + x as usize
            }
        };

        Ok(index as usize)
    }
}