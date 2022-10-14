use std::{error, fmt, io};
use std::collections::VecDeque;
use std::num::{ParseFloatError, ParseIntError};
use std::sync::mpsc::{RecvError, Sender, SendError};
use std::sync::{MutexGuard, PoisonError};
use concurrent_queue::{PopError, PushError};
use csaparser::error::CsaParserError;
use nncombinator::error::{ConfigReadError, CudaError, DeviceError, EvaluateError, PersistenceError, TrainingError};
use packedsfen::error::ReadError;
use usiagent::error::{EventDispatchError, PlayerError, SfenStringConvertError};
use usiagent::event::{EventQueue, SystemEvent, SystemEventKind};
use usiagent::rule::LegalMove;
use crate::nn::{BatchItem, Message};

#[derive(Debug)]
pub enum ApplicationError {
    StartupError(String),
    SfenStringConvertError(SfenStringConvertError),
    EventDispatchError(String),
    IOError(io::Error),
    ParseIntError(ParseIntError),
    ParseFloatError(ParseFloatError),
    ParseSfenError(ReadError),
    AgentRunningError(String),
    CsaParserError(CsaParserError),
    LogicError(String),
    LearningError(String),
    SerdeError(toml::ser::Error),
    ConfigReadError(ConfigReadError),
    InvalidSettingError(String),
    TrainingError(TrainingError),
    EvaluateError(EvaluateError),
    DeviceError(DeviceError),
    PersistenceError(PersistenceError),
    CudaError(CudaError),
    RecvError(RecvError),
    NNSendError(SendError<Message>),
    ResultSendError(SendError<(LegalMove,i32)>),
    AllResultSendError(SendError<Vec<(f32,f32)>>),
    EndTransactionSendError(SendError<()>),
    PoisonError(String),
    TransactionPushError(PushError<Sender<()>>),
    BatchItemPushError(PushError<BatchItem>),
    ConcurrentQueuePopError(PopError),
}
impl fmt::Display for ApplicationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ApplicationError::StartupError(ref s) => write!(f, "{}",s),
            ApplicationError::SfenStringConvertError(ref e) => write!(f, "{}",e),
            ApplicationError::EventDispatchError(ref s) => write!(f,"{}",s),
            ApplicationError::IOError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseIntError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseFloatError(ref e) => write!(f, "{}",e),
            ApplicationError::ParseSfenError(ref e) => write!(f,"{}",e),
            ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
            ApplicationError::CsaParserError(ref e) => write!(f, "{}",e),
            ApplicationError::LogicError(ref s) => write!(f,"{}",s),
            ApplicationError::LearningError(ref s) => write!(f,"{}",s),
            ApplicationError::SerdeError(ref e) => write!(f,"{}",e),
            ApplicationError::InvalidSettingError(ref s) => write!(f,"{}",s),
            ApplicationError::ConfigReadError(ref e) => write!(f,"{}",e),
            ApplicationError::TrainingError(ref e) => write!(f,"{}",e),
            ApplicationError::EvaluateError(ref e) => write!(f,"{}",e),
            ApplicationError::DeviceError(ref e) => write!(f,"{}",e),
            ApplicationError::PersistenceError(ref e) => write!(f,"{}",e),
            ApplicationError::CudaError(ref e) => write!(f, "An error occurred in the process of cuda. ({})",e),
            ApplicationError::RecvError(ref e) => write!(f, "{}",e),
            ApplicationError::NNSendError(ref e) => write!(f,"{}",e),
            ApplicationError::ResultSendError(ref e) => write!(f,"{}",e),
            ApplicationError::AllResultSendError(ref e) => write!(f,"{}",e),
            ApplicationError::EndTransactionSendError(ref e) => write!(f,"{}",e),
            ApplicationError::PoisonError(ref s) => write!(f,"{}",s),
            ApplicationError::TransactionPushError(ref e) => write!(f,"{}",e),
            ApplicationError::BatchItemPushError(ref e) => write!(f,"{}",e),
            ApplicationError::ConcurrentQueuePopError(ref e) => write!(f,"{}",e)
        }
    }
}
impl error::Error for ApplicationError {
    fn description(&self) -> &str {
        match *self {
            ApplicationError::StartupError(_) => "Startup Error.",
            ApplicationError::SfenStringConvertError(_) => "An error occurred during conversion to sfen string.",
            ApplicationError::EventDispatchError(_) => "An error occurred while processing the event.",
            ApplicationError::IOError(_) => "IO Error.",
            ApplicationError::ParseIntError(_) => "An error occurred parsing the integer string.",
            ApplicationError::ParseFloatError(_) => "An error occurred parsing the float string.",
            ApplicationError::ParseSfenError(_) => "An error occurred parsing the packed sfen.",
            ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
            ApplicationError::CsaParserError(_) => "An error occurred parsing the csa file.",
            ApplicationError::LogicError(_) => "Logic error.",
            ApplicationError::LearningError(_) => "An error occurred while learning the neural network.",
            ApplicationError::SerdeError(_) => "An error occurred during serialization or deserialization.",
            ApplicationError::ConfigReadError(_) => "An error occurred while loading the neural network model.",
            ApplicationError::InvalidSettingError(_) => "Invalid setting.",
            ApplicationError::TrainingError(_) => "An error occurred while training the model.",
            ApplicationError::EvaluateError(_) => "An error occurred when running the neural network.",
            ApplicationError::DeviceError(_) => "An error occurred during device initialization.",
            ApplicationError::PersistenceError(_) => "An error occurred when saving model information.",
            ApplicationError::CudaError(_) => "An error occurred in the process of cuda.",
            ApplicationError::RecvError(_) => "An error occurred while receiving the message.",
            ApplicationError::NNSendError(_) => "An error occurred in the communication process with the neural network thread.",
            ApplicationError::ResultSendError(_) => "An error occurred in the process of sending the results of the neural network calculation.",
            ApplicationError::AllResultSendError(_) => "An error occurred in the process of sending the all results of the neural network calculation.",
            ApplicationError::EndTransactionSendError(_) => "An error occurred when sending the transaction termination notification.",
            ApplicationError::PoisonError(_) => "panic occurred during thread execution.",
            ApplicationError::TransactionPushError(_) => "An error occurred in adding the transaction to the queue.",
            ApplicationError::BatchItemPushError(_) => "An error occurred while adding a batch item to the queue.",
            ApplicationError::ConcurrentQueuePopError(_) => "Error retrieving element from concurrent queue.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ApplicationError::StartupError(_) => None,
            ApplicationError::SfenStringConvertError(ref e) => Some(e),
            ApplicationError::EventDispatchError(_) => None,
            ApplicationError::IOError(ref e) => Some(e),
            ApplicationError::ParseIntError(ref e) => Some(e),
            ApplicationError::ParseFloatError(ref e) => Some(e),
            ApplicationError::ParseSfenError(ref e) => Some(e),
            ApplicationError::AgentRunningError(_) => None,
            ApplicationError::CsaParserError(ref e) => Some(e),
            ApplicationError::LogicError(_) => None,
            ApplicationError::LearningError(_) => None,
            ApplicationError::SerdeError(ref e) => Some(e),
            ApplicationError::ConfigReadError(ref e) => Some(e),
            ApplicationError::InvalidSettingError(_) => None,
            ApplicationError::TrainingError(ref e) => Some(e),
            ApplicationError::EvaluateError(ref e) => Some(e),
            ApplicationError::DeviceError(ref e) => Some(e),
            ApplicationError::PersistenceError(ref e) => Some(e),
            ApplicationError::CudaError(_) => None,
            ApplicationError::RecvError(ref e) => Some(e),
            ApplicationError::NNSendError(ref e) => Some(e),
            ApplicationError::ResultSendError(ref e) =>  Some(e),
            ApplicationError::AllResultSendError(ref e) => Some(e),
            ApplicationError::EndTransactionSendError(ref e) => Some(e),
            ApplicationError::PoisonError(_) => None,
            ApplicationError::TransactionPushError(ref e) => Some(e),
            ApplicationError::BatchItemPushError(ref e) => Some(e),
            ApplicationError::ConcurrentQueuePopError(ref e) => Some(e),
        }
    }
}
impl PlayerError for ApplicationError {}
impl From<io::Error> for ApplicationError {
    fn from(err: io::Error) -> ApplicationError {
        ApplicationError::IOError(err)
    }
}
impl From<ParseIntError> for ApplicationError {
    fn from(err: ParseIntError) -> ApplicationError {
        ApplicationError::ParseIntError(err)
    }
}
impl From<ParseFloatError> for ApplicationError {
    fn from(err: ParseFloatError) -> ApplicationError {
        ApplicationError::ParseFloatError(err)
    }
}
impl From<ReadError> for ApplicationError {
    fn from(err: ReadError) -> ApplicationError {
        ApplicationError::ParseSfenError(err)
    }
}
impl From<SfenStringConvertError> for ApplicationError {
    fn from(err: SfenStringConvertError) -> ApplicationError {
        ApplicationError::SfenStringConvertError(err)
    }
}
impl<'a> From<EventDispatchError<'a,EventQueue<SystemEvent,SystemEventKind>,SystemEvent,ApplicationError>>
for ApplicationError {
    fn from(err: EventDispatchError<'a, EventQueue<SystemEvent, SystemEventKind>, SystemEvent, ApplicationError>)
            -> ApplicationError {
        ApplicationError::EventDispatchError(format!("{}",err))
    }
}
impl From<CsaParserError> for ApplicationError {
    fn from(err: CsaParserError) -> ApplicationError {
        ApplicationError::CsaParserError(err)
    }
}
impl From<toml::ser::Error> for ApplicationError {
    fn from(err: toml::ser::Error) -> ApplicationError {
        ApplicationError::SerdeError(err)
    }
}
impl From<ConfigReadError> for ApplicationError {
    fn from(err: ConfigReadError) -> ApplicationError {
        ApplicationError::ConfigReadError(err)
    }
}
impl From<TrainingError> for ApplicationError {
    fn from(err: TrainingError) -> ApplicationError {
        ApplicationError::TrainingError(err)
    }
}
impl From<EvaluateError> for ApplicationError {
    fn from(err: EvaluateError) -> ApplicationError {
        ApplicationError::EvaluateError(err)
    }
}
impl From<DeviceError> for ApplicationError {
    fn from(err: DeviceError) -> ApplicationError {
        ApplicationError::DeviceError(err)
    }
}
impl From<PersistenceError> for ApplicationError {
    fn from(err: PersistenceError) -> ApplicationError {
        ApplicationError::PersistenceError(err)
    }
}
impl From<CudaError> for ApplicationError {
    fn from(err: CudaError) -> ApplicationError {
        ApplicationError::CudaError(err)
    }
}
impl From<RecvError> for ApplicationError {
    fn from(err: RecvError) -> ApplicationError {
        ApplicationError::RecvError(err)
    }
}
impl From<SendError<Message>> for ApplicationError {
    fn from(err: SendError<Message>) -> ApplicationError {
        ApplicationError::NNSendError(err)
    }
}
impl From<SendError<(LegalMove,i32)>> for ApplicationError {
    fn from(err: SendError<(LegalMove,i32)>) -> ApplicationError {
        ApplicationError::ResultSendError(err)
    }
}
impl From<SendError<Vec<(f32,f32)>>> for ApplicationError {
    fn from(err: SendError<Vec<(f32,f32)>>) -> ApplicationError {
        ApplicationError::AllResultSendError(err)
    }
}
impl From<SendError<()>> for ApplicationError {
    fn from(err: SendError<()>) -> ApplicationError {
        ApplicationError::EndTransactionSendError(err)
    }
}
impl From<PoisonError<MutexGuard<'_, VecDeque<std::sync::mpsc::Sender<()>>>>> for ApplicationError {
    fn from(err: PoisonError<MutexGuard<'_, VecDeque<std::sync::mpsc::Sender<()>>>>) -> ApplicationError {
        ApplicationError::PoisonError(format!("{}",err))
    }
}
impl From<PushError<Sender<()>>> for ApplicationError {
    fn from(err: PushError<Sender<()>>) -> ApplicationError {
        ApplicationError::TransactionPushError(err)
    }
}
impl From<PushError<BatchItem>> for ApplicationError {
    fn from(err: PushError<BatchItem>) -> ApplicationError {
        ApplicationError::BatchItemPushError(err)
    }
}
impl From<PopError> for ApplicationError {
    fn from(err: PopError) -> ApplicationError {
        ApplicationError::ConcurrentQueuePopError(err)
    }
}
#[derive(Debug)]
pub enum EvaluationError {
    InternalError(ApplicationError),
    Timeout,
}
impl From<ApplicationError> for EvaluationError {
    fn from(err: ApplicationError) -> EvaluationError {
        EvaluationError::InternalError(err)
    }
}
