use std::{error, fmt, io};
use std::num::{ParseFloatError, ParseIntError};
use csaparser::error::CsaParserError;
use nncombinator::error::{ConfigReadError, CudaError, DeviceError, EvaluateError, PersistenceError, TrainingError};
use usiagent::error::{EventDispatchError, KifuWriteError, SfenStringConvertError};
use usiagent::event::{EventQueue, SystemEvent, SystemEventKind};

#[derive(Debug)]
pub enum ApplicationError {
    StartupError(String),
    SfenStringConvertError(SfenStringConvertError),
    EventDispatchError(String),
    IOError(io::Error),
    ParseIntError(ParseIntError),
    ParseFloatError(ParseFloatError),
    AgentRunningError(String),
    CsaParserError(CsaParserError),
    LogicError(String),
    LearningError(String),
    KifuWriteError(KifuWriteError),
    SerdeError(toml::ser::Error),
    ConfigReadError(ConfigReadError),
    TrainingError(TrainingError),
    EvaluateError(EvaluateError),
    DeviceError(DeviceError),
    PersistenceError(PersistenceError),
    CudaError(CudaError),
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
            ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
            ApplicationError::CsaParserError(ref e) => write!(f, "{}",e),
            ApplicationError::LogicError(ref s) => write!(f,"{}",s),
            ApplicationError::LearningError(ref s) => write!(f,"{}",s),
            ApplicationError::KifuWriteError(ref s) => write!(f,"{}",s),
            ApplicationError::SerdeError(ref e) => write!(f,"{}",e),
            ApplicationError::ConfigReadError(ref e) => write!(f,"{}",e),
            ApplicationError::TrainingError(ref e) => write!(f,"{}",e),
            ApplicationError::EvaluateError(ref e) => write!(f,"{}",e),
            ApplicationError::DeviceError(ref e) => write!(f,"{}",e),
            ApplicationError::PersistenceError(ref e) => write!(f,"{}",e),
            ApplicationError::CudaError(ref e) => write!(f, "An error occurred in the process of cuda. ({})",e),
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
            ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
            ApplicationError::CsaParserError(_) => "An error occurred parsing the csa file.",
            ApplicationError::LogicError(_) => "Logic error.",
            ApplicationError::LearningError(_) => "An error occurred while learning the neural network.",
            ApplicationError::KifuWriteError(_) => "An error occurred when recording kifu or initialize KifuWriter.",
            ApplicationError::SerdeError(_) => "An error occurred during serialization or deserialization.",
            ApplicationError::ConfigReadError(_) => "An error occurred while loading the neural network model.",
            ApplicationError::TrainingError(_) => "An error occurred while training the model.",
            ApplicationError::EvaluateError(_) => "An error occurred when running the neural network.",
            ApplicationError::DeviceError(_) => "An error occurred during device initialization.",
            ApplicationError::PersistenceError(_) => "An error occurred when saving model information.",
            ApplicationError::CudaError(_) => "An error occurred in the process of cuda.",
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
            ApplicationError::AgentRunningError(_) => None,
            ApplicationError::CsaParserError(ref e) => Some(e),
            ApplicationError::LogicError(_) => None,
            ApplicationError::LearningError(_) => None,
            ApplicationError::KifuWriteError(ref e) => Some(e),
            ApplicationError::SerdeError(ref e) => Some(e),
            ApplicationError::ConfigReadError(ref e) => Some(e),
            ApplicationError::TrainingError(ref e) => Some(e),
            ApplicationError::EvaluateError(ref e) => Some(e),
            ApplicationError::DeviceError(ref e) => Some(e),
            ApplicationError::PersistenceError(ref e) => Some(e),
            ApplicationError::CudaError(_) => None
        }
    }
}
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
impl From<SfenStringConvertError> for ApplicationError {
    fn from(err: SfenStringConvertError) -> ApplicationError {
        ApplicationError::SfenStringConvertError(err)
    }
}
impl From<KifuWriteError> for ApplicationError {
    fn from(err: KifuWriteError) -> ApplicationError {
        ApplicationError::KifuWriteError(err)
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
