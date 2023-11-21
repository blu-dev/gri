use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    time::Instant,
};

pub(crate) struct PollingMetric {
    pub id: String,
    pub timestamp: Instant,
}

pub struct PollingMetrics {
    sender: Sender<PollingMetric>,
    receiver: Receiver<PollingMetric>,
    history: HashMap<String, Vec<Instant>>,
}

pub(crate) struct InputSnapshotMetric(pub Instant);

pub struct InputSnapshotMetrics {
    sender: Sender<InputSnapshotMetric>,
    receiver: Receiver<InputSnapshotMetric>,
    history: Vec<Instant>,
}

impl PollingMetrics {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender,
            receiver,
            history: HashMap::new(),
        }
    }

    pub fn process_updates(&mut self) {
        loop {
            match self.receiver.try_recv() {
                Ok(metric) => self
                    .history
                    .entry(metric.id)
                    .or_default()
                    .push(metric.timestamp),
                Err(TryRecvError::Disconnected) => {
                    panic!("PollingMetrics sender has disconnected");
                }
                Err(TryRecvError::Empty) => break,
            }
        }
    }

    pub fn enumerate_polled_devices<'a>(&'a self) -> impl Iterator<Item = &'a str> {
        self.history.keys().map(|key| key.as_str())
    }

    pub fn history_for_device(&self, device_id: impl AsRef<str>) -> Option<&[Instant]> {
        self.history
            .get(device_id.as_ref())
            .map(|history| history.as_slice())
    }

    pub(crate) fn new_sender(&self) -> Sender<PollingMetric> {
        self.sender.clone()
    }
}

impl InputSnapshotMetrics {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            sender: tx,
            receiver: rx,
            history: vec![],
        }
    }

    pub fn process_updates(&mut self) {
        loop {
            match self.receiver.try_recv() {
                Ok(metric) => self.history.push(metric.0),
                Err(TryRecvError::Disconnected) => {
                    panic!("InputSnapshotMetrics sender has disconnected");
                }
                Err(TryRecvError::Empty) => break,
            }
        }
    }

    pub fn history(&self) -> &[Instant] {
        &self.history
    }

    pub(crate) fn new_sender(&self) -> Sender<InputSnapshotMetric> {
        self.sender.clone()
    }
}

pub struct Metrics {
    polling: PollingMetrics,
    input_snapshots: InputSnapshotMetrics,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            polling: PollingMetrics::new(),
            input_snapshots: InputSnapshotMetrics::new(),
        }
    }

    pub fn update(&mut self) {
        self.polling.process_updates();
        self.input_snapshots.process_updates();
    }

    pub fn polling(&self) -> &PollingMetrics {
        &self.polling
    }

    pub fn input_snapshots(&self) -> &InputSnapshotMetrics {
        &self.input_snapshots
    }
}
