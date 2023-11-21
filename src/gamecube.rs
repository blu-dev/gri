use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        mpsc::{self, Receiver, Sender, TryRecvError},
        Arc, Weak,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use bitflags::bitflags;
use rusb::{Context, Device, DeviceHandle, Hotplug, HotplugBuilder, Registration, UsbContext};

use crate::metrics::{InputSnapshotMetric, Metrics, PollingMetric};

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ControllerKind {
    Disconnected = 0,
    Normal = 1,
    Wavebird = 2,
    Invalid = 3,
}

impl From<u8> for ControllerKind {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Disconnected,
            1 => Self::Normal,
            2 => Self::Wavebird,
            _ => Self::Invalid,
        }
    }
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct GamecubeButtons: u16 {
        const A = 1 << 0;
        const B = 1 << 1;
        const X = 1 << 2;
        const Y = 1 << 3;
        const D_LEFT = 1 << 4;
        const D_RIGHT = 1 << 5;
        const D_DOWN = 1 << 6;
        const D_UP = 1 << 7;
        const START = 1 << 8;
        const Z = 1 << 9;
        const R = 1 << 10;
        const L = 1 << 11;
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct TriggerAxis {
    value: u8,
    ground: u8,
    shallow_deadzone: Option<u8>,
    deep_deadzone: Option<u8>,
}

impl Debug for TriggerAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.sign_plus() {
            return f
                .debug_struct("TriggerAxis")
                .field("value", &self.value)
                .field("ground", &self.ground)
                .field("shallow_deadzone", &self.shallow_deadzone)
                .field("deep_deadzone", &self.deep_deadzone)
                .finish();
        }

        let raw = self.normalized_raw();
        let clamped = self.normalized_clamped();
        let clamped_dz = self.normalized_deadzone();
        let clamped_intp = self.normalized_deadzone_reinterpolated();

        f.debug_struct("TriggerAxis")
            .field("raw", &raw)
            .field("clamped", &clamped)
            .field("clamped_dz", &clamped_dz)
            .field("clamped_intp", &clamped_intp)
            .finish()
    }
}

impl TriggerAxis {
    pub fn new(initial_value: u8) -> Self {
        Self {
            value: initial_value,
            ground: initial_value,
            shallow_deadzone: None,
            deep_deadzone: None,
        }
    }

    pub fn set_value(&mut self, value: u8) {
        self.value = value;
    }

    pub fn set_deadzones(&mut self, shallow: u8, deep: u8) {
        let (shallow, deep) = match shallow.cmp(&deep) {
            Ordering::Less => (shallow, deep),
            Ordering::Equal => (0, 0),
            Ordering::Greater => (deep, shallow),
        };

        self.shallow_deadzone = Some(shallow);
        self.deep_deadzone = Some(deep);
    }

    pub fn get_raw(&self) -> i16 {
        self.value as i16 - self.ground as i16
    }

    pub fn get_raw_clamped(&self) -> u8 {
        self.get_raw().max(0) as u8
    }

    pub fn get_deadzone_clamped(&self) -> u8 {
        let lower_bound = self.shallow_deadzone.unwrap_or(0);
        let upper_bound = self.deep_deadzone.unwrap_or(u8::MAX);

        let raw = self.get_raw_clamped();

        if raw < lower_bound {
            0
        } else if raw >= upper_bound {
            u8::MAX
        } else {
            raw
        }
    }

    pub fn get_deadzone_reinterpolated(&self) -> u8 {
        let lower_bound = self.shallow_deadzone.unwrap_or(0);
        let upper_bound = self.deep_deadzone.unwrap_or(u8::MAX);

        let raw = self.get_deadzone_clamped();
        let numer = raw as f32 - lower_bound as f32;
        let denom = (upper_bound - lower_bound) as f32;
        (255.0 * (numer / denom).clamp(0.0, 1.0)) as u8
    }

    pub fn normalized_raw(&self) -> f32 {
        self.get_raw() as f32 / u8::MAX as f32
    }

    pub fn normalized_clamped(&self) -> f32 {
        self.get_raw_clamped() as f32 / u8::MAX as f32
    }

    pub fn normalized_deadzone(&self) -> f32 {
        self.get_deadzone_clamped() as f32 / u8::MAX as f32
    }

    pub fn normalized_deadzone_reinterpolated(&self) -> f32 {
        self.get_deadzone_reinterpolated() as f32 / u8::MAX as f32
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct StickAxis {
    value: u8,
    center: u8,
    negative_outer_deadzone: Option<i8>,
    negative_inner_deadzone: Option<i8>,
    positive_outer_deadzone: Option<i8>,
    positive_inner_deadzone: Option<i8>,
}

impl Debug for StickAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.sign_plus() {
            return f
                .debug_struct("StickAxis")
                .field("value", &self.value)
                .field("center", &self.center)
                .field("negative_outer_deadzone", &self.negative_outer_deadzone)
                .field("negative_inner_deadzone", &self.negative_inner_deadzone)
                .field("positive_outer_deadzone", &self.positive_outer_deadzone)
                .field("positive_inner_deadzone", &self.positive_inner_deadzone)
                .finish();
        }

        let raw = self.normalized_raw();
        let clamped = self.normalized_clamped();
        let clamped_dz = self.normalized_deadzone();
        let clamped_intp = self.normalized_deadzone_reinterpolated();

        f.debug_struct("StickAxis")
            .field("raw", &raw)
            .field("clamped", &clamped)
            .field("clamped_dz", &clamped_dz)
            .field("clamped_intp", &clamped_intp)
            .finish()
    }
}

impl StickAxis {
    const MAX: i8 = 127;
    const MIN: i8 = -127;

    pub fn new(initial_value: u8) -> Self {
        Self {
            value: initial_value,
            center: initial_value,
            negative_outer_deadzone: None,
            negative_inner_deadzone: None,
            positive_outer_deadzone: None,
            positive_inner_deadzone: None,
        }
    }

    pub fn set_value(&mut self, value: u8) {
        self.value = value;
    }

    pub fn set_positive_deadzones(&mut self, inner: u8, outer: u8) {
        let (inner, outer) = match inner.cmp(&outer) {
            Ordering::Greater => (outer, inner),
            Ordering::Equal => (0, 0),
            Ordering::Less => (inner, outer),
        };

        let inner: i8 = inner.try_into().unwrap_or(Self::MAX as i8);
        let outer: i8 = outer.try_into().unwrap_or(Self::MAX as i8);
        self.positive_inner_deadzone = Some(inner);
        self.positive_outer_deadzone = Some(outer);
    }

    pub fn set_negative_deadzones(&mut self, inner: u8, outer: u8) {
        let (inner, outer) = match inner.cmp(&outer) {
            Ordering::Greater => (outer, inner),
            Ordering::Equal => (0, 0),
            Ordering::Less => (inner, outer),
        };

        let inner: i8 = -inner.try_into().unwrap_or(Self::MAX as i8);
        let outer: i8 = -outer.try_into().unwrap_or(Self::MAX as i8);
        self.negative_inner_deadzone = Some(inner);
        self.negative_outer_deadzone = Some(outer);
    }

    pub fn get_raw(&self) -> i16 {
        self.value as i16 - self.center as i16
    }

    pub fn get_raw_clamped(&self) -> i8 {
        self.get_raw().clamp(Self::MIN as i16, Self::MAX as i16) as i8
    }

    pub fn get_deadzone_clamped(&self) -> i8 {
        let raw = self.get_raw();
        match raw.cmp(&0) {
            Ordering::Less => {
                let lower_bound = self.negative_outer_deadzone.unwrap_or(Self::MIN) as i16;
                let upper_bound = self.negative_inner_deadzone.unwrap_or(0) as i16;
                if raw <= lower_bound {
                    Self::MIN
                } else if raw > upper_bound {
                    0
                } else {
                    raw as i8
                }
            }
            Ordering::Equal => 0,
            Ordering::Greater => {
                let lower_bound = self.positive_inner_deadzone.unwrap_or(0) as i16;
                let upper_bound = self.positive_outer_deadzone.unwrap_or(Self::MAX) as i16;
                if raw < lower_bound {
                    0
                } else if raw >= upper_bound {
                    Self::MAX
                } else {
                    raw as i8
                }
            }
        }
    }

    pub fn get_deadzone_reinterpolated(&self) -> i8 {
        let clamped = self.get_deadzone_clamped();

        match clamped.cmp(&0) {
            Ordering::Equal => 0,
            Ordering::Greater => {
                let inner = self.positive_inner_deadzone.unwrap_or(0);
                let outer = self.positive_outer_deadzone.unwrap_or(Self::MAX);

                let numer = clamped as f32 - inner as f32;
                let denom = (outer - inner) as f32;
                (127.0 * (numer / denom).clamp(0.0, 1.0)) as i8
            }
            Ordering::Less => {
                let inner = self.negative_inner_deadzone.unwrap_or(0);
                let outer = self.negative_outer_deadzone.unwrap_or(Self::MIN);

                let numer = inner as f32 - clamped as f32;
                let denom = (inner - outer) as f32;
                (-127.0 * (numer / denom).clamp(0.0, 1.0)) as i8
            }
        }
    }

    pub fn normalized_raw(&self) -> f32 {
        self.get_raw() as f32 / Self::MAX as f32
    }

    pub fn normalized_clamped(&self) -> f32 {
        self.get_raw_clamped() as f32 / Self::MAX as f32
    }

    pub fn normalized_deadzone(&self) -> f32 {
        self.get_deadzone_clamped() as f32 / Self::MAX as f32
    }

    pub fn normalized_deadzone_reinterpolated(&self) -> f32 {
        self.get_deadzone_reinterpolated() as f32 / Self::MAX as f32
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GamecubeId {
    adapter_id: AdapterId,
    port: u8,
}

impl Display for GamecubeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.adapter_id, self.port)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct GamecubeStick {
    pub x: StickAxis,
    pub y: StickAxis,
}

#[derive(Debug, Copy, Clone)]
pub struct GamecubeController {
    pub rumble_enabled: bool,
    pub poll_count: usize,
    pub id: GamecubeId,
    pub kind: ControllerKind,
    pub buttons: GamecubeButtons,
    pub left_stick: GamecubeStick,
    pub right_stick: GamecubeStick,
    pub left_trigger: TriggerAxis,
    pub right_trigger: TriggerAxis,
}

struct AdapterThreadCreationEvent {
    pub adapter_id: AdapterId,
    pub start_signal: oneshot::Sender<()>,
    pub rumble_msg_sender: Sender<(GamecubeId, bool)>,
}

pub struct GamecubeControllers {
    connection_manager_thread: JoinHandle<()>,
    packet_receiver: Receiver<AdapterPacket>,
    disconnected_adapters: Receiver<AdapterId>,
    waiting_threads: Receiver<AdapterThreadCreationEvent>,
    shutdown_signal: oneshot::Sender<()>,
    snapshot_sender: Option<Sender<InputSnapshotMetric>>,
    adapters: Vec<(AdapterId, [Option<GamecubeController>; 4])>,
    rumble_broadcasters: Vec<(AdapterId, Sender<(GamecubeId, bool)>)>,
    _hotplug_handle: Option<Registration<Context>>,
}

impl GamecubeControllers {
    pub fn new(metrics: Option<&Metrics>) -> Self {
        let context = Context::new().expect("failed to initialize rusb context");

        let (packet_tx, packet_rx) = mpsc::channel();
        let (waiting_tx, waiting_rx) = mpsc::channel();

        let (hp_handle, event_receiver) = {
            let (tx, rx) = mpsc::channel();
            match HotplugBuilder::new()
                .product_id(OEM_GAMECUBE_ADAPTER_PID)
                .vendor_id(OEM_GAMECUBE_ADAPTER_VID)
                .enumerate(true)
                .register(&context, Box::new(AdapterHotplugHandler { sender: tx }))
            {
                Ok(reg) => (Some(reg), Some(rx)),
                Err(rusb::Error::NotSupported) => {
                    log::info!("Failed to register USB hotplug as it is not supported, falling back to manual device scanning.");
                    (None, None)
                }
                Err(e) => {
                    log::warn!("Failed to register USB hotplug, falling back to manual device scanning: {e}");
                    (None, None)
                }
            }
        };

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let (disconnect_tx, disconnect_rx) = mpsc::channel();

        let metrics_sender = metrics.map(|metrics| metrics.polling().new_sender());
        let snapshot_sender = metrics.map(|metrics| metrics.input_snapshots().new_sender());

        let thread = std::thread::Builder::new()
            .name("GCAdapter Connection Manager Thread".to_string())
            .spawn(move || {
                rusb_connection_manager(
                    context,
                    waiting_tx,
                    packet_tx,
                    disconnect_tx,
                    event_receiver,
                    metrics_sender,
                    shutdown_rx,
                )
            })
            .expect("failed to spawn connection manager thread");

        Self {
            connection_manager_thread: thread,
            packet_receiver: packet_rx,
            disconnected_adapters: disconnect_rx,
            waiting_threads: waiting_rx,
            shutdown_signal: shutdown_tx,
            snapshot_sender,
            adapters: vec![],
            rumble_broadcasters: vec![],
            _hotplug_handle: hp_handle,
        }
    }

    pub fn update(&mut self) {
        loop {
            match self.waiting_threads.try_recv() {
                Ok(signal) => {
                    if let Err(_error) = signal.start_signal.send(()) {
                        log::error!("Failed to send start signal to thread while updating, thread must've already terminated");
                    } else {
                        self.rumble_broadcasters
                            .push((signal.adapter_id, signal.rumble_msg_sender));
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    panic!("Failed to receive new thread thread signal from rusb_connection_manager, thread must have prematurely terminated.");
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        loop {
            match self.disconnected_adapters.try_recv() {
                Ok(id) => {
                    let Some(pos) = self
                        .adapters
                        .iter()
                        .position(|(adapter_id, _)| *adapter_id == id)
                    else {
                        log::info!("Adapter {id} was disconnected, but no controller info was able to be removed");
                        continue;
                    };

                    self.adapters.remove(pos);
                }
                Err(TryRecvError::Disconnected) => {
                    panic!("Failed to receive disconnected adapter event from rusb_connection_manager, thread must've prematurely terminated.");
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        self.adapters.iter_mut().for_each(|(_, adapter)| {
            adapter.iter_mut().for_each(|controller| {
                if let Some(controller) = controller {
                    controller.buttons = GamecubeButtons::empty();
                }
            })
        });

        loop {
            match self.packet_receiver.try_recv() {
                Ok(packet) => {
                    let adapter_pos = if let Some(pos) = self
                        .adapters
                        .iter()
                        .position(|(id, _)| *id == packet.adapter_id)
                    {
                        pos
                    } else {
                        self.adapters.push((packet.adapter_id, [None; 4]));
                        self.adapters.len() - 1
                    };

                    let adapter = &mut self.adapters[adapter_pos].1;

                    for x in 0..4 {
                        let sub_packet = &packet.packet[(1 + x * 9)..=(1 + x) * 9];
                        let kind = ControllerKind::from((sub_packet[0] & 0x30) >> 4);
                        if kind == ControllerKind::Disconnected {
                            adapter[x] = None;
                            continue;
                        }

                        let rumble_enabled = (sub_packet[0] & 0x4) != 0;
                        let buttons = GamecubeButtons::from_bits_truncate(u16::from_le_bytes([
                            sub_packet[1],
                            sub_packet[2],
                        ]));
                        let lx = sub_packet[3];
                        let ly = sub_packet[4];
                        let rx = sub_packet[5];
                        let ry = sub_packet[6];
                        let l = sub_packet[7];
                        let r = sub_packet[8];

                        // When connecting and disconnecting controllers, early packets can have all these fields zeroed
                        // which is bad because then our sticks don't get centered on neutral positions.
                        if lx == 0 && ly == 0 && rx == 0 && ry == 0 && r == 0 && l == 0 {
                            adapter[x] = None;
                            continue;
                        }

                        if let Some(controller) = adapter[x].as_mut() {
                            controller.rumble_enabled = rumble_enabled;
                            controller.poll_count += 1;
                            controller.kind = kind;
                            controller.buttons |= buttons;
                            controller.left_stick.x.set_value(lx);
                            controller.left_stick.y.set_value(ly);
                            controller.right_stick.x.set_value(rx);
                            controller.right_stick.y.set_value(ry);
                            controller.left_trigger.set_value(l);
                            controller.right_trigger.set_value(r);
                        } else {
                            adapter[x] = Some(GamecubeController {
                                rumble_enabled,
                                poll_count: 1,
                                id: GamecubeId {
                                    adapter_id: packet.adapter_id,
                                    port: x as u8,
                                },
                                kind,
                                buttons,
                                left_stick: GamecubeStick {
                                    x: StickAxis::new(lx),
                                    y: StickAxis::new(ly),
                                },
                                right_stick: GamecubeStick {
                                    x: StickAxis::new(rx),
                                    y: StickAxis::new(ry),
                                },
                                left_trigger: TriggerAxis::new(l),
                                right_trigger: TriggerAxis::new(r),
                            });
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    panic!("Failed to receive new packets because all senders have disconnected, connection manager must have been prematurely terminated.");
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        if let Some(sender) = self.snapshot_sender.as_ref() {
            if let Err(_error) = sender.send(InputSnapshotMetric(Instant::now())) {
                log::warn!("Failed to send input snapshot metric");
            }
        }
    }

    pub fn enumerate_connected_controllers<'a>(&'a self) -> impl Iterator<Item = GamecubeId> + 'a {
        self.adapters.iter().flat_map(|(adapter_id, controllers)| {
            controllers
                .iter()
                .enumerate()
                .filter_map(|(port, controller)| {
                    controller.is_some().then_some(GamecubeId {
                        adapter_id: *adapter_id,
                        port: port as u8,
                    })
                })
        })
    }

    pub fn get_controller(&self, id: GamecubeId) -> Option<&GamecubeController> {
        self.adapters
            .iter()
            .find_map(|(adapter_id, controllers)| {
                (*adapter_id == id.adapter_id).then_some(controllers[id.port as usize].as_ref())
            })
            .flatten()
    }

    pub fn set_rumble(&self, id: GamecubeId, enable: bool) {
        if let Some((_, broadcaster)) = self
            .rumble_broadcasters
            .iter()
            .find(|(adapter_id, _)| *adapter_id == id.adapter_id)
        {
            match broadcaster.send((id, enable)) {
                Ok(_) => {}
                Err(_error) => {
                    log::warn!("Failed to broadcast rumble change for {id}");
                }
            }
        } else {
            log::warn!("Gamecube Controller {id}'s adapter was not found to signal a rumble event");
        }
    }

    pub fn shutdown(self) {
        if self.shutdown_signal.send(()).is_err() {
            log::warn!("Failed to send shutdown signal, not waiting on thread to terminate (this may have undefined behavior)");
            return;
        }

        self.connection_manager_thread
            .join()
            .expect("failed to join thread");
    }
}

struct AdapterHotplugHandler {
    sender: Sender<GCAdapterConnectionEvent>,
}

impl Hotplug<Context> for AdapterHotplugHandler {
    fn device_arrived(&mut self, device: rusb::Device<Context>) {
        let desc = match device.device_descriptor() {
            Ok(desc) => desc,
            Err(e) => {
                log::warn!(
                    "AdapterHotplugHandler failed to get device descriptor for new device: {e}"
                );
                return;
            }
        };

        let vid = desc.vendor_id();
        let pid = desc.product_id();
        if vid != OEM_GAMECUBE_ADAPTER_VID || pid != OEM_GAMECUBE_ADAPTER_PID {
            log::info!("AdapterHotplugHandler got VID/PID {vid:04}/{pid:04}, ignoring.");
            return;
        }

        let id = AdapterId {
            bus: device.bus_number(),
            port_number: device.port_number(),
        };

        match self.sender.send(GCAdapterConnectionEvent::Connected {
            adapter_id: id,
            device,
        }) {
            Ok(_) => {}
            Err(e) => {
                log::error!("Failed to send connection event for adapter {id}: {e}")
            }
        }
    }

    fn device_left(&mut self, device: rusb::Device<Context>) {
        let desc = match device.device_descriptor() {
            Ok(desc) => desc,
            Err(e) => {
                log::warn!(
                    "AdapterHotplugHandler failed to get device descriptor for removed device: {e}"
                );
                return;
            }
        };

        let vid = desc.vendor_id();
        let pid = desc.product_id();
        if vid != OEM_GAMECUBE_ADAPTER_VID || pid != OEM_GAMECUBE_ADAPTER_PID {
            log::info!("AdapterHotplugHandler got VID/PID {vid:04}/{pid:04}, ignoring.");
            return;
        }

        let id = AdapterId {
            bus: device.bus_number(),
            port_number: device.port_number(),
        };

        match self
            .sender
            .send(GCAdapterConnectionEvent::Disconnected { adapter_id: id })
        {
            Ok(_) => {}
            Err(e) => {
                log::error!("Failed to send connection event for adapter {id}: {e}")
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct AdapterId {
    bus: u8,
    port_number: u8,
}

impl Display for AdapterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}@{}", self.bus, self.port_number)
    }
}

struct AdapterPacket {
    adapter_id: AdapterId,
    packet: [u8; 37],
}

enum GCAdapterConnectionEvent {
    Connected {
        adapter_id: AdapterId,
        device: Device<Context>,
    },
    Disconnected {
        adapter_id: AdapterId,
    },
}

const OEM_GAMECUBE_ADAPTER_VID: u16 = 0x057E;
const OEM_GAMECUBE_ADAPTER_PID: u16 = 0x0337;

const OEM_GAMECUBE_ADAPTER_WRITE: u8 = 0x2;
const OEM_GAMECUBE_ADAPTER_READ: u8 = 0x81;

fn rusb_adapter_management_thread(
    id: AdapterId,
    mut device: DeviceHandle<Context>,
    shutdown_signal: Weak<AtomicBool>,
    packet_sender: Sender<AdapterPacket>,
    metrics_sender: Option<Sender<PollingMetric>>,
    start_thread_signal: oneshot::Receiver<()>,
    rumble_msg_receiver: Receiver<(GamecubeId, bool)>,
) {
    const BLOCKING_DURATION: Duration = Duration::from_nanos(0);

    fn should_continue(signal: &Weak<AtomicBool>) -> bool {
        signal
            .upgrade()
            .map(|signal| !signal.load(AtomicOrdering::Acquire))
            .unwrap_or(false)
    }

    match start_thread_signal.recv() {
        Ok(_) => {
            log::info!("rusb_adapter_management_thread starting polling thread for {id}");
        }
        Err(_) => {
            log::error!("rusb_adapter_management_thread({id}) could not start polling thread");
            return;
        }
    }

    let has_kernel_driver = match device.kernel_driver_active(0) {
        Ok(active) => active,
        Err(rusb::Error::NotSupported) => false,
        Err(e) => {
            log::error!("rusb_adapter_management_thread({id}) could not check kernel driver state and could not initialize: {e}");
            return;
        }
    };

    if has_kernel_driver {
        if let Err(e) = device.detach_kernel_driver(0) {
            log::error!("rusb_adapter_management_thread({id}) could not detach kernal driver and could not initialize: {e}");
            return;
        }
    }

    if let Err(e) = device.claim_interface(0) {
        log::error!("rusb_adapter_management_thread({id}) could not claim default interface and could not initialize: {e}");
    }

    if let Err(e) = device.write_interrupt(OEM_GAMECUBE_ADAPTER_WRITE, &[0x13], BLOCKING_DURATION) {
        log::error!(
            "rusb_adapter_management_thread({id}) could not send initialization packet: {e}"
        );
        return;
    }

    let mut rumble_state = [false; 4];

    let mut input_packet_buffer = [0u8; 37];
    while should_continue(&shutdown_signal) {
        log::trace!("rusb_adapter_management_thread({id}) preparing to poll");
        let time = Instant::now();
        match device.read_interrupt(
            OEM_GAMECUBE_ADAPTER_READ,
            &mut input_packet_buffer,
            Duration::from_nanos(0),
        ) {
            Ok(num_bytes) => {
                if num_bytes != input_packet_buffer.len() {
                    log::error!("rusb_adapter_management_thread({id}) failed to read whole buffer (read {num_bytes} / {})", input_packet_buffer.len());
                    continue;
                }

                if let Some(metrics) = metrics_sender.as_ref() {
                    if let Err(_error) = metrics.send(PollingMetric {
                        id: id.to_string(),
                        timestamp: Instant::now(),
                    }) {
                        log::warn!("rusb_adapter_management_thread({id}) to send polling metrics");
                    }
                }

                log::trace!(
                    "rusb_adapter_management_thread({id}) read from device in {}us",
                    time.elapsed().as_micros()
                );

                if let Err(e) = packet_sender.send(AdapterPacket {
                    adapter_id: id,
                    packet: input_packet_buffer,
                }) {
                    log::error!(
                        "rusb_adapter_management_thread({id}) failed to send adapter packet: {e}"
                    )
                }
            }
            Err(e) => {
                log::error!("rusb_adapter_management_thread({id}) failed to read from device: {e}");
                continue;
            }
        }

        let mut rumble_msg = None;
        loop {
            match rumble_msg_receiver.try_recv() {
                Ok((id, enable)) => {
                    if rumble_msg.is_none() {
                        rumble_msg = Some([None; 4]);
                    }

                    rumble_msg.as_mut().unwrap()[id.port as usize] = Some(enable);
                }
                Err(TryRecvError::Disconnected) => {
                    log::error!(
                        "rusb_adapter_management_thread({id}) rumble_msg_receiver is disconnected"
                    );
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        if let Some(msg) = rumble_msg {
            for idx in 0..4 {
                rumble_state[idx] = msg[idx].unwrap_or(rumble_state[idx]);
            }

            let payload = [
                0x11,
                rumble_state[0] as u8,
                rumble_state[1] as u8,
                rumble_state[2] as u8,
                rumble_state[3] as u8,
            ];

            match device.write_interrupt(
                OEM_GAMECUBE_ADAPTER_WRITE,
                &payload,
                Duration::from_nanos(0),
            ) {
                Ok(num_bytes) => {
                    if num_bytes != payload.len() {
                        log::error!("rusb_adapter_management_thread({id}) failed to write whole rumble packet: {num_bytes} / {}", payload.len());
                    }
                }
                Err(e) => {
                    log::error!(
                        "rusb_adapter_management_thread({id}) failed to write rumble packet: {e}"
                    );
                }
            }
        }
    }

    if has_kernel_driver {
        if let Err(e) = device.attach_kernel_driver(0) {
            log::warn!(
                "rusb_adapter_management_thread({id}) failed to reattach kernel driver: {e}"
            );
        }
    }

    log::info!("rusb_adapter_management_thread({id}) shutting down");
}

fn rusb_connection_manager(
    context: Context,
    waiting_sender: Sender<AdapterThreadCreationEvent>,
    packet_sender: Sender<AdapterPacket>,
    disconnect_sender: Sender<AdapterId>,
    receiver: Option<Receiver<GCAdapterConnectionEvent>>,
    metrics_sender: Option<Sender<PollingMetric>>,
    shutdown_signal: oneshot::Receiver<()>,
) {
    const HANDLE_EVENTS_TIMEOUT: Duration = Duration::from_millis(500);

    let mut adapter_threads: Vec<(AdapterId, JoinHandle<()>, Arc<AtomicBool>)> = vec![];
    let mut join_list: Vec<(JoinHandle<()>, Arc<AtomicBool>)> = vec![];

    log::info!("rusb_connection_manager thread starting");

    'outer_loop: loop {
        match shutdown_signal.try_recv() {
            Ok(_) => break,
            Err(oneshot::TryRecvError::Disconnected) => {
                log::warn!("rusb_connection_manager owner has shut down without stopping the thread, stopping thread now");
                break;
            }
            Err(oneshot::TryRecvError::Empty) => {}
        }

        match context.handle_events(Some(HANDLE_EVENTS_TIMEOUT)) {
            Ok(_) => {}
            Err(e) => {
                log::error!("rusb_connection_manager failed to handle events: {e}");
            }
        }

        match receiver.as_ref() {
            Some(receiver) => {
                loop {
                    match receiver.try_recv() {
                        Ok(event) => match event {
                            GCAdapterConnectionEvent::Connected { adapter_id, device } => {
                                let device = match device.open() {
                                    Ok(device) => device,
                                    Err(e) => {
                                        log::error!("rusb_connection_manager failed to open adapter {adapter_id}: {e}");
                                        continue;
                                    }
                                };

                                let shutdown_signal = Arc::new(AtomicBool::new(false));
                                let weak_signal = Arc::downgrade(&shutdown_signal);

                                let (waiting_tx, waiting_rx) = oneshot::channel();

                                let packet_sender = packet_sender.clone();
                                let metrics_sender = metrics_sender.clone();

                                let (rumble_msg_tx, rumble_msg_rx) = mpsc::channel();

                                let event = AdapterThreadCreationEvent {
                                    start_signal: waiting_tx,
                                    adapter_id,
                                    rumble_msg_sender: rumble_msg_tx,
                                };

                                let handle = std::thread::Builder::new()
                                    .name(format!("GCAdapter {adapter_id} Polling Thread"))
                                    .spawn(move || rusb_adapter_management_thread(adapter_id, device, weak_signal, packet_sender, metrics_sender, waiting_rx, rumble_msg_rx))
                                    .unwrap_or_else(|_| panic!("Failed to spawn polling thread for GCAdapter {adapter_id}"));

                                if let Err(e) = waiting_sender.send(event) {
                                    log::error!("rusb_connection_manager failed to hand off start signaler, starting thread immediately with no rumble support");
                                    e.0.start_signal
                                        .send(())
                                        .expect("failed to send start signal");
                                }

                                adapter_threads.push((adapter_id, handle, shutdown_signal));
                            }
                            GCAdapterConnectionEvent::Disconnected { adapter_id } => {
                                let Some(position) = adapter_threads
                                    .iter()
                                    .position(|(id, _handle, _signal)| *id == adapter_id)
                                else {
                                    log::warn!("rusb_connection_manager received disconnection event for {adapter_id}, but was not managing that device");
                                    continue;
                                };

                                let (id, handle, signal) = adapter_threads.remove(position);
                                signal.store(true, AtomicOrdering::Release);

                                if let Err(_error) = disconnect_sender.send(id) {
                                    log::error!("rusb_connection_manager failed to send disconnection event to handle");
                                }

                                join_list.push((handle, signal));
                            }
                        },
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            log::error!("rusb_connection_manager receiver has disconnected");
                            break 'outer_loop;
                        }
                    }
                }
            }
            None => loop {
                let devices = match context.devices() {
                    Ok(devices) => devices,
                    Err(e) => {
                        log::error!("rusb_connection_manager failed to query for devices: {e}");
                        break 'outer_loop;
                    }
                };

                for device in devices.iter() {
                    let desc = match device.device_descriptor() {
                        Ok(desc) => desc,
                        Err(e) => {
                            log::error!(
                                "rusb_connection_manager failed to get device descriptor: {e}"
                            );
                            continue;
                        }
                    };

                    if desc.vendor_id() != OEM_GAMECUBE_ADAPTER_VID
                        && desc.product_id() != OEM_GAMECUBE_ADAPTER_PID
                    {
                        continue;
                    }
                }
            },
        }
    }
}
