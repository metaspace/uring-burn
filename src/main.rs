use anyhow::{anyhow, Context, Result};
use byte_unit::Byte;
use clap::Parser;
use log::{error, info};
use rand::{distributions::uniform::SampleUniform, prelude::Distribution};
use rand_core::SeedableRng;
use std::{
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::mpsc::Sender,
    time::{Duration, Instant},
};

#[derive(Parser, Clone, Debug)]
struct Args {
    #[clap(long)]
    device: PathBuf,

    #[clap(long, default_value_t = 8)]
    depth: u32,

    #[clap(long, default_value = "4 GiB")]
    size: Byte,

    #[clap(long, default_value = "4 KiB")]
    bs: Byte,

    #[clap(long, default_value = "5s")]
    time: humantime::Duration,

    #[clap(long, default_value_t = 1)]
    threads: usize,

    #[clap(long)]
    ramp: Option<humantime::Duration>,

    #[clap(long)]
    indirect: bool,
}

enum Message {
    Progress {
        iops: u64,
        duration: std::time::Duration,
        id: usize,
    },
    Done {
        iops: f64,
    },
}

fn main() -> Result<()> {
    colog::init();
    info!("uring-burn running");
    let args = Args::parse();

    if args.threads > num_cpus::get() {
        return Err(anyhow!("too many threads"));
    }

    let (send, recv) = std::sync::mpsc::channel();

    std::thread::scope(|s| -> Result<()> {
        for id in 0..args.threads {
            let args = args.clone();
            let send = send.clone();
            s.spawn(move || run_workload(args, id, send));
        }

        let mut done = 0;
        let mut total_iops = 0f64;

        let mut status = vec![None; args.threads];
        while done < args.threads {
            match recv.recv()? {
                Message::Progress { iops, duration, id } => {
                    status[id] = Some(iops as f64 / duration.as_secs_f64());
                    if status.iter().all(|v| v.is_some()) {
                        let iops = status.iter().fold(0f64, |acc, item| acc + item.unwrap());
                        info!("iops: {iops:+.3e}");
                        status.fill(None);
                    }
                }
                Message::Done { iops } => {
                    done += 1;
                    total_iops += iops;
                }
            }
        }

        info!("Total iops: {:.3e}", total_iops);
        Ok(())
    })?;

    Ok(())
}

struct State<T>
where
    T: SampleUniform,
{
    file: std::fs::File,
    ring: io_uring::IoUring,
    depth: u32,
    bs: u32,
    buf: Vec<NonNull<core::ffi::c_void>>,
    dist: rand::distributions::Uniform<T>,
    rng: lineargen::Linear64,
    prepped: u32,
    in_flight: u32,
    done: u64,
    check: u64,
    shift: usize,
    free_list: smallset::SmallSet<[u8; 8]>,
}

impl State<u64> {
    fn new(
        device: impl AsRef<Path>,
        depth: u32,
        size: u64,
        bs: Byte,
        direct: bool,
    ) -> Result<Self> {
        let shift = bs.as_u64().ilog2().try_into()?;

        let mut free_list = smallset::SmallSet::new();
        for i in 0..depth {
            free_list.insert(i.try_into()?);
        }

        let file = {
            use libc;
            use std::os::unix::fs::OpenOptionsExt;
            let mut options = std::fs::OpenOptions::new();
            options.read(true);

            let mut flags = libc::O_NOATIME;

            if direct {
                flags |= libc::O_DIRECT;
            }

            options.custom_flags(flags);

            options.open(device.as_ref())?
        };

        let mut buffers = Vec::new();
        for _ in 0..depth {
            buffers.push(NonNull::new(unsafe { libc::memalign(bs.as_u64().try_into()?, bs.as_u64().try_into()?) }).ok_or(anyhow!("Allocation failed"))?);
        }

        let ring  = io_uring::IoUring::new(depth)?;
        ring.submitter().register_files(&[file.as_raw_fd()])?;

        Ok(Self {
            file,
            ring,
            depth,
            bs: bs.as_u64().try_into()?,
            buf: buffers,
            dist: rand::distributions::Uniform::from(0..(size >> shift)),
            rng: lineargen::Linear64::seed_from_u64(384324),
            prepped: 0,
            in_flight: 0,
            done: 0,
            check: 0,
            shift,
            free_list,
        })
    }

    fn get_offset(&mut self) -> u64 {
        self.dist.sample(&mut self.rng) << self.shift
    }
}

fn run_workload(args: Args, id: usize, send: Sender<Message>) -> Result<()> {
    if let Err(e) = run_workload_inner(args, id, send) {
        error!("{e}");
        Err(e)
    } else {
        Ok(())
    }
}

fn run_workload_inner(args: Args, id: usize, mut send: Sender<Message>) -> Result<()> {
    info!("Thrad {id} running");

    let mut state = State::new(
        &args.device,
        args.depth,
        args.size
            .as_u64_checked()
            .ok_or(anyhow!("Size too large"))?,
        args.bs,
        !args.indirect,
    )?;

    core_affinity::set_for_current(core_affinity::CoreId { id });

    if let Some(ramp) = args.ramp {
        let start = Instant::now();
        burn::<true>(&mut state, start, &mut send, id, ramp.into())?;
        state.done = 0;
    }

    let start = Instant::now();
    burn::<false>(&mut state, start, &mut send, id, args.time.into())?;

    let duration = start.elapsed();
    let iops = state.done as f64 / duration.as_secs_f64();

    info!("Thread {id} done");

    send.send(Message::Done { iops })?;

    Ok(())
}

fn burn<const RAMP: bool>(
    state: &mut State<u64>,
    start: Instant,
    tx: &mut Sender<Message>,
    id: usize,
    time: Duration,
) -> Result<()> {
    let check_at = 2_000_000 / (state.bs as u64 / 4096);
    let mut old_duration = start.elapsed();
    loop {
        enqueue_ios(state)?;
        state.ring.submit_and_wait(1)?;
        reap(state)?;
        if state.check >= check_at {
            state.check = 0;
            let duration = start.elapsed();

            if !RAMP {
                if duration - old_duration > std::time::Duration::from_secs(1) {
                    tx.send(Message::Progress {
                        iops: state.done,
                        duration,
                        id,
                    })?;
                    old_duration = duration;
                }
            }

            if duration > time {
                break;
            }
        }
    }

    Ok(())
}

fn enqueue_ios(state: &mut State<u64>) -> Result<()> {
    let to_prep = state.depth - state.in_flight;
    state.prepped = 0;

    while state.prepped < to_prep {
        enqueue_io(state)?;
    }

    Ok(())
}

fn enqueue_io(state: &mut State<u64>) -> Result<()> {
    let buffer_index = *state.free_list.iter().next().unwrap();
    state.free_list.remove(&buffer_index);

    let offset = state.get_offset();

    let opcode = io_uring::opcode::Read::new(
        //io_uring::types::Fd(state.file.as_raw_fd()),
        io_uring::types::Fixed(0),
        state.buf[buffer_index as usize].as_ptr().cast(),
        state.bs as _,
    )
    .offset(offset)
    .build()
    .user_data(buffer_index.into());

    unsafe {
        state
            .ring
            .submission()
            .push(&opcode)
            .context("Could not queue entry")?
    };
    state.prepped += 1;
    state.in_flight += 1;

    Ok(())
}

fn reap(state: &mut State<u64>) -> Result<()> {
    for cqe in state.ring.completion() {
        state.in_flight -= 1;
        state.done += 1;
        state.check += 1;

        let buffer_index = cqe.user_data() as u8;
        state.free_list.insert(buffer_index);

        if cqe.result() != state.bs as i32 {
            error!("Read error: {}", errno::Errno(-cqe.result()));
            return Err(anyhow!("Read error"));
        }
    }

    Ok(())
}
