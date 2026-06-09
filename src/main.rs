use anyhow::{anyhow, Context, Result};
use byte_unit::Byte;
use clap::Parser;
use log::{error, info};
use rand::{distributions::uniform::SampleUniform, prelude::Distribution};
use rand_core::SeedableRng;
use std::{
    fs::{File, OpenOptions},
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::PathBuf,
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

    /// Workload direction.
    #[clap(long, value_enum, default_value_t = Rw::Read)]
    rw: Rw,

    /// Limit the number of blocks submitted per thread. The run stops when this
    /// limit or `--time` is reached, whichever comes first.
    #[clap(long)]
    blocks: Option<u64>,

    /// Payload source for writes. Block `i` is filled from byte `i * bs` of this
    /// file. Only valid with `--rw write`; enables sequential, single-threaded
    /// (functional) mode.
    #[clap(long)]
    input: Option<PathBuf>,

    /// Sink for reads. Block `i` read from the device is written to byte
    /// `i * bs` of this file. Only valid with `--rw read`; enables sequential,
    /// single-threaded (functional) mode.
    #[clap(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Rw {
    Read,
    Write,
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

    // Validate combinations up front: a worker that fails in State::new never
    // sends a Done message, which would wedge the recv loop below.
    if args.input.is_some() && args.rw != Rw::Write {
        return Err(anyhow!("--input is only valid with --rw write"));
    }
    if args.output.is_some() && args.rw != Rw::Read {
        return Err(anyhow!("--output is only valid with --rw read"));
    }
    if (args.input.is_some() || args.output.is_some()) && args.threads > 1 {
        return Err(anyhow!(
            "file-backed (functional) mode requires --threads 1"
        ));
    }

    let (send, recv) = std::sync::mpsc::channel();

    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::with_capacity(args.threads);
        for id in 0..args.threads {
            let args = args.clone();
            let send = send.clone();
            handles.push(s.spawn(move || run_workload(args, id, send)));
        }
        // Drop our own sender so the channel closes once every worker has
        // exited, whether normally or with an error. Without this the recv
        // loop below could block forever waiting on a Done that a failed
        // worker will never send.
        drop(send);

        let mut total_iops = 0f64;
        let mut status = vec![None; args.threads];
        loop {
            match recv.recv() {
                Ok(Message::Progress { iops, duration, id }) => {
                    status[id] = Some(iops as f64 / duration.as_secs_f64());
                    if status.iter().all(|v| v.is_some()) {
                        let iops = status.iter().fold(0f64, |acc, item| acc + item.unwrap());
                        info!("iops: {iops:+.3e}");
                        status.fill(None);
                    }
                }
                Ok(Message::Done { iops }) => {
                    total_iops += iops;
                }
                // All senders dropped: every worker has exited.
                Err(_) => break,
            }
        }

        // Join the workers and surface any failure. A worker that errored has
        // already logged its message; a panic has not.
        let mut failures = 0;
        for (id, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(_)) => failures += 1,
                Err(_) => {
                    error!("Thread {id} panicked");
                    failures += 1;
                }
            }
        }
        if failures > 0 {
            return Err(anyhow!("{failures} worker thread(s) failed"));
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
    _file: std::fs::File,
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
    rw: Rw,
    /// When set, offsets are walked sequentially from 0 instead of sampled
    /// randomly, giving a stable file-position <-> device-offset mapping.
    sequential: bool,
    seq_next: u64,
    /// Per-thread cap on submitted blocks, if any.
    limit: Option<u64>,
    submitted: u64,
    /// Payload source for writes (`--input`).
    input: Option<File>,
    /// Sink for reads (`--output`).
    output: Option<File>,
    /// Device offset of the in-flight IO using each buffer, indexed by buffer.
    offsets: Vec<u64>,
}

impl State<u64> {
    fn new(args: &Args) -> Result<Self> {
        let depth = args.depth;
        let direct = !args.indirect;
        let size = args.size.as_u64_checked().ok_or(anyhow!("Size too large"))?;
        let bs = args
            .bs
            .as_u64_checked()
            .ok_or(anyhow!("Failed to convert bs"))?;
        let shift = bs.ilog2().try_into()?;

        let mut free_list = smallset::SmallSet::new();
        for i in 0..depth {
            free_list.insert(i.try_into()?);
        }

        let file = {
            use libc;
            use std::os::unix::fs::OpenOptionsExt;
            let mut options = OpenOptions::new();
            options.read(true);
            if args.rw == Rw::Write {
                options.write(true);
            }

            let mut flags = libc::O_NOATIME;

            if direct {
                flags |= libc::O_DIRECT;
            }

            options.custom_flags(flags);

            options.open(&args.device)?
        };

        // File-backed (functional) mode: writes pull payload from --input,
        // reads dump into --output, offsets are walked sequentially so the
        // file position and device offset correspond.
        let input = match (&args.input, args.rw) {
            (Some(path), Rw::Write) => Some(File::open(path)?),
            (Some(_), Rw::Read) => return Err(anyhow!("--input is only valid with --rw write")),
            (None, _) => None,
        };
        let output = match (&args.output, args.rw) {
            (Some(path), Rw::Read) => Some(File::create(path)?),
            (Some(_), Rw::Write) => return Err(anyhow!("--output is only valid with --rw read")),
            (None, _) => None,
        };
        let sequential = input.is_some() || output.is_some();

        let region_blocks = size >> shift;
        let mut limit = args.blocks;
        if sequential {
            // Bound by the input file (for writes) and by the region.
            if let Some(inp) = &input {
                let file_blocks = inp.metadata()?.len() >> shift;
                limit = Some(limit.map_or(file_blocks, |b| b.min(file_blocks)));
            }
            limit = Some(limit.map_or(region_blocks, |b| b.min(region_blocks)));
        }

        let mut buffers = Vec::new();
        for _ in 0..depth {
            buffers.push(
                NonNull::new(unsafe { libc::memalign(bs as usize, bs as usize) })
                    .ok_or(anyhow!("Allocation failed"))?,
            );
        }

        let ring = io_uring::IoUring::new(depth)?;
        ring.submitter().register_files(&[file.as_raw_fd()])?;
        let iovs: Vec<libc::iovec> = buffers
            .iter()
            .map(|p| libc::iovec {
                iov_base: p.as_ptr(),
                iov_len: bs as usize,
            })
            .collect();
        unsafe { ring.submitter().register_buffers(&iovs)? };

        Ok(Self {
            _file: file,
            ring,
            depth,
            bs: bs.try_into()?,
            buf: buffers,
            dist: rand::distributions::Uniform::from(0..(size >> shift)),
            rng: lineargen::Linear64::seed_from_u64(384324),
            prepped: 0,
            in_flight: 0,
            done: 0,
            check: 0,
            shift,
            free_list,
            rw: args.rw,
            sequential,
            seq_next: 0,
            limit,
            submitted: 0,
            input,
            output,
            offsets: vec![0; depth as usize],
        })
    }

    fn get_offset(&mut self) -> u64 {
        if self.sequential {
            let block = self.seq_next;
            self.seq_next += 1;
            block << self.shift
        } else {
            self.dist.sample(&mut self.rng) << self.shift
        }
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

    let mut state = State::new(&args)?;

    core_affinity::set_for_current(core_affinity::CoreId { id });

    if let Some(ramp) = args.ramp {
        let start = Instant::now();
        burn::<true>(&mut state, start, &mut send, id, ramp.into())?;
        state.done = 0;
        state.submitted = 0;
        state.seq_next = 0;
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
        // Nothing left to submit and the queue is drained: the block limit has
        // been reached.
        if state.prepped == 0 && state.in_flight == 0 {
            break;
        }
        state.ring.submit_and_wait(1)?;
        reap(state)?;
        if state.check >= check_at {
            state.check = 0;
            let duration = start.elapsed();

            if !RAMP && duration - old_duration > std::time::Duration::from_secs(1) {
                tx.send(Message::Progress {
                    iops: state.done,
                    duration,
                    id,
                })?;
                old_duration = duration;
            }

            if duration > time {
                break;
            }
        }
    }

    Ok(())
}

fn enqueue_ios(state: &mut State<u64>) -> Result<()> {
    let mut to_prep = state.depth - state.in_flight;
    if let Some(limit) = state.limit {
        let remaining = limit.saturating_sub(state.submitted);
        to_prep = to_prep.min(remaining.min(u32::MAX as u64) as u32);
    }
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
    state.offsets[buffer_index as usize] = offset;

    let buf_ptr = state.buf[buffer_index as usize].as_ptr();

    // For file-backed writes, fill the buffer with the payload at the matching
    // position in the input file before submitting.
    if state.rw == Rw::Write {
        if let Some(input) = &state.input {
            let buf = unsafe {
                std::slice::from_raw_parts_mut(buf_ptr.cast::<u8>(), state.bs as usize)
            };
            input.read_exact_at(buf, offset)?;
        }
    }

    let opcode = match state.rw {
        Rw::Read => io_uring::opcode::ReadFixed::new(
            io_uring::types::Fixed(0),
            buf_ptr.cast(),
            state.bs as _,
            buffer_index as u16,
        )
        .offset(offset)
        .build()
        .user_data(buffer_index.into()),
        Rw::Write => io_uring::opcode::WriteFixed::new(
            io_uring::types::Fixed(0),
            buf_ptr.cast(),
            state.bs as _,
            buffer_index as u16,
        )
        .offset(offset)
        .build()
        .user_data(buffer_index.into()),
    };

    unsafe {
        state
            .ring
            .submission()
            .push(&opcode)
            .context("Could not queue entry")?
    };
    state.prepped += 1;
    state.in_flight += 1;
    state.submitted += 1;

    Ok(())
}

fn reap(state: &mut State<u64>) -> Result<()> {
    for cqe in state.ring.completion() {
        state.in_flight -= 1;
        state.done += 1;
        state.check += 1;

        let buffer_index = cqe.user_data() as u8;

        if cqe.result() != state.bs as i32 {
            error!("IO error: {}", errno::Errno(-cqe.result()));
            return Err(anyhow!("IO error"));
        }

        // For file-backed reads, dump the data into the matching position of
        // the output file.
        if state.rw == Rw::Read {
            if let Some(output) = &state.output {
                let offset = state.offsets[buffer_index as usize];
                let buf = unsafe {
                    std::slice::from_raw_parts(
                        state.buf[buffer_index as usize].as_ptr().cast::<u8>(),
                        state.bs as usize,
                    )
                };
                output.write_all_at(buf, offset)?;
            }
        }

        state.free_list.insert(buffer_index);
    }

    Ok(())
}
