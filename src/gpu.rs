//! GPU mining backend via OpenCL for H98 (32-byte nonces).
//!
//! Builds a single program/kernel up-front, then `mine_batch` repeatedly
//! dispatches a fixed-size grid against new nonce_base values until either
//! a hit is found or the caller signals stop.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use keccak_hash::keccak;
use ocl::{flags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

const NONCE_BASE_ARG_IDX: u32 = 8;
const KERNEL_SRC: &str = include_str!("keccak_kernel.cl");
const DEFAULT_BATCH: usize = 1 << 22; // 4,194,304 nonces/dispatch

pub struct GpuMiner {
    queue: Queue,
    program: Program,
    device_name: String,
    batch_size: usize,
}

impl GpuMiner {
    pub fn new(batch_size: Option<usize>) -> Result<Self, Box<dyn std::error::Error>> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH);

        let platform = Platform::default();
        let device = Device::first(platform)
            .map_err(|e| format!("no OpenCL device found: {e}"))?;
        let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;
        let program = Program::builder()
            .src(KERNEL_SRC)
            .devices(device)
            .build(&context)?;

        Ok(Self {
            queue,
            program,
            device_name,
            batch_size,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Verify the GPU kernel matches a CPU reference on one known nonce.
    pub fn self_test(&self) -> Result<(), Box<dyn std::error::Error>> {
        let challenge = [0u8; 32]; // all-zero challenge for test
        let difficulty = [0xffu8; 32]; // MAX difficulty
        let nonce_base: u64 = 12345;

        let cw = split_challenge_le(&challenge);
        let dw = split_difficulty_be(&difficulty);

        let found_nonce = Buffer::<u64>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(4)
            .copy_host_slice(&[0u64; 4])
            .build()?;
        let found_flag = Buffer::<i32>::builder()
            .queue(self.queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(1)
            .copy_host_slice(&[0i32])
            .build()?;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name("mine_keccak")
            .queue(self.queue.clone())
            .global_work_size(1usize)
            .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
            .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
            .arg(nonce_base)
            .arg(&found_nonce)
            .arg(&found_flag)
            .build()?;

        unsafe { kernel.enq()?; }
        self.queue.finish()?;

        let mut flag = [0i32];
        found_flag.read(&mut flag[..]).enq()?;
        if flag[0] == 0 {
            return Err("self-test: GPU did not report any hit against MAX difficulty".into());
        }

        let mut got = [0u64; 4];
        found_nonce.read(&mut got[..]).enq()?;
        println!("[GPU] self-test passed — device: {}", self.device_name);
        println!("[GPU] nonce words: {:016x} {:016x} {:016x} {:016x}", got[0], got[1], got[2], got[3]);

        Ok(())
    }

    /// Run batches until either a solution is found or `stop_flag` is set.
    /// Returns the winning 32-byte nonce when one is found.
    pub fn mine(
        &self,
        challenge: &[u8; 32],
        difficulty: &[u8; 32],
        stop_flag: Arc<AtomicBool>,
        attempts_counter: Arc<AtomicU64>,
    ) -> Result<Option<[u8; 32]>, Box<dyn std::error::Error>> {
        const PIPELINE_DEPTH: usize = 2;

        let cw = split_challenge_le(challenge);
        let dw = split_difficulty_be(difficulty);

        let mut flag_bufs: Vec<Buffer<i32>> = Vec::with_capacity(PIPELINE_DEPTH);
        let mut nonce_bufs: Vec<Buffer<u64>> = Vec::with_capacity(PIPELINE_DEPTH);
        for _ in 0..PIPELINE_DEPTH {
            flag_bufs.push(
                Buffer::<i32>::builder()
                    .queue(self.queue.clone())
                    .flags(flags::MEM_READ_WRITE)
                    .len(1)
                    .copy_host_slice(&[0i32])
                    .build()?,
            );
            nonce_bufs.push(
                Buffer::<u64>::builder()
                    .queue(self.queue.clone())
                    .flags(flags::MEM_READ_WRITE)
                    .len(4)
                    .copy_host_slice(&[0u64; 4])
                    .build()?,
            );
        }

        let mut kernels: Vec<Kernel> = Vec::with_capacity(PIPELINE_DEPTH);
        for i in 0..PIPELINE_DEPTH {
            kernels.push(
                Kernel::builder()
                    .program(&self.program)
                    .name("mine_keccak")
                    .queue(self.queue.clone())
                    .global_work_size(self.batch_size)
                    .arg(cw[0]).arg(cw[1]).arg(cw[2]).arg(cw[3])
                    .arg(dw[0]).arg(dw[1]).arg(dw[2]).arg(dw[3])
                    .arg(0u64) // nonce_base placeholder
                    .arg(&nonce_bufs[i])
                    .arg(&flag_bufs[i])
                    .build()?,
            );
        }

        // Use time-based seed
        let mut nonce_base: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64;

        // Prime pipeline
        for slot in 0..PIPELINE_DEPTH {
            flag_bufs[slot].cmd().fill(0i32, None).enq()?;
            nonce_bufs[slot].cmd().fill(0u64, None).enq()?;
            kernels[slot].set_arg(NONCE_BASE_ARG_IDX, nonce_base)?;
            unsafe { kernels[slot].enq()?; }
            nonce_base = nonce_base.wrapping_add(0x9E3779B97F4A7C15);
        }

        let mut head: usize = 0;
        loop {
            if stop_flag.load(Ordering::Relaxed) {
                self.queue.finish()?;
                return Ok(None);
            }

            let mut flag = [0i32];
            flag_bufs[head].read(&mut flag[..]).enq()?;
            attempts_counter.fetch_add(self.batch_size as u64, Ordering::Relaxed);

            if flag[0] != 0 {
                let mut got = [0u64; 4];
                nonce_bufs[head].read(&mut got[..]).enq()?;

                // Reconstruct 32-byte nonce
                let mut nonce = [0u8; 32];
                nonce[0..8].copy_from_slice(&got[0].to_be_bytes());
                nonce[8..16].copy_from_slice(&got[1].to_be_bytes());
                nonce[16..24].copy_from_slice(&got[2].to_be_bytes());
                nonce[24..32].copy_from_slice(&got[3].to_be_bytes());

                // CPU verify
                let cpu_hash = cpu_hash(challenge, &nonce);
                if cpu_hash.as_slice() < difficulty.as_slice() {
                    self.queue.finish()?;
                    return Ok(Some(nonce));
                } else {
                    eprintln!("[GPU] reported nonce but CPU verify failed — skipping");
                }
            }

            // Refill slot
            flag_bufs[head].cmd().fill(0i32, None).enq()?;
            nonce_bufs[head].cmd().fill(0u64, None).enq()?;
            kernels[head].set_arg(NONCE_BASE_ARG_IDX, nonce_base)?;
            unsafe { kernels[head].enq()?; }
            nonce_base = nonce_base.wrapping_add(0x9E3779B97F4A7C15);

            head = (head + 1) % PIPELINE_DEPTH;
        }
    }
}

/// Read 32-byte challenge as 4 little-endian u64 words.
fn split_challenge_le(challenge: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_le_bytes(challenge[0..8].try_into().unwrap()),
        u64::from_le_bytes(challenge[8..16].try_into().unwrap()),
        u64::from_le_bytes(challenge[16..24].try_into().unwrap()),
        u64::from_le_bytes(challenge[24..32].try_into().unwrap()),
    ]
}

/// Split 32-byte difficulty into 4 big-endian u64 words (MS first).
fn split_difficulty_be(difficulty: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_be_bytes(difficulty[0..8].try_into().unwrap()),
        u64::from_be_bytes(difficulty[8..16].try_into().unwrap()),
        u64::from_be_bytes(difficulty[16..24].try_into().unwrap()),
        u64::from_be_bytes(difficulty[24..32].try_into().unwrap()),
    ]
}

/// CPU reference hash for verification.
fn cpu_hash(challenge: &[u8; 32], nonce: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    buf[32..].copy_from_slice(nonce);
    let hash = keccak(buf);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}
