use keccak_hash::keccak;
use rand::RngCore;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
mod gpu;

const CONTRACT: &str = "0x1E5adF70321CA28b3Ead70Eac545E6055E969e6f";

fn keccak256(data: &[u8; 64]) -> [u8; 32] {
    let hash = keccak(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}

fn check_proof(challenge: &[u8; 32], nonce: &[u8; 32], difficulty: &[u8; 32]) -> bool {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    buf[32..].copy_from_slice(nonce);
    let hash = keccak256(&buf);
    hash.as_slice() < difficulty.as_slice()
}

async fn rpc_call(
    rpc: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client
        .post(rpc)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }))
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let body: serde_json::Value = resp.json().await?;
    if let Some(err) = body.get("error") {
        return Err(format!("RPC error: {}", err).into());
    }
    Ok(body)
}

async fn eth_call(rpc: &str, to: &str, data: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let body = rpc_call(
        rpc,
        "eth_call",
        serde_json::json!([{"to": to, "data": data}, "latest"]),
    )
    .await?;
    let hex_str = body["result"].as_str().ok_or("No result")?;
    Ok(hex::decode(&hex_str[2..])?)
}

async fn get_challenge(rpc: &str, wallet: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    let padded = format!("{:0>64}", wallet.trim_start_matches("0x"));
    let data = format!("0xb52f2c33{}", padded);
    let bytes = eth_call(rpc, CONTRACT, &data).await?;
    let mut challenge = [0u8; 32];
    challenge.copy_from_slice(&bytes[..32]);
    Ok(challenge)
}

async fn get_difficulty(rpc: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    let bytes = eth_call(rpc, CONTRACT, "0x3f5c3e87").await?;
    let mut difficulty = [0u8; 32];
    difficulty.copy_from_slice(&bytes[..32]);
    Ok(difficulty)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let rpc = env::var("RPC_URL")
        .unwrap_or_else(|_| "https://ethereum-rpc.publicnode.com".to_string());
    let wallet = env::var("WALLET").expect("WALLET not set in .env");
    let _private_key = env::var("PRIVATE_KEY").unwrap_or_default();
    let use_gpu = env::var("GPU").map(|v| v == "1").unwrap_or(false);
    let _gpu_batch: usize = env::var("GPU_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 22);

    println!("=== H98 Miner v0.2 ===");
    println!("Wallet: {}", wallet);
    println!("Contract: {}", CONTRACT);
    println!("RPC: {}", rpc);
    #[cfg(feature = "gpu")]
    println!("GPU: {}", if use_gpu { "enabled" } else { "disabled" });
    #[cfg(not(feature = "gpu"))]
    println!("GPU: not compiled (build with --features gpu)");

    let rt = tokio::runtime::Runtime::new()?;
    let challenge = rt.block_on(get_challenge(&rpc, &wallet))?;
    let difficulty = rt.block_on(get_difficulty(&rpc))?;

    println!("Challenge: 0x{}", hex::encode(&challenge));
    println!("Difficulty: 0x{}", hex::encode(&difficulty));

    let challenge = Arc::new(challenge);
    let difficulty = Arc::new(difficulty);
    let found = Arc::new(AtomicBool::new(false));
    let total_hashes = Arc::new(AtomicU64::new(0));
    let found_nonce: Arc<std::sync::Mutex<Option<[u8; 32]>>> =
        Arc::new(std::sync::Mutex::new(None));

    let start = Instant::now();

    #[cfg(feature = "gpu")]
    if use_gpu {
        println!("[GPU] Initializing OpenCL...");
        let gpu_miner = gpu::GpuMiner::new(Some(_gpu_batch))?;
        println!("[GPU] Device: {}", gpu_miner.device_name());
        println!("[GPU] Batch size: {}", gpu_miner.batch_size());

        gpu_miner.self_test()?;

        let stop_flag = found.clone();
        let counter = total_hashes.clone();
        let ch = *challenge;
        let diff = *difficulty;
        let fn_ref = found_nonce.clone();

        let gpu_handle = std::thread::spawn(move || {
            match gpu_miner.mine(&ch, &diff, stop_flag, counter) {
                Ok(Some(nonce)) => {
                    *fn_ref.lock().unwrap() = Some(nonce);
                }
                Ok(None) => {}
                Err(e) => eprintln!("[GPU] Error: {}", e),
            }
        });

        // Stats thread
        let stats_hashes = total_hashes.clone();
        let stats_found = found.clone();
        std::thread::spawn(move || {
            let mut last_hashes = 0u64;
            let mut last_time = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(3));
                let h = stats_hashes.load(Ordering::Relaxed);
                let now = Instant::now();
                let dt = last_time.elapsed().as_secs_f64();
                let rate = (h - last_hashes) as f64 / dt;
                println!(
                    "[STATS] {:.2} MH/s | {} total | {:.0}s",
                    rate / 1_000_000.0,
                    h,
                    start.elapsed().as_secs_f64()
                );
                last_hashes = h;
                last_time = now;
                if stats_found.load(Ordering::Relaxed) {
                    break;
                }
            }
        });

        gpu_handle.join().unwrap();
    } else {
        run_cpu(&challenge, &difficulty, &found, &total_hashes, &found_nonce, start)?;
    }

    #[cfg(not(feature = "gpu"))]
    run_cpu(&challenge, &difficulty, &found, &total_hashes, &found_nonce, start)?;

    let elapsed = start.elapsed();
    let hashes = total_hashes.load(Ordering::Relaxed);

    if let Some(nonce) = *found_nonce.lock().unwrap() {
        println!("\n=== MINING SUCCESS ===");
        println!("Nonce: 0x{}", hex::encode(&nonce));
        println!("Time: {:.1}s", elapsed.as_secs_f64());
        println!("Total hashes: {}", hashes);
        println!(
            "Average rate: {:.2} MH/s",
            hashes as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
        println!(
            "\nTo submit: send tx to {} with data: 0xacc8f306{}",
            CONTRACT,
            hex::encode(&nonce)
        );
    } else {
        println!("No solution found in {:.1}s", elapsed.as_secs_f64());
    }

    Ok(())
}

fn run_cpu(
    challenge: &Arc<[u8; 32]>,
    difficulty: &Arc<[u8; 32]>,
    found: &Arc<AtomicBool>,
    total_hashes: &Arc<AtomicU64>,
    found_nonce: &Arc<std::sync::Mutex<Option<[u8; 32]>>>,
    start: Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    let num_workers = num_cpus::get().saturating_sub(1).max(1);
    println!("[CPU] Starting {} workers...", num_workers);

    let handles: Vec<_> = (0..num_workers)
        .map(|wid| {
            let challenge = challenge.clone();
            let difficulty = difficulty.clone();
            let found = found.clone();
            let total_hashes = total_hashes.clone();
            let found_nonce = found_nonce.clone();

            std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                let mut nonce = [0u8; 32];
                rng.fill_bytes(&mut nonce);

                let batch_size = 100_000u64;

                loop {
                    if found.load(Ordering::Relaxed) {
                        return;
                    }

                    for _ in 0..batch_size {
                        if check_proof(&challenge, &nonce, &difficulty) {
                            found.store(true, Ordering::Relaxed);
                            *found_nonce.lock().unwrap() = Some(nonce);
                            println!(
                                "\n[WORKER {}] FOUND: 0x{}",
                                wid,
                                hex::encode(&nonce)
                            );
                            return;
                        }

                        for i in 0..32 {
                            nonce[i] = nonce[i].wrapping_add(1);
                            if nonce[i] != 0 {
                                break;
                            }
                        }
                    }

                    total_hashes.fetch_add(batch_size, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Stats thread
    let stats_hashes = total_hashes.clone();
    let stats_found = found.clone();
    std::thread::spawn(move || {
        let mut last_hashes = 0u64;
        let mut last_time = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(3));
            let h = stats_hashes.load(Ordering::Relaxed);
            let now = Instant::now();
            let dt = last_time.elapsed().as_secs_f64();
            let rate = (h - last_hashes) as f64 / dt;
            println!(
                "[STATS] {:.2} MH/s | {} total | {:.0}s",
                rate / 1_000_000.0,
                h,
                start.elapsed().as_secs_f64()
            );
            last_hashes = h;
            last_time = now;
            if stats_found.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    for handle in handles {
        let _ = handle.join();
    }

    Ok(())
}
