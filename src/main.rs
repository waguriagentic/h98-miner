use alloy_primitives::Address;
use keccak_hash::keccak;
use rand::RngCore;
use std::env;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const CONTRACT: &str = "0x1E5adF70321CA28b3Ead70Eac545E6055E969e6f";

/// Keccak256 hash of 64 bytes, returns 32 bytes
fn keccak256(data: &[u8; 64]) -> [u8; 32] {
    let hash = keccak(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash.as_bytes());
    out
}

/// Check if a nonce produces a valid proof:
/// keccak256(abi.encode(challenge, nonce)) < difficulty
fn check_proof(challenge: &[u8; 32], nonce: &[u8; 32], difficulty: &[u8; 32]) -> bool {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    buf[32..].copy_from_slice(nonce);
    let hash = keccak256(&buf);
    hash.as_slice() < difficulty.as_slice()
}

/// JSON-RPC helper
async fn rpc_call(rpc: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client.post(rpc)
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }))
        .timeout(Duration::from_secs(10))
        .send().await?;
    let body: serde_json::Value = resp.json().await?;
    if let Some(err) = body.get("error") {
        return Err(format!("RPC error: {}", err).into());
    }
    Ok(body)
}

/// eth_call helper
async fn eth_call(rpc: &str, to: &str, data: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let body = rpc_call(rpc, "eth_call", serde_json::json!([
        {"to": to, "data": data}, "latest"
    ])).await?;
    let hex_str = body["result"].as_str().ok_or("No result")?;
    Ok(hex::decode(&hex_str[2..])?)
}

/// Get challenge for a miner address from contract
/// getChallenge(address) → bytes32  [selector: 0xb52f2c33]
async fn get_challenge(rpc: &str, wallet: Address) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    let padded = format!("{:0>64x}", wallet);
    let data = format!("0xb52f2c33{}", padded);
    let bytes = eth_call(rpc, CONTRACT, &data).await?;
    let mut challenge = [0u8; 32];
    challenge.copy_from_slice(&bytes[..32]);
    Ok(challenge)
}

/// Get mining difficulty from contract
async fn get_difficulty(rpc: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    // Try common difficulty selectors
    // 0x3f5c3e87 returned 500 for the miner
    // Let's also try a generic approach
    let bytes = eth_call(rpc, CONTRACT, "0x3f5c3e87").await?;
    let mut difficulty = [0u8; 32];
    difficulty.copy_from_slice(&bytes[..32]);
    Ok(difficulty)
}

/// Submit a found nonce to the contract
async fn submit_nonce(rpc: &str, wallet: Address, nonce: &[u8; 32]) -> Result<String, Box<dyn std::error::Error>> {
    // Build calldata: 0xacc8f306 + nonce (32 bytes)
    let calldata = format!("0xacc8f306{}", hex::encode(nonce));
    
    // Get tx count
    let body = rpc_call(rpc, "eth_getTransactionCount", serde_json::json!([
        format!("{:?}", wallet), "latest"
    ])).await?;
    let tx_nonce = body["result"].as_str().unwrap_or("0x0");
    
    // Get gas price
    let body = rpc_call(rpc, "eth_gasPrice", serde_json::json!([])).await?;
    let gas_price = body["result"].as_str().unwrap_or("0x0");
    
    // Get chain ID
    let body = rpc_call(rpc, "eth_chainId", serde_json::json!([])).await?;
    let chain_id = body["result"].as_str().unwrap_or("0x1");
    
    println!("[TX] nonce={}, gasPrice={}, chainId={}", tx_nonce, gas_price, chain_id);
    println!("[TX] calldata={}", calldata);
    
    // Return calldata for now - actual signing requires private key integration
    Ok(calldata)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    
    let rpc = env::var("RPC_URL").unwrap_or_else(|_| "https://ethereum-rpc.publicnode.com".to_string());
    let wallet_str = env::var("WALLET").expect("WALLET not set in .env");
    let private_key = env::var("PRIVATE_KEY").unwrap_or_default();
    let gpu_batch: u64 = env::var("GPU_BATCH").unwrap_or_else(|_| "0".to_string()).parse().unwrap_or(0);
    
    let wallet = Address::from_str(&wallet_str)?;
    
    println!("=== H98 Miner v0.1 ===");
    println!("Wallet: {:?}", wallet);
    println!("Contract: {}", CONTRACT);
    println!("RPC: {}", rpc);
    
    // Get challenge via async runtime
    let rt = tokio::runtime::Runtime::new()?;
    let challenge = rt.block_on(get_challenge(&rpc, wallet))?;
    let difficulty = rt.block_on(get_difficulty(&rpc))?;
    
    println!("Challenge: 0x{}", hex::encode(&challenge));
    println!("Difficulty: 0x{}", hex::encode(&difficulty));
    
    let challenge = Arc::new(challenge);
    let difficulty = Arc::new(difficulty);
    let found = Arc::new(AtomicBool::new(false));
    let total_hashes = Arc::new(AtomicU64::new(0));
    let found_nonce = Arc::new(std::sync::Mutex::new(None::<[u8; 32]>));
    
    let num_workers = num_cpus::get().saturating_sub(1).max(1);
    println!("Starting {} CPU workers...", num_workers);
    
    let start = Instant::now();
    
    // Spawn worker threads
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
                            let mut guard = found_nonce.lock().unwrap();
                            *guard = Some(nonce);
                            println!("\n[WORKER {}] FOUND VALID NONCE: 0x{}", wid, hex::encode(&nonce));
                            return;
                        }
                        
                        // Increment nonce
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
    
    // Stats reporting thread
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
    
    // Wait for workers
    for handle in handles {
        let _ = handle.join();
    }
    
    let elapsed = start.elapsed();
    let hashes = total_hashes.load(Ordering::Relaxed);
    
    if let Some(nonce) = *found_nonce.lock().unwrap() {
        println!("\n=== MINING SUCCESS ===");
        println!("Nonce: 0x{}", hex::encode(&nonce));
        println!("Time: {:.1}s", elapsed.as_secs_f64());
        println!("Total hashes: {}", hashes);
        println!("Average rate: {:.2} MH/s", hashes as f64 / elapsed.as_secs_f64() / 1_000_000.0);
        
        // Submit
        if !private_key.is_empty() {
            match rt.block_on(submit_nonce(&rpc, wallet, &nonce)) {
                Ok(tx) => println!("TX calldata: {}", tx),
                Err(e) => println!("Submit error: {}", e),
            }
        } else {
            println!("No PRIVATE_KEY set, skipping on-chain submission");
            println!("To submit manually:");
            println!("  Send tx to {} with data: 0xacc8f306{}", CONTRACT, hex::encode(&nonce));
        }
    } else {
        println!("No solution found in {:.1}s", elapsed.as_secs_f64());
    }
    
    Ok(())
}
