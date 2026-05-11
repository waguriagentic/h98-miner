# HASH Token Miner (Rust + OpenCL)

Port Rust dari `miner.js` di folder induk, plus backend GPU OpenCL. Native, multi-threaded, jauh lebih cepat dari versi Node.js.

## Kenapa Rust + GPU?

| Hal | Node.js (`miner.js`) | Rust CPU | Rust GPU (OpenCL) |
|---|---|---|---|
| Hashing | single-threaded JS | semua core via `std::thread::scope` | ribuan work-item per dispatch |
| Per-attempt cost | `await contract.currentEpoch()` per nonce (RPC roundtrip!) | murni CPU | murni GPU |
| Hash rate (laptop 8-core) | ~1k–5k H/s | ~MH/s s.d. puluhan MH/s | tergantung GPU (lihat di bawah) |
| Binary | butuh Node + 200 MB `node_modules` | satu binary statis | satu binary statis |

Hash rate yang sudah diukur:
- **NVIDIA GT 1030** (384 CUDA cores) + Xeon E5-2698 v4 (19 thread): **~127 MH/s**
- Hardware modern (RTX 3060+) seharusnya tembus GH/s — optimasi pipeline + midstate yang ada di sini akan memberi gain lebih signifikan di GPU yang lebih powerful.

## Requirements

- Rust toolchain (>= 1.75) — install via [rustup.rs](https://rustup.rs)
- Wallet Ethereum dengan ETH untuk gas
- RPC endpoint (default `https://eth.llamarpc.com` — publik, ganti ke Alchemy/Infura/punyamu kalau bisa)
- **Untuk GPU:** OpenCL runtime + driver:
  - NVIDIA: driver resmi (CUDA Toolkit opsional, OpenCL ICD sudah include di driver)
  - AMD: ROCm atau AMDGPU-PRO
  - Intel: Intel Compute Runtime

## Build

```bash
cargo build --release
```

Binary keluar di `target/release/hash-miner-rs` (atau `.exe` di Windows).

Build CPU-only (skip OpenCL dependency):
```bash
cargo build --release --no-default-features
```

## Run

### Cara 1: env var (paling aman)

Bash:
```bash
PRIVATE_KEY=0xabc... GPU=1 ./target/release/hash-miner-rs
```

PowerShell:
```powershell
$env:PRIVATE_KEY = "0xabc..."
$env:GPU = "1"                                                  # optional, hidupin GPU
$env:RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"  # optional
$env:MINER_THREADS = "8"                                        # optional, default = num CPUs - 1
.\target\release\hash-miner-rs.exe
```

### Cara 2: prompt interaktif

```bash
./target/release/hash-miner-rs
# masukin private key pas diminta (input disembunyiin via rpassword)
```

### Cara 3: .env file

Copy `.env.example` ke `.env`, isi nilai, lalu jalankan binary biasa — `dotenvy` otomatis load.

## Environment variables

| Variable | Default | Keterangan |
|---|---|---|
| `PRIVATE_KEY` | — (prompt) | Private key 64 hex char, optional `0x` prefix. |
| `RPC_URL` | `https://eth.llamarpc.com` | RPC HTTP endpoint. WS belum disupport. |
| `MINER_THREADS` | `num_cpus - 1` | Jumlah CPU worker thread. |
| `GPU` | `0` | Set `1` untuk pakai backend OpenCL. |
| `GPU_BATCH` | `4194304` (`1 << 22`) | Nonce per GPU dispatch. Naikkan kalau VRAM cukup dan GPU underutilized. |
| `PRIORITY_GWEI` | `5.0` | Tip miner EIP-1559 (gwei). |
| `MAX_FEE_GWEI` | `100.0` | Ceiling max-fee EIP-1559 (gwei). |
| `GAS_LIMIT_OVERRIDE` | (auto-estimate) | Force gas limit ke nilai fix. |

## Konfigurasi non-env

Konstanta lain ada di atas `src/main.rs`:
- `HASH_CONTRACT_ADDRESS` — `0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc`
- `EPOCH_BLOCKS` — 100 (matches contract)
- `EPOCH_POLL_INTERVAL` — 15 detik (seberapa sering watchdog cek epoch berubah)
- `STATS_INTERVAL` — 2 detik (refresh hash rate display)

## Cara kerja mining

Algoritma sama dengan versi JS:
1. **Challenge** = `keccak256(abi.encodePacked(chainId, contract, miner, epoch))` — contract recomputes ini, miner cuma fetch.
2. **Proof** = `keccak256(abi.encode(bytes32 challenge, uint256 nonce)) < currentDifficulty`
3. **Reward** = `100 HASH >> (totalMints / 100_000)` — halving tiap 100k mints.

### Di Rust

**Per ronde:**
- Fetch `miningState()` + `getChallenge()` paralel via `tokio::join!` — 2 RPC round-trip ganti 3 yang serial.
- Random `start_nonce: u64` per ronde supaya miner di wallet yang sama gak overlap.
- Watchdog `tokio` task: poll block number tiap 15 detik. Begitu epoch berubah, signal stop ke worker via `AtomicBool` dan ronde di-restart.

**CPU backend:**
- N worker thread, tiap thread iterasi `nonce += num_threads` mulai dari `start_nonce + tid` — stride memastikan tidak ada dua thread yang hash nonce sama.
- `nonce` di-track sebagai `u64` (native ALU ops, bukan U256 yang heavyweight).
- Difficulty di-compare via lexicographic byte compare langsung pada output Keccak — hindari `U256::from_be_bytes` per attempt.

**GPU backend (OpenCL):**
- Kernel Keccak-f[1600] ditulis tangan di `src/keccak_kernel.cl`.
- **Pipeline depth 2** — selalu ada 2 batch antri di queue. Saat host blocking-read hasil batch N, GPU sudah running batch N+1.
- **Midstate precompute** — round 1 Theta delta (D0, D2, D4) precomputed di host dari challenge, dipasang sebagai kernel arg. Kernel skip Theta di round 1.
- **Kernel build sekali per ronde** — bukan per dispatch. Hanya `set_arg(nonce_base)` di hot loop.
- **`clEnqueueFillBuffer`** untuk reset flag/nonce buffer — driver path lebih cepat dari host write.
- **Self-test di startup** — verifikasi GPU output match CPU reference dengan known challenge. Kalau kernel salah, binary abort sebelum mining.

## Stop

`Ctrl+C` — signal worker untuk berhenti setelah attempt sekarang, lalu print statistik akhir. GPU queue di-drain bersih sebelum exit.

## Security

- Jangan commit private key. Pakai env var, prompt, atau `.env` (yang ada di `.gitignore`).
- RPC publik bisa rate-limit / di-MITM. Pakai punyamu kalau bisa.
- Mining butuh ETH untuk gas. Kalau hash rate ketemu solusi tapi gas-mu kurang, tx revert.
- GPU self-test verifikasi correctness — kalau gagal, jangan dipakai (kemungkinan driver/OpenCL bermasalah).

## Troubleshooting

**`GPU=1` tapi miner pakai CPU:**
- Binary di-build tanpa feature `gpu`? Cek pesan "binary built without the `gpu` feature".
- OpenCL runtime/driver belum terinstall? Cek `clinfo` di Linux atau cek Device Manager di Windows.

**GPU self-test FAILED:**
- Driver outdated atau OpenCL ICD bermasalah. Update driver.
- Kalau persisten, build CPU-only dan bug report dengan device name.

**Hash rate jauh lebih rendah dari yang diharapkan:**
- CPU: cek `MINER_THREADS` — default `num_cpus - 1` udah optimal kebanyakan kasus.
- GPU: coba naikkan `GPU_BATCH` ke `8388608` (`1 << 23`) atau lebih kalau VRAM cukup.

## License

MIT — risiko ditanggung sendiri.
