# H98 Miner

Rust CPU miner for the H98 (hash98) token on Ethereum.

## Contract

- **Address**: `0x1E5adF70321CA28b3Ead70Eac545E6055E969e6f`
- **Token**: H98 / hash98 (18 decimals, 21M supply)
- **Mining**: `0xacc8f306(uint256 nonce)` — find nonce where `keccak256(abi.encode(challenge, nonce)) < difficulty`
- **Reward**: 1000 H98 per successful mine

## Setup

```bash
cp .env.example .env
# Edit .env with your wallet and RPC

cargo build --release
./target/release/h98-miner
```

## How It Works

1. Fetches your mining challenge from the contract (`getChallenge(address)`)
2. Fetches current difficulty
3. Spawns N-1 CPU threads to brute-force nonces
4. On finding a valid nonce, prints it for on-chain submission

## Requirements

- Rust 1.70+
- Registered wallet on the H98 contract (must call registration function first)
- Ethereum RPC endpoint

## Based On

[hash256-cli-with-gpu](https://github.com/waguriagentic/hash256-cli-with-gpu) — adapted for H98 contract ABI and 32-byte nonces.
