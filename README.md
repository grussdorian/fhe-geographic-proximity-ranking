# Private Proximity Matching with Threshold Fully Homomorphic Encryption

## Web demo

```bash
python web_server.py
```

Then open `localhost:5000`

## Manifesto

**The problem is simple: you want to find who's nearby. The cost should not be your privacy.**

Every proximity-based system today — ride-sharing, dating apps, friend finders, local marketplaces — demands that you surrender your exact location to a central server. That server knows where you are, where you've been, and who you've been near. It becomes a honeypot for surveillance, breaches, and abuse. Users are forced into a binary choice: participate and lose privacy, or opt out entirely.

**We reject that tradeoff.**

This project demonstrates that proximity matching can be done *without anyone learning anyone else's location* — not the server, not other users, not even a colluding subset of participants. The entire computation happens on encrypted data using **Fully Homomorphic Encryption (FHE)**, and decryption requires the **unanimous participation of every party** through a threshold protocol.

No trusted third party. No plaintext locations on any server. No single point of compromise.

---

## What This Is

A working proof-of-concept for **privacy-preserving proximity ranking** using **threshold FHE** (specifically, the CKKS scheme via [OpenFHE](https://github.com/openfheorg/openfhe-development)).

Given *N* parties (up to 20), the system determines who is closest to an initiator — without revealing:

- Any party's actual coordinates
- The distances between parties
- The final ranking to the server

### The Protocol

```
1. SETUP        All N parties collaboratively generate a shared encryption key.
                No single party holds the full secret — each holds only a share.

2. ENCRYPT      The initiator encrypts their coordinates under the joint public key
                and broadcasts the ciphertext.

3. COMPUTE      Each party locally computes their encrypted distance to the initiator
                using their own plaintext coordinates and the initiator's ciphertext.
                Their coordinates never leave their device.

4. SCORE        The server performs pairwise scoring on the encrypted distances
                using a polynomial approximation — entirely on ciphertext.
                The server learns nothing.

5. DECRYPT      ALL N parties contribute a partial decryption.
                The server fuses these into the plaintext result.
                If even one party withholds their share, decryption fails entirely.

6. RESOLVE      The initiator reads the result: a one-hot vector indicating
                the nearest party. Only the initiator learns who is closest.
```

### Security Properties

| Property | Guarantee |
|---|---|
| **Server learns nothing** | The server operates exclusively on ciphertexts. It never sees coordinates, distances, or results in plaintext. |
| **No single-party decryption** | The secret key is split across all *N* parties. Decryption requires every share — this is **N-of-N threshold**, not majority. |
| **Collusion resistance** | Even if the initiator and server collude, they cannot decrypt without the remaining parties' shares. Any *N-1* subset is insufficient. |
| **Local distance computation** | Each party computes their distance on their own device using their plaintext coordinates and the initiator's ciphertext. Coordinates never leave the device. |
| **128-bit security** | CKKS parameters use `HEStd_128_classic`, the standard post-quantum security level. |

### What Happens If a Party Refuses to Decrypt?

Decryption fails completely. OpenFHE throws a `RuntimeError` — not garbage output, not a degraded result, but a hard cryptographic failure. This is by design. Privacy is not probabilistic here; it is absolute.

---

## Architecture

```
┌─────────────┐  ┌─────────────┐       ┌─────────────┐
│   Party 0   │  │   Party 1   │  ...  │   Party N   │
│   (Lead)    │  │   (Joiner)  │       │   (Joiner)  │
│             │  │             │       │             │
│ - KeyGen()  │  │ - Multi     │       │ - Multi     │
│ - Encrypt   │  │   KeyGen()  │       │   KeyGen()  │
│   coords    │  │ - Compute   │       │ - Compute   │
│ - Partial   │  │   distance  │       │   distance  │
│   decrypt   │  │   locally   │       │   locally   │
│   (Lead)    │  │ - Partial   │       │ - Partial   │
│ - Resolve   │  │   decrypt   │       │   decrypt   │
│   result    │  │   (Main)    │       │   (Main)    │
└──────┬──────┘  └──────┬──────┘       └───────┬─────┘
       │                │                      │
       └────────────────┼──────────────────────┘
                        │  ciphertexts only
                        ▼
               ┌─────────────────┐
               │     Server      │
               │                 │
               │ - Holds NO keys │
               │ - Pairwise      │
               │   scoring on    │
               │   ciphertexts   │
               │ - Fuses partial │
               │   decryptions   │
               │ - Learns NOTHING│
               └─────────────────┘
```

### Depth Budget

The CKKS scheme has a fixed multiplicative depth (set to **7**). Every homomorphic multiplication consumes one level. Our protocol uses **5 levels**:

| Operation | Who | Depth Cost |
|---|---|---|
| Distance² = Δlat² + Δlon² | Client (local) | 1 level |
| Selector polynomial (Horner form) | Server | 3 levels |
| Score × one-hot ID | Server | 1 level |
| **Total** | | **5 of 7** |

The selector polynomial approximates a proximity comparator:

$$s(x) = 0.5 + x \cdot (0.1125 - 0.00084375 \cdot x^2)$$

This smooth polynomial maps distance differences to scores without branching — which is impossible on encrypted data.

Coordinates are normalised by `MAX_COORD = 0.5` (≈ 55 km at the equator) instead of 180, amplifying within-city squared-distance differences by **129 600×**. The polynomial is monotonically above 0.5 for positive inputs up to $x \approx 11.55$, which covers cities up to ~100 km in diameter. Combined with 4-digit nonces (max 9 999 — keeping CKKS noise low through the `score × ID` multiplication), this gives reliable sub-km resolution for dense urban scenarios.

---

## Files

| File | Purpose |
|---|---|
| `setup_keys.py` | Generates the shared CKKS crypto context with `MULTIPARTY` enabled. Run once before starting the system. |
| `server.py` | HTTP server implementing the full threshold FHE protocol. 12 endpoints covering key generation, matching, scoring, and decryption. Holds no secret keys. |
| `client.py` | Two-role client (`--lead` / `--join`). Handles key share generation, local distance computation, and partial decryption. |
| `demo.py` | Self-contained orchestrator that runs the entire protocol in one process. Useful for testing and demonstration without HTTP. |
| `test_fhe_proximity.py` | 38 tests covering encryption roundtrips, local distance computation, proximity matching (2–20 parties), nonce commitment/extraction, shared key derivation, edge cases, symmetry, and N-of-N threshold enforcement. |
| `test_dense_urban.py` | 34 tests across 5 dense urban environments (Manhattan, SF, Tokyo, London, Mumbai) with 20 users each, plus same-block precision (7m–50m), cluster detection, and threshold security tests. |
| `test_e2e_client_server.py` | 11 end-to-end scenarios testing the full HTTP client–server protocol with nonce-commitment, match proof, and E2E encrypted chat. Covers cities worldwide (Bay Area, Manhattan, London, Tokyo, Sydney, Paris, Mumbai, LA, SF) at distances from 200m to 67km. Auto-manages server lifecycle per test. |
| `test_logger.py` | Custom test runner with failure logging. |

---

## Getting Started

### Prerequisites

- **macOS** (ARM64 tested) or Linux
- **Python 3.13+**
- **OpenFHE** built from source with Python bindings

### Build OpenFHE

```bash
# Clone and build OpenFHE
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
mkdir build && cd build
cmake .. -DBUILD_EXTRAS=ON
make -j$(sysctl -n hw.ncpu)

# Build Python bindings
cd ../openfhe-python
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Set Environment

```bash
export PYTHONPATH="/path/to/openfhe-development/openfhe-python/build"
```

### Generate Crypto Context

```bash
python3.13 setup_keys.py
```

### Run the Demo

```bash
# 3-party demo (default)
python3.13 demo.py

# 5-party demo
python3.13 demo.py --parties 5
```

### Run Client–Server Mode

Terminal 1:
```bash
python3.13 server.py
```

Terminal 2 (lead party):
```bash
python3.13 client.py --lead --name Alice --parties 3 --lat 40.7580 --lon -73.9855
```

Terminal 3 (joining party):
```bash
python3.13 client.py --join --name Bob --lat 40.7614 --lon -73.9776
```

Terminal 4 (joining party):
```bash
python3.13 client.py --join --name Carol --lat 40.7484 --lon -73.9856
```

| Flag | Description |
|---|---|
| `--lead` / `--join` | Role: first party (lead) or subsequent joiner |
| `--name` | Display name for this party |
| `--parties` | Total expected parties (lead only, default 3) |
| `--lat`, `--lon` | GPS coordinates |
| `--server` | Server URL (default `http://localhost:8080`) |

---

## Testing

82 tests total. All passing.

### Run All Tests

```bash
# General proximity tests (33 tests, ~60s)
python3.13 test_fhe_proximity.py

# Dense urban tests (34 tests, ~5min)
python3.13 test_dense_urban.py

# End-to-end client–server tests (15 scenarios, ~3min)
python3.13 test_e2e_client_server.py
```

### What the Tests Cover

**Correctness:**
- Encryption/decryption roundtrips (locations, one-hot vectors, negative coordinates, precision)
- Local distance computation across scales (1km to 16,000km)  
- Proximity ranking with 2, 3, 5, 10, and 20 parties
- Dense urban environments: Manhattan, San Francisco, Tokyo, London, Mumbai
- Same-block resolution (users 30–50m apart in Times Square and Shibuya)
- Cluster detection with geographic outliers

**Security (N-of-N threshold enforcement):**
- Missing any non-lead party → `RuntimeError`
- Missing the lead party → `RuntimeError`
- Single party attempting solo decryption → `RuntimeError`
- Verified for party counts: 2, 3, 5, 10, 20

**End-to-end client–server (15 scenarios):**
- Bay Area (SF → San Jose, Oakland, Palo Alto)
- Manhattan (Times Square → Central Park, Wall St, Brooklyn)
- London (Big Ben → Tower Bridge, Camden, Greenwich)
- Tokyo (Shibuya → Shinjuku, Akihabara, Asakusa)
- Stanford campus (~200m separation)
- Cross-country USA (NYC → LA, Chicago, Miami)
- Sydney (Opera House → Harbour Bridge, Bondi, Manly)
- Paris 5-party (Eiffel Tower → Louvre, Notre-Dame, Sacré-Cœur, Bastille)
- Mumbai (Gateway → CST, Bandra, Juhu)
- Arctic (Svalbard, ~0.5° lat separation)
- Equatorial Kenya (Nairobi → Mombasa, Kisumu, Nakuru)
- Ride-share (pickup → 3 nearby drivers)
- Hospital (ER → pharmacy, lab, radiology)
- Food delivery 5-party (restaurant → 4 couriers)
- New Zealand (Wellington → Auckland, Christchurch, Queenstown)

### With Failure Logging

```bash
python3.13 test_fhe_proximity.py --log
python3.13 test_dense_urban.py --log
```

The `--log` flag generates detailed failure reports in the `logs/` directory, including timestamps, device information, and full tracebacks.

---

## Performance

Measured on Apple M-series (ARM64), single-threaded:

| Scenario | Parties | Time |
|---|---|---|
| Demo (SF Bay Area) | 3 | ~0.7s |
| Demo (SF Bay Area) | 5 | ~1.2s |
| Full test suite (proximity) | 2–20 | ~60s |
| Full test suite (dense urban) | 20 | ~5 min |
| E2E client–server suite | 3–5 | ~3 min |

Key sizes are modest: multiplicative depth 7 yields ~26 MB of key material (down from 312 MB at depth 20 in an earlier iteration).

---

## Why Threshold FHE, Not Just FHE

Standard FHE has a single secret key holder. That party can decrypt everything. In a proximity system, this means either:

1. A central server holds the key → it sees all locations (defeats the purpose)
2. One user holds the key → they see all distances (unacceptable)

**Threshold FHE** splits the key across all participants. The math enforces that decryption is impossible without unanimous cooperation. This isn't a policy or a pinky promise — it's a cryptographic invariant. OpenFHE's implementation makes this a hard failure: attempt to decrypt with missing shares and the library throws an exception, not a wrong answer.

This makes the system **collusion-resistant by construction**. Any coalition of N-1 parties, including the server, learns nothing about the missing party's data.

---

## Limitations and Future Work

- **Quadratic comparisons**: Pairwise scoring is O(N²). At N=20, this means 190 comparisons — feasible but not cheap. Sublinear approaches (e.g., tree-based protocols) are an open research area.
- **Single round**: The current protocol finds the nearest neighbor in one round. Iterative refinement or k-nearest queries would require protocol extensions.
- **No persistent identity**: Parties are ephemeral. Integrating with identity systems (e.g., DIDs) is left for future work.
- **Approximate arithmetic**: CKKS is an approximate scheme. Very close distances (< 30m) may be indistinguishable depending on coordinate precision.
- **Normalized distance metric**: Distances are computed as `(Δlat/180)² + (Δlon/180)²` — a normalized Euclidean approximation, not haversine. This works well for city-scale applications where latitude/longitude degrees are roughly equal in physical distance. At extreme latitudes (e.g., near the poles), 1° longitude shrinks physically, so the metric overestimates east-west distances. Near the antimeridian (±180° longitude), the linear subtraction produces maximum distance instead of minimum. Both are acceptable for the intended city-based use case.
- **Network overhead**: Ciphertext serialization adds bandwidth cost. The current HTTP protocol is unoptimized.

---

## License

This is a research prototype and proof-of-concept. OpenFHE is licensed under BSD 2-Clause.