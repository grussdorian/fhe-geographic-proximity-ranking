# Engineering Log: Dense-Area Precision & Nonce Overflow Fixes

### A homage to Claude's coding and hypothesis testing capabilities

**Date:** February 2026  
**Scope:** Fixing two critical bugs in the threshold FHE proximity matching system — a runtime crash from nonce slot overflow and systematic scoring failures in dense urban environments (sub-km distances).

---

## Table of Contents

1. [The Bugs](#1-the-bugs)
2. [Bug 1: Nonce Slot Overflow — The IndexError Crash](#2-bug-1-nonce-slot-overflow)
3. [Bug 2: Dense Area Scoring — The Precision Wall](#3-bug-2-dense-area-scoring)
4. [Attempt 1: Scale the Polynomial Coefficients (Failed)](#4-attempt-1-scale-the-polynomial-coefficients)
5. [Attempt 2: Shrink the Nonces (Partial Success)](#5-attempt-2-shrink-the-nonces)
6. [Attempt 3: Strip the MSBs — City-Scale Normalisation (The Fix)](#6-attempt-3-strip-the-msbs)
7. [The Flaky Test Fix](#7-the-flaky-test-fix)
8. [Final Results](#8-final-results)
9. [Files Changed](#9-files-changed)
10. [Key Insights & Lessons](#10-key-insights--lessons)

---

## 1. The Bugs

Two problems surfaced during a 3-party live test (`python client.py --lead` with 2 joiners):

### Bug A — Runtime Crash (IndexError)
```
IndexError: list index out of range
  values[NONCE_OFFSET + winner_slot]  →  values[33]  (BATCH_SIZE is 32)
```

### Bug B — Wrong Winner in Dense Areas
Manhattan scenario: Alice at Times Square, Bob at Rockefeller Center (~0.8 km), Charlie at Empire State (~1.1 km). The system picked **Charlie** as nearest instead of Bob. When printing scores with full precision:
```
Bob=4.999999997284590e-01  Charlie=5.000000001477929e-01
Diff = -4.193339009361807e-10  (negative — wrong direction!)
```

The score difference was **4×10⁻¹⁰** — a fraction of a billionth. The scoring polynomial couldn't distinguish these distances.

Later, the Shibuya same-block test (7m vs 45m) also failed for the same root cause.

---

## 2. Bug 1: Nonce Slot Overflow

### How the Batch is Laid Out

The CKKS batch has 32 slots, split into two halves:

```
Slots 0–15:   Identity slots  (one-hot ID: 1.0 at the party's assigned slot)
Slots 16–31:  Nonce slots     (nonce_value at NONCE_OFFSET + slot)
```

After scoring, each identity slot holds `score_i` and each nonce slot holds `score_i × nonce_i`.

### The Bug

The winner-detection code scanned **all** slots to find the maximum:

```python
active_slots = [i for i in range(len(values)) if i != initiator_slot and abs(values[i]) > 0.01]
winner_slot = max(active_slots, key=lambda s: values[s])
```

With 6-digit nonces (100,000–999,999), the nonce slots held values like `0.45 × 344,002 = 154,800` — massively larger than the identity slot scores (~0.5). So `winner_slot` would land on slot 17 (a nonce slot), then the code tried `values[NONCE_OFFSET + 17]` = `values[33]` — **boom**, IndexError.

### The Fix

Restrict the scan to identity slots only:

```python
num_assigned = poll_server_status()["slots_assigned"]
id_limit = min(num_assigned, NONCE_OFFSET)  # never scan past slot 15
active_slots = [i for i in range(id_limit) if i != initiator_slot and abs(values[i]) > 0.01]
```

Same fix applied to `test_fhe_proximity.py`'s `run_proximity()` and `test_dense_urban.py`'s `run_scoring()` — both had `max(range(len(values)), ...)` which scanned all 32 slots.

### Why It Wasn't Caught Earlier

Before the nonce-commitment scheme was added, slots 16–31 were always zero. The `range(len(values))` scan was harmless. The bug was introduced when nonces were embedded in the upper half of the batch — a latent interaction between two independently-correct features.

---

## 3. Bug 2: Dense Area Scoring — The Precision Wall

### The Mathematics of the Problem

The scoring pipeline works like this:

1. **Normalise:** Each GPS coordinate is divided by `MAX_COORD` (was 180.0) to get values in `[-1, 1]`
2. **Distance:** `dist² = Δlat² + Δlon²` in normalised space
3. **Selector:** For each pair (i,j), compute `f(dist_j² - dist_i²)` where `f(x) = 0.5 + 0.1125x - 0.00084375x³`
4. **Accumulate:** Sum selectors → score per party
5. **Multiply:** `score × one_hot_id` → result vector

For Manhattan (Rockefeller ~0.8km vs Empire State ~1.1km from Times Square):

```
Δlat_bob = 0.0034/180 = 1.889e-05    Δlon_bob = 0.0079/180 = 4.389e-05
dist²_bob = (1.889e-05)² + (4.389e-05)² = 2.283e-09

Δlat_charlie = 0.0096/180 = 5.333e-05    Δlon_charlie = 0.0002/180 = 1.111e-06
dist²_charlie = (5.333e-05)² + (1.111e-06)² = 2.844e-09

diff = dist²_charlie - dist²_bob = 5.63e-10
```

Feeding this into the selector: `f(5.63e-10) = 0.5 + 0.1125 × 5.63e-10 = 0.500000000000063`

That's a score difference of **6.3×10⁻¹¹** — well below CKKS noise (~10⁻¹⁰ to 10⁻⁹). The result is effectively a **coin flip**.

### The Root Cause: Global Normalisation

Dividing by 180 maps the entire planet to `[-1, 1]`. The MSBs (most significant bits — the integer part of coordinates like `40.xxxx`) consume almost all the representational range. The LSBs (the fractional digits that distinguish Rockefeller from Empire State, like `40.7614` vs `40.7484`) end up in the ~10⁻⁵ range after normalisation, and their **squares** land at ~10⁻¹⁰.

This is the "MSB problem" the user identified: for a city-wide app, the integer parts of the coordinates are redundant. Everyone in Manhattan shares `40.7xxx, -73.9xxx`. Only the last few decimal places matter, and they're being drowned out.

---

## 4. Attempt 1: Scale the Polynomial Coefficients (Failed)

### The Idea

Scale the polynomial coefficients by 10⁸ to amplify the tiny differences:

```
f(x) = 0.5 + 11,250,000·x - 150,000·x³
```

Mathematically, this was sound:
- For Manhattan (x = 5.63e-10): f ≈ 0.506 (clear separation!)
- Still monotonically increasing for |x| ≤ 5 (the full valid range)
- Same number of FHE operations — just different constants, no extra depth

### What Happened

```
CKKS decode error: "The decryption failed because the approximation error is too high"
```

**Every single scenario crashed.** All three (Bay Area, Manhattan, College Campus).

### Why It Failed

This was a subtle CKKS trap. The polynomial evaluates `inner = 11,250,000.0` as a plaintext constant added to the ciphertext. In CKKS, the **magnitude** of plaintext values consumed during homomorphic operations affects the noise budget. The OpenFHE decode check at `ckkspackedencoding.cpp:455` fires when:

```cpp
if (logstd > p - 5.0)  // p = ScalingModSize = 50
```

With 50-bit scaling modulus, adding a constant of ~10⁷ pushes the noise above 2⁴⁵, exceeding the precision budget. The "approximation error" isn't about the polynomial approximation — it's about CKKS's internal noise tracking detecting that the ciphertext has become too noisy to decrypt meaningfully.

### The Lesson

**In CKKS, you can't just scale coefficients arbitrarily.** The crypto system tracks noise proportional to the magnitude of values flowing through the circuit. Large plaintext constants poison the noise budget even though they're "just constants." This is fundamentally different from cleartext arithmetic where scaling is free.

---

## 5. Attempt 2: Shrink the Nonces (Partial Success)

### The Idea

If we can't amplify the signal, reduce the noise. The `score × id_ct` multiplication amplifies CKKS noise proportionally to the nonce magnitude. Reducing nonces from 6-digit (100,000–999,999) to 4-digit (1,000–9,999) cuts noise amplification by ~50×.

### Changes

```python
# Before:
nonce_val = secrets.randbelow(900000) + 100000  # 100000–999999

# After:
nonce_val = secrets.randbelow(9000) + 1000      # 1000–9999
```

Changed in `client.py`, `demo.py`, and `test_fhe_proximity.py`.

### Results

With `MAX_COORD = 180`:

| Scenario | Score Diff | Result |
|---|---|---|
| Bay Area (34 vs 67 km) | 2.55e-06 | ✅ PASS |
| Manhattan (0.8 vs 1.1 km) | 5.42e-10 | ⚠️ Barely passes (printed as 0.500000, but full precision shows Bob wins) |
| College Campus (220m vs 490m) | 9.70e-10 | ⚠️ Barely passes |
| Shibuya (7m vs 45m) | ~3e-13 | ❌ FAIL (noise > signal) |

The nonce reduction helped but didn't solve the fundamental problem: the normalised differences were still too tiny. Manhattan worked by luck — the score difference was within 1 order of magnitude of the noise floor. Shibuya (7m) was still hopeless.

### The Lesson

Reducing noise is necessary but not sufficient. The signal itself needs to be amplified. But we can't do it with polynomial coefficients (Attempt 1 showed that). We need to amplify the signal **before** it enters the FHE circuit.

---

## 6. Attempt 3: Strip the MSBs — City-Scale Normalisation (The Fix)

### The User's Insight

> "Note that I'm making a city wide app... you may get rid of MSBs and only keep LSBs (rightmost bits which capture small changes)"

This was the key insight. If the app is city-scale, `MAX_COORD = 180` is absurd overkill. Everyone in Manhattan shares the same `40.7, -73.9` prefix. We're wasting 99.7% of the normalised range on information that's identical across all participants.

### The Fix

```python
MAX_COORD = 0.5  # ≈ 55 km at the equator
```

This is conceptually equivalent to "stripping the MSBs." Instead of mapping [-180, 180] → [-1, 1], we map [-0.5, 0.5] → [-1, 1]. The raw coordinates (like 40.7580) still get divided by 0.5, producing values like 81.516 — but the **differences** between coordinates are what matter in the distance computation:

```
dlat = (40.7614 - 40.7580) / 0.5 = 0.0068
dlon = (-73.9776 - (-73.9855)) / 0.5 = 0.0158
dist² = 0.0068² + 0.0158² = 2.96e-04    ← was 2.28e-09 with MAX_COORD=180
```

The amplification factor is `(180/0.5)² = 129,600×`.

### Why This Works and Attempt 1 Didn't

| Approach | Where Signal is Amplified | Noise Impact |
|---|---|---|
| Scale polynomial coefficients | Inside the FHE circuit (server-side) | Amplifies noise equally — S/N unchanged; large constants poison CKKS noise budget |
| Scale MAX_COORD | Before encryption (client-side, in plaintext) | No noise impact whatsoever — both parties independently normalise using the same constant before anything touches the FHE circuit |

**Attempt 1 tried to fix noise-drowned signals by shouting louder in a noisy room. Attempt 3 moved the conversation to a quiet room before anyone started talking.**

The normalisation happens in `encrypt_location()` and `compute_distance_local()` on the **client side**, before the values are encrypted. By the time the ciphertext reaches the FHE circuit, the differences are already 129,600× larger. The polynomial sees them at full strength with no extra noise penalty.

### The Signal Improvement

| Scenario | Old Score Diff (MAX_COORD=180) | New Score Diff (MAX_COORD=0.5) | Improvement |
|---|---|---|---|
| Bay Area (34 vs 67 km) | 2.55e-06 | 0.325 | **127,000×** |
| Manhattan (0.8 vs 1.1 km) | 5.42e-10 | 1.64e-05 | **30,000×** |
| College Campus (220 vs 490m) | 9.70e-10 | ~1.5e-05 | **15,000×** |
| Shibuya (7m vs 45m) | ~3e-13 | ~3e-08 | **100,000×** |

### Monotonicity Check

The selector polynomial `f(x) = 0.5 + 0.1125x - 0.00084375x³` is monotonically increasing for:

```
f'(x) = 0.1125 - 3 × 0.00084375 × x² > 0
x² < 0.1125 / 0.00253125 = 44.44
|x| < 6.67
```

Wait — actually let me be precise. The polynomial is `f(x) = 0.5 + 0.1125x - 0.00084375x³`:

```
f'(x) = 0.1125 - 0.00253125x² = 0  →  x = ±6.67
```

And it crosses 0.5 (indifference) when `0.1125x - 0.00084375x³ = 0`, i.e. `x(0.1125 - 0.00084375x²) = 0` → `x = ±11.55`.

For the largest city scenario (Bay Area, 67 km apart):
```
diff = dist²_charlie - dist²_bob with MAX_COORD=0.5
     ≈ 2*(0.3/0.5)² = 0.72  (well within the monotonic range of 6.67)
```

For an extreme city boundary (~100 km): `diff ≈ 2*(0.9/0.5)² ≈ 6.5` — still within range. ✓

### Removing Non-City E2E Scenarios

With `MAX_COORD = 0.5`, parties separated by hundreds of km would produce normalised differences exceeding the polynomial's valid range. Since this is explicitly a city-wide app, four E2E scenarios were removed:

- **Cross-country road trip** (Denver–Chicago, 1500 km)
- **Arctic research stations** (Svalbard, 192 km)
- **Equatorial meetup** (Kenya, 123 km)
- **New Zealand meetup** (Auckland–Wellington, 493 km)

The remaining 11 scenarios cover 6 continents, 3–5 party counts, and distances from 200m (College Campus) to 67 km (Bay Area).

---

## 7. The Flaky Test Fix

### The Bug

`test_non_winner_nonces_are_near_zero` in `test_fhe_proximity.py` compared **nonce-weighted** values:

```python
winner_nonce_val = abs(values[NONCE_OFFSET + winner])   # = score_winner × nonce_winner
loser_nonce_val = abs(values[NONCE_OFFSET + i])         # = score_loser × nonce_loser
self.assertGreater(winner_nonce_val, loser_nonce_val)   # ← WRONG
```

With random 4-digit nonces, a loser with nonce=9800 and score=0.3 produces `2940`, while the winner with nonce=1200 and score=0.7 produces `840`. The losers' nonce-weighted value can easily exceed the winner's — the test was logically incorrect.

### The Fix

Compare **identity-slot scores** (which don't carry random nonce scaling) and add a nonce round-trip extraction check:

```python
winner_score = values[winner]
for i in range(1, len(locs)):
    if i == winner:
        continue
    loser_score = values[i]
    self.assertGreater(winner_score, loser_score)

# Also verify nonce extraction round-trip
nonce_weighted = values[NONCE_OFFSET + winner]
extracted = nonce_weighted / winner_score
self.assertAlmostEqual(extracted, expected, delta=20)
```

---

## 8. Final Results

### test_fhe_proximity.py: 38/38 ✅

Global-scale test suite (uses `MAX_COORD = 180` independently — these are algorithm correctness tests, not city-specific).

Includes:
- Encryption roundtrips
- Local distance computation
- 2, 3, 5, 10, 20-party proximity matching
- Nonce commitment/extraction (3-party and 5-party)
- Nonce commitment verification via brute-force
- Non-winner score suppression
- Shared key derivation
- N-of-N threshold enforcement (2, 3, 5, 10, 20 parties)

### test_dense_urban.py: 34/34 ✅

City-scale tests with 20 users per city:

- 5 cities × 20 users × 5 initiator queries = 25 city tests
- 2 same-block tests (Times Square ~30m, Shibuya ~7m)
- 3 security tests (full decrypt, partial decrypt fails, single-party fails)
- 4 Manhattan cluster + borough outlier tests
- 3 SF cluster + ocean outlier tests

All pass including Shibuya 7m resolution — the test that was failing before.

### test_e2e_client_server.py: 11/11 ✅

Full HTTP client-server protocol tests with fresh server per scenario:

Every scenario verifies **all seven** properties:
1. ✅ Correct nearest neighbour identification
2. ✅ Nonce commitment verification
3. ✅ Match proof verification
4. ✅ Shared key derivation (initiator = winner)
5. ✅ Local encrypt → decrypt roundtrip
6. ✅ Reverse-direction encrypt → decrypt
7. ✅ Server-relayed E2E encrypted chat message

---

## 9. Files Changed

| File | Changes |
|---|---|
| `client.py` | `MAX_COORD: 180→0.5` with detailed comment; `active_slots` restricted to identity range; nonces reduced to 4-digit |
| `server.py` | `MAX_COORD: 180→0.5`; updated selector polynomial docstring |
| `demo.py` | `MAX_COORD: 180→0.5`; nonces reduced to 4-digit |
| `test_dense_urban.py` | `MAX_COORD: 180→0.5`; `winner_slot` scan restricted to participant slots |
| `test_fhe_proximity.py` | `run_proximity()` winner scan restricted; nonces reduced to 4-digit; `test_non_winner_nonces_are_near_zero` logic fixed |
| `test_e2e_client_server.py` | Removed 4 non-city scenarios (Cross-country, Arctic, Equatorial, New Zealand) |
| `README.md` | Updated MAX_COORD explanation, added dense-area resolution paragraph |

---

## 10. Key Insights & Lessons

### 1. "Where you amplify matters more than how much you amplify"

The critical distinction between Attempt 1 (failed) and Attempt 3 (succeeded) is **where** the signal amplification happens:

- **Inside the FHE circuit** (polynomial coefficients): the noise is amplified equally. S/N ratio is unchanged. And large constants can exceed the CKKS noise budget entirely, causing decode failures.
- **Before encryption** (coordinate normalisation): the amplification is free. Both parties independently apply the same constant in plaintext, and the FHE circuit sees already-amplified differences with no extra noise.

This is perhaps the most important lesson for anyone building CKKS applications: **push as much computation as possible into the plaintext domain**, especially scaling and normalisation. The FHE circuit should only see the minimal, well-conditioned version of the actual computation.

### 2. "CKKS constants are not free"

In normal arithmetic, `x + 11250000` is the same operation regardless of the constant. In CKKS, large constants interact with the noise budget. The `ScalingModSize` (50 bits in our setup) sets an effective precision ceiling — the OpenFHE decode check fires when `log₂(noise) > 50 - 5 = 45`. Adding a 24-bit constant (~10⁷) to a ciphertext that's already consumed 4 levels of depth was enough to push noise past this threshold.

### 3. "The MSB stripping intuition is exactly right"

The user's intuition — "out of the 64 bits only the rightmost bits capture meaningful info in a city" — maps perfectly to the MAX_COORD fix. Dividing by 0.5 instead of 180 is the floating-point equivalent of right-shifting: the high-order bits (the `40` in `40.7580`) become large but **cancel out** in the subtraction `Δlat = lat_a/0.5 - lat_b/0.5`, while the low-order bits (the `.0034` difference) stay proportionally large.

### 4. "Nonces interact with scoring precision in non-obvious ways"

The nonce-commitment scheme places the nonce value in the same ciphertext as the one-hot identity. After `score × id_ct`, the nonce slot holds `score × nonce`. A 6-digit nonce (999,999) amplifies the remaining CKKS noise by ~10⁶ in those slots. While identity slots (carrying just `score ≈ 0.5`) have noise ~10⁻⁹, nonce slots have noise ~10⁻³. If the winner detection accidentally scans nonce slots, these dominate.

This created a subtle two-bug interaction: the nonce overflow crash only happened **because** nonce slots had large values that won the argmax. With 4-digit nonces AND the identity-slot-only scan, both failure modes are eliminated.

### 5. "Test what you check, check what you test"

The `test_non_winner_nonces_are_near_zero` test was comparing nonce-weighted values (score × nonce), asserting the winner's must be largest. But with random nonces, a loser with a high random nonce easily exceeds the winner with a low one. The test was asserting a property that **doesn't hold by design** — it passed by luck with 6-digit nonces (where the magnitude difference between winner and loser nonces was smaller relative to the score difference) and broke with 4-digit nonces.

### 6. "Tolerance is a feature, not a limitation"

The user noted: "extreme precision is not required when differences are not really meaningful e.g. 7m vs 11m." This tolerance assumption is fundamental to the design. CKKS is an **approximate** encryption scheme — it trades exact precision for the ability to compute on real numbers. For proximity matching, whether someone is 7m or 11m away is meaningless for the use case (finding who's nearby for a meetup, ride-share, etc.). The system's job is to reliably rank 500m vs 5km, or even 200m vs 800m. The dense urban tests showing 7m resolution are a bonus of the city-scale normalisation, not a requirement.

### 7. The Combined Fix: Defence in Depth

The final solution combines three orthogonal fixes:

| Fix | What It Prevents | Signal Improvement |
|---|---|---|
| `MAX_COORD = 0.5` | Signal loss from global normalisation | 129,600× |
| 4-digit nonces | Noise amplification in score × id_ct | ~50× noise reduction |
| Identity-slot-only scan | Nonce slots winning argmax | Correctness (was crashing) |

Each fix addresses a different failure mode. Together, they provide robust sub-km resolution in dense urban environments with a comfortable margin.
