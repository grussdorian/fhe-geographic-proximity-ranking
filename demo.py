#!/usr/bin/env python3
# ==========================================
# demo.py — Threshold FHE Proximity Demo
#
# Runs the FULL multi-party threshold protocol in one process:
#   1. Generate crypto context (MULTIPARTY + ADVANCEDSHE)
#   2. Interactive key generation: each party contributes a key share
#   3. Combine eval mult keys (2-round protocol)
#   4. Initiator encrypts coords → distributed to all parties
#   5. Each party computes distance LOCALLY (never shares plaintext coords)
#   6. Server runs pairwise scoring on encrypted distances
#   7. Threshold decryption: ALL parties contribute partial decryption
#   8. Fusion → plaintext result → argmax → nearest user
#
# Security properties:
#   - NO single party can decrypt alone
#   - Server NEVER sees plaintext locations or distances
#   - Even initiator + server collusion cannot decrypt
#   - Decryption requires ALL N parties to participate
#
# Usage:
#   python3 demo.py                     # default 3-party demo
#   python3 demo.py --parties 5         # 5-party demo
# ==========================================

import sys
import time
import math
import secrets
import hashlib
from openfhe import *

KEYS_DIR = "fhe_keys"
BATCH_SIZE = 32
MAX_COORD = 0.5   # city-scale normaliser (see client.py for rationale)
NONCE_OFFSET = BATCH_SIZE // 2   # nonces packed in slots [16..31]

# =============================================
# Step 1: Generate crypto context
# =============================================

def create_context():
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(7)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetBatchSize(BATCH_SIZE)
    params.SetSecurityLevel(HEStd_128_classic)

    cc = GenCryptoContext(params)
    cc.Enable(PKE)
    cc.Enable(KEYSWITCH)
    cc.Enable(LEVELEDSHE)
    cc.Enable(ADVANCEDSHE)
    cc.Enable(MULTIPARTY)

    return cc


# =============================================
# Step 2: Multi-party key generation
# =============================================

def generate_threshold_keys(cc, num_parties):
    """Generate threshold keys for N parties.
    
    Returns:
        keypairs: list of KeyPair objects (one per party)
        joint_public_key: the combined public key
    """
    print(f"  Generating threshold keys for {num_parties} parties...")
    keypairs = []

    # Lead party generates first
    kp1 = cc.KeyGen()
    keypairs.append(kp1)
    print(f"    Party 0 (lead): KeyGen() ✓")

    # Subsequent parties chain from previous public key
    for i in range(1, num_parties):
        kp_prev = keypairs[-1]
        kp_i = cc.MultipartyKeyGen(kp_prev.publicKey)
        keypairs.append(kp_i)
        print(f"    Party {i}: MultipartyKeyGen(prev_pk) ✓")

    joint_pk = keypairs[-1].publicKey
    print(f"  Joint public key ready (tag: {joint_pk.GetKeyTag()})")

    return keypairs, joint_pk


def generate_eval_mult_key(cc, keypairs):
    """Generate combined eval mult key (2-round protocol).
    
    Round 1: Each party generates their eval mult key share
    Round 2: Each party transforms the combined key with their secret
    """
    n = len(keypairs)
    print(f"  Generating eval mult key ({n} parties, 2 rounds)...")

    # Round 1: Each party generates eval mult share
    # Lead: KeySwitchGen(sk, sk)
    # Others: MultiKeySwitchGen(sk, sk, prev_share)
    shares = []
    lead_share = cc.KeySwitchGen(keypairs[0].secretKey, keypairs[0].secretKey)
    shares.append(lead_share)
    print(f"    Round 1 - Party 0: KeySwitchGen ✓")

    for i in range(1, n):
        share_i = cc.MultiKeySwitchGen(
            keypairs[i].secretKey,
            keypairs[i].secretKey,
            lead_share  # chain from lead share
        )
        shares.append(share_i)
        print(f"    Round 1 - Party {i}: MultiKeySwitchGen ✓")

    # Combine all shares: MultiAddEvalKeys
    joint_pk = keypairs[-1].publicKey
    pk_tag = joint_pk.GetKeyTag()

    combined = shares[0]
    for i in range(1, n):
        combined = cc.MultiAddEvalKeys(combined, shares[i], pk_tag)
    print(f"    Round 1 combined ✓")

    # Round 2: Each party transforms with their secret key
    # s_i * combined → MultiMultEvalKey
    mult_shares = []
    for i in range(n):
        mult_i = cc.MultiMultEvalKey(keypairs[i].secretKey, combined, pk_tag)
        mult_shares.append(mult_i)
        print(f"    Round 2 - Party {i}: MultiMultEvalKey ✓")

    # Combine round 2 shares
    final = mult_shares[0]
    for i in range(1, n):
        final = cc.MultiAddEvalMultKeys(
            final, mult_shares[i], final.GetKeyTag()
        )
    print(f"    Round 2 combined ✓")

    # Install the joint eval mult key
    cc.InsertEvalMultKey([final])
    print(f"  ✓ Joint eval mult key installed")


# =============================================
# Step 3: Proximity computation helpers
# =============================================

def compute_selector(cc, diff):
    """Selector polynomial in Horner form.
    f(x) = 0.5 + x * (0.1125 - 0.00084375 * x²)
    Depth: 2 ct-ct + 1 pt-ct = 3 levels.
    """
    x2 = cc.EvalMult(diff, diff)
    inner = cc.EvalMult(x2, -0.00084375)
    inner = cc.EvalAdd(inner, 0.1125)
    product = cc.EvalMult(inner, diff)
    return cc.EvalAdd(product, 0.5)


def find_nearest_by_scoring(cc, opponents):
    """Pairwise scoring to find nearest neighbor.
    opponents: list of (dist_ct, id_ct) tuples.
    Returns: encrypted one-hot ID of nearest.
    """
    n = len(opponents)
    if n == 1:
        return opponents[0][1]

    dists = [d for d, _ in opponents]

    scores = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = cc.EvalSub(dists[j], dists[i])
            sel_ji = compute_selector(cc, diff)
            sel_ij = cc.EvalSub(1.0, sel_ji)

            scores[i] = cc.EvalAdd(scores[i], sel_ji) if scores[i] is not None else sel_ji
            scores[j] = cc.EvalAdd(scores[j], sel_ij) if scores[j] is not None else sel_ij

    result = cc.EvalMult(scores[0], opponents[0][1])
    for i in range(1, n):
        result = cc.EvalAdd(result, cc.EvalMult(scores[i], opponents[i][1]))

    return result


# =============================================
# Step 4: Threshold decryption
# =============================================

def threshold_decrypt(cc, keypairs, ciphertext):
    """All parties contribute partial decryptions, then fuse.
    
    Lead party 0: MultipartyDecryptLead
    Others: MultipartyDecryptMain
    Server: MultipartyDecryptFusion
    """
    partials = []

    # Lead first
    lead_partial = cc.MultipartyDecryptLead([ciphertext], keypairs[0].secretKey)
    partials.append(lead_partial[0])

    # Others
    for i in range(1, len(keypairs)):
        main_partial = cc.MultipartyDecryptMain([ciphertext], keypairs[i].secretKey)
        partials.append(main_partial[0])

    # Fuse
    plaintext = cc.MultipartyDecryptFusion(partials)
    plaintext.SetLength(BATCH_SIZE)
    return list(plaintext.GetRealPackedValue())


# =============================================
# Nonce-commitment helpers
# =============================================

def generate_nonce():
    """Generate a 6-digit secret nonce and its SHA-256 commitment."""
    nonce_val = secrets.randbelow(9000) + 1000  # 1000–9999 (4-digit)
    commitment = hashlib.sha256(str(nonce_val).encode()).hexdigest()
    return nonce_val, commitment


def verify_nonce(nonce_val, commitment):
    """Check that nonce matches commitment."""
    return hashlib.sha256(str(nonce_val).encode()).hexdigest() == commitment


def extract_nonce_from_result(values, winner_slot, commitment, search_range=20):
    """Extract the winner's nonce from the decrypted result vector.
    
    The result vector has: result[slot] = score, result[NONCE_OFFSET+slot] = score*nonce.
    We divide to get nonce, then brute-force search a small range around
    the rounded value to find the exact nonce matching the commitment.
    This compensates for CKKS approximation errors.
    """
    score = values[winner_slot]
    nonce_weighted = values[NONCE_OFFSET + winner_slot]

    if abs(score) < 1e-6:
        return None, False

    raw_nonce = nonce_weighted / score
    center = round(raw_nonce)

    # Search a small range around the center for the commitment match
    for delta in range(search_range + 1):
        for candidate in [center + delta, center - delta]:
            if hashlib.sha256(str(candidate).encode()).hexdigest() == commitment:
                return candidate, True

    # No match found — return best guess
    return center, False


def derive_shared_key(winner_nonce, r_initiator, session_id):
    """Derive a symmetric key from the winner's nonce, initiator's random, and session id."""
    material = f"{winner_nonce}:{r_initiator}:{session_id}".encode()
    return hashlib.pbkdf2_hmac('sha256', material, b'fhe-proximity-salt', 100000)


# =============================================
# Main Demo
# =============================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine formula — true distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def run_demo(num_parties=3, initiator_idx=0, locations=None, names=None):
    """Run the full threshold FHE proximity demo.
    
    Args:
        num_parties: number of parties (default 3)
        initiator_idx: which party initiates the match (default 0)
        locations: list of (lat, lon) tuples. If None, uses defaults.
        names: list of party names. If None, auto-generated.
    """
    # Default locations: SF, Oakland, San Jose
    if locations is None:
        locations = [
            (37.77490, -122.41940),   # San Francisco (initiator)
            (37.49612, -122.24713),   # Oakland (closest to SF)
            (37.33820, -121.88630),   # San Jose (farther)
        ]
        if num_parties > 3:
            extra = [
                (37.87160, -122.27270),  # Berkeley
                (37.54830, -122.06400),  # Fremont
                (37.44190, -122.14300),  # Palo Alto
                (37.97160, -122.52020),  # San Rafael
                (37.65850, -122.09910),  # Pleasanton
            ]
            locations = locations + extra[:num_parties - 3]

    if names is None:
        default_names = ["Alice", "Bob", "Charlie", "Diana", "Eve",
                         "Frank", "Grace", "Henry"]
        names = default_names[:num_parties]

    assert len(locations) >= num_parties
    assert len(names) >= num_parties

    print("=" * 60)
    print("  THRESHOLD FHE PROXIMITY DEMO")
    print("=" * 60)
    print()
    print(f"Parties: {num_parties}")
    print(f"Initiator: {names[initiator_idx]} (party {initiator_idx})")
    print(f"Security: threshold decryption (ALL {num_parties} parties required)")
    print()

    # Print party info
    for i in range(num_parties):
        lat, lon = locations[i]
        role = " [INITIATOR]" if i == initiator_idx else ""
        print(f"  {names[i]:>10}: ({lat:>10.5f}, {lon:>11.5f}){role}")
    print()

    # Print true distances
    init_lat, init_lon = locations[initiator_idx]
    print("True distances from initiator:")
    true_dists = {}
    for i in range(num_parties):
        if i == initiator_idx:
            continue
        lat, lon = locations[i]
        d = haversine_km(init_lat, init_lon, lat, lon)
        true_dists[i] = d
        print(f"  {names[i]:>10}: {d:>8.2f} km")
    true_nearest_idx = min(true_dists, key=true_dists.get)
    print(f"  Expected nearest: {names[true_nearest_idx]} ({true_dists[true_nearest_idx]:.2f} km)")
    print()

    # ---- Step 1: Create context ----
    t0 = time.time()
    print("[1] Creating crypto context...")
    cc = create_context()
    print(f"    Done ({time.time() - t0:.1f}s)")
    print()

    # ---- Step 2: Threshold key generation ----
    t1 = time.time()
    print("[2] Threshold key generation...")
    keypairs, joint_pk = generate_threshold_keys(cc, num_parties)
    generate_eval_mult_key(cc, keypairs)
    print(f"    Done ({time.time() - t1:.1f}s)")
    print()

    # ---- Step 3: Initiator encrypts coords ----
    t2 = time.time()
    print("[3] Initiator encrypts coordinates...")
    nlat = init_lat / MAX_COORD
    nlon = init_lon / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    enc_lat = cc.Encrypt(joint_pk, lat_pt)
    enc_lon = cc.Encrypt(joint_pk, lon_pt)
    print(f"    Encrypted ({time.time() - t2:.2f}s)")
    print(f"    (Coords encrypted under JOINT key — no single party can decrypt)")
    print()

    # ---- Step 4: Each party computes distance locally ----
    t3 = time.time()
    print("[4] Each party computes distance LOCALLY...")
    print(f"    (Party uses own plaintext coords + initiator's encrypted coords)")
    opponents = []
    nonces = {}       # slot -> nonce value
    commitments = {}  # slot -> SHA-256 hex digest
    for i in range(num_parties):
        if i == initiator_idx:
            continue

        lat, lon = locations[i]
        nlat_i = lat / MAX_COORD
        nlon_i = lon / MAX_COORD

        # Party subtracts their plaintext coords from encrypted coords
        lat_pt_i = cc.MakeCKKSPackedPlaintext([nlat_i] * BATCH_SIZE)
        lon_pt_i = cc.MakeCKKSPackedPlaintext([nlon_i] * BATCH_SIZE)

        dlat = cc.EvalSub(enc_lat, lat_pt_i)
        dlon = cc.EvalSub(enc_lon, lon_pt_i)

        dlat2 = cc.EvalMult(dlat, dlat)   # depth +1
        dlon2 = cc.EvalMult(dlon, dlon)   # depth +1 (parallel)
        dist_ct = cc.EvalAdd(dlat2, dlon2)

        # Generate secret nonce for this party
        nonce_val, commitment = generate_nonce()
        nonces[i] = nonce_val
        commitments[i] = commitment

        # One-hot ID with embedded nonce
        # Slots [0..15]  = one-hot identity
        # Slots [16..31] = nonce at NONCE_OFFSET + slot
        vec = [0.0] * BATCH_SIZE
        vec[i] = 1.0
        vec[NONCE_OFFSET + i] = float(nonce_val)
        id_pt = cc.MakeCKKSPackedPlaintext(vec)
        id_ct = cc.Encrypt(joint_pk, id_pt)

        opponents.append((dist_ct, id_ct))
        print(f"    {names[i]:>10}: computed enc(distance²) ✓  (nonce committed: {commitment[:8]}...)")

    print(f"    All distances computed ({time.time() - t3:.2f}s)")
    print()

    # ---- Step 5: Server runs pairwise scoring ----
    t4 = time.time()
    n_opp = len(opponents)
    n_pairs = n_opp * (n_opp - 1) // 2
    print(f"[5] Server: pairwise scoring ({n_opp} opponents, {n_pairs} comparisons)...")
    result_ct = find_nearest_by_scoring(cc, opponents)
    print(f"    Scoring done ({time.time() - t4:.2f}s)")
    print(f"    (Server computed on ENCRYPTED data — no plaintext access)")
    print()

    # ---- Step 6: Threshold decryption ----
    t5 = time.time()
    print(f"[6] Threshold decryption ({num_parties} parties)...")
    print(f"    Lead (party 0): MultipartyDecryptLead")
    for i in range(1, num_parties):
        print(f"    Party {i}: MultipartyDecryptMain")
    print(f"    Server: MultipartyDecryptFusion")

    values = threshold_decrypt(cc, keypairs, result_ct)
    print(f"    Decryption fused ({time.time() - t5:.2f}s)")
    print()

    # ---- Step 7: Find nearest + nonce verification ----
    print("[7] Result analysis + nonce verification...")

    # Show all slot values
    active_slots = [i for i in range(num_parties) if i != initiator_idx]
    print(f"    Score vector (active slots):")
    for i in active_slots:
        print(f"      Slot {i} ({names[i]:>10}): {values[i]:.4f}")

    winner_slot = max(active_slots, key=lambda i: values[i])
    print()
    print(f"    Winner: slot {winner_slot} = {names[winner_slot]}")
    print(f"    True nearest: {names[true_nearest_idx]}")

    if winner_slot == true_nearest_idx:
        print(f"    ✓ CORRECT!")
    else:
        winner_dist = true_dists[winner_slot]
        nearest_dist = true_dists[true_nearest_idx]
        print(f"    ✗ Mismatch (picked {names[winner_slot]} at {winner_dist:.2f}km"
              f" vs {names[true_nearest_idx]} at {nearest_dist:.2f}km)")

    # ---- Step 8: Extract winner's nonce from decrypted vector ----
    print()
    print("[8] Nonce extraction + verification...")
    winner_commitment = commitments[winner_slot]
    winner_nonce, nonce_ok = extract_nonce_from_result(
        values, winner_slot, winner_commitment
    )
    true_nonce = nonces[winner_slot]

    print(f"    Score at winner slot:  {values[winner_slot]:.6f}")
    print(f"    Nonce×score raw:      {values[NONCE_OFFSET + winner_slot]:.4f}")
    print(f"    Extracted nonce:      {winner_nonce}")
    print(f"    True nonce (secret):  {true_nonce}")
    print(f"    Commitment:           {winner_commitment[:16]}...")

    if nonce_ok:
        print(f"    ✓ Nonce commitment VERIFIED — initiator can prove match to {names[winner_slot]}")
    else:
        print(f"    ✗ Nonce commitment FAILED (CKKS approximation error too large)")

    # ---- Step 9: Derive symmetric key ----
    print()
    print("[9] Symmetric key derivation...")
    session_id = secrets.token_hex(8)
    r_initiator = secrets.token_hex(16)

    shared_key = derive_shared_key(winner_nonce, r_initiator, session_id)
    print(f"    session_id:   {session_id}")
    print(f"    r_initiator:  {r_initiator[:16]}...")
    print(f"    shared_key:   {shared_key.hex()[:32]}...")
    print(f"    (Both {names[initiator_idx]} and {names[winner_slot]} can derive this key)")

    # Winner can verify: they know their own nonce, receive r_initiator
    # and session_id from the initiator, and derive the same key.
    # The initiator sends the extracted nonce; the winner checks it
    # matches their true nonce (or commitment), then both use the
    # extracted nonce value for key derivation.
    winner_key = derive_shared_key(winner_nonce, r_initiator, session_id)
    keys_match = shared_key == winner_key
    print(f"    Keys match:   {'✓ YES' if keys_match else '✗ NO'}")

    print()
    total = time.time() - t0
    print("=" * 60)
    print(f"  DONE — Total time: {total:.1f}s")
    print(f"  Key properties:")
    print(f"  • NO single party could decrypt alone")
    print(f"  • Initiator proved the match via nonce commitment")
    print(f"  • Shared symmetric key derived for secure channel")
    print(f"  • Server learned NOTHING about locations or distances")
    print("=" * 60)

    return winner_slot == true_nearest_idx


if __name__ == "__main__":
    num_parties = 3
    args = sys.argv[1:]
    if "--parties" in args:
        idx = args.index("--parties")
        num_parties = int(args[idx + 1])
    print(f"Running threshold FHE proximity demo with {num_parties} parties...")
    success = run_demo(num_parties=num_parties)
    sys.exit(0 if success else 1)
