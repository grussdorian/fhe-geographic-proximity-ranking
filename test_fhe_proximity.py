#!/usr/bin/env python3.13
# ==========================================
# test_fhe_proximity.py
#
# Comprehensive unit tests for the THRESHOLD FHE
# privacy-preserving proximity matching system.
#
# Tests the FHE core directly (no HTTP layer)
# using real-world GPS coordinates.
#
# KEY SECURITY TESTS:
#   - ALL parties must participate in decryption
#   - Missing even ONE party → garbage output
#   - No single party can decrypt alone
#
# Run:  python3.13 test_fhe_proximity.py
#
# NOTE: FHE operations are computationally expensive.
#       The full suite may take several minutes.
# ==========================================

from openfhe import *
import test_logger
import unittest
import time
import sys
import math

# ==========================================
# Config
# ==========================================

BATCH_SIZE = 32
MAX_COORD = 180.0
MAX_USERS = 20
NUM_BASE_PARTIES = 3  # default for fast tests


# ==========================================
# Threshold FHE infrastructure
# (self-contained — copied from demo.py to
#  avoid module-level import side effects)
# ==========================================

def create_context():
    """Create a fresh CKKS crypto context with MULTIPARTY enabled."""
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


def generate_threshold_keys(cc, num_parties):
    """Generate threshold keys for N parties.
    Returns (keypairs, joint_public_key)."""
    keypairs = []
    kp1 = cc.KeyGen()
    keypairs.append(kp1)
    for i in range(1, num_parties):
        kp_i = cc.MultipartyKeyGen(keypairs[-1].publicKey)
        keypairs.append(kp_i)
    joint_pk = keypairs[-1].publicKey
    return keypairs, joint_pk


def generate_eval_mult_key(cc, keypairs):
    """Generate combined eval mult key (2-round protocol)."""
    n = len(keypairs)
    joint_pk = keypairs[-1].publicKey
    pk_tag = joint_pk.GetKeyTag()

    # Round 1
    lead_share = cc.KeySwitchGen(keypairs[0].secretKey, keypairs[0].secretKey)
    shares = [lead_share]
    for i in range(1, n):
        share_i = cc.MultiKeySwitchGen(
            keypairs[i].secretKey, keypairs[i].secretKey, lead_share
        )
        shares.append(share_i)

    combined = shares[0]
    for i in range(1, n):
        combined = cc.MultiAddEvalKeys(combined, shares[i], pk_tag)

    # Round 2
    mult_shares = []
    for i in range(n):
        mult_i = cc.MultiMultEvalKey(keypairs[i].secretKey, combined, pk_tag)
        mult_shares.append(mult_i)

    final = mult_shares[0]
    for i in range(1, n):
        final = cc.MultiAddEvalMultKeys(final, mult_shares[i], final.GetKeyTag())

    cc.InsertEvalMultKey([final])


def threshold_decrypt(cc, keypairs, ciphertext):
    """ALL parties contribute partial decryptions, then fuse."""
    partials = []
    lead_partial = cc.MultipartyDecryptLead([ciphertext], keypairs[0].secretKey)
    partials.append(lead_partial[0])
    for i in range(1, len(keypairs)):
        main_partial = cc.MultipartyDecryptMain([ciphertext], keypairs[i].secretKey)
        partials.append(main_partial[0])
    pt = cc.MultipartyDecryptFusion(partials)
    pt.SetLength(BATCH_SIZE)
    return list(pt.GetRealPackedValue())


def partial_decrypt(cc, keypairs, ciphertext, party_indices):
    """Attempt decryption using only a SUBSET of parties.
    Returns the fused result (garbage) or raises RuntimeError if
    OpenFHE detects the approximation error is too high."""
    partials = []
    for idx in party_indices:
        if idx == 0:
            p = cc.MultipartyDecryptLead([ciphertext], keypairs[0].secretKey)
        else:
            p = cc.MultipartyDecryptMain([ciphertext], keypairs[idx].secretKey)
        partials.append(p[0])
    # This may raise RuntimeError("approximation error is too high")
    pt = cc.MultipartyDecryptFusion(partials)
    pt.SetLength(BATCH_SIZE)
    return list(pt.GetRealPackedValue())


# ==========================================
# Proximity scoring helpers
# ==========================================

def compute_selector(cc, diff):
    """Selector polynomial (Horner form).
    0.5 + x * (0.1125 - 0.00084375 * x^2)"""
    x2 = cc.EvalMult(diff, diff)
    inner = cc.EvalMult(x2, -0.00084375)
    inner = cc.EvalAdd(inner, 0.1125)
    product = cc.EvalMult(inner, diff)
    return cc.EvalAdd(product, 0.5)


def find_nearest_by_scoring(cc, opponents):
    """Pairwise scoring to find nearest neighbor.
    opponents: list of (dist_ct, id_ct) tuples."""
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


def compute_distance_local(cc, enc_lat, enc_lon, lat, lon):
    """Client-side: subtract own plaintext coords from encrypted, then square.
    This is ct - pt, then (ct)^2. Depth: 1 level (ct*ct for squaring)."""
    nlat = lat / MAX_COORD
    nlon = lon / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    dlat = cc.EvalSub(enc_lat, lat_pt)
    dlon = cc.EvalSub(enc_lon, lon_pt)
    dlat2 = cc.EvalMult(dlat, dlat)
    dlon2 = cc.EvalMult(dlon, dlon)
    return cc.EvalAdd(dlat2, dlon2)


# ==========================================
# Plaintext reference helpers
# ==========================================

def plaintext_sq_dist(loc1, loc2):
    """Squared Euclidean distance on normalized coords."""
    dlat = (loc1[0] - loc2[0]) / MAX_COORD
    dlon = (loc1[1] - loc2[1]) / MAX_COORD
    return dlat ** 2 + dlon ** 2


def plaintext_nearest(target, others):
    """Return index into `others` of nearest by norm_sq."""
    dists = [plaintext_sq_dist(target, o) for o in others]
    return min(range(len(dists)), key=lambda i: dists[i])


def haversine_km(loc1, loc2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ==========================================
# Full proximity pipeline
# ==========================================

def run_proximity(target_idx, locations, num_parties=NUM_BASE_PARTIES, cc=None, keypairs=None, joint_pk=None):
    """Full threshold FHE proximity pipeline.

    1. Create context + threshold keys (or reuse provided ones)
    2. Initiator encrypts coords
    3. Each other party computes distance locally (ct - pt)
    4. Server runs pairwise scoring
    5. Threshold decryption (ALL parties)
    6. Return winner slot

    Returns: (winner_slot, decrypted_values, elapsed_seconds)
    """
    n_locs = len(locations)
    assert n_locs <= BATCH_SIZE

    # Use provided context or create new one
    if cc is None:
        cc = create_context()
        keypairs, joint_pk = generate_threshold_keys(cc, num_parties)
        generate_eval_mult_key(cc, keypairs)

    # Initiator encrypts their coords
    init_lat, init_lon = locations[target_idx]
    nlat = init_lat / MAX_COORD
    nlon = init_lon / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    enc_lat = cc.Encrypt(joint_pk, lat_pt)
    enc_lon = cc.Encrypt(joint_pk, lon_pt)

    # Each other party computes distance locally
    opponents = []
    for i in range(n_locs):
        if i == target_idx:
            continue
        lat, lon = locations[i]
        dist_ct = compute_distance_local(cc, enc_lat, enc_lon, lat, lon)

        # One-hot ID
        vec = [0.0] * BATCH_SIZE
        vec[i] = 1.0
        id_pt = cc.MakeCKKSPackedPlaintext(vec)
        id_ct = cc.Encrypt(joint_pk, id_pt)
        opponents.append((dist_ct, id_ct))

    # Pairwise scoring
    t0 = time.time()
    result_ct = find_nearest_by_scoring(cc, opponents)
    elapsed = time.time() - t0

    # Threshold decryption
    values = threshold_decrypt(cc, keypairs, result_ct)
    winner_slot = max(range(len(values)), key=lambda i: values[i])

    return winner_slot, values, elapsed


# ==========================================
# Real-world locations
# ==========================================

NYC_TIMES_SQUARE       = (40.75800, -73.98557)
NYC_ROCKEFELLER        = (40.75890, -73.97931)
NYC_EMPIRE_STATE       = (40.74844, -73.98566)
NYC_CENTRAL_PARK       = (40.77260, -73.97093)
NYC_STATUE_LIBERTY     = (40.68925, -74.04451)
NYC_BROOKLYN_BRIDGE    = (40.70608, -73.99691)

SF_GOLDEN_GATE         = (37.81990, -122.47840)
SF_FISHERMANS_WHARF    = (37.80830, -122.41780)
SF_UNION_SQUARE        = (37.78780, -122.40760)
SF_ALCATRAZ            = (37.82680, -122.42290)

LONDON_BIG_BEN         = (51.50070, -0.12460)
PARIS_EIFFEL           = (48.85830,  2.29450)
TOKYO_TOWER            = (35.65860, 139.74540)
SYDNEY_OPERA           = (-33.85680, 151.21530)
MUMBAI_GATEWAY         = (18.92200, 72.83470)
CAIRO_PYRAMIDS         = (29.97940, 31.13420)
SAO_PAULO              = (-23.55050, -46.63330)
TORONTO_CN_TOWER       = (43.64260, -79.38710)
DUBAI_BURJ             = (25.19720, 55.27440)
SINGAPORE_MARINA       = (1.28140, 103.85860)
ROME_COLOSSEUM         = (41.89020, 12.49220)
BERLIN_BRANDENBURG     = (52.51630, 13.37770)
SEOUL_N_TOWER          = (37.55120, 126.98820)
MEXICO_CITY_ZOCALO     = (19.43200, -99.13320)
BANGKOK_GRAND_PALACE   = (13.75000, 100.49140)
NAIROBI                = (-1.29210, 36.82190)
MOSCOW_RED_SQUARE      = (55.75390, 37.62090)
LOS_ANGELES_HOLLYWOOD  = (34.09820, -118.32570)


# ==========================================
# Shared context for fast tests
# (created once, reused across test classes)
# ==========================================

print("Creating threshold FHE context (3-party)...")
t0 = time.time()
_cc = create_context()
_keypairs, _joint_pk = generate_threshold_keys(_cc, NUM_BASE_PARTIES)
generate_eval_mult_key(_cc, _keypairs)
print(f"  Ready in {time.time() - t0:.1f}s (3-party threshold)")


# ==========================================
# Test Cases
# ==========================================

class TestThresholdEncryptionRoundtrip(unittest.TestCase):
    """Verify encryption/decryption works under threshold FHE."""

    def test_location_roundtrip(self):
        """Encrypt and threshold-decrypt a location."""
        lat, lon = NYC_TIMES_SQUARE
        nlat, nlon = lat / MAX_COORD, lon / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
        lon_pt = _cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
        lat_ct = _cc.Encrypt(_joint_pk, lat_pt)
        lon_ct = _cc.Encrypt(_joint_pk, lon_pt)

        lat_vals = threshold_decrypt(_cc, _keypairs, lat_ct)
        lon_vals = threshold_decrypt(_cc, _keypairs, lon_ct)

        for v in lat_vals[:4]:
            self.assertAlmostEqual(v, nlat, places=5,
                                   msg=f"Lat mismatch: {v} vs {nlat}")
        for v in lon_vals[:4]:
            self.assertAlmostEqual(v, nlon, places=5,
                                   msg=f"Lon mismatch: {v} vs {nlon}")
        print(f"  [roundtrip] lat={lat:.5f} -> threshold decrypt -> {lat_vals[0]*MAX_COORD:.5f}  OK")

    def test_onehot_roundtrip(self):
        """Encrypt a one-hot ID and threshold-decrypt."""
        for slot in [0, 5, 19, 31]:
            vec = [0.0] * BATCH_SIZE
            vec[slot] = 1.0
            pt = _cc.MakeCKKSPackedPlaintext(vec)
            ct = _cc.Encrypt(_joint_pk, pt)

            vals = threshold_decrypt(_cc, _keypairs, ct)
            winner = max(range(len(vals)), key=lambda i: vals[i])
            self.assertEqual(winner, slot, f"Slot {slot}: argmax={winner}")
            self.assertAlmostEqual(vals[slot], 1.0, places=4)
        print("  [roundtrip] one-hot slots 0, 5, 19, 31  OK")

    def test_negative_coords(self):
        """Encrypt negative coords (southern/western hemisphere)."""
        lat, lon = SYDNEY_OPERA
        nlat, nlon = lat / MAX_COORD, lon / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
        lat_ct = _cc.Encrypt(_joint_pk, lat_pt)

        vals = threshold_decrypt(_cc, _keypairs, lat_ct)
        self.assertAlmostEqual(vals[0], nlat, places=5)
        print(f"  [roundtrip] negative coords Sydney ({lat}, {lon})  OK")

    def test_high_precision_coords(self):
        """5-decimal-place precision survives threshold encrypt/decrypt."""
        loc_a = (40.74844, -73.98566)
        loc_b = (40.74845, -73.98567)

        a_pt = _cc.MakeCKKSPackedPlaintext([loc_a[0] / MAX_COORD] * BATCH_SIZE)
        b_pt = _cc.MakeCKKSPackedPlaintext([loc_b[0] / MAX_COORD] * BATCH_SIZE)
        a_ct = _cc.Encrypt(_joint_pk, a_pt)
        b_ct = _cc.Encrypt(_joint_pk, b_pt)

        a_vals = threshold_decrypt(_cc, _keypairs, a_ct)
        b_vals = threshold_decrypt(_cc, _keypairs, b_ct)

        diff = abs(a_vals[0] - b_vals[0])
        expected_diff = abs(loc_a[0] - loc_b[0]) / MAX_COORD
        self.assertAlmostEqual(diff, expected_diff, places=7)
        print(f"  [roundtrip] 5-decimal precision diff = {diff:.2e}  OK")


class TestAllPartyDecryptionRequired(unittest.TestCase):
    """CRITICAL SECURITY TEST: Verify that ALL parties must
    participate in decryption. Missing even one → garbage."""

    def test_missing_one_non_lead_party(self):
        """Decrypt with parties [0,1] out of [0,1,2] → fails."""
        val = 42.0
        pt = _cc.MakeCKKSPackedPlaintext([val / MAX_COORD] * BATCH_SIZE)
        ct = _cc.Encrypt(_joint_pk, pt)

        # Full decrypt (all 3) → correct
        full_vals = threshold_decrypt(_cc, _keypairs, ct)
        self.assertAlmostEqual(full_vals[0], val / MAX_COORD, places=5,
                               msg="Full threshold decrypt should work")

        # Partial decrypt (missing party 2) → error or garbage
        try:
            partial_vals = partial_decrypt(_cc, _keypairs, ct, [0, 1])
            # If it doesn't throw, values must be garbage
            error = abs(partial_vals[0] - val / MAX_COORD)
            self.assertGreater(error, 0.01,
                               f"Missing party 2 should give garbage, but got error={error:.6f}")
            print(f"  [security] Missing party 2: error={error:.4f} (garbage)  OK")
        except RuntimeError:
            # OpenFHE detected approximation error too high — even better!
            print(f"  [security] Missing party 2: RuntimeError (decode refused)  OK")

    def test_missing_lead_party(self):
        """Decrypt with parties [1,2] (no lead) out of [0,1,2] → fails."""
        val = 42.0
        pt = _cc.MakeCKKSPackedPlaintext([val / MAX_COORD] * BATCH_SIZE)
        ct = _cc.Encrypt(_joint_pk, pt)

        # Partial with parties 1,2 only (party 1 acts as lead)
        try:
            partials = []
            p1 = _cc.MultipartyDecryptLead([ct], _keypairs[1].secretKey)
            partials.append(p1[0])
            p2 = _cc.MultipartyDecryptMain([ct], _keypairs[2].secretKey)
            partials.append(p2[0])
            fused = _cc.MultipartyDecryptFusion(partials)
            fused.SetLength(BATCH_SIZE)
            partial_vals = list(fused.GetRealPackedValue())

            error = abs(partial_vals[0] - val / MAX_COORD)
            self.assertGreater(error, 0.01,
                               f"Missing lead should give garbage, but got error={error:.6f}")
            print(f"  [security] Missing lead (party 0): error={error:.4f} (garbage)  OK")
        except RuntimeError:
            print(f"  [security] Missing lead (party 0): RuntimeError (decode refused)  OK")

    def test_single_party_alone(self):
        """A single party alone cannot decrypt."""
        val = 42.0
        pt = _cc.MakeCKKSPackedPlaintext([val / MAX_COORD] * BATCH_SIZE)
        ct = _cc.Encrypt(_joint_pk, pt)

        for party_idx in range(NUM_BASE_PARTIES):
            try:
                single_vals = partial_decrypt(_cc, _keypairs, ct, [party_idx])
                error = abs(single_vals[0] - val / MAX_COORD)
                self.assertGreater(error, 0.01,
                                   f"Party {party_idx} alone should give garbage, error={error:.6f}")
            except RuntimeError:
                pass  # Expected: OpenFHE refuses to decode
        print(f"  [security] No single party can decrypt alone  OK")

    def test_partial_decrypt_of_scoring_result(self):
        """Even the scored result can't be decrypted without all parties."""
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_STATUE_LIBERTY]
        # Run full pipeline
        winner, full_vals, _ = run_proximity(0, locs, NUM_BASE_PARTIES,
                                             cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1, "Nearest to TS should be Rockefeller")

        # Now try to decrypt scoring result with only 2 parties
        # Recompute the scoring result ciphertext
        init_lat, init_lon = locs[0]
        nlat = init_lat / MAX_COORD
        nlon = init_lon / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
        lon_pt = _cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
        enc_lat = _cc.Encrypt(_joint_pk, lat_pt)
        enc_lon = _cc.Encrypt(_joint_pk, lon_pt)

        opponents = []
        for i in [1, 2]:
            lat, lon = locs[i]
            dist_ct = compute_distance_local(_cc, enc_lat, enc_lon, lat, lon)
            vec = [0.0] * BATCH_SIZE
            vec[i] = 1.0
            id_pt = _cc.MakeCKKSPackedPlaintext(vec)
            id_ct = _cc.Encrypt(_joint_pk, id_pt)
            opponents.append((dist_ct, id_ct))

        result_ct = find_nearest_by_scoring(_cc, opponents)

        # Full decrypt should pick slot 1 (Rockefeller)
        full_result = threshold_decrypt(_cc, _keypairs, result_ct)
        full_winner = max(range(len(full_result)), key=lambda i: full_result[i])
        self.assertEqual(full_winner, 1, "Full decrypt: Rockefeller should win")

        # Partial decrypt (missing party 2) → error or garbage
        try:
            partial_result = partial_decrypt(_cc, _keypairs, result_ct, [0, 1])
            error_slot1 = abs(partial_result[1] - full_result[1])
            self.assertGreater(error_slot1, 0.01,
                               f"Partial decrypt of scoring should be garbage, error={error_slot1}")
            print(f"  [security] Scoring result: partial decrypt error={error_slot1:.4f}  OK")
        except RuntimeError:
            print(f"  [security] Scoring result: RuntimeError (decode refused)  OK")


class TestLocalDistanceComputation(unittest.TestCase):
    """Test the local (ct-pt) distance computation."""

    def _check_distance_local(self, loc1, loc2, label):
        """Encrypt loc1, compute distance to loc2 locally, verify."""
        nlat1 = loc1[0] / MAX_COORD
        nlon1 = loc1[1] / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat1] * BATCH_SIZE)
        lon_pt = _cc.MakeCKKSPackedPlaintext([nlon1] * BATCH_SIZE)
        enc_lat = _cc.Encrypt(_joint_pk, lat_pt)
        enc_lon = _cc.Encrypt(_joint_pk, lon_pt)

        dist_ct = compute_distance_local(_cc, enc_lat, enc_lon, loc2[0], loc2[1])
        dist_vals = threshold_decrypt(_cc, _keypairs, dist_ct)
        fhe_dist = dist_vals[0]

        expected = plaintext_sq_dist(loc1, loc2)
        km = haversine_km(loc1, loc2)
        self.assertAlmostEqual(fhe_dist, expected, places=6,
                               msg=f"{label}: FHE={fhe_dist}, expected={expected}")
        print(f"  [distance] {label}: FHE={fhe_dist:.8f}, expected={expected:.8f}, real={km:.1f}km  OK")

    def test_nearby_points_nyc(self):
        self._check_distance_local(NYC_TIMES_SQUARE, NYC_ROCKEFELLER, "TS→Rockefeller")

    def test_moderate_distance_nyc(self):
        self._check_distance_local(NYC_TIMES_SQUARE, NYC_STATUE_LIBERTY, "TS→Statue")

    def test_cross_city(self):
        self._check_distance_local(NYC_TIMES_SQUARE, SF_GOLDEN_GATE, "NYC→SF")

    def test_intercontinental(self):
        self._check_distance_local(LONDON_BIG_BEN, TOKYO_TOWER, "London→Tokyo")

    def test_southern_hemisphere(self):
        self._check_distance_local(SYDNEY_OPERA, SAO_PAULO, "Sydney→SãoPaulo")

    def test_same_point(self):
        """Distance to self should be ~0."""
        loc = NYC_TIMES_SQUARE
        nlat = loc[0] / MAX_COORD
        nlon = loc[1] / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
        lon_pt = _cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
        enc_lat = _cc.Encrypt(_joint_pk, lat_pt)
        enc_lon = _cc.Encrypt(_joint_pk, lon_pt)

        dist_ct = compute_distance_local(_cc, enc_lat, enc_lon, loc[0], loc[1])
        val = threshold_decrypt(_cc, _keypairs, dist_ct)[0]
        self.assertAlmostEqual(val, 0.0, places=6)
        print(f"  [distance] self-distance = {val:.2e}  OK")


class TestProximity2Users(unittest.TestCase):
    """2-user proximity: each user's nearest is the other."""

    def test_two_nyc_landmarks(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER]
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1, "Nearest to TS should be Rockefeller")
        print(f"  [2-user] TS→Rockefeller: slot {winner}  ({elapsed:.1f}s)  OK")

        winner, vals, elapsed = run_proximity(1, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 0, "Nearest to Rockefeller should be TS")
        print(f"  [2-user] Rockefeller→TS: slot {winner}  ({elapsed:.1f}s)  OK")

    def test_two_far_apart(self):
        locs = [NYC_TIMES_SQUARE, TOKYO_TOWER]
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1)
        print(f"  [2-user] NYC→Tokyo: slot {winner}  ({elapsed:.1f}s)  OK")


class TestProximity3Users(unittest.TestCase):
    """3-user proximity with clear nearest-neighbor answers."""

    def test_nyc_triangle(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_STATUE_LIBERTY]
        expected = plaintext_nearest(NYC_TIMES_SQUARE, [NYC_ROCKEFELLER, NYC_STATUE_LIBERTY])
        expected_slot = [1, 2][expected]

        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot)
        print(f"  [3-user] TS nearest: slot {winner} "
              f"(Rockefeller={vals[1]:.3f}, Statue={vals[2]:.3f})  ({elapsed:.1f}s)  OK")

    def test_global_triangle(self):
        locs = [LONDON_BIG_BEN, PARIS_EIFFEL, TOKYO_TOWER]
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1, "London's nearest should be Paris")
        print(f"  [3-user] London nearest: slot {winner} ({elapsed:.1f}s)  OK")


class TestProximity5Users(unittest.TestCase):
    """5-user scenarios."""

    def test_nyc_landmarks(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_EMPIRE_STATE,
                NYC_CENTRAL_PARK, NYC_STATUE_LIBERTY]
        others = [locs[i] for i in range(5) if i != 0]
        expected_idx = plaintext_nearest(locs[0], others)
        expected_slot = expected_idx + 1

        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot)
        print(f"  [5-user] TS nearest: slot {winner} ({elapsed:.1f}s)  OK")

    def test_sf_landmarks(self):
        locs = [SF_GOLDEN_GATE, SF_FISHERMANS_WHARF, SF_UNION_SQUARE,
                SF_ALCATRAZ, NYC_TIMES_SQUARE]
        others = [locs[i] for i in range(5) if i != 0]
        expected_idx = plaintext_nearest(locs[0], others)
        expected_slot = expected_idx + 1

        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot)
        self.assertNotEqual(winner, 4, "Should not pick NYC from SF!")
        print(f"  [5-user] Golden Gate nearest: slot {winner} ({elapsed:.1f}s)  OK")


class TestProximityClusters(unittest.TestCase):
    """Two geographic clusters — verify intra-cluster matching."""

    def test_nyc_vs_sf_clusters(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_EMPIRE_STATE,
                SF_GOLDEN_GATE, SF_FISHERMANS_WHARF, SF_UNION_SQUARE]

        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertIn(winner, [1, 2], f"TS nearest in NYC cluster, got slot {winner}")
        print(f"  [cluster] TS nearest: slot {winner} (NYC cluster)  ({elapsed:.1f}s)  OK")

        winner, vals, elapsed = run_proximity(3, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertIn(winner, [4, 5], f"Golden Gate nearest in SF cluster, got slot {winner}")
        print(f"  [cluster] Golden Gate nearest: slot {winner} (SF cluster)  ({elapsed:.1f}s)  OK")


class TestProximity10Users(unittest.TestCase):
    """10 users across the globe."""

    def test_10_global_cities(self):
        locs = [
            NYC_TIMES_SQUARE, LONDON_BIG_BEN, PARIS_EIFFEL, TOKYO_TOWER,
            SYDNEY_OPERA, CAIRO_PYRAMIDS, SAO_PAULO, TORONTO_CN_TOWER,
            DUBAI_BURJ, MUMBAI_GATEWAY,
        ]
        names = ["NYC", "London", "Paris", "Tokyo", "Sydney",
                 "Cairo", "São Paulo", "Toronto", "Dubai", "Mumbai"]

        # NYC → Toronto
        slots = [i for i in range(10) if i != 0]
        others = [locs[i] for i in slots]
        expected_slot = slots[plaintext_nearest(locs[0], others)]

        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot,
                         f"NYC nearest: expected {names[expected_slot]}, got {names[winner]}")
        print(f"  [10-user NYC] winner: slot {winner} ({names[winner]})  ({elapsed:.1f}s)  OK")

        # London → Paris
        slots = [i for i in range(10) if i != 1]
        others = [locs[i] for i in slots]
        expected_slot = slots[plaintext_nearest(locs[1], others)]

        winner, vals, elapsed = run_proximity(1, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot,
                         f"London nearest: expected {names[expected_slot]}, got {names[winner]}")
        print(f"  [10-user London] winner: slot {winner} ({names[winner]})  ({elapsed:.1f}s)  OK")


class TestProximityMax20Users(unittest.TestCase):
    """THE BIG TEST: 20 simultaneous users (system maximum)."""

    def test_20_global_cities(self):
        locs = [
            NYC_TIMES_SQUARE, LONDON_BIG_BEN, PARIS_EIFFEL, TOKYO_TOWER,
            SYDNEY_OPERA, CAIRO_PYRAMIDS, SAO_PAULO, TORONTO_CN_TOWER,
            DUBAI_BURJ, MUMBAI_GATEWAY, SINGAPORE_MARINA, ROME_COLOSSEUM,
            BERLIN_BRANDENBURG, SEOUL_N_TOWER, MEXICO_CITY_ZOCALO,
            BANGKOK_GRAND_PALACE, NAIROBI, MOSCOW_RED_SQUARE,
            LOS_ANGELES_HOLLYWOOD, SF_GOLDEN_GATE,
        ]
        names = [
            "NYC", "London", "Paris", "Tokyo", "Sydney",
            "Cairo", "São Paulo", "Toronto", "Dubai", "Mumbai",
            "Singapore", "Rome", "Berlin", "Seoul", "Mexico City",
            "Bangkok", "Nairobi", "Moscow", "Los Angeles", "San Francisco",
        ]
        assert len(locs) == MAX_USERS

        test_targets = [
            (0, "NYC"), (1, "London"), (4, "Sydney"),
            (14, "Mexico City"), (19, "San Francisco"),
        ]

        for target_idx, target_name in test_targets:
            slots = [i for i in range(20) if i != target_idx]
            others = [locs[i] for i in slots]
            expected_idx = plaintext_nearest(locs[target_idx], others)
            expected_slot = slots[expected_idx]

            dists = [(i, plaintext_sq_dist(locs[target_idx], locs[i]),
                       haversine_km(locs[target_idx], locs[i]))
                      for i in range(20) if i != target_idx]
            dists.sort(key=lambda x: x[1])
            top3 = dists[:3]
            print(f"\n  [20-user] {target_name} nearest:")
            for rank, (idx, sq, km) in enumerate(top3):
                marker = " <--" if idx == expected_slot else ""
                print(f"    {rank+1}. {names[idx]:15s}: norm_sq={sq:.6f}, ~{km:.0f}km{marker}")

            winner, vals, elapsed = run_proximity(target_idx, locs, NUM_BASE_PARTIES,
                                                  cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
            self.assertEqual(winner, expected_slot,
                             f"{target_name} nearest: expected {names[expected_slot]}, got {names[winner]}")
            print(f"  [20-user {target_name}] winner: slot {winner} ({names[winner]})  ({elapsed:.1f}s)  OK")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_nearly_equidistant(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_EMPIRE_STATE]
        expected = plaintext_nearest(NYC_TIMES_SQUARE, [NYC_ROCKEFELLER, NYC_EMPIRE_STATE])
        expected_slot = expected + 1
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot)
        print(f"  [edge] near-equidistant: slot {winner}  ({elapsed:.1f}s)  OK")

    def test_single_opponent(self):
        locs = [LONDON_BIG_BEN, MOSCOW_RED_SQUARE]
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1)
        self.assertGreater(vals[1], 0.5)
        print(f"  [edge] single opponent: slot {winner}, val={vals[1]:.3f}  ({elapsed:.1f}s)  OK")

    def test_all_negative_coords(self):
        locs = [SAO_PAULO, SYDNEY_OPERA, NAIROBI]
        expected = plaintext_nearest(SAO_PAULO, [SYDNEY_OPERA, NAIROBI])
        expected_slot = expected + 1
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, expected_slot)
        print(f"  [edge] negative coords: slot {winner}  ({elapsed:.1f}s)  OK")

    def test_very_close_vs_far(self):
        target = (40.75800, -73.98557)
        nearby = (40.75810, -73.98550)   # ~12m
        far    = (34.09820, -118.32570)  # LA
        locs = [target, nearby, far]
        winner, vals, elapsed = run_proximity(0, locs, NUM_BASE_PARTIES,
                                              cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner, 1, "Should pick 12m away, not LA")
        print(f"  [edge] very close vs far: slot {winner}  ({elapsed:.1f}s)  OK")


class TestSymmetry(unittest.TestCase):
    """Verify nearest-neighbor symmetry."""

    def test_mutual_nearest(self):
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, TOKYO_TOWER]
        winner_a, _, _ = run_proximity(0, locs, NUM_BASE_PARTIES,
                                       cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        winner_b, _, _ = run_proximity(1, locs, NUM_BASE_PARTIES,
                                       cc=_cc, keypairs=_keypairs, joint_pk=_joint_pk)
        self.assertEqual(winner_a, 1, "A's nearest should be B")
        self.assertEqual(winner_b, 0, "B's nearest should be A")
        print(f"  [symmetry] A→B (slot {winner_a}), B→A (slot {winner_b})  OK")


class TestThresholdWithDifferentPartyCounts(unittest.TestCase):
    """Test threshold FHE with varying numbers of parties (2-20).
    For each party count, verify:
      1. Full threshold decrypt works correctly
      2. Missing ANY party gives garbage"""

    def _test_n_parties(self, n):
        """Run a proximity test with N threshold parties and verify ALL required."""
        cc = create_context()
        keypairs, joint_pk = generate_threshold_keys(cc, n)
        generate_eval_mult_key(cc, keypairs)

        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_STATUE_LIBERTY]

        # Run proximity with this N-party setup
        winner, vals, elapsed = run_proximity(0, locs, n, cc=cc, keypairs=keypairs, joint_pk=joint_pk)
        expected = plaintext_nearest(NYC_TIMES_SQUARE, [NYC_ROCKEFELLER, NYC_STATUE_LIBERTY])
        expected_slot = [1, 2][expected]
        self.assertEqual(winner, expected_slot,
                         f"{n}-party: expected slot {expected_slot}, got {winner}")
        print(f"  [{n}-party] Full decrypt: slot {winner}  ({elapsed:.1f}s)  OK")

        # Verify missing party → garbage
        # Encrypt a simple value and try partial decrypt
        val = 42.0
        pt = cc.MakeCKKSPackedPlaintext([val / MAX_COORD] * BATCH_SIZE)
        ct = cc.Encrypt(joint_pk, pt)

        # Full decrypt should work
        full_vals = threshold_decrypt(cc, keypairs, ct)
        self.assertAlmostEqual(full_vals[0], val / MAX_COORD, places=5)

        # Missing last party → error or garbage
        missing_last = list(range(n - 1))  # all except last
        try:
            partial_vals = partial_decrypt(cc, keypairs, ct, missing_last)
            error = abs(partial_vals[0] - val / MAX_COORD)
            self.assertGreater(error, 0.01,
                               f"{n}-party: missing party {n-1} should give garbage, error={error}")
            print(f"  [{n}-party] Missing party {n-1}: error={error:.4f} (garbage)  OK")
        except RuntimeError:
            print(f"  [{n}-party] Missing party {n-1}: RuntimeError (decode refused)  OK")

    def test_2_parties(self):
        self._test_n_parties(2)

    def test_3_parties(self):
        self._test_n_parties(3)

    def test_5_parties(self):
        self._test_n_parties(5)

    def test_10_parties(self):
        self._test_n_parties(10)

    def test_20_parties(self):
        self._test_n_parties(20)


# ==========================================
# Runner
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FHE Proximity — Threshold Multi-Party Test Suite")
    print("=" * 60)
    print(f"  BATCH_SIZE={BATCH_SIZE}, MAX_USERS={MAX_USERS}")
    print(f"  Depth=7, Scoring: selector(x) = 0.5 + 0.1125x - 0.00084375x^3")
    print(f"  Threshold: N-of-N (ALL parties required for decryption)")
    print(f"  Base parties for fast tests: {NUM_BASE_PARTIES}")
    print()
    print("SECURITY TESTS:")
    print("  - TestAllPartyDecryptionRequired: missing party → garbage")
    print("  - TestThresholdWithDifferentPartyCounts: 2,3,5,10,20 parties")
    print("=" * 60 + "\n")

    start = time.time()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Fast tests first
    suite.addTests(loader.loadTestsFromTestCase(TestThresholdEncryptionRoundtrip))
    suite.addTests(loader.loadTestsFromTestCase(TestAllPartyDecryptionRequired))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalDistanceComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestProximity2Users))
    suite.addTests(loader.loadTestsFromTestCase(TestProximity3Users))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSymmetry))
    suite.addTests(loader.loadTestsFromTestCase(TestProximity5Users))
    suite.addTests(loader.loadTestsFromTestCase(TestProximityClusters))
    suite.addTests(loader.loadTestsFromTestCase(TestProximity10Users))
    suite.addTests(loader.loadTestsFromTestCase(TestProximityMax20Users))
    # Scaling tests (create fresh contexts — slower)
    suite.addTests(loader.loadTestsFromTestCase(TestThresholdWithDifferentPartyCounts))

    runner = test_logger.get_runner()
    result = runner.run(suite)

    total_time = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'=' * 60}")

    sys.exit(0 if result.wasSuccessful() else 1)
