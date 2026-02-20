#!/usr/bin/env python3.13
# ==========================================
# test_fhe_proximity.py
#
# Comprehensive unit tests for the FHE
# privacy-preserving proximity matching system.
#
# Tests the FHE core directly (no HTTP layer)
# using real-world GPS coordinates.
#
# Run:  python3.13 test_fhe_proximity.py
# Requires: keys generated via setup_keys.py
#
# NOTE: FHE operations are computationally expensive.
#       The full suite (especially the 20-user test)
#       may take several minutes to complete.
# ==========================================

from openfhe import *
import test_logger

# ==========================================
# Config
# ==========================================

KEYS_DIR = "fhe_keys"
BATCH_SIZE = 32
MAX_COORD = 180.0
MAX_USERS = 20

# ==========================================
# Load crypto context + all keys (once)
# ==========================================

def load_all():
    cc, ok = DeserializeCryptoContext(f"{KEYS_DIR}/cryptocontext.bin", BINARY)
    if not ok:
        raise RuntimeError("Failed to load crypto context. Run setup_keys.py first!")

    pk, ok = DeserializePublicKey(f"{KEYS_DIR}/publickey.bin", BINARY)
    if not ok:
        raise RuntimeError("Failed to load public key")

    sk, ok = DeserializePrivateKey(f"{KEYS_DIR}/secretkey.bin", BINARY)
    if not ok:
        raise RuntimeError("Failed to load secret key")

    if not cc.DeserializeEvalMultKey(f"{KEYS_DIR}/evalmultkey.bin", BINARY):
        raise RuntimeError("Failed to load eval mult key")

    return cc, pk, sk


print("Loading crypto context and keys...")
t0 = time.time()
cc, public_key, secret_key = load_all()
print(f"  Loaded in {time.time() - t0:.1f}s")


# ==========================================
# FHE helpers (self-contained, no imports
# from server/client to avoid module-level
# side effects)
# ==========================================

def encrypt_location(lat, lon):
    """Encrypt lat/lon as two replicated ciphertexts."""
    nlat = float(lat) / MAX_COORD
    nlon = float(lon) / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    return cc.Encrypt(public_key, lat_pt), cc.Encrypt(public_key, lon_pt)


def encrypt_onehot_id(slot_index):
    """Encrypt a one-hot vector with 1.0 at the given slot."""
    vec = [0.0] * BATCH_SIZE
    vec[slot_index] = 1.0
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(public_key, pt)


def decrypt_vector(ct, length=BATCH_SIZE):
    """Decrypt a ciphertext and return the real packed values."""
    pt = cc.Decrypt(secret_key, ct)
    pt.SetLength(length)
    return list(pt.GetRealPackedValue())


def compute_distance(x1, y1, x2, y2):
    """Squared Euclidean distance on encrypted replicated coords."""
    dx = cc.EvalSub(x1, x2)
    dy = cc.EvalSub(y1, y2)
    dx2 = cc.EvalMult(dx, dx)
    dy2 = cc.EvalMult(dy, dy)
    return cc.EvalAdd(dx2, dy2)


def compute_selector(diff):
    """Selector polynomial: ≈ 1 if diff > 0, ≈ 0 if diff < 0.

    Horner form: 0.5 + x * (0.1125 - 0.00084375 * x²)
    Derived from sign(0.15*x) = 1.5*(0.15x) - 0.5*(0.15x)³,
    then selector = (sign + 1) / 2.

    Depth cost: 2 ct-ct + 1 pt-ct = 3 levels.
    """
    x2 = cc.EvalMult(diff, diff)                 # ct-ct
    inner = cc.EvalMult(x2, -0.00084375)         # pt-ct
    inner = cc.EvalAdd(inner, 0.1125)            # add: free
    product = cc.EvalMult(inner, diff)            # ct-ct
    return cc.EvalAdd(product, 0.5)              # add: free


def find_nearest_by_scoring(target_x, target_y, opponents):
    """Find nearest opponent using pairwise scoring.

    For each opponent i, score_i = sum of selector(d_j - d_i)
    for all j != i. Highest score = nearest to target.

    Depth: 1 (distance) + 3 (selector) + 1 (score*id) = 5 levels.
    """
    n = len(opponents)

    # Trivial case: single opponent is the nearest
    if n == 1:
        return opponents[0][2]

    # Step 1: Compute all distances to target
    dists = []
    for x_ct, y_ct, _ in opponents:
        d = compute_distance(target_x, target_y, x_ct, y_ct)
        dists.append(d)

    # Step 2: Pairwise selectors → accumulate scores
    scores = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = cc.EvalSub(dists[j], dists[i])
            sel_ji = compute_selector(diff)
            sel_ij = cc.EvalSub(1.0, sel_ji)

            scores[i] = cc.EvalAdd(scores[i], sel_ji) if scores[i] is not None else sel_ji
            scores[j] = cc.EvalAdd(scores[j], sel_ij) if scores[j] is not None else sel_ij

    # Step 3: Weighted sum of one-hot IDs
    result = cc.EvalMult(scores[0], opponents[0][2])
    for i in range(1, n):
        result = cc.EvalAdd(result, cc.EvalMult(scores[i], opponents[i][2]))

    return result


# ==========================================
# Plaintext reference helpers
# ==========================================

def plaintext_sq_dist(loc1, loc2):
    """Squared Euclidean distance on normalized coords (the metric the FHE system uses)."""
    dlat = (loc1[0] - loc2[0]) / MAX_COORD
    dlon = (loc1[1] - loc2[1]) / MAX_COORD
    return dlat ** 2 + dlon ** 2


def plaintext_nearest(target, others):
    """
    Return the index (into `others`) of the nearest location to `target`
    using the same distance metric as the FHE system.
    """
    dists = [plaintext_sq_dist(target, o) for o in others]
    return min(range(len(dists)), key=lambda i: dists[i])


def haversine_km(loc1, loc2):
    """Great-circle distance in km (for human-readable output only)."""
    R = 6371.0
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ==========================================
# Full scoring helper
#
# Encrypts all locations, runs pairwise scoring
# from one target against all others, returns
# the winning slot index.
# ==========================================

def run_tournament(target_idx, locations):
    """
    Encrypt all locations, compute pairwise scores from
    target_idx to every other user, decrypt and return
    the winning slot index.

    locations: list of (lat, lon) tuples
    Returns: (winner_slot, decrypted_vector, elapsed_seconds)
    """
    n = len(locations)
    assert n <= BATCH_SIZE, f"Too many users ({n} > {BATCH_SIZE})"

    # Encrypt all locations and one-hot IDs
    encrypted = []
    for i, (lat, lon) in enumerate(locations):
        lat_ct, lon_ct = encrypt_location(lat, lon)
        id_ct = encrypt_onehot_id(i)
        encrypted.append((lat_ct, lon_ct, id_ct))

    target_lat, target_lon, _ = encrypted[target_idx]

    # Build opponents list (everyone except target)
    opponents = []
    slot_map = []  # maps opponent index to original slot
    for i, (lat_ct, lon_ct, id_ct) in enumerate(encrypted):
        if i == target_idx:
            continue
        opponents.append((lat_ct, lon_ct, id_ct))
        slot_map.append(i)

    t0 = time.time()
    winner_id_ct = find_nearest_by_scoring(target_lat, target_lon, opponents)
    elapsed = time.time() - t0

    values = decrypt_vector(winner_id_ct)
    winner_slot = max(range(len(values)), key=lambda i: values[i])

    return winner_slot, values, elapsed


# ==========================================
# Real-world locations
# ==========================================

# --- New York City area ---
NYC_TIMES_SQUARE       = (40.75800, -73.98557)
NYC_ROCKEFELLER        = (40.75890, -73.97931)
NYC_EMPIRE_STATE       = (40.74844, -73.98566)
NYC_CENTRAL_PARK       = (40.77260, -73.97093)
NYC_STATUE_LIBERTY     = (40.68925, -74.04451)
NYC_BROOKLYN_BRIDGE    = (40.70608, -73.99691)

# --- San Francisco area ---
SF_GOLDEN_GATE         = (37.81990, -122.47840)
SF_FISHERMANS_WHARF    = (37.80830, -122.41780)
SF_UNION_SQUARE        = (37.78780, -122.40760)
SF_ALCATRAZ            = (37.82680, -122.42290)

# --- Global landmarks ---
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
# Test Cases
# ==========================================

class TestEncryptionRoundtrip(unittest.TestCase):
    """Verify encryption/decryption preserves values."""

    def test_location_roundtrip(self):
        """Encrypt and decrypt a location — values should survive."""
        lat, lon = NYC_TIMES_SQUARE
        lat_ct, lon_ct = encrypt_location(lat, lon)

        lat_vals = decrypt_vector(lat_ct, 4)
        lon_vals = decrypt_vector(lon_ct, 4)

        expected_lat = lat / MAX_COORD
        expected_lon = lon / MAX_COORD

        for v in lat_vals:
            self.assertAlmostEqual(v, expected_lat, places=5,
                                   msg=f"Lat mismatch: {v} vs {expected_lat}")
        for v in lon_vals:
            self.assertAlmostEqual(v, expected_lon, places=5,
                                   msg=f"Lon mismatch: {v} vs {expected_lon}")
        print(f"  [roundtrip] lat={lat:.5f} -> encrypted -> decrypted -> {lat_vals[0]*MAX_COORD:.5f}  OK")

    def test_onehot_roundtrip(self):
        """Encrypt a one-hot ID at various slots and verify."""
        for slot in [0, 5, 19, 31]:
            id_ct = encrypt_onehot_id(slot)
            vals = decrypt_vector(id_ct)

            winner = max(range(len(vals)), key=lambda i: vals[i])
            self.assertEqual(winner, slot,
                             msg=f"Slot {slot}: argmax={winner}")
            self.assertAlmostEqual(vals[slot], 1.0, places=4)

            # All other slots should be ~0
            for i, v in enumerate(vals):
                if i != slot:
                    self.assertAlmostEqual(v, 0.0, places=4,
                                           msg=f"Slot {slot}: non-zero at {i}: {v}")
        print("  [roundtrip] one-hot encoding at slots 0, 5, 19, 31  OK")

    def test_negative_coords(self):
        """Encrypt negative coordinates (southern/western hemisphere)."""
        lat, lon = SYDNEY_OPERA  # -33.85680, 151.21530
        lat_ct, lon_ct = encrypt_location(lat, lon)

        lat_vals = decrypt_vector(lat_ct, 2)
        lon_vals = decrypt_vector(lon_ct, 2)

        self.assertAlmostEqual(lat_vals[0], lat / MAX_COORD, places=5)
        self.assertAlmostEqual(lon_vals[0], lon / MAX_COORD, places=5)
        print(f"  [roundtrip] negative coords Sydney ({lat}, {lon})  OK")

    def test_high_precision_coords(self):
        """Verify 5-decimal-place precision survives encryption."""
        # These two points differ only at the 5th decimal place (~1.1m apart)
        loc_a = (40.74844, -73.98566)
        loc_b = (40.74845, -73.98567)

        a_lat_ct, a_lon_ct = encrypt_location(*loc_a)
        b_lat_ct, b_lon_ct = encrypt_location(*loc_b)

        a_lat = decrypt_vector(a_lat_ct, 1)[0]
        b_lat = decrypt_vector(b_lat_ct, 1)[0]

        # The difference at the 5th decimal place is 1e-5 / 180 ≈ 5.6e-8
        # CKKS should preserve this
        diff = abs(a_lat - b_lat)
        expected_diff = abs(loc_a[0] - loc_b[0]) / MAX_COORD
        self.assertAlmostEqual(diff, expected_diff, places=7,
                               msg=f"Precision loss: diff={diff}, expected={expected_diff}")
        print(f"  [roundtrip] 5-decimal precision difference = {diff:.2e}  OK")


class TestDistanceComputation(unittest.TestCase):
    """Verify FHE distance computation matches plaintext."""

    def _check_distance(self, loc1, loc2, label):
        lat1_ct, lon1_ct = encrypt_location(*loc1)
        lat2_ct, lon2_ct = encrypt_location(*loc2)

        dist_ct = compute_distance(lat1_ct, lon1_ct, lat2_ct, lon2_ct)
        dist_vals = decrypt_vector(dist_ct, 1)
        fhe_dist = dist_vals[0]

        expected = plaintext_sq_dist(loc1, loc2)
        km = haversine_km(loc1, loc2)

        # Relative tolerance: CKKS introduces small noise
        self.assertAlmostEqual(fhe_dist, expected, places=6,
                               msg=f"{label}: FHE={fhe_dist}, expected={expected}")
        print(f"  [distance] {label}: FHE={fhe_dist:.8f}, expected={expected:.8f}, "
              f"real={km:.1f}km  OK")

    def test_nearby_points_nyc(self):
        """Times Square → Rockefeller Center (~1.2km)."""
        self._check_distance(NYC_TIMES_SQUARE, NYC_ROCKEFELLER, "TS→Rockefeller")

    def test_moderate_distance_nyc(self):
        """Times Square → Statue of Liberty (~8km)."""
        self._check_distance(NYC_TIMES_SQUARE, NYC_STATUE_LIBERTY, "TS→Statue")

    def test_cross_city(self):
        """NYC → SF (~4100km)."""
        self._check_distance(NYC_TIMES_SQUARE, SF_GOLDEN_GATE, "NYC→SF")

    def test_intercontinental(self):
        """London → Tokyo (~9600km)."""
        self._check_distance(LONDON_BIG_BEN, TOKYO_TOWER, "London→Tokyo")

    def test_southern_hemisphere(self):
        """Sydney → São Paulo (~13500km)."""
        self._check_distance(SYDNEY_OPERA, SAO_PAULO, "Sydney→SãoPaulo")

    def test_same_point(self):
        """Distance to self should be ~0."""
        lat_ct, lon_ct = encrypt_location(*NYC_TIMES_SQUARE)
        dist_ct = compute_distance(lat_ct, lon_ct, lat_ct, lon_ct)
        val = decrypt_vector(dist_ct, 1)[0]
        self.assertAlmostEqual(val, 0.0, places=6,
                               msg=f"Self-distance should be ~0, got {val}")
        print(f"  [distance] self-distance = {val:.2e}  OK")


class TestTournament2Users(unittest.TestCase):
    """With only 2 users, each user's nearest is the other."""

    def test_two_nyc_landmarks(self):
        """Times Square + Rockefeller → each is nearest to the other."""
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, 1, "Nearest to TS should be Rockefeller (slot 1)")
        print(f"  [2-user] TS→Rockefeller: winner=slot {winner}  ({elapsed:.1f}s)  OK")

        winner, vals, elapsed = run_tournament(1, locs)
        self.assertEqual(winner, 0, "Nearest to Rockefeller should be TS (slot 0)")
        print(f"  [2-user] Rockefeller→TS: winner=slot {winner}  ({elapsed:.1f}s)  OK")

    def test_two_far_apart(self):
        """NYC + Tokyo → each is nearest (only option)."""
        locs = [NYC_TIMES_SQUARE, TOKYO_TOWER]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, 1)
        print(f"  [2-user] NYC→Tokyo: winner=slot {winner}  ({elapsed:.1f}s)  OK")


class TestTournament3Users(unittest.TestCase):
    """3-user scenarios with clear nearest-neighbor answers."""

    def test_nyc_triangle(self):
        """
        Times Square (0), Rockefeller (1), Statue of Liberty (2).
        Nearest to TS → Rockefeller (~1.2km), not Statue (~8km).
        """
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_STATUE_LIBERTY]

        expected = plaintext_nearest(
            NYC_TIMES_SQUARE,
            [NYC_ROCKEFELLER, NYC_STATUE_LIBERTY]
        )
        # expected=0 in others list → slot 1 (Rockefeller)
        expected_slot = [1, 2][expected]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot,
                         f"Expected slot {expected_slot}, got {winner}")
        print(f"  [3-user] TS nearest: slot {winner} "
              f"(Rockefeller={vals[1]:.3f}, Statue={vals[2]:.3f})  "
              f"({elapsed:.1f}s)  OK")

    def test_global_triangle(self):
        """
        London (0), Paris (1), Tokyo (2).
        Nearest to London → Paris (~340km), not Tokyo (~9600km).
        """
        locs = [LONDON_BIG_BEN, PARIS_EIFFEL, TOKYO_TOWER]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, 1, "London's nearest should be Paris")
        print(f"  [3-user] London nearest: slot {winner} "
              f"(Paris={vals[1]:.3f}, Tokyo={vals[2]:.3f})  "
              f"({elapsed:.1f}s)  OK")


class TestTournament5Users(unittest.TestCase):
    """5-user scenarios — 4 opponents, 2 tournament rounds."""

    def test_nyc_landmarks(self):
        """
        0: Times Square      3: Central Park
        1: Rockefeller        4: Statue of Liberty
        2: Empire State

        Nearest to TS → Rockefeller (~0.6km) or Empire State (~1.1km).
        Plaintext reference decides.
        """
        locs = [
            NYC_TIMES_SQUARE,
            NYC_ROCKEFELLER,
            NYC_EMPIRE_STATE,
            NYC_CENTRAL_PARK,
            NYC_STATUE_LIBERTY,
        ]

        others = [locs[i] for i in range(5) if i != 0]
        expected_idx = plaintext_nearest(locs[0], others)
        # Map back: others[0]=slot1, others[1]=slot2, ...
        expected_slot = expected_idx + 1

        km_vals = {
            "Rockefeller": haversine_km(locs[0], locs[1]),
            "Empire St": haversine_km(locs[0], locs[2]),
            "Central Pk": haversine_km(locs[0], locs[3]),
            "Statue":     haversine_km(locs[0], locs[4]),
        }
        print(f"  [5-user] Distances from TS: {', '.join(f'{k}={v:.1f}km' for k,v in km_vals.items())}")

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot,
                         f"Expected slot {expected_slot}, got {winner}")
        print(f"  [5-user] TS nearest: slot {winner} (expected {expected_slot})  "
              f"({elapsed:.1f}s)  OK")

    def test_sf_landmarks(self):
        """
        0: Golden Gate     3: Alcatraz
        1: Fisherman's     4: Times Square (far away!)
        2: Union Square

        Nearest to Golden Gate → Fisherman's or Alcatraz (both ~2-3km).
        """
        locs = [
            SF_GOLDEN_GATE,
            SF_FISHERMANS_WHARF,
            SF_UNION_SQUARE,
            SF_ALCATRAZ,
            NYC_TIMES_SQUARE,  # very far away
        ]

        others = [locs[i] for i in range(5) if i != 0]
        expected_idx = plaintext_nearest(locs[0], others)
        expected_slot = expected_idx + 1

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot,
                         f"Expected slot {expected_slot}, got {winner}")

        # Also verify it did NOT pick NYC (slot 4)
        self.assertNotEqual(winner, 4, "Should not pick NYC from SF!")
        print(f"  [5-user] Golden Gate nearest: slot {winner} "
              f"({elapsed:.1f}s)  OK")


class TestTournamentClusters(unittest.TestCase):
    """Two geographic clusters — verify intra-cluster matching."""

    def test_nyc_vs_sf_clusters(self):
        """
        NYC cluster: TS (0), Rockefeller (1), Empire State (2)
        SF cluster:  Golden Gate (3), Fisherman's (4), Union Sq (5)

        Nearest to TS → should be in NYC (slot 1 or 2), NOT SF.
        Nearest to Golden Gate → should be in SF (slot 4 or 5), NOT NYC.
        """
        locs = [
            NYC_TIMES_SQUARE,
            NYC_ROCKEFELLER,
            NYC_EMPIRE_STATE,
            SF_GOLDEN_GATE,
            SF_FISHERMANS_WHARF,
            SF_UNION_SQUARE,
        ]

        # Test from NYC perspective
        winner, vals, elapsed = run_tournament(0, locs)
        self.assertIn(winner, [1, 2],
                      f"TS nearest should be in NYC cluster, got slot {winner}")
        print(f"  [cluster] TS nearest: slot {winner} (NYC cluster)  ({elapsed:.1f}s)  OK")

        # Test from SF perspective
        winner, vals, elapsed = run_tournament(3, locs)
        self.assertIn(winner, [4, 5],
                      f"Golden Gate nearest should be in SF cluster, got slot {winner}")
        print(f"  [cluster] Golden Gate nearest: slot {winner} (SF cluster)  ({elapsed:.1f}s)  OK")


class TestTournament10Users(unittest.TestCase):
    """10 users across the globe — 9 opponents, ~4 tournament rounds."""

    def test_10_global_cities(self):
        """
        0: Times Square      5: Cairo
        1: London             6: São Paulo
        2: Paris              7: Toronto
        3: Tokyo              8: Dubai
        4: Sydney             9: Mumbai
        """
        locs = [
            NYC_TIMES_SQUARE,     # 0
            LONDON_BIG_BEN,       # 1
            PARIS_EIFFEL,         # 2
            TOKYO_TOWER,          # 3
            SYDNEY_OPERA,         # 4
            CAIRO_PYRAMIDS,       # 5
            SAO_PAULO,            # 6
            TORONTO_CN_TOWER,     # 7
            DUBAI_BURJ,           # 8
            MUMBAI_GATEWAY,       # 9
        ]

        names = [
            "NYC", "London", "Paris", "Tokyo", "Sydney",
            "Cairo", "São Paulo", "Toronto", "Dubai", "Mumbai",
        ]

        # Test from NYC — expected nearest is Toronto (~550km)
        others_0 = [locs[i] for i in range(10) if i != 0]
        expected_idx_0 = plaintext_nearest(locs[0], others_0)
        expected_slot_0 = expected_idx_0 + (1 if expected_idx_0 < 0 + 1 else 0)
        # Simpler: map others list index back to slot
        slots_for_0 = [i for i in range(10) if i != 0]
        expected_slot_0 = slots_for_0[expected_idx_0]

        print(f"  [10-user] Distances from NYC:")
        for i in range(10):
            if i == 0:
                continue
            km = haversine_km(locs[0], locs[i])
            sq = plaintext_sq_dist(locs[0], locs[i])
            print(f"    → {names[i]:12s} (slot {i}): {km:8.1f}km  norm_sq={sq:.6f}")

        print(f"  [10-user] Expected nearest to NYC: {names[expected_slot_0]} (slot {expected_slot_0})")

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot_0,
                         f"NYC nearest: expected slot {expected_slot_0} ({names[expected_slot_0]}), "
                         f"got slot {winner} ({names[winner]})")
        print(f"  [10-user NYC] winner: slot {winner} ({names[winner]})  "
              f"({elapsed:.1f}s)  OK")

        # Test from London — expected nearest is Paris (~340km)
        slots_for_1 = [i for i in range(10) if i != 1]
        others_1 = [locs[i] for i in slots_for_1]
        expected_idx_1 = plaintext_nearest(locs[1], others_1)
        expected_slot_1 = slots_for_1[expected_idx_1]

        winner, vals, elapsed = run_tournament(1, locs)
        self.assertEqual(winner, expected_slot_1,
                         f"London nearest: expected slot {expected_slot_1} ({names[expected_slot_1]}), "
                         f"got slot {winner} ({names[winner]})")
        print(f"  [10-user London] winner: slot {winner} ({names[winner]})  "
              f"({elapsed:.1f}s)  OK")

        # Test from Tokyo — nearest is Seoul by Euclidean-norm metric
        slots_for_3 = [i for i in range(10) if i != 3]
        others_3 = [locs[i] for i in slots_for_3]
        expected_idx_3 = plaintext_nearest(locs[3], others_3)
        expected_slot_3 = slots_for_3[expected_idx_3]

        winner, vals, elapsed = run_tournament(3, locs)
        self.assertEqual(winner, expected_slot_3,
                         f"Tokyo nearest: expected slot {expected_slot_3} ({names[expected_slot_3]}), "
                         f"got slot {winner} ({names[winner]})")
        print(f"  [10-user Tokyo] winner: slot {winner} ({names[winner]})  "
              f"({elapsed:.1f}s)  OK")


class TestTournamentMax20Users(unittest.TestCase):
    """
    THE BIG TEST: 20 simultaneous users (the system maximum).

    19 opponents → 5 tournament rounds → 16 multiplicative levels.
    This is the stress test for both correctness and depth budget.
    """

    def test_20_global_cities(self):
        """
        20 landmarks across the globe. Verify nearest-neighbor
        from multiple perspectives.
        """
        locs = [
            NYC_TIMES_SQUARE,       # 0
            LONDON_BIG_BEN,         # 1
            PARIS_EIFFEL,           # 2
            TOKYO_TOWER,            # 3
            SYDNEY_OPERA,           # 4
            CAIRO_PYRAMIDS,         # 5
            SAO_PAULO,              # 6
            TORONTO_CN_TOWER,       # 7
            DUBAI_BURJ,             # 8
            MUMBAI_GATEWAY,         # 9
            SINGAPORE_MARINA,       # 10
            ROME_COLOSSEUM,         # 11
            BERLIN_BRANDENBURG,     # 12
            SEOUL_N_TOWER,          # 13
            MEXICO_CITY_ZOCALO,     # 14
            BANGKOK_GRAND_PALACE,   # 15
            NAIROBI,                # 16
            MOSCOW_RED_SQUARE,      # 17
            LOS_ANGELES_HOLLYWOOD,  # 18
            SF_GOLDEN_GATE,         # 19
        ]

        names = [
            "NYC", "London", "Paris", "Tokyo", "Sydney",
            "Cairo", "São Paulo", "Toronto", "Dubai", "Mumbai",
            "Singapore", "Rome", "Berlin", "Seoul", "Mexico City",
            "Bangkok", "Nairobi", "Moscow", "Los Angeles", "San Francisco",
        ]

        assert len(locs) == MAX_USERS, f"Expected {MAX_USERS} locations, got {len(locs)}"

        # ---------- Print distance table for reference ----------
        print(f"\n  [20-user] Full distance table from key cities:")

        # --- Test from multiple perspectives ---
        test_targets = [
            (0,  "NYC"),
            (1,  "London"),
            (4,  "Sydney"),
            (14, "Mexico City"),
            (19, "San Francisco"),
        ]

        for target_idx, target_name in test_targets:
            # Compute expected nearest
            slots_for_target = [i for i in range(20) if i != target_idx]
            others = [locs[i] for i in slots_for_target]
            expected_idx = plaintext_nearest(locs[target_idx], others)
            expected_slot = slots_for_target[expected_idx]

            # Print top 3 nearest for context
            dists = [(i, plaintext_sq_dist(locs[target_idx], locs[i]),
                       haversine_km(locs[target_idx], locs[i]))
                      for i in range(20) if i != target_idx]
            dists.sort(key=lambda x: x[1])
            top3 = dists[:3]
            print(f"\n  [20-user] {target_name} nearest (by norm_sq):")
            for rank, (idx, sq, km) in enumerate(top3):
                marker = " <-- expected" if idx == expected_slot else ""
                print(f"    {rank+1}. {names[idx]:15s} (slot {idx:2d}): "
                      f"norm_sq={sq:.6f}, ~{km:.0f}km{marker}")

            # Run tournament
            print(f"  [20-user] Running tournament from {target_name} (19 opponents)...")
            t0 = time.time()
            winner, vals, elapsed = run_tournament(target_idx, locs)

            self.assertEqual(winner, expected_slot,
                             f"{target_name} nearest: expected slot {expected_slot} "
                             f"({names[expected_slot]}), got slot {winner} ({names[winner]})\n"
                             f"  Top values: {sorted(enumerate(vals), key=lambda x: -x[1])[:5]}")

            print(f"  [20-user {target_name}] winner: slot {winner} ({names[winner]})  "
                  f"({elapsed:.1f}s)  OK")

            # Print the top 3 slots by decrypted value
            top_slots = sorted(enumerate(vals), key=lambda x: -x[1])[:3]
            for rank, (idx, val) in enumerate(top_slots):
                print(f"    argmax rank {rank+1}: slot {idx:2d} ({names[idx]:15s}) = {val:.4f}")


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_nearly_equidistant(self):
        """
        Two users nearly the same distance from target.
        The system should pick one of the two (may be either).
        """
        # Two points roughly equidistant from Times Square
        # Rockefeller: ~0.6km NE
        # Empire State: ~1.1km S
        # NOT truly equidistant, so the system should reliably pick Rockefeller
        locs = [NYC_TIMES_SQUARE, NYC_ROCKEFELLER, NYC_EMPIRE_STATE]

        expected = plaintext_nearest(
            NYC_TIMES_SQUARE,
            [NYC_ROCKEFELLER, NYC_EMPIRE_STATE]
        )
        expected_slot = expected + 1

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot)
        print(f"  [edge] near-equidistant: winner=slot {winner}  ({elapsed:.1f}s)  OK")

    def test_single_opponent(self):
        """Only 1 opponent — tournament should still work (trivial case)."""
        locs = [LONDON_BIG_BEN, MOSCOW_RED_SQUARE]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, 1, "Only opponent should win")
        self.assertGreater(vals[1], 0.5, "Winner slot should have high value")
        print(f"  [edge] single opponent: winner=slot {winner}, val={vals[1]:.3f}  "
              f"({elapsed:.1f}s)  OK")

    def test_all_negative_coords(self):
        """Both lat and lon negative (southern + western hemisphere)."""
        locs = [
            SAO_PAULO,        # -23.55, -46.63
            SYDNEY_OPERA,     # -33.86, 151.22
            NAIROBI,          # -1.29,  36.82
        ]

        expected = plaintext_nearest(SAO_PAULO, [SYDNEY_OPERA, NAIROBI])
        expected_slot = expected + 1

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot)
        print(f"  [edge] negative coords: winner=slot {winner}  ({elapsed:.1f}s)  OK")

    def test_cross_hemisphere(self):
        """Target in northern hemisphere, opponents in southern."""
        locs = [
            NYC_TIMES_SQUARE,   # 40.76, -73.99
            SYDNEY_OPERA,       # -33.86, 151.22
            SAO_PAULO,          # -23.55, -46.63
            NAIROBI,            # -1.29, 36.82
        ]

        # São Paulo is closest to NYC by Euclidean normalized dist
        expected = plaintext_nearest(NYC_TIMES_SQUARE, [SYDNEY_OPERA, SAO_PAULO, NAIROBI])
        expected_slot = expected + 1

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, expected_slot)
        km = haversine_km(NYC_TIMES_SQUARE, locs[winner])
        print(f"  [edge] cross-hemisphere: winner=slot {winner} (~{km:.0f}km)  "
              f"({elapsed:.1f}s)  OK")

    def test_very_close_vs_far(self):
        """
        Two users at nearly the same spot vs one far away.
        Should always pick the close one.
        """
        target = (40.75800, -73.98557)   # Times Square
        nearby = (40.75810, -73.98550)   # ~12 meters away!
        far    = (34.09820, -118.32570)  # Los Angeles

        locs = [target, nearby, far]

        winner, vals, elapsed = run_tournament(0, locs)
        self.assertEqual(winner, 1, "Should pick the point 12m away, not LA")
        print(f"  [edge] very close ({haversine_km(target, nearby)*1000:.0f}m) vs far "
              f"({haversine_km(target, far):.0f}km): winner=slot {winner}  ({elapsed:.1f}s)  OK")


class TestSymmetry(unittest.TestCase):
    """Verify nearest-neighbor symmetry in small groups."""

    def test_mutual_nearest(self):
        """
        In a 3-user setup where A and B are very close and C is far:
        A's nearest should be B, and B's nearest should be A.
        """
        locs = [
            NYC_TIMES_SQUARE,    # 0: A
            NYC_ROCKEFELLER,     # 1: B  (~0.6km from A)
            TOKYO_TOWER,         # 2: C  (~10,000km from A)
        ]

        winner_a, _, _ = run_tournament(0, locs)
        winner_b, _, _ = run_tournament(1, locs)

        self.assertEqual(winner_a, 1, "A's nearest should be B")
        self.assertEqual(winner_b, 0, "B's nearest should be A")
        print(f"  [symmetry] A→B (slot {winner_a}), B→A (slot {winner_b})  OK")


# ==========================================
# Runner
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FHE Proximity Matching — Comprehensive Test Suite")
    print("=" * 60)
    print(f"  BATCH_SIZE={BATCH_SIZE}, MAX_USERS={MAX_USERS}, MAX_COORD={MAX_COORD}")
    print(f"  Depth=7, Scoring approach: selector(x) = 0.5 + 0.1125x - 0.00084375x³")
    print(f"  Depth budget: 5 levels out of 7 (independent of user count)")
    print()
    print("WARNING: FHE operations are computationally expensive.")
    print("  2-3 user tests:  ~5-15s each")
    print("  5-10 user tests: ~30-120s each")
    print("  20 user tests:   ~5-15 min each (190 pairwise comparisons)")
    print("=" * 60 + "\n")

    start = time.time()

    # Use a custom test runner with verbosity
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Order: fast tests first, slow tests last
    suite.addTests(loader.loadTestsFromTestCase(TestEncryptionRoundtrip))
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestTournament2Users))
    suite.addTests(loader.loadTestsFromTestCase(TestTournament3Users))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSymmetry))
    suite.addTests(loader.loadTestsFromTestCase(TestTournament5Users))
    suite.addTests(loader.loadTestsFromTestCase(TestTournamentClusters))
    suite.addTests(loader.loadTestsFromTestCase(TestTournament10Users))
    suite.addTests(loader.loadTestsFromTestCase(TestTournamentMax20Users))

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
