#!/usr/bin/env python3.13
# ==========================================
# test_dense_urban.py
#
# Threshold FHE proximity tests focused on
# densely populated urban areas where users
# are within the same city, often just blocks
# or hundreds of meters apart.
#
# Scenarios:
#   - Manhattan grid (20 users, ~50m to ~5km)
#   - Downtown San Francisco (20 users)
#   - Central Tokyo (20 users)
#   - Central London (20 users)
#   - Mumbai (20 users)
#   - Same-block precision (users <50m apart)
#   - Mixed: one dense cluster + a few outliers
#
# Security:
#   - ALL parties must participate in decryption
#   - Threshold N-of-N (no single party can decrypt)
#
# Run:  python3.13 test_dense_urban.py
# ==========================================

from openfhe import *
import test_logger
import time
import unittest
import sys
import math

# ==========================================
# Config
# ==========================================

BATCH_SIZE = 32
MAX_COORD = 180.0
MAX_USERS = 20
NUM_BASE_PARTIES = 3


# ==========================================
# Threshold FHE infrastructure
# ==========================================

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


def generate_threshold_keys(cc, num_parties):
    keypairs = []
    kp1 = cc.KeyGen()
    keypairs.append(kp1)
    for i in range(1, num_parties):
        kp_i = cc.MultipartyKeyGen(keypairs[-1].publicKey)
        keypairs.append(kp_i)
    joint_pk = keypairs[-1].publicKey
    return keypairs, joint_pk


def generate_eval_mult_key(cc, keypairs):
    n = len(keypairs)
    joint_pk = keypairs[-1].publicKey
    pk_tag = joint_pk.GetKeyTag()

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

    mult_shares = []
    for i in range(n):
        mult_i = cc.MultiMultEvalKey(keypairs[i].secretKey, combined, pk_tag)
        mult_shares.append(mult_i)

    final = mult_shares[0]
    for i in range(1, n):
        final = cc.MultiAddEvalMultKeys(final, mult_shares[i], final.GetKeyTag())

    cc.InsertEvalMultKey([final])


def threshold_decrypt(cc, keypairs, ciphertext):
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
    """Attempt decryption with a SUBSET of parties.
    Returns garbage values or raises RuntimeError."""
    partials = []
    for idx in party_indices:
        if idx == 0:
            p = cc.MultipartyDecryptLead([ciphertext], keypairs[0].secretKey)
        else:
            p = cc.MultipartyDecryptMain([ciphertext], keypairs[idx].secretKey)
        partials.append(p[0])
    # May raise RuntimeError("approximation error is too high")
    pt = cc.MultipartyDecryptFusion(partials)
    pt.SetLength(BATCH_SIZE)
    return list(pt.GetRealPackedValue())


# ==========================================
# Scoring helpers
# ==========================================

def compute_selector(cc, diff):
    x2 = cc.EvalMult(diff, diff)
    inner = cc.EvalMult(x2, -0.00084375)
    inner = cc.EvalAdd(inner, 0.1125)
    product = cc.EvalMult(inner, diff)
    return cc.EvalAdd(product, 0.5)


def find_nearest_by_scoring(cc, opponents):
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

def haversine_m(loc1, loc2):
    R = 6371000.0
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def plaintext_sq_dist(loc1, loc2):
    dlat = (loc1[0] - loc2[0]) / MAX_COORD
    dlon = (loc1[1] - loc2[1]) / MAX_COORD
    return dlat ** 2 + dlon ** 2


def plaintext_nearest(target, others):
    dists = [plaintext_sq_dist(target, o) for o in others]
    return min(range(len(dists)), key=lambda i: dists[i])


# ==========================================
# Full scoring pipeline (threshold FHE)
# ==========================================

def run_scoring(target_idx, locs, names, cc=None, keypairs=None, joint_pk=None):
    """Run full threshold FHE proximity pipeline.
    Returns (winner_slot, values, elapsed)."""
    n = len(locs)
    assert n <= BATCH_SIZE

    if cc is None:
        cc = _cc
        keypairs = _keypairs
        joint_pk = _joint_pk

    # Initiator encrypts
    init_lat, init_lon = locs[target_idx]
    nlat = init_lat / MAX_COORD
    nlon = init_lon / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    enc_lat = cc.Encrypt(joint_pk, lat_pt)
    enc_lon = cc.Encrypt(joint_pk, lon_pt)

    # Each party computes distance locally
    opponents = []
    for i in range(n):
        if i == target_idx:
            continue
        dist_ct = compute_distance_local(cc, enc_lat, enc_lon, locs[i][0], locs[i][1])
        vec = [0.0] * BATCH_SIZE
        vec[i] = 1.0
        id_pt = cc.MakeCKKSPackedPlaintext(vec)
        id_ct = cc.Encrypt(joint_pk, id_pt)
        opponents.append((dist_ct, id_ct))

    t0 = time.time()
    result_ct = find_nearest_by_scoring(cc, opponents)
    elapsed = time.time() - t0

    values = threshold_decrypt(cc, keypairs, result_ct)
    winner_slot = max(range(len(values)), key=lambda i: values[i])

    return winner_slot, values, elapsed


def verify_nearest(test_case, target_idx, locs, names, label, sq_tolerance=0.25):
    """Verify FHE picks correct nearest neighbor.
    Returns the winning slot index.

    sq_tolerance: relative tolerance on squared distances for near-ties.
    If the top-2 are within sq_tolerance of each other, accept either.
    """
    n = len(locs)
    target = locs[target_idx]

    # Compute plaintext distances
    pt_dists = []
    for i in range(n):
        if i == target_idx:
            continue
        sq = plaintext_sq_dist(target, locs[i])
        m = haversine_m(target, locs[i])
        pt_dists.append((i, sq, m))
    pt_dists.sort(key=lambda x: x[1])

    expected_slot = pt_dists[0][0]
    expected_sq = pt_dists[0][1]

    # Acceptable slots (within tolerance for near-ties)
    acceptable = {expected_slot}
    for i in range(1, len(pt_dists)):
        ratio = pt_dists[i][1] / expected_sq if expected_sq > 0 else float('inf')
        if ratio - 1.0 < sq_tolerance:
            acceptable.add(pt_dists[i][0])
        else:
            break

    winner, values, elapsed = run_scoring(target_idx, locs, names)

    test_case.assertIn(
        winner, acceptable,
        f"[{label}] {names[target_idx]} (slot {target_idx}): "
        f"expected one of {[names[s] for s in acceptable]}, "
        f"got {names[winner]} (slot {winner})"
    )

    print(f"  [{label}] {names[target_idx]} → {names[winner]} "
          f"({haversine_m(target, locs[winner]):.0f}m, {elapsed:.1f}s)  OK")

    return winner


# ==========================================
# Create shared 3-party threshold context
# ==========================================

print("Creating threshold FHE context (3-party) for dense urban tests...")
t0 = time.time()
_cc = create_context()
_keypairs, _joint_pk = generate_threshold_keys(_cc, NUM_BASE_PARTIES)
generate_eval_mult_key(_cc, _keypairs)
print(f"  Ready in {time.time() - t0:.1f}s")


# ==========================================
# Dense city datasets (20 locations each)
# ==========================================

MANHATTAN = {
    "names": [
        "Times Square",              # 0
        "Bryant Park",               # 1  ~350m
        "Grand Central",             # 2  ~1km
        "Empire State",              # 3  ~1.1km
        "Rockefeller Center",        # 4  ~600m
        "MoMA",                      # 5  ~800m
        "Carnegie Hall",             # 6  ~900m
        "Columbus Circle",           # 7  ~1.2km
        "Penn Station",              # 8  ~900m
        "Madison Sq Garden",         # 9  ~900m (right next to Penn)
        "Flatiron Building",         # 10 ~1.8km
        "Union Square",              # 11 ~2.5km
        "Washington Sq Park",        # 12 ~3km
        "Wall Street",               # 13 ~6km
        "Brooklyn Bridge",           # 14 ~5km
        "One World Trade",           # 15 ~5.5km
        "High Line (14th)",          # 16 ~2.5km
        "Met Museum",                # 17 ~3km
        "Intrepid Museum",           # 18 ~1.5km
        "Central Park Zoo",          # 19 ~2km
    ],
    "locs": [
        (40.75800, -73.98557),   # 0  Times Square
        (40.75360, -73.98340),   # 1  Bryant Park
        (40.75270, -73.97720),   # 2  Grand Central
        (40.74844, -73.98566),   # 3  Empire State
        (40.75890, -73.97931),   # 4  Rockefeller
        (40.76160, -73.97770),   # 5  MoMA
        (40.76500, -73.97990),   # 6  Carnegie Hall
        (40.76800, -73.98190),   # 7  Columbus Circle
        (40.75050, -73.99340),   # 8  Penn Station
        (40.75060, -73.99350),   # 9  MSG (11m from Penn!)
        (40.74110, -73.98970),   # 10 Flatiron
        (40.73570, -73.99070),   # 11 Union Square
        (40.73080, -73.99720),   # 12 Washington Sq Park
        (40.70560, -74.00900),   # 13 Wall Street
        (40.70608, -73.99691),   # 14 Brooklyn Bridge
        (40.71270, -74.01340),   # 15 One World Trade
        (40.74250, -74.00690),   # 16 High Line (14th)
        (40.77920, -73.96340),   # 17 Met Museum
        (40.76460, -73.99970),   # 18 Intrepid Museum
        (40.76770, -73.97180),   # 19 Central Park Zoo
    ],
}

SF_DOWNTOWN = {
    "names": [
        "Salesforce Tower",          # 0
        "Ferry Building",            # 1  ~600m
        "Transamerica Pyramid",      # 2  ~700m
        "Moscone Center",            # 3  ~600m
        "Union Square SF",           # 4  ~900m
        "Powell St Station",         # 5  ~800m
        "Coit Tower",                # 6  ~1.5km
        "Embarcadero Center",        # 7  ~400m
        "SF City Hall",              # 8  ~2km
        "Twitter/X HQ",              # 9  ~1.8km
        "Oracle Park",               # 10 ~1.5km
        "Pier 39",                   # 11 ~2.5km
        "Ghirardelli Square",        # 12 ~3km
        "Palace of Fine Arts",       # 13 ~4km
        "Lombard Street",            # 14 ~2.5km
        "Chinatown Gate",            # 15 ~1km
        "AT&T Park Lot",             # 16 ~1.5km
        "SFMOMA",                    # 17 ~400m
        "Yerba Buena Gardens",       # 18 ~500m
        "Rincon Park",               # 19 ~300m
    ],
    "locs": [
        (37.78950, -122.39640),  # 0  Salesforce Tower
        (37.79550, -122.39340),  # 1  Ferry Building
        (37.79520, -122.40280),  # 2  Transamerica Pyramid
        (37.78390, -122.40060),  # 3  Moscone Center
        (37.78780, -122.40760),  # 4  Union Square
        (37.78480, -122.40790),  # 5  Powell St Station
        (37.80240, -122.40590),  # 6  Coit Tower
        (37.79480, -122.39880),  # 7  Embarcadero Center
        (37.77930, -122.41880),  # 8  SF City Hall
        (37.77680, -122.41650),  # 9  Twitter/X HQ
        (37.77860, -122.38930),  # 10 Oracle Park
        (37.80870, -122.40990),  # 11 Pier 39
        (37.80570, -122.42270),  # 12 Ghirardelli Square
        (37.80280, -122.44820),  # 13 Palace of Fine Arts
        (37.80200, -122.41870),  # 14 Lombard Street
        (37.79050, -122.40570),  # 15 Chinatown Gate
        (37.77840, -122.38820),  # 16 AT&T Park Lot
        (37.78560, -122.40110),  # 17 SFMOMA
        (37.78510, -122.40270),  # 18 Yerba Buena Gardens
        (37.79080, -122.38850),  # 19 Rincon Park
    ],
}

TOKYO_CENTRAL = {
    "names": [
        "Shibuya Crossing",          # 0
        "Shibuya 109",               # 1  ~160m
        "Harajuku Station",          # 2  ~1.3km
        "Meiji Shrine",              # 3  ~2km
        "Shinjuku Station",          # 4  ~3.5km
        "Tokyo Tower",               # 5  ~5km
        "Roppongi Hills",            # 6  ~3km
        "Tokyo Station",             # 7  ~8km
        "Ginza Crossing",            # 8  ~7km
        "Akihabara",                 # 9  ~9km
        "Ueno Park",                 # 10 ~9km
        "Asakusa/Sensoji",           # 11 ~10km
        "Ikebukuro Station",         # 12 ~5km
        "Ebisu Garden Place",        # 13 ~1.5km
        "Yoyogi Park",               # 14 ~1km from Shibuya
        "Shinagawa Station",         # 15 ~5km
        "Odaiba",                    # 16 ~8km
        "Tsukiji Market",            # 17 ~6km
        "Nihonbashi",                # 18 ~7km
        "Hachiko Statue",            # 19 ~50m from crossing
    ],
    "locs": [
        (35.65940, 139.70060),   # 0  Shibuya Crossing
        (35.65910, 139.69900),   # 1  Shibuya 109
        (35.67040, 139.70270),   # 2  Harajuku Station
        (35.67640, 139.69930),   # 3  Meiji Shrine
        (35.68950, 139.70040),   # 4  Shinjuku Station
        (35.65860, 139.74540),   # 5  Tokyo Tower
        (35.66050, 139.72910),   # 6  Roppongi Hills
        (35.68120, 139.76710),   # 7  Tokyo Station
        (35.67110, 139.76390),   # 8  Ginza Crossing
        (35.69840, 139.77310),   # 9  Akihabara
        (35.71460, 139.77350),   # 10 Ueno Park
        (35.71480, 139.79670),   # 11 Asakusa/Sensoji
        (35.72890, 139.71100),   # 12 Ikebukuro Station
        (35.64680, 139.71370),   # 13 Ebisu Garden Place
        (35.67170, 139.69470),   # 14 Yoyogi Park
        (35.62870, 139.73880),   # 15 Shinagawa Station
        (35.62500, 139.77520),   # 16 Odaiba
        (35.66530, 139.76990),   # 17 Tsukiji Market
        (35.68360, 139.77390),   # 18 Nihonbashi
        (35.65900, 139.70040),   # 19 Hachiko Statue
    ],
}

LONDON_CENTRAL = {
    "names": [
        "Big Ben",                   # 0
        "Westminster Abbey",         # 1  ~200m
        "London Eye",                # 2  ~500m
        "10 Downing Street",         # 3  ~300m from Big Ben
        "Trafalgar Square",          # 4  ~700m
        "Buckingham Palace",         # 5  ~800m
        "Piccadilly Circus",         # 6  ~1.1km
        "Covent Garden",             # 7  ~1.3km
        "Leicester Square",          # 8  ~1.1km
        "St Paul's Cathedral",       # 9  ~2.5km
        "Tower of London",           # 10 ~3.5km
        "Tower Bridge",              # 11 ~3.6km
        "The Shard",                 # 12 ~2.8km
        "Tate Modern",               # 13 ~2km
        "Borough Market",            # 14 ~2.5km
        "Oxford Circus",             # 15 ~1.5km
        "British Museum",            # 16 ~1.8km
        "King's Cross",              # 17 ~3.2km
        "Soho Square",               # 18 ~1.3km
        "Embankment Station",        # 19 ~500m
    ],
    "locs": [
        (51.50070, -0.12460),    # 0  Big Ben
        (51.49940, -0.12720),    # 1  Westminster Abbey
        (51.50330, -0.11950),    # 2  London Eye
        (51.50340, -0.12570),    # 3  10 Downing Street
        (51.50800, -0.12810),    # 4  Trafalgar Square
        (51.50140, -0.14190),    # 5  Buckingham Palace
        (51.50990, -0.13430),    # 6  Piccadilly Circus
        (51.51190, -0.12410),    # 7  Covent Garden
        (51.51100, -0.12830),    # 8  Leicester Square
        (51.51380, -0.09840),    # 9  St Paul's Cathedral
        (51.50810, -0.07590),    # 10 Tower of London
        (51.50550, -0.07540),    # 11 Tower Bridge
        (51.50450, -0.08660),    # 12 The Shard
        (51.50760, -0.09930),    # 13 Tate Modern
        (51.50540, -0.09100),    # 14 Borough Market
        (51.51520, -0.14160),    # 15 Oxford Circus
        (51.51940, -0.12690),    # 16 British Museum
        (51.53050, -0.12370),    # 17 King's Cross
        (51.51550, -0.13190),    # 18 Soho Square
        (51.50710, -0.12230),    # 19 Embankment Station
    ],
}

MUMBAI = {
    "names": [
        "Gateway of India",          # 0
        "Taj Mahal Palace",          # 1  ~100m
        "Colaba Causeway",           # 2  ~500m
        "Nariman Point",             # 3  ~1.5km
        "Marine Drive (south)",      # 4  ~2km
        "Chowpatty Beach",           # 5  ~3.5km
        "Horniman Circle",           # 6  ~800m from Gateway
        "CST Station",               # 7  ~2km
        "Crawford Market",           # 8  ~2.5km
        "Haji Ali Dargah",           # 9  ~5km
        "Mahalaxmi Temple",          # 10 ~5.5km
        "Bandra-Worli Sea Link",     # 11 ~8km
        "Mount Mary Church",         # 12 ~10km
        "Bandra Station",            # 13 ~10.5km
        "Siddhivinayak Temple",      # 14 ~7km
        "Dadar Station",             # 15 ~7.5km
        "Nehru Planetarium",         # 16 ~4.5km
        "Mani Bhavan",               # 17 ~4km
        "Flora Fountain",            # 18 ~1.2km
        "Oval Maidan",               # 19 ~1km
    ],
    "locs": [
        (18.92200, 72.83470),    # 0  Gateway of India
        (18.92170, 72.83330),    # 1  Taj Mahal Palace
        (18.91670, 72.83250),    # 2  Colaba Causeway
        (18.92540, 72.82250),    # 3  Nariman Point
        (18.94370, 72.82340),    # 4  Marine Drive (south end)
        (18.95530, 72.81510),    # 5  Chowpatty Beach
        (18.93170, 72.84190),    # 6  Horniman Circle
        (18.94000, 72.83530),    # 7  CST Station
        (18.94750, 72.83330),    # 8  Crawford Market
        (18.98270, 72.80880),    # 9  Haji Ali Dargah
        (18.98250, 72.81240),    # 10 Mahalaxmi Temple
        (19.02820, 72.81560),    # 11 Bandra-Worli Sea Link
        (19.04300, 72.82830),    # 12 Mount Mary Church
        (19.05420, 72.84010),    # 13 Bandra Station
        (19.01710, 72.83020),    # 14 Siddhivinayak Temple
        (19.01780, 72.84290),    # 15 Dadar Station
        (18.97300, 72.81230),    # 16 Nehru Planetarium
        (18.96150, 72.81110),    # 17 Mani Bhavan
        (18.93350, 72.83080),    # 18 Flora Fountain
        (18.93010, 72.83100),    # 19 Oval Maidan
    ],
}


# ======================================================================
# Test classes
# ======================================================================

class TestManhattan20Users(unittest.TestCase):
    """20 users scattered across Manhattan, ~50m to ~5km apart."""

    def test_from_times_square(self):
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 0, locs, names, "Manhattan")

    def test_from_wall_street(self):
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 13, locs, names, "Manhattan")

    def test_from_met_museum(self):
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 17, locs, names, "Manhattan")

    def test_adjacent_penn_msg(self):
        """Penn Station and MSG are ~11m apart."""
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        dist_m = haversine_m(locs[8], locs[9])
        print(f"\n  [Manhattan] Penn Station <-> MSG: {dist_m:.0f}m")
        verify_nearest(self, 8, locs, names, "Manhattan-adjacent")

    def test_from_washington_sq(self):
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 12, locs, names, "Manhattan")


class TestSFDowntown20Users(unittest.TestCase):
    """20 users in downtown San Francisco, ~100m to ~3km apart."""

    def test_from_salesforce_tower(self):
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 0, locs, names, "SF")

    def test_from_sfmoma(self):
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 17, locs, names, "SF")

    def test_from_pier39(self):
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 11, locs, names, "SF")

    def test_from_city_hall(self):
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 8, locs, names, "SF")

    def test_from_oracle_park(self):
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 10, locs, names, "SF")


class TestTokyoCentral20Users(unittest.TestCase):
    """20 users across central Tokyo, ~50m to ~10km apart."""

    def test_from_shibuya_crossing(self):
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        dist_19 = haversine_m(locs[0], locs[19])
        dist_1 = haversine_m(locs[0], locs[1])
        print(f"\n  [Tokyo] Shibuya Crossing -> Hachiko: {dist_19:.0f}m, -> 109: {dist_1:.0f}m")
        verify_nearest(self, 0, locs, names, "Tokyo")

    def test_from_tokyo_station(self):
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 7, locs, names, "Tokyo")

    def test_from_asakusa(self):
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 11, locs, names, "Tokyo")

    def test_from_shinjuku(self):
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 4, locs, names, "Tokyo")

    def test_from_ebisu(self):
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 13, locs, names, "Tokyo")


class TestLondonCentral20Users(unittest.TestCase):
    """20 users across central London, ~100m to ~4km."""

    def test_from_big_ben(self):
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 0, locs, names, "London")

    def test_from_tower_bridge(self):
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 11, locs, names, "London")

    def test_from_leicester_sq(self):
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 8, locs, names, "London")

    def test_from_kings_cross(self):
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 17, locs, names, "London")

    def test_from_buckingham(self):
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 5, locs, names, "London")


class TestMumbai20Users(unittest.TestCase):
    """20 users from Colaba to Bandra, ~100m to ~10km."""

    def test_from_gateway(self):
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        d = haversine_m(locs[0], locs[1])
        print(f"\n  [Mumbai] Gateway -> Taj: {d:.0f}m")
        verify_nearest(self, 0, locs, names, "Mumbai")

    def test_from_cst_station(self):
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 7, locs, names, "Mumbai")

    def test_from_bandra(self):
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 13, locs, names, "Mumbai")

    def test_from_haji_ali(self):
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 9, locs, names, "Mumbai")

    def test_from_nariman_point(self):
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 3, locs, names, "Mumbai")


class TestSameBlock(unittest.TestCase):
    """Extreme precision: users < 50m apart."""

    def test_times_square_block(self):
        locs = [
            (40.75800, -73.98557),  # 0  TS center
            (40.75825, -73.98540),  # 1  ~30m NE
            (40.75780, -73.98580),  # 2  ~30m SW
            (40.75810, -73.98500),  # 3  ~50m E
            (40.75760, -73.98600),  # 4  ~55m SW
        ]
        names = ["TS center", "30m NE", "30m SW", "50m E", "55m SW"]

        print("\n  [same-block] Distances from TS center:")
        for i in range(1, 5):
            print(f"    -> {names[i]:10s}: {haversine_m(locs[0], locs[i]):.0f}m")

        verify_nearest(self, 0, locs, names, "same-block")

    def test_shibuya_scramble(self):
        locs = [
            (35.65940, 139.70060),   # 0  Crossing center
            (35.65935, 139.70055),   # 1  ~7m
            (35.65960, 139.70080),   # 2  ~28m NE
            (35.65910, 139.70030),   # 3  ~45m SW
            (35.65900, 139.70040),   # 4  Hachiko ~50m
        ]
        names = ["Scramble", "7m away", "28m NE", "45m SW", "Hachiko"]

        print("\n  [same-block] Distances from Shibuya Scramble:")
        for i in range(1, 5):
            print(f"    -> {names[i]:10s}: {haversine_m(locs[0], locs[i]):.0f}m")

        verify_nearest(self, 0, locs, names, "same-block-shibuya")


class TestDenseClusterWithOutliers(unittest.TestCase):
    """Dense cluster + far away outliers. Outliers should never be nearest."""

    def test_manhattan_cluster_plus_boroughs(self):
        midtown = [
            (40.75800, -73.98557),  # 0  Times Square
            (40.75360, -73.98340),  # 1  Bryant Park
            (40.75270, -73.97720),  # 2  Grand Central
            (40.74844, -73.98566),  # 3  Empire State
            (40.75890, -73.97931),  # 4  Rockefeller
            (40.76160, -73.97770),  # 5  MoMA
            (40.76500, -73.97990),  # 6  Carnegie Hall
            (40.76800, -73.98190),  # 7  Columbus Circle
            (40.75050, -73.99340),  # 8  Penn / MSG
            (40.74110, -73.98970),  # 9  Flatiron
            (40.73570, -73.99070),  # 10 Union Square
            (40.77220, -73.98350),  # 11 Lincoln Center
            (40.76770, -73.97180),  # 12 Central Park Zoo
            (40.75060, -73.99350),  # 13 Penn Station
            (40.76160, -73.97770),  # 14 (duplicate MoMA for density)
        ]
        outer = [
            (40.82920, -73.92620),  # 15 Yankee Stadium (~10km)
            (40.68920, -74.04450),  # 16 Statue of Liberty (~9km)
            (40.67130, -73.96340),  # 17 Prospect Park Brooklyn (~10km)
            (40.74480, -73.94880),  # 18 Long Island City (~3.5km)
            (40.58920, -73.95100),  # 19 Coney Island (~19km)
        ]
        locs = midtown + outer
        names = [
            "Times Square", "Bryant Park", "Grand Central", "Empire State",
            "Rockefeller", "MoMA", "Carnegie Hall", "Columbus Circle",
            "Penn/MSG", "Flatiron", "Union Square", "Lincoln Center",
            "CP Zoo", "Penn Station", "MoMA-2",
            "Yankee Stadium", "Statue Liberty", "Prospect Pk", "LIC", "Coney Island",
        ]

        midtown_slots = set(range(15))
        for target in [0, 3, 7, 10]:
            winner = verify_nearest(self, target, locs, names, "cluster")
            self.assertIn(
                winner, midtown_slots,
                f"{names[target]}: nearest should be in Midtown, got {names[winner]}"
            )

    def test_sf_soma_plus_east_bay(self):
        soma = [
            (37.78950, -122.39640),  # 0  Salesforce Tower
            (37.78560, -122.40110),  # 1  SFMOMA
            (37.78510, -122.40270),  # 2  Yerba Buena
            (37.78390, -122.40060),  # 3  Moscone
            (37.78230, -122.39410),  # 4  South Park
            (37.78480, -122.40790),  # 5  Powell St
            (37.78840, -122.39230),  # 6  Rincon Hill
            (37.79080, -122.38850),  # 7  Rincon Park
            (37.78660, -122.39950),  # 8  Mint Plaza
            (37.78770, -122.39800),  # 9  Salesforce Park
        ]
        east_bay = [
            (37.79520, -122.27280),  # 10 Jack London Sq, Oakland
            (37.80440, -122.27160),  # 11 Oakland City Hall
            (37.87120, -122.27280),  # 12 UC Berkeley
            (37.77440, -122.21790),  # 13 Oakland Airport
            (37.85340, -122.24200),  # 14 Rockridge
        ]
        locs = soma + east_bay
        names = [
            "Salesforce", "SFMOMA", "YB Gardens", "Moscone", "South Park",
            "Powell St", "Rincon Hill", "Rincon Park", "Mint Plaza", "SF Park",
            "Jack London", "Oakland CH", "UC Berkeley", "OAK Airport", "Rockridge",
        ]

        soma_slots = set(range(10))
        for target in [0, 3, 5]:
            winner = verify_nearest(self, target, locs, names, "SF-cluster")
            self.assertIn(
                winner, soma_slots,
                f"{names[target]}: nearest should be in SoMa, got {names[winner]}"
            )


class TestPrecisionDistance(unittest.TestCase):
    """Verify FHE distinguishes small distance differences."""

    def test_can_distinguish_30m_vs_50m(self):
        target = (40.75000, -73.98000)
        close  = (40.75027, -73.98000)   # ~30m north
        far    = (40.75045, -73.98000)   # ~50m north
        locs = [target, close, far]
        names = ["Target", "30m N", "50m N"]

        d1 = haversine_m(target, close)
        d2 = haversine_m(target, far)
        print(f"\n  [precision] 30m vs 50m: actual {d1:.0f}m vs {d2:.0f}m")

        winner, _, elapsed = run_scoring(0, locs, names)
        self.assertEqual(winner, 1, f"Should pick 30m (slot 1), got slot {winner}")
        print(f"  [precision] winner=slot {winner} ({elapsed:.1f}s)  OK")

    def test_can_distinguish_100m_vs_200m(self):
        target = (51.50700, -0.12760)
        close  = (51.50790, -0.12760)   # ~100m north
        far    = (51.50880, -0.12760)   # ~200m north
        locs = [target, close, far]
        names = ["Target", "100m N", "200m N"]

        d1 = haversine_m(target, close)
        d2 = haversine_m(target, far)
        print(f"\n  [precision] 100m vs 200m: actual {d1:.0f}m vs {d2:.0f}m")

        winner, _, elapsed = run_scoring(0, locs, names)
        self.assertEqual(winner, 1, f"Should pick 100m (slot 1), got slot {winner}")
        print(f"  [precision] winner=slot {winner} ({elapsed:.1f}s)  OK")

    def test_can_distinguish_500m_vs_700m(self):
        target = (35.65940, 139.70060)   # Shibuya
        close  = (35.66390, 139.70060)   # ~500m north
        far    = (35.66570, 139.70060)   # ~700m north
        locs = [target, close, far]
        names = ["Shibuya", "500m N", "700m N"]

        d1 = haversine_m(target, close)
        d2 = haversine_m(target, far)
        print(f"\n  [precision] 500m vs 700m: actual {d1:.0f}m vs {d2:.0f}m")

        winner, _, elapsed = run_scoring(0, locs, names)
        self.assertEqual(winner, 1, f"Should pick 500m (slot 1), got slot {winner}")
        print(f"  [precision] winner=slot {winner} ({elapsed:.1f}s)  OK")


class TestDenseAllPartyRequired(unittest.TestCase):
    """SECURITY: Verify ALL parties required even for dense urban scenarios.
    Uses a Manhattan 5-user scenario to ensure partial decrypt → garbage."""

    def test_manhattan_partial_decrypt_fails(self):
        """Run Manhattan proximity, then verify partial decrypt gives garbage."""
        locs = MANHATTAN["locs"][:5]
        names = MANHATTAN["names"][:5]

        # Run full pipeline
        winner, full_vals, elapsed = run_scoring(0, locs, names)
        print(f"  [security] Manhattan full: slot {winner} ({elapsed:.1f}s)")

        # Now recompute the scoring ciphertext
        init_lat, init_lon = locs[0]
        nlat = init_lat / MAX_COORD
        nlon = init_lon / MAX_COORD
        lat_pt = _cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
        lon_pt = _cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
        enc_lat = _cc.Encrypt(_joint_pk, lat_pt)
        enc_lon = _cc.Encrypt(_joint_pk, lon_pt)

        opponents = []
        for i in range(1, 5):
            dist_ct = compute_distance_local(_cc, enc_lat, enc_lon, locs[i][0], locs[i][1])
            vec = [0.0] * BATCH_SIZE
            vec[i] = 1.0
            id_pt = _cc.MakeCKKSPackedPlaintext(vec)
            id_ct = _cc.Encrypt(_joint_pk, id_pt)
            opponents.append((dist_ct, id_ct))

        result_ct = find_nearest_by_scoring(_cc, opponents)

        # Full threshold decrypt works
        full_result = threshold_decrypt(_cc, _keypairs, result_ct)
        full_winner = max(range(len(full_result)), key=lambda i: full_result[i])

        # Partial decrypt (missing party 2) → error or garbage
        try:
            partial_result = partial_decrypt(_cc, _keypairs, result_ct, [0, 1])
            error = abs(partial_result[full_winner] - full_result[full_winner])
            self.assertGreater(error, 0.01,
                               f"Missing party should give garbage, error={error}")
            print(f"  [security] Manhattan partial: error={error:.4f} (garbage)  OK")
        except RuntimeError:
            print(f"  [security] Manhattan partial: RuntimeError (decode refused)  OK")

    def test_tokyo_single_party_fails(self):
        """No single party can decrypt Tokyo dense results."""
        val = 99.0
        pt = _cc.MakeCKKSPackedPlaintext([val / MAX_COORD] * BATCH_SIZE)
        ct = _cc.Encrypt(_joint_pk, pt)

        for party_idx in range(NUM_BASE_PARTIES):
            try:
                single_vals = partial_decrypt(_cc, _keypairs, ct, [party_idx])
                error = abs(single_vals[0] - val / MAX_COORD)
                self.assertGreater(error, 0.01,
                                   f"Party {party_idx} alone should fail, error={error}")
            except RuntimeError:
                pass  # Expected: OpenFHE refuses to decode
        print(f"  [security] Tokyo: no single party can decrypt  OK")


# ==========================================
# Runner
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FHE Proximity — Dense Urban Area Tests (Threshold)")
    print("=" * 60)
    print(f"  BATCH_SIZE={BATCH_SIZE}, MAX_USERS={MAX_USERS}")
    print(f"  Threshold: N-of-N (ALL {NUM_BASE_PARTIES} parties required)")
    print(f"  Pairwise scoring, depth budget 5 levels")
    print()
    print("5 cities x 20 users each + precision + cluster + security")
    print("=" * 60 + "\n")

    start = time.time()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Fast tests first
    suite.addTests(loader.loadTestsFromTestCase(TestPrecisionDistance))
    suite.addTests(loader.loadTestsFromTestCase(TestSameBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestDenseAllPartyRequired))
    suite.addTests(loader.loadTestsFromTestCase(TestDenseClusterWithOutliers))
    suite.addTests(loader.loadTestsFromTestCase(TestManhattan20Users))
    suite.addTests(loader.loadTestsFromTestCase(TestSFDowntown20Users))
    suite.addTests(loader.loadTestsFromTestCase(TestTokyoCentral20Users))
    suite.addTests(loader.loadTestsFromTestCase(TestLondonCentral20Users))
    suite.addTests(loader.loadTestsFromTestCase(TestMumbai20Users))

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
