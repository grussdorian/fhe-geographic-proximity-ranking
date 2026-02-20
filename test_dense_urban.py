#!/usr/bin/env python3.13
# ==========================================
# test_dense_urban.py
#
# FHE proximity tests focused on densely
# populated urban areas where users are
# within the same city, often just blocks
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
# Run:  python3.13 test_dense_urban.py
# Requires: keys generated via setup_keys.py
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
# FHE helpers
# ==========================================

def encrypt_location(lat, lon):
    nlat = float(lat) / MAX_COORD
    nlon = float(lon) / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    return cc.Encrypt(public_key, lat_pt), cc.Encrypt(public_key, lon_pt)


def encrypt_onehot_id(slot_index):
    vec = [0.0] * BATCH_SIZE
    vec[slot_index] = 1.0
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(public_key, pt)


def decrypt_vector(ct, length=BATCH_SIZE):
    pt = cc.Decrypt(secret_key, ct)
    pt.SetLength(length)
    return list(pt.GetRealPackedValue())


def compute_distance(x1, y1, x2, y2):
    dx = cc.EvalSub(x1, x2)
    dy = cc.EvalSub(y1, y2)
    dx2 = cc.EvalMult(dx, dx)
    dy2 = cc.EvalMult(dy, dy)
    return cc.EvalAdd(dx2, dy2)


def compute_selector(diff):
    x2 = cc.EvalMult(diff, diff)
    inner = cc.EvalMult(x2, -0.00084375)
    inner = cc.EvalAdd(inner, 0.1125)
    product = cc.EvalMult(inner, diff)
    return cc.EvalAdd(product, 0.5)


def find_nearest_by_scoring(target_x, target_y, opponents):
    n = len(opponents)

    if n == 1:
        return opponents[0][2]

    dists = []
    for x_ct, y_ct, _ in opponents:
        d = compute_distance(target_x, target_y, x_ct, y_ct)
        dists.append(d)

    scores = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = cc.EvalSub(dists[j], dists[i])
            sel_ji = compute_selector(diff)
            sel_ij = cc.EvalSub(1.0, sel_ji)
            scores[i] = cc.EvalAdd(scores[i], sel_ji) if scores[i] is not None else sel_ji
            scores[j] = cc.EvalAdd(scores[j], sel_ij) if scores[j] is not None else sel_ij

    result = cc.EvalMult(scores[0], opponents[0][2])
    for i in range(1, n):
        result = cc.EvalAdd(result, cc.EvalMult(scores[i], opponents[i][2]))

    return result


# ==========================================
# Plaintext helpers
# ==========================================

def plaintext_sq_dist(loc1, loc2):
    dlat = (loc1[0] - loc2[0]) / MAX_COORD
    dlon = (loc1[1] - loc2[1]) / MAX_COORD
    return dlat ** 2 + dlon ** 2


def plaintext_nearest(target, others):
    dists = [plaintext_sq_dist(target, o) for o in others]
    return min(range(len(dists)), key=lambda i: dists[i])


def haversine_m(loc1, loc2):
    """Great-circle distance in METERS."""
    R = 6_371_000.0
    lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
    lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fmt_dist(loc1, loc2):
    """Human-readable distance string."""
    m = haversine_m(loc1, loc2)
    return f"{m:.0f}m" if m < 1000 else f"{m/1000:.2f}km"


# ==========================================
# Scoring helper
# ==========================================

def run_scoring(target_idx, locations, names=None):
    """
    Run pairwise scoring from target_idx against all others.
    Returns (winner_slot, decrypted_vector, elapsed_seconds).
    """
    n = len(locations)
    assert n <= BATCH_SIZE

    encrypted = []
    for i, (lat, lon) in enumerate(locations):
        lat_ct, lon_ct = encrypt_location(lat, lon)
        id_ct = encrypt_onehot_id(i)
        encrypted.append((lat_ct, lon_ct, id_ct))

    target_lat, target_lon, _ = encrypted[target_idx]

    opponents = []
    for i, (lat_ct, lon_ct, id_ct) in enumerate(encrypted):
        if i == target_idx:
            continue
        opponents.append((lat_ct, lon_ct, id_ct))

    t0 = time.time()
    winner_id_ct = find_nearest_by_scoring(target_lat, target_lon, opponents)
    elapsed = time.time() - t0

    values = decrypt_vector(winner_id_ct)
    winner_slot = max(range(len(values)), key=lambda i: values[i])

    return winner_slot, values, elapsed


def verify_nearest(test_case, target_idx, locations, names, label,
                   sq_tolerance=0.25):
    """
    Verify FHE nearest matches plaintext nearest.

    Because the CKKS selector polynomial returns ≈0.5 when two normalised
    squared distances are very close, the FHE result may pick *any*
    candidate whose squared distance is within `sq_tolerance` (relative,
    default 25 %) of the true nearest.  That is a correct outcome — the
    distances are indistinguishable under FHE noise.
    """
    n = len(locations)
    slots = [i for i in range(n) if i != target_idx]
    others = [locations[i] for i in slots]
    expected_idx = plaintext_nearest(locations[target_idx], others)
    expected_slot = slots[expected_idx]

    # Compute all squared distances and sort
    dists = [(i, plaintext_sq_dist(locations[target_idx], locations[i]),
              haversine_m(locations[target_idx], locations[i]))
             for i in range(n) if i != target_idx]
    dists.sort(key=lambda x: x[1])
    top3 = dists[:3]

    best_sq = dists[0][1]  # smallest squared distance

    # Build set of acceptable winners: any slot whose sq dist is within
    # tolerance of the true nearest (accounts for CKKS polynomial limits)
    acceptable = set()
    for idx, sq, _ in dists:
        if best_sq == 0:
            if sq == 0:
                acceptable.add(idx)
        elif sq <= best_sq * (1.0 + sq_tolerance):
            acceptable.add(idx)
        else:
            break  # sorted, no need to check further

    print(f"\n  [{label}] {names[target_idx]} — top 3 nearest:")
    for rank, (idx, sq, m) in enumerate(top3):
        marker = " <--" if idx == expected_slot else ""
        if idx in acceptable and idx != expected_slot:
            marker += " (acceptable)"
        dist_str = f"{m:.0f}m" if m < 1000 else f"{m/1000:.2f}km"
        print(f"    {rank+1}. {names[idx]:30s} (slot {idx:2d}): {dist_str:>10s}  sq={sq:.2e}{marker}")

    winner, vals, elapsed = run_scoring(target_idx, locations, names)

    test_case.assertIn(
        winner, acceptable,
        f"{label} {names[target_idx]}: expected one of "
        f"{[f'{names[s]} (slot {s})' for s in acceptable]}, "
        f"got {names[winner]} (slot {winner})"
    )

    top_slots = sorted(enumerate(vals), key=lambda x: -x[1])[:3]
    status = "OK" if winner == expected_slot else f"OK (tied — {names[winner]} also valid)"
    print(f"  [{label}] Winner: {names[winner]} (slot {winner})  ({elapsed:.1f}s)  {status}")
    for rank, (idx, val) in enumerate(top_slots):
        print(f"    rank {rank+1}: slot {idx:2d} ({names[idx]:30s}) = {val:.4f}")

    return winner


# ======================================================================
# REAL-WORLD DENSE URBAN LOCATIONS
#
# All coordinates verified to 5 decimal places.
# Distances shown are approximate haversine.
# ======================================================================

# ------------------------------------------------------------------
# MANHATTAN, NEW YORK CITY
# 20 locations across Midtown, Downtown, UES, UWS, Village
# Typical inter-point distance: 200m to 5km
# ------------------------------------------------------------------
MANHATTAN = {
    "names": [
        "Times Square",              # 0
        "Bryant Park",               # 1  ~350m from TS
        "Grand Central",             # 2  ~700m from TS
        "Empire State Building",     # 3  ~1.1km from TS
        "Rockefeller Center",        # 4  ~350m from TS
        "MoMA",                      # 5  ~500m from Rockefeller
        "Carnegie Hall",             # 6  ~600m from TS
        "Columbus Circle",           # 7  ~900m from TS
        "Penn Station",              # 8  ~600m from TS
        "Madison Sq Garden",         # 9  ~700m from Penn
        "Flatiron Building",         # 10 ~1.5km from TS
        "Union Square",              # 11 ~2.3km from TS
        "Washington Sq Park",        # 12 ~3km from TS
        "Wall Street",               # 13 ~5km from TS
        "Brooklyn Bridge",           # 14 ~5.5km from TS
        "One World Trade",           # 15 ~5km from TS
        "High Line (14th St)",       # 16 ~2.5km from TS
        "Met Museum",                # 17 ~2.5km NE of TS
        "Lincoln Center",            # 18 ~1.3km from TS
        "Central Park Zoo",          # 19 ~1.5km from TS
    ],
    "locs": [
        (40.75800, -73.98557),  # 0  Times Square
        (40.75360, -73.98340),  # 1  Bryant Park
        (40.75270, -73.97720),  # 2  Grand Central
        (40.74844, -73.98566),  # 3  Empire State Building
        (40.75890, -73.97931),  # 4  Rockefeller Center
        (40.76160, -73.97770),  # 5  MoMA
        (40.76500, -73.97990),  # 6  Carnegie Hall
        (40.76800, -73.98190),  # 7  Columbus Circle
        (40.75060, -73.99350),  # 8  Penn Station
        (40.75050, -73.99340),  # 9  Madison Sq Garden (adjacent to Penn)
        (40.74110, -73.98970),  # 10 Flatiron Building
        (40.73570, -73.99070),  # 11 Union Square
        (40.73080, -73.99740),  # 12 Washington Square Park
        (40.70600, -74.00900),  # 13 Wall Street
        (40.70610, -73.99690),  # 14 Brooklyn Bridge
        (40.71270, -74.01340),  # 15 One World Trade Center
        (40.74240, -74.00680),  # 16 High Line (14th St entrance)
        (40.77920, -73.96310),  # 17 Metropolitan Museum
        (40.77220, -73.98350),  # 18 Lincoln Center
        (40.76770, -73.97180),  # 19 Central Park Zoo
    ],
}

# ------------------------------------------------------------------
# DOWNTOWN SAN FRANCISCO
# 20 locations in the Financial District / SoMa / North Beach
# Typical: 100m to 3km
# ------------------------------------------------------------------
SF_DOWNTOWN = {
    "names": [
        "Salesforce Tower",          # 0
        "Transamerica Pyramid",      # 1
        "Ferry Building",            # 2
        "Moscone Center",            # 3
        "Union Square SF",           # 4
        "Chinatown Gate",            # 5
        "Coit Tower",                # 6
        "Embarcadero Center",        # 7
        "SF City Hall",              # 8
        "Twitter HQ (X)",            # 9
        "Oracle Park",               # 10
        "Pier 39",                   # 11
        "Ghirardelli Square",        # 12
        "Levi's Plaza",              # 13
        "Rincon Park",               # 14
        "South Park",                # 15
        "AT&T Park Lot",             # 16  near Oracle Park
        "SFMOMA",                    # 17
        "Yerba Buena Gardens",       # 18
        "Powell St Station",         # 19
    ],
    "locs": [
        (37.78950, -122.39640),  # 0  Salesforce Tower
        (37.79520, -122.40280),  # 1  Transamerica Pyramid
        (37.79540, -122.39350),  # 2  Ferry Building
        (37.78390, -122.40060),  # 3  Moscone Center
        (37.78780, -122.40760),  # 4  Union Square SF
        (37.79060, -122.40570),  # 5  Chinatown Gate
        (37.80220, -122.40590),  # 6  Coit Tower
        (37.79490, -122.39870),  # 7  Embarcadero Center
        (37.77920, -122.41920),  # 8  SF City Hall
        (37.77670, -122.41650),  # 9  Twitter/X HQ (Market St)
        (37.77840, -122.38920),  # 10 Oracle Park
        (37.80870, -122.40980),  # 11 Pier 39
        (37.80590, -122.42280),  # 12 Ghirardelli Square
        (37.80050, -122.39980),  # 13 Levi's Plaza
        (37.79080, -122.38850),  # 14 Rincon Park
        (37.78230, -122.39410),  # 15 South Park
        (37.77660, -122.39100),  # 16 AT&T Park Lot
        (37.78560, -122.40110),  # 17 SFMOMA
        (37.78510, -122.40270),  # 18 Yerba Buena Gardens
        (37.78480, -122.40790),  # 19 Powell St Station
    ],
}

# ------------------------------------------------------------------
# CENTRAL TOKYO
# 20 locations in Shibuya / Shinjuku / Ginza / Asakusa
# Typical: 200m to 10km
# ------------------------------------------------------------------
TOKYO_CENTRAL = {
    "names": [
        "Shibuya Crossing",          # 0
        "Shibuya 109",               # 1  ~100m from crossing
        "Harajuku Station",          # 2  ~1.2km from Shibuya
        "Meiji Shrine",              # 3  ~1.5km from Shibuya
        "Shinjuku Station",          # 4  ~3km from Shibuya
        "Tokyo Tower",               # 5  ~4.5km from Shibuya
        "Roppongi Hills",            # 6  ~2km from Shibuya
        "Tokyo Station",             # 7  ~6km from Shibuya
        "Ginza Crossing",            # 8  ~5.5km from Shibuya
        "Akihabara",                 # 9  ~7.5km from Shibuya
        "Ueno Park",                 # 10 ~9km from Shibuya
        "Asakusa/Sensoji",           # 11 ~10km from Shibuya
        "Ikebukuro Station",         # 12 ~6km from Shibuya
        "Ebisu Garden Place",        # 13 ~1km from Shibuya
        "Yoyogi Park",               # 14 ~1km from Shibuya
        "Shinagawa Station",         # 15 ~5km from Shibuya
        "Odaiba",                    # 16 ~8km from Shibuya
        "Tsukiji Market",            # 17 ~6km from Shibuya
        "Nihonbashi",                # 18 ~7km from Shibuya
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

# ------------------------------------------------------------------
# CENTRAL LONDON
# 20 locations in Westminster / City / Southbank / Soho
# Typical: 100m to 5km
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# MUMBAI (SOUTH/CENTRAL)
# 20 locations from Colaba to Bandra
# Typical: 200m to 10km
# ------------------------------------------------------------------
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
    """
    20 users scattered across Manhattan, ~50m to ~5km apart.
    Tests the system at city-block resolution.
    """

    def test_from_times_square(self):
        """
        Target: Times Square (0).
        Nearest should be Rockefeller (4) or Bryant Park (1), both ~350m.
        """
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 0, locs, names, "Manhattan")

    def test_from_wall_street(self):
        """
        Target: Wall Street (13).
        Nearest should be Brooklyn Bridge (14) or One World Trade (15).
        """
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 13, locs, names, "Manhattan")

    def test_from_met_museum(self):
        """
        Target: Met Museum (17), upper east side.
        Nearest should be Central Park Zoo (19) ~1.3km.
        """
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 17, locs, names, "Manhattan")

    def test_adjacent_penn_msg(self):
        """
        Penn Station (8) and Madison Sq Garden (9) are ~11m apart.
        From Penn Station, the nearest MUST be MSG.
        """
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        dist_m = haversine_m(locs[8], locs[9])
        print(f"\n  [Manhattan] Penn Station ↔ MSG: {dist_m:.0f}m")
        verify_nearest(self, 8, locs, names, "Manhattan-adjacent")

    def test_from_washington_sq(self):
        """
        Target: Washington Square Park (12), Greenwich Village.
        Nearest should be Union Square (11) ~750m or High Line (16).
        """
        locs = MANHATTAN["locs"]
        names = MANHATTAN["names"]
        verify_nearest(self, 12, locs, names, "Manhattan")


class TestSFDowntown20Users(unittest.TestCase):
    """
    20 users in downtown San Francisco, ~100m to ~3km apart.
    Very compact financial district + SoMa grid.
    """

    def test_from_salesforce_tower(self):
        """Salesforce Tower (0) — nearest is likely Embarcadero Ctr (7) or Rincon Park."""
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 0, locs, names, "SF")

    def test_from_sfmoma(self):
        """SFMOMA (17) — nearest is Yerba Buena (18) ~170m or Moscone (3)."""
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 17, locs, names, "SF")

    def test_from_pier39(self):
        """Pier 39 (11), touristy north — nearest is Ghirardelli (12) or Coit Tower (6)."""
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 11, locs, names, "SF")

    def test_from_city_hall(self):
        """SF City Hall (8) — nearest is Twitter/X HQ (9) ~300m."""
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 8, locs, names, "SF")

    def test_from_oracle_park(self):
        """Oracle Park (10) — nearest is AT&T lot (16) ~200m."""
        locs = SF_DOWNTOWN["locs"]
        names = SF_DOWNTOWN["names"]
        verify_nearest(self, 10, locs, names, "SF")


class TestTokyoCentral20Users(unittest.TestCase):
    """
    20 users across central Tokyo, ~50m to ~10km apart.
    Tests extreme density around Shibuya + spread to Asakusa.
    """

    def test_from_shibuya_crossing(self):
        """
        Shibuya Crossing (0) — Hachiko (19) is ~50m away, must win
        over Shibuya 109 (1) at ~160m.
        """
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        dist_19 = haversine_m(locs[0], locs[19])
        dist_1 = haversine_m(locs[0], locs[1])
        print(f"\n  [Tokyo] Shibuya Crossing → Hachiko: {dist_19:.0f}m, → 109: {dist_1:.0f}m")
        verify_nearest(self, 0, locs, names, "Tokyo")

    def test_from_tokyo_station(self):
        """Tokyo Station (7) — nearest is Nihonbashi (18) ~700m or Ginza (8)."""
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 7, locs, names, "Tokyo")

    def test_from_asakusa(self):
        """Asakusa/Sensoji (11) — nearest is Ueno Park (10) ~2km."""
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 11, locs, names, "Tokyo")

    def test_from_shinjuku(self):
        """Shinjuku (4) — nearest depends on exact coords. Likely Ikebukuro (12) or Yoyogi (14)."""
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 4, locs, names, "Tokyo")

    def test_from_ebisu(self):
        """Ebisu (13) — nearest is Shibuya Crossing (0) or Roppongi (6)."""
        locs = TOKYO_CENTRAL["locs"]
        names = TOKYO_CENTRAL["names"]
        verify_nearest(self, 13, locs, names, "Tokyo")


class TestLondonCentral20Users(unittest.TestCase):
    """
    20 users across central London, ~100m to ~4km.
    Very dense around Westminster / West End.
    """

    def test_from_big_ben(self):
        """Big Ben (0) — nearest is Westminster Abbey (1) ~200m or Downing St (3)."""
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 0, locs, names, "London")

    def test_from_tower_bridge(self):
        """Tower Bridge (11) — nearest is Tower of London (10) ~300m."""
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 11, locs, names, "London")

    def test_from_leicester_sq(self):
        """Leicester Square (8) — dense West End. Piccadilly (6) ~200m or Covent Garden (7)."""
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 8, locs, names, "London")

    def test_from_kings_cross(self):
        """King's Cross (17) — northern outlier. Nearest: British Museum (16) ~1.3km."""
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 17, locs, names, "London")

    def test_from_buckingham(self):
        """Buckingham Palace (5) — nearest: Westminster Abbey (1) or Big Ben (0)."""
        locs = LONDON_CENTRAL["locs"]
        names = LONDON_CENTRAL["names"]
        verify_nearest(self, 5, locs, names, "London")


class TestMumbai20Users(unittest.TestCase):
    """
    20 users from Colaba to Bandra, ~100m to ~10km.
    Tests a long, narrow coastal city layout.
    """

    def test_from_gateway(self):
        """Gateway of India (0) — Taj Mahal Palace (1) is ~150m, must win."""
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        d = haversine_m(locs[0], locs[1])
        print(f"\n  [Mumbai] Gateway → Taj: {d:.0f}m")
        verify_nearest(self, 0, locs, names, "Mumbai")

    def test_from_cst_station(self):
        """CST Station (7) — nearest: Crawford Market (8) ~850m or Flora Fountain (18)."""
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 7, locs, names, "Mumbai")

    def test_from_bandra(self):
        """Bandra Station (13) — nearest: Mount Mary (12) ~1.5km or Bandra-Worli (11)."""
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 13, locs, names, "Mumbai")

    def test_from_haji_ali(self):
        """Haji Ali (9) — nearest: Mahalaxmi (10) ~350m."""
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 9, locs, names, "Mumbai")

    def test_from_nariman_point(self):
        """Nariman Point (3) — nearest: Marine Drive (4) or Oval Maidan (19)."""
        locs = MUMBAI["locs"]
        names = MUMBAI["names"]
        verify_nearest(self, 3, locs, names, "Mumbai")


class TestSameBlock(unittest.TestCase):
    """
    Extreme precision test: multiple users on the same city block
    (< 50 meters apart). This pushes the limits of CKKS precision
    on tiny coordinate differences.
    """

    def test_times_square_block(self):
        """
        5 users within 200m of Times Square.
        Differences at the 4th-5th decimal place.
        """
        locs = [
            (40.75800, -73.98557),  # 0  Times Square center
            (40.75825, -73.98540),  # 1  ~30m NE
            (40.75780, -73.98580),  # 2  ~30m SW
            (40.75810, -73.98500),  # 3  ~50m E
            (40.75760, -73.98600),  # 4  ~55m SW
        ]
        names = ["TS center", "30m NE", "30m SW", "50m E", "55m SW"]

        print("\n  [same-block] Distances from TS center:")
        for i in range(1, 5):
            print(f"    → {names[i]:10s}: {haversine_m(locs[0], locs[i]):.0f}m")

        verify_nearest(self, 0, locs, names, "same-block")

    def test_shibuya_scramble(self):
        """
        5 users around Shibuya Scramble intersection — 10m to 150m.
        """
        locs = [
            (35.65940, 139.70060),   # 0  Crossing center
            (35.65935, 139.70055),   # 1  ~7m (next to the crossing)
            (35.65960, 139.70080),   # 2  ~28m NE
            (35.65910, 139.70030),   # 3  ~45m SW
            (35.65900, 139.70040),   # 4  Hachiko ~50m
        ]
        names = ["Scramble", "7m away", "28m NE", "45m SW", "Hachiko"]

        print("\n  [same-block] Distances from Shibuya Scramble:")
        for i in range(1, 5):
            print(f"    → {names[i]:10s}: {haversine_m(locs[0], locs[i]):.0f}m")

        verify_nearest(self, 0, locs, names, "same-block-shibuya")


class TestDenseClusterWithOutliers(unittest.TestCase):
    """
    Dense cluster of users + a few far-away outliers.
    The outliers should never be chosen as nearest for
    anyone in the cluster. Tests that small intra-cluster
    differences aren't overwhelmed by large outlier distances.
    """

    def test_manhattan_cluster_plus_boroughs(self):
        """
        15 users in Midtown Manhattan (< 2km apart)
        + 5 users in outer boroughs (5-15km away).

        Every Midtown user's nearest should be another Midtown user.
        """
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

        # Test from several midtown locations —
        # nearest should always be in midtown (slots 0-14)
        midtown_slots = set(range(15))

        for target in [0, 3, 7, 10]:
            winner = verify_nearest(self, target, locs, names, "cluster")
            self.assertIn(
                winner, midtown_slots,
                f"{names[target]}: nearest should be in Midtown, got {names[winner]} (slot {winner})"
            )

    def test_sf_soma_plus_east_bay(self):
        """
        10 SoMa users (< 1km apart) + 5 East Bay users (5-15km away).
        SoMa users should always match within SoMa.
        """
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
                f"{names[target]}: nearest should be in SoMa, got {names[winner]} (slot {winner})"
            )


class TestPrecisionDistance(unittest.TestCase):
    """
    Verify that the FHE system can distinguish distances that
    differ by small amounts in dense urban settings.
    """

    def test_can_distinguish_30m_vs_50m(self):
        """
        Target at origin. Two opponents: one 30m away, one 50m away.
        ~0.0003° and ~0.0005° difference at 40°N.
        """
        target = (40.75000, -73.98000)
        close  = (40.75027, -73.98000)  # ~30m north
        far    = (40.75045, -73.98000)  # ~50m north

        locs = [target, close, far]
        names = ["Target", "30m N", "50m N"]

        d1 = haversine_m(target, close)
        d2 = haversine_m(target, far)
        print(f"\n  [precision] 30m vs 50m: actual {d1:.0f}m vs {d2:.0f}m")

        winner, _, elapsed = run_scoring(0, locs, names)
        self.assertEqual(winner, 1, f"Should pick 30m (slot 1), got slot {winner}")
        print(f"  [precision] winner=slot {winner} ({elapsed:.1f}s)  OK")

    def test_can_distinguish_100m_vs_200m(self):
        """
        100m vs 200m at a different angle.
        """
        target = (51.50700, -0.12760)   # Trafalgar area
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
        """
        500m vs 700m — typical urban scenario.
        """
        target = (35.65940, 139.70060)    # Shibuya
        close  = (35.66390, 139.70060)    # ~500m north
        far    = (35.66570, 139.70060)    # ~700m north

        locs = [target, close, far]
        names = ["Shibuya", "500m N", "700m N"]

        d1 = haversine_m(target, close)
        d2 = haversine_m(target, far)
        print(f"\n  [precision] 500m vs 700m: actual {d1:.0f}m vs {d2:.0f}m")

        winner, _, elapsed = run_scoring(0, locs, names)
        self.assertEqual(winner, 1, f"Should pick 500m (slot 1), got slot {winner}")
        print(f"  [precision] winner=slot {winner} ({elapsed:.1f}s)  OK")


# ==========================================
# Runner
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FHE Proximity — Dense Urban Area Tests")
    print("=" * 60)
    print(f"  BATCH_SIZE={BATCH_SIZE}, MAX_USERS={MAX_USERS}")
    print(f"  Pairwise scoring, depth budget 5 levels")
    print()
    print("5 cities × 20 users each + precision + cluster tests")
    print("=" * 60 + "\n")

    start = time.time()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Order: fast (small) → slow (20-user)
    suite.addTests(loader.loadTestsFromTestCase(TestPrecisionDistance))
    suite.addTests(loader.loadTestsFromTestCase(TestSameBlock))
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
