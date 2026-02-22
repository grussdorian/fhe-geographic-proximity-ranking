#!/usr/bin/env python3
"""
End-to-end client-server tests — real-world proximity scenarios.

Each test:
  1. Starts a fresh server subprocess
  2. Runs the full threshold FHE protocol over HTTP
  3. Validates the nearest-neighbor result
  4. Kills the server

Prerequisites:
  - setup_keys.py has been run (cryptocontext.bin exists)
  - PYTHONPATH includes the openfhe-python build directory

Run:
  PYTHONPATH="..." python3.13 test_e2e_client_server.py
"""

import sys, os, time, signal, subprocess, math, requests

sys.path.insert(0, ".")
import client as C

PYTHON = sys.executable
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__) or ".", "server.py")
SERVER_PORT = 8000
SERVER_URL = f"http://localhost:{SERVER_PORT}"

# ============================================================
# Test infrastructure
# ============================================================

def kill_port(port):
    """Kill any process listening on the given port."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        for pid in out.split("\n"):
            pid = pid.strip()
            if pid:
                os.kill(int(pid), signal.SIGKILL)
        time.sleep(0.5)
    except (subprocess.CalledProcessError, ValueError):
        pass  # nothing on port


def reset_client_context():
    """Reload a fresh crypto context in the client module so eval mult keys
    from a previous test don't pollute the next one."""
    C.cc = C.load_context()


def start_server():
    """Start a fresh server subprocess. Returns the Popen object."""
    # Make sure port is free first
    kill_port(SERVER_PORT)

    env = os.environ.copy()
    proc = subprocess.Popen(
        [PYTHON, SERVER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for server to be ready
    for _ in range(30):
        try:
            r = requests.get(f"{SERVER_URL}/status", timeout=1)
            if r.status_code == 200:
                data = r.json()
                # Verify this is a FRESH server (no parties joined yet)
                if data.get("parties_joined", 0) == 0:
                    return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Server failed to start within 15 seconds")


def stop_server(proc):
    """Kill the server subprocess and make sure the port is free."""
    proc.kill()
    proc.wait()
    kill_port(SERVER_PORT)
    time.sleep(0.5)


def run_scenario(n_parties, locations, names, expected_nearest_name, description):
    """
    Run a full threshold FHE proximity match scenario.

    Args:
        n_parties:   Number of parties (first is the initiator/lead)
        locations:   List of (lat, lon) tuples — first is the initiator
        names:       List of display names
        expected_nearest_name: The name we expect to be the nearest to the initiator
        description: Human-readable scenario description

    Returns:
        (passed: bool, actual_winner: str, score_vector: list, active_slots: list)
    """
    assert len(locations) == n_parties == len(names)

    # Reset client crypto context so eval mult keys from prior tests don't leak
    reset_client_context()

    party_ids = [f"party_{names[i].lower()}_{i}" for i in range(n_parties)]

    # ====== Phase 1: Threshold key generation ======
    keypairs = []
    slots = []
    lead_eval_share = None

    for i in range(n_parties):
        info = C.join_server(party_ids[i], names[i])
        slots.append(info["slot"])
        is_lead = info["is_lead"]
        kp = C.generate_keys(is_lead, info["prev_public_key"])
        keypairs.append(kp)

        eval_share = C.generate_eval_mult_share(
            kp, is_lead=is_lead, prev_eval_mult_key=lead_eval_share
        )
        if is_lead:
            lead_eval_share = eval_share
        C.submit_key_share(party_ids[i], kp, eval_share)

    fin_resp = C.finalize_keys()
    combined_b64 = fin_resp["combined_eval_key"]
    joint_pk_tag = fin_resp["joint_pk_tag"]

    for i in range(n_parties):
        C.submit_mult_eval_key(party_ids[i], keypairs[i], combined_b64, joint_pk_tag)

    C.fetch_and_install_eval_mult_key()
    joint_pk = keypairs[-1].publicKey

    # ====== Phase 2: Proximity match ======
    # Initiator starts match
    C.start_match(party_ids[0], locations[0][0], locations[0][1], joint_pk)

    # Other parties compute & submit distances
    for i in range(1, n_parties):
        resp = requests.post(f"{SERVER_URL}/get_match", json={"party_id": party_ids[i]})
        match_data = resp.json()

        enc_lat = C.deserialize_ciphertext(match_data["enc_lat"])
        enc_lon = C.deserialize_ciphertext(match_data["enc_lon"])

        dist_ct = C.compute_distance_local(
            enc_lat, enc_lon,
            locations[i][0], locations[i][1],
            joint_pk,
        )
        id_ct = C.encrypt_onehot_id(slots[i], joint_pk)
        C.submit_distance(party_ids[i], dist_ct, id_ct)

    # Server computes pairwise scoring
    resp = requests.post(f"{SERVER_URL}/compute_nearest", json={})
    assert resp.status_code == 200, f"compute_nearest failed: {resp.text}"

    # All parties do partial decryption
    for i in range(n_parties):
        C.do_partial_decrypt(party_ids[i], keypairs[i], is_lead=(i == 0))

    # Fuse & resolve
    values = C.get_decrypted_result(party_ids[0])
    initiator_slot = slots[0]
    active = [s for s in range(n_parties) if s != initiator_slot]

    winner_slot = max(active, key=lambda s: values[s])
    winner_name = C.resolve_slot(winner_slot)

    passed = (winner_name == expected_nearest_name)
    return passed, winner_name, values, active


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km (for display/verification only)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# Test scenarios
# ============================================================

SCENARIOS = []

def scenario(name, n, locations, names, expected, description):
    SCENARIOS.append({
        "name": name,
        "n": n,
        "locations": locations,
        "names": names,
        "expected": expected,
        "description": description,
    })


# -----------------------------------------------------------
# 1. SF Bay Area commuters — classic 3 parties
# -----------------------------------------------------------
scenario(
    "Bay Area commuters",
    3,
    [
        (37.7749, -122.4194),   # Alice: San Francisco
        (37.4961, -122.2471),   # Bob: Palo Alto (~31km)
        (37.3382, -121.8863),   # Charlie: San Jose (~54km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Alice queries who's nearest. Bob (Palo Alto) is closer than Charlie (San Jose).",
)

# -----------------------------------------------------------
# 2. Manhattan coffee meetup — very close distances
# -----------------------------------------------------------
scenario(
    "Manhattan coffee meetup",
    3,
    [
        (40.7580, -73.9855),    # Alice: Times Square
        (40.7614, -73.9776),    # Bob: Rockefeller Center (~0.7km)
        (40.7484, -73.9857),    # Charlie: Empire State Building (~1.1km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Dense urban: Alice at Times Square, Bob at Rockefeller (closest), "
    "Charlie at Empire State.",
)

# -----------------------------------------------------------
# 3. London pub crawl — 4 parties scattered across central London
# -----------------------------------------------------------
scenario(
    "London pub crawl",
    4,
    [
        (51.5074, -0.1278),     # Alice: Trafalgar Square
        (51.5115, -0.1197),     # Bob: Covent Garden (~0.7km)
        (51.5014, -0.1419),     # Charlie: Westminster (~1.1km)
        (51.5155, -0.0922),     # Dave: St Paul's Cathedral (~2.6km)
    ],
    ["Alice", "Bob", "Charlie", "Dave"],
    "Bob",
    "4 friends in central London. Covent Garden is the closest to Trafalgar Square.",
)

# -----------------------------------------------------------
# 4. Tokyo neighborhoods — 4 parties, very tight spacing
# -----------------------------------------------------------
scenario(
    "Tokyo neighborhoods",
    4,
    [
        (35.6762, 139.6503),    # Alice: Shibuya Crossing
        (35.6595, 139.7004),    # Bob: Roppongi (~5km)
        (35.6938, 139.7034),    # Charlie: Shinjuku (~5.2km)
        (35.6684, 139.6833),    # Dave: Ebisu (~3.1km)
    ],
    ["Alice", "Bob", "Charlie", "Dave"],
    "Dave",
    "Dense Tokyo: Alice at Shibuya, Dave (Ebisu) is nearest.",
)

# -----------------------------------------------------------
# 5. College campus — 3 parties, extremely close (~100-300m)
# -----------------------------------------------------------
scenario(
    "College campus",
    3,
    [
        (37.4275, -122.1697),   # Alice: Stanford Main Quad
        (37.4267, -122.1672),   # Bob: Green Library (~220m)
        (37.4300, -122.1740),   # Charlie: Hoover Tower (~490m)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Ultra-close campus scenario. Bob at Green Library (~220m) vs "
    "Charlie at Hoover Tower (~490m).",
)

# -----------------------------------------------------------
# 6. Cross-country road trip — very spread out (hundreds of km)
# -----------------------------------------------------------
scenario(
    "Cross-country road trip",
    3,
    [
        (39.7392, -104.9903),   # Alice: Denver
        (38.6270, -90.1994),    # Bob: St Louis (~1300km)
        (41.8781, -87.6298),    # Charlie: Chicago (~1500km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Long-distance: Denver to St. Louis is closer than Denver to Chicago.",
)

# -----------------------------------------------------------
# 7. Sydney harbour walk — Southern hemisphere
# -----------------------------------------------------------
scenario(
    "Sydney harbour walk",
    3,
    [
        (-33.8568, 151.2153),   # Alice: Opera House
        (-33.8523, 151.2108),   # Bob: Circular Quay (~600m)
        (-33.8688, 151.2093),   # Charlie: Darling Harbour (~1.4km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Southern hemisphere near-zero: Opera House -> Circular Quay (closest).",
)

# -----------------------------------------------------------
# 8. Paris landmarks — 5 parties
# -----------------------------------------------------------
scenario(
    "Paris landmarks",
    5,
    [
        (48.8584, 2.2945),      # Alice: Eiffel Tower
        (48.8606, 2.3376),      # Bob: Louvre (~3.1km)
        (48.8530, 2.3499),      # Charlie: Notre-Dame (~4.0km)
        (48.8738, 2.2950),      # Dave: Arc de Triomphe (~1.7km)
        (48.8462, 2.3464),      # Eve: Jardin du Luxembourg (~3.8km)
    ],
    ["Alice", "Bob", "Charlie", "Dave", "Eve"],
    "Dave",
    "5-party Paris: Arc de Triomphe (~1.7km) is closest to the Eiffel Tower.",
)

# -----------------------------------------------------------
# 9. Mumbai rush hour — dense Indian metro, 4 parties
# -----------------------------------------------------------
scenario(
    "Mumbai rush hour",
    4,
    [
        (19.0760, 72.8777),     # Alice: Chhatrapati Shivaji Terminus
        (19.0896, 72.8656),     # Bob: Marine Lines (~1.8km)
        (19.0596, 72.8295),     # Charlie: Haji Ali (~5.3km)
        (19.0825, 72.8812),     # Dave: Fort area (~0.8km)
    ],
    ["Alice", "Bob", "Charlie", "Dave"],
    "Dave",
    "Mumbai: Dave at Fort area is closest to CST.",
)

# -----------------------------------------------------------
# 10. Arctic edge case — very high latitude (near pole)
#     NOTE: Our system uses normalized (lat/180, lon/180) Euclidean
#     distance, NOT haversine. At high latitudes, longitude degrees
#     are physically short but carry equal weight in our metric.
#     So we separate parties mainly by latitude to stay consistent.
# -----------------------------------------------------------
scenario(
    "Arctic research stations",
    3,
    [
        (78.2232, 15.6267),     # Alice: Longyearbyen, Svalbard
        (78.9000, 15.8000),     # Bob: north of Longyearbyen (~75km)
        (76.5000, 15.5000),     # Charlie: south of Svalbard (~192km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Extreme high latitude. Bob (~75km north) is closer than Charlie (~192km south).",
)

# -----------------------------------------------------------
# 11. Equatorial meetup — near lat=0
# -----------------------------------------------------------
scenario(
    "Equatorial meetup",
    3,
    [
        (-0.1807, 36.8070),     # Alice: Nanyuki, Kenya (equator town)
        (0.0236, 37.0662),      # Bob: Meru (~32km)
        (-1.2921, 36.8219),     # Charlie: Nairobi (~123km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "Straddling the equator in Kenya. Meru is much closer than Nairobi.",
)

# -----------------------------------------------------------
# 12. Ride-share pickup — who's the closest driver?
# -----------------------------------------------------------
scenario(
    "Ride-share pickup",
    4,
    [
        (40.7128, -74.0060),    # Alice (rider): Lower Manhattan
        (40.7282, -73.7949),    # Bob: JFK area (~19km)
        (40.7178, -74.0126),    # Charlie: Tribeca (~0.8km) — closest
        (40.6892, -74.0445),    # Dave: Statue of Liberty ferry (~4.4km)
    ],
    ["Alice", "Bob", "Charlie", "Dave"],
    "Charlie",
    "Ride-share: Alice requests, Charlie (Tribeca) is the closest driver.",
)

# -----------------------------------------------------------
# 13. Emergency services — hospital proximity
# -----------------------------------------------------------
scenario(
    "Hospital proximity",
    4,
    [
        (34.0522, -118.2437),   # Patient: Downtown LA
        (34.0635, -118.2577),   # Hospital A: Good Samaritan (~1.7km)
        (34.0733, -118.3786),   # Hospital B: Cedars-Sinai (~11km)
        (34.0459, -118.2590),   # Hospital C: LA Convention area (~1.4km)
    ],
    ["Patient", "GoodSam", "CedarsSinai", "LAConvention"],
    "LAConvention",
    "Emergency: find closest hospital to downtown LA patient.",
)

# -----------------------------------------------------------
# 14. Food delivery — closest restaurant
# -----------------------------------------------------------
scenario(
    "Food delivery",
    5,
    [
        (37.7749, -122.4194),   # Customer: SF
        (37.7870, -122.4089),   # Restaurant A: Chinatown (~1.6km)
        (37.7599, -122.4148),   # Restaurant B: Mission District (~1.7km)
        (37.7790, -122.4174),   # Restaurant C: Nob Hill (~0.5km) — closest
        (37.8044, -122.2712),   # Restaurant D: Oakland (~13km)
    ],
    ["Customer", "Chinatown", "Mission", "NobHill", "Oakland"],
    "NobHill",
    "Delivery: 4 restaurants, Nob Hill (~0.5km) is nearest to the customer in SF.",
)

# -----------------------------------------------------------
# 15. New Zealand — large eastern longitudes
# -----------------------------------------------------------
scenario(
    "New Zealand meetup",
    3,
    [
        (-36.8485, 174.7633),   # Alice: Auckland
        (-37.7870, 175.2793),   # Bob: Hamilton (~116km)
        (-41.2865, 174.7762),   # Charlie: Wellington (~493km)
    ],
    ["Alice", "Bob", "Charlie"],
    "Bob",
    "New Zealand longitudes near 175 deg. Hamilton is closer to Auckland than Wellington.",
)


# ============================================================
# Test runner
# ============================================================

def run_all():
    total = len(SCENARIOS)
    passed_count = 0
    failed_tests = []

    print("=" * 65)
    print("  THRESHOLD FHE END-TO-END TESTS — REAL-WORLD SCENARIOS")
    print(f"  {total} scenarios | server restarted per test")
    print("=" * 65)
    print()

    for idx, s in enumerate(SCENARIOS):
        test_num = idx + 1
        print(f"[{test_num}/{total}] {s['name']}")
        print(f"  {s['description']}")
        print(f"  Parties: {s['n']}  Expected nearest: {s['expected']}")

        # Print distances for reference
        init_lat, init_lon = s["locations"][0]
        for i in range(1, s["n"]):
            lat2, lon2 = s["locations"][i]
            d = haversine_km(init_lat, init_lon, lat2, lon2)
            print(f"    {s['names'][i]:15s}: {d:8.1f} km")

        server = start_server()
        try:
            t0 = time.time()
            ok, actual, values, active = run_scenario(
                s["n"], s["locations"], s["names"], s["expected"], s["description"]
            )
            elapsed = time.time() - t0

            # Print score vector
            print(f"  Scores: ", end="")
            for si in active:
                print(f"{s['names'][si]}={values[si]:.3f}  ", end="")
            print()

            if ok:
                print(f"  PASS  ({elapsed:.1f}s)  nearest='{actual}'")
                passed_count += 1
            else:
                print(f"  FAIL  ({elapsed:.1f}s)  expected='{s['expected']}', got='{actual}'")
                failed_tests.append(s["name"])
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_tests.append(s["name"])
        finally:
            stop_server(server)

        print()

    # ---- Summary ----
    print("=" * 65)
    print(f"  RESULTS: {passed_count}/{total} passed")
    if failed_tests:
        print(f"  FAILED:")
        for name in failed_tests:
            print(f"    - {name}")
    else:
        print(f"  ALL TESTS PASSED")
    print("=" * 65)

    return len(failed_tests) == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
