# ==========================================
# client.py — Threshold FHE Proximity Client
#
# Two roles:
#   --lead    : First party — calls cc.KeyGen(), initiates matches
#   --join    : Other parties — calls cc.MultipartyKeyGen(prev_pk)
#
# Protocol:
#   Phase 1 — Key generation (all parties):
#     1. POST /join → get slot, order, previous public key
#     2. Generate key pair (lead: KeyGen, others: MultipartyKeyGen)
#     3. Generate eval mult key share (KeySwitchGen / MultiKeySwitchGen)
#     4. POST /submit_key_share → send public key + eval mult share
#     5. POST /finalize_keys (any party, once all joined)
#     6. GET combined eval key, compute MultiMultEvalKey share
#     7. POST /submit_mult_eval_key → send round-2 share
#
#   Phase 2 — Match (per query):
#     Lead/initiator:
#       8. Encrypts own coords → POST /start_match
#       9. Waits for all parties to submit distances
#      10. POST /compute_nearest → server scores
#      11. POST /get_result → gets encrypted result
#      12. MultipartyDecryptLead → POST /submit_partial_decrypt
#      13. POST /get_decrypted → fused plaintext → argmax → /resolve
#
#     Other parties:
#       8. POST /get_match → get initiator's encrypted coords
#       9. Compute distance LOCALLY (using own plaintext coords + enc coords)
#      10. POST /submit_distance → send enc(dist²) + enc(one_hot_id)
#      11. POST /get_result → gets encrypted result
#      12. MultipartyDecryptMain → POST /submit_partial_decrypt
#
# Security:
#   - Each party holds only their secret key share
#   - No single party can decrypt (need ALL shares)
#   - Server never sees plaintext coords or distances
#   - Even initiator+server collusion cannot decrypt
# ==========================================

import requests
import uuid
import time
import base64
import tempfile
import os
import sys
import json
from openfhe import *

SERVER_URL = "http://localhost:8000"
KEYS_DIR = "fhe_keys"
BATCH_SIZE = 32
MAX_COORD = 180.0

# -----------------------------
# Load crypto context (no keys — we generate them)
# -----------------------------

def load_context():
    cc, success = DeserializeCryptoContext(f"{KEYS_DIR}/cryptocontext.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load crypto context. Run setup_keys.py first!")
    return cc


cc = load_context()
print("Loaded crypto context (threshold mode)")


# -----------------------------
# Serialization helpers
# -----------------------------

def serialize_ciphertext(ct):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        fname = f.name
    SerializeToFile(fname, ct, BINARY)
    with open(fname, 'rb') as f:
        data = f.read()
    os.unlink(fname)
    return base64.b64encode(data).decode()


def deserialize_ciphertext(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        fname = f.name
        f.write(base64.b64decode(data))
    ct, success = DeserializeCiphertext(fname, BINARY)
    os.unlink(fname)
    if not success:
        raise ValueError("Failed to deserialize ciphertext")
    return ct


def serialize_key(key):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
    SerializeToFile(fname, key, BINARY)
    with open(fname, 'rb') as f:
        data = f.read()
    os.unlink(fname)
    return base64.b64encode(data).decode()


def deserialize_public_key(data):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
        f.write(base64.b64decode(data))
    key, success = DeserializePublicKey(fname, BINARY)
    os.unlink(fname)
    if not success:
        raise ValueError("Failed to deserialize public key")
    return key


def deserialize_eval_key(data):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
        f.write(base64.b64decode(data))
    key, success = DeserializeEvalKey(fname, BINARY)
    os.unlink(fname)
    if not success:
        raise ValueError("Failed to deserialize eval key")
    return key


# =============================================
# Phase 1: Key Generation
# =============================================

def join_server(party_id, actual_user):
    """Join the threshold key generation protocol."""
    resp = requests.post(f"{SERVER_URL}/join", json={
        "party_id": party_id,
        "actual_user": actual_user,
    })
    if resp.status_code != 200:
        raise RuntimeError(f"Join failed: {resp.json()}")
    return resp.json()


def generate_keys(is_lead, prev_public_key_b64=None):
    """Generate this party's key pair.
    Lead: cc.KeyGen()
    Others: cc.MultipartyKeyGen(prev_public_key)
    """
    if is_lead:
        kp = cc.KeyGen()
    else:
        prev_pk = deserialize_public_key(prev_public_key_b64)
        kp = cc.MultipartyKeyGen(prev_pk)
    return kp


def generate_eval_mult_share(kp, is_lead, prev_eval_mult_key=None):
    """Generate eval mult key share.
    Lead: cc.KeySwitchGen(sk, sk)
    Others: cc.MultiKeySwitchGen(sk, sk, prev_eval_mult_key)
    """
    if is_lead:
        return cc.KeySwitchGen(kp.secretKey, kp.secretKey)
    else:
        return cc.MultiKeySwitchGen(kp.secretKey, kp.secretKey, prev_eval_mult_key)


def submit_key_share(party_id, kp, eval_mult_share):
    """Send public key and eval mult key share to server."""
    resp = requests.post(f"{SERVER_URL}/submit_key_share", json={
        "party_id": party_id,
        "public_key": serialize_key(kp.publicKey),
        "eval_mult_share": serialize_key(eval_mult_share),
    })
    if resp.status_code != 200:
        raise RuntimeError(f"submit_key_share failed: {resp.json()}")
    return resp.json()


def finalize_keys():
    """Tell server to combine eval mult key shares (round 1)."""
    resp = requests.post(f"{SERVER_URL}/finalize_keys", json={})
    if resp.status_code != 200:
        raise RuntimeError(f"finalize_keys failed: {resp.json()}")
    return resp.json()


def submit_mult_eval_key(party_id, kp, combined_eval_key_b64, joint_pk_tag):
    """Round 2: compute s_i * combined_eval_key, send to server."""
    combined = deserialize_eval_key(combined_eval_key_b64)
    mult_share = cc.MultiMultEvalKey(kp.secretKey, combined, joint_pk_tag)
    resp = requests.post(f"{SERVER_URL}/submit_mult_eval_key", json={
        "party_id": party_id,
        "mult_eval_share": serialize_key(mult_share),
    })
    if resp.status_code != 200:
        raise RuntimeError(f"submit_mult_eval_key failed: {resp.json()}")
    return resp.json()


# =============================================
# Phase 2: Match Protocol
# =============================================

def fetch_and_install_eval_mult_key():
    """Fetch the finalized eval mult key from server and install locally.
    Required before any EvalMult operations (e.g., computing distance²).
    """
    resp = requests.get(f"{SERVER_URL}/eval_mult_key")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch eval mult key: {resp.text}")
    key_b64 = resp.json()["eval_mult_key"]
    key = deserialize_eval_key(key_b64)
    cc.InsertEvalMultKey([key])


def fetch_joint_public_key():
    """Fetch the joint public key from server.
    In threshold FHE, all encryption must use the JOINT public key
    (the last party's pk in the chain), not any individual party's pk.
    """
    resp = requests.get(f"{SERVER_URL}/joint_public_key")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch joint public key: {resp.text}")
    return deserialize_public_key(resp.json()["joint_public_key"])


def encrypt_location(lat, lon, public_key):
    """Encrypt coords replicated across all slots. Uses joint public key."""
    nlat = float(lat) / MAX_COORD
    nlon = float(lon) / MAX_COORD
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    lat_ct = cc.Encrypt(public_key, lat_pt)
    lon_ct = cc.Encrypt(public_key, lon_pt)
    return lat_ct, lon_ct


def encrypt_onehot_id(slot_index, public_key):
    """Encrypt a one-hot vector with 1.0 at the given slot."""
    vec = [0.0] * BATCH_SIZE
    vec[slot_index] = 1.0
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(public_key, pt)


def compute_distance_local(enc_lat, enc_lon, my_lat, my_lon, public_key):
    """Compute enc(distance²) using initiator's encrypted coords and
    this party's plaintext coords. Depth cost: 1 level.
    
    Key insight: the party uses their own PLAINTEXT location as a
    plaintext vector, and subtracts it from the encrypted location.
    Then squares the differences. The party's location NEVER leaves
    their device.
    """
    nlat = float(my_lat) / MAX_COORD
    nlon = float(my_lon) / MAX_COORD

    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)

    dlat = cc.EvalSub(enc_lat, lat_pt)   # enc(init_lat - my_lat)
    dlon = cc.EvalSub(enc_lon, lon_pt)   # enc(init_lon - my_lon)

    dlat2 = cc.EvalMult(dlat, dlat)      # enc(dlat²) — depth +1
    dlon2 = cc.EvalMult(dlon, dlon)      # enc(dlon²) — depth +1 (parallel)

    return cc.EvalAdd(dlat2, dlon2)       # enc(dlat² + dlon²)


def start_match(party_id, lat, lon, public_key):
    """Initiator: encrypt coords and send to server."""
    lat_ct, lon_ct = encrypt_location(lat, lon, public_key)

    resp = requests.post(f"{SERVER_URL}/start_match", json={
        "party_id": party_id,
        "enc_lat": serialize_ciphertext(lat_ct),
        "enc_lon": serialize_ciphertext(lon_ct),
    })
    if resp.status_code != 200:
        raise RuntimeError(f"start_match failed: {resp.json()}")
    return resp.json()


def submit_distance(party_id, dist_ct, id_ct):
    """Non-initiator: submit encrypted distance + one-hot ID."""
    resp = requests.post(f"{SERVER_URL}/submit_distance", json={
        "party_id": party_id,
        "dist_ct": serialize_ciphertext(dist_ct),
        "id_ct": serialize_ciphertext(id_ct),
    })
    if resp.status_code != 200:
        raise RuntimeError(f"submit_distance failed: {resp.json()}")
    return resp.json()


def do_partial_decrypt(party_id, kp, is_lead):
    """Get encrypted result from server, produce partial decryption."""
    resp = requests.post(f"{SERVER_URL}/get_result", json={"party_id": party_id})
    if resp.status_code != 200:
        raise RuntimeError(f"get_result failed: {resp.json()}")

    result_ct = deserialize_ciphertext(resp.json()["encrypted_result"])

    if is_lead:
        partial = cc.MultipartyDecryptLead([result_ct], kp.secretKey)
    else:
        partial = cc.MultipartyDecryptMain([result_ct], kp.secretKey)

    # partial is a list, we want the first (and only) element
    partial_ct = partial[0]

    # Serialize and submit
    partial_b64 = serialize_ciphertext(partial_ct)
    resp2 = requests.post(f"{SERVER_URL}/submit_partial_decrypt", json={
        "party_id": party_id,
        "partial_ct": partial_b64,
    })
    if resp2.status_code != 200:
        raise RuntimeError(f"submit_partial_decrypt failed: {resp2.json()}")
    return resp2.json()


def get_decrypted_result(party_id):
    """Initiator: get fused plaintext result."""
    resp = requests.post(f"{SERVER_URL}/get_decrypted", json={
        "party_id": party_id,
    })
    if resp.status_code != 200:
        raise RuntimeError(f"get_decrypted failed: {resp.json()}")
    return resp.json()["values"]


def resolve_slot(slot):
    """Map winning slot to actual user identity."""
    resp = requests.post(f"{SERVER_URL}/resolve", json={"slot": slot})
    if resp.status_code != 200:
        raise RuntimeError(f"resolve failed: {resp.json()}")
    return resp.json()["actual_user"]


# =============================================
# Library API for programmatic use (demo.py)
# =============================================

def create_party(party_id, actual_user, lat, lon):
    """Create a party dict for use by the demo orchestrator."""
    return {
        "party_id": party_id,
        "actual_user": actual_user,
        "lat": lat,
        "lon": lon,
        "slot": None,
        "kp": None,
        "is_lead": False,
        "order": None,
    }


# =============================================
# Helpers
# =============================================

def poll_server_status():
    """Get current server state."""
    resp = requests.get(f"{SERVER_URL}/status")
    if resp.status_code != 200:
        raise RuntimeError(f"status failed: {resp.text}")
    return resp.json()


def wait_for_parties(expected_count, poll_interval=2.0):
    """Poll server until expected number of parties have joined."""
    while True:
        status = poll_server_status()
        joined = status["parties_with_keys"]
        print(f"\r  Waiting for parties... {joined}/{expected_count} key shares received", end="", flush=True)
        if joined >= expected_count:
            print()
            return status
        time.sleep(poll_interval)


def wait_for_match(party_id, poll_interval=2.0):
    """Poll /get_match until a match is available."""
    while True:
        try:
            resp = requests.post(f"{SERVER_URL}/get_match", json={"party_id": party_id})
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        print(f"\r  Waiting for match to start...", end="", flush=True)
        time.sleep(poll_interval)


def wait_for_distances(expected, poll_interval=2.0):
    """Poll server until all distances are submitted."""
    while True:
        status = poll_server_status()
        received = status["match_distances_received"]
        print(f"\r  Waiting for distances... {received}/{expected}", end="", flush=True)
        if received >= expected:
            print()
            return
        time.sleep(poll_interval)


def wait_for_result(poll_interval=2.0):
    """Poll /get_result until the encrypted result is available."""
    while True:
        try:
            resp = requests.post(f"{SERVER_URL}/get_result", json={})
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        time.sleep(poll_interval)


def wait_for_partials(expected, poll_interval=2.0):
    """Poll server until all partial decryptions are in."""
    while True:
        status = poll_server_status()
        received = status["match_partials_received"]
        print(f"\r  Waiting for partial decryptions... {received}/{expected}", end="", flush=True)
        if received >= expected:
            print()
            return
        time.sleep(poll_interval)


# =============================================
# Main — Interactive Threshold FHE Client
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Threshold FHE Proximity Client (interactive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lead party with specific location (Times Square)
  python3.13 client.py --lead --lat 40.7580 --lon -73.9855 --name Alice --parties 3

  # Joining party with specific location (Central Park)
  python3.13 client.py --join --lat 40.7829 --lon -73.9654 --name Bob

  # Joining party with random US location
  python3.13 client.py --join --name Charlie
        """,
    )
    parser.add_argument("--lead", action="store_true", help="Be the lead (first) party")
    parser.add_argument("--join", action="store_true", help="Join as a non-lead party")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Longitude")
    parser.add_argument("--name", type=str, default=None, help="Display name for this party")
    parser.add_argument("--parties", type=int, default=None,
                        help="Expected number of parties (lead only — used to know when to finalize)")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    if not args.lead and not args.join:
        parser.error("Must specify --lead or --join")

    SERVER_URL = args.server

    # Generate identity
    import random
    actual_user = args.name or f"user_{uuid.uuid4().hex[:6]}"
    party_id = f"party_{uuid.uuid4().hex[:8]}"

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        # Random US location
        lat = round(random.uniform(25.0, 48.0), 5)
        lon = round(random.uniform(-125.0, -70.0), 5)

    is_lead = args.lead

    print("=" * 55)
    print("  THRESHOLD FHE PROXIMITY CLIENT")
    print("=" * 55)
    print(f"  Role:     {'LEAD (initiator)' if is_lead else 'JOIN (responder)'}")
    print(f"  Name:     {actual_user}")
    print(f"  Party ID: {party_id}")
    print(f"  Location: ({lat:.5f}, {lon:.5f})")
    print(f"  Server:   {SERVER_URL}")
    print(f"  Coords stay LOCAL — never sent to server")
    print("=" * 55)
    print()

    # ================================================================
    # PHASE 1: Threshold Key Generation
    # ================================================================

    print("[Phase 1] Threshold key generation")
    print("-" * 40)

    # Step 1: Join the server
    print("  Joining server...")
    join_info = join_server(party_id, actual_user)
    slot = join_info["slot"]
    order = join_info["order"]
    is_lead = join_info["is_lead"]
    prev_pk_b64 = join_info["prev_public_key"]
    print(f"  Joined: slot={slot}, order={order}, is_lead={is_lead}")

    # Step 2: Generate key pair
    print("  Generating key pair...")
    kp = generate_keys(is_lead, prev_pk_b64)
    print(f"  Key pair generated ({'KeyGen' if is_lead else 'MultipartyKeyGen'})")

    # Step 3: Generate eval mult key share
    # Lead: KeySwitchGen(sk, sk)
    # Others: MultiKeySwitchGen(sk, sk, lead_share) — must fetch lead's share
    print("  Generating eval mult key share...")
    if is_lead:
        eval_share = generate_eval_mult_share(kp, is_lead=True)
        print("  Eval mult key share ready (KeySwitchGen)")
    else:
        # Wait for lead's share to be available, then fetch it
        print("  Waiting for lead's eval share...")
        while True:
            try:
                resp = requests.get(f"{SERVER_URL}/lead_eval_share")
                if resp.status_code == 200:
                    lead_share_b64 = resp.json()["lead_eval_share"]
                    break
            except Exception:
                pass
            time.sleep(1.0)
        lead_share = deserialize_eval_key(lead_share_b64)
        eval_share = generate_eval_mult_share(kp, is_lead=False, prev_eval_mult_key=lead_share)
        print("  Eval mult key share ready (MultiKeySwitchGen)")

    # Step 4: Submit key share
    print("  Submitting key share to server...")
    submit_key_share(party_id, kp, eval_share)
    print("  Key share submitted")
    print()

    # Step 5: Wait for all parties to join and submit keys
    if is_lead:
        expected_parties = args.parties
        if expected_parties is None:
            expected_parties = int(input("  How many total parties? "))

        print(f"  Waiting for {expected_parties} parties to submit key shares...")
        wait_for_parties(expected_parties)

        # Step 6: Finalize keys (round 1)
        print("  Finalizing keys (round 1: combine eval mult shares)...")
        finalize_resp = finalize_keys()
        combined_eval_key_b64 = finalize_resp["combined_eval_key"]
        print(f"  Round 1 complete ({finalize_resp['parties']} parties)")

        # Step 7: Round 2 — compute s_i * combined_eval_key
        # MUST use the joint public key tag (not our own pk tag)
        print("  Computing round 2 eval mult key share...")
        joint_pk_tag = finalize_resp["joint_pk_tag"]
        submit_mult_eval_key(party_id, kp, combined_eval_key_b64, joint_pk_tag)
        print("  Round 2 share submitted")

        # Wait for all round 2 shares
        print("  Waiting for all parties to complete round 2...")
        while True:
            try:
                resp = requests.get(f"{SERVER_URL}/eval_mult_key")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        print("  All round 2 shares complete")
    else:
        # Non-lead: wait for finalization, then do round 2
        print("  Waiting for lead to finalize keys...")
        while True:
            status = poll_server_status()
            if status["keys_finalized"]:
                break
            time.sleep(2.0)
        print("  Keys finalized by lead")

        print("  Fetching combined eval key for round 2...")
        resp = requests.get(f"{SERVER_URL}/combined_eval_key")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get combined eval key: {resp.text}")
        resp_json = resp.json()
        combined_eval_key_b64 = resp_json["combined_eval_key"]

        # MUST use the joint public key tag (not our own pk tag)
        print("  Computing round 2 eval mult key share...")
        joint_pk_tag = resp_json["joint_pk_tag"]
        submit_mult_eval_key(party_id, kp, combined_eval_key_b64, joint_pk_tag)
        print("  Round 2 share submitted")

        # Wait for all round 2 to complete
        print("  Waiting for eval mult key to be finalized...")
        while True:
            try:
                resp = requests.get(f"{SERVER_URL}/eval_mult_key")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        print("  Eval mult key finalized")

    # Install the eval mult key locally (needed for EvalMult operations)
    print("  Installing eval mult key locally...")
    fetch_and_install_eval_mult_key()
    print("  Eval mult key installed")

    # Fetch the joint public key (must use this for ALL encryption)
    print("  Fetching joint public key...")
    joint_pk = fetch_joint_public_key()
    print("  Joint public key ready")

    print()
    print("  [Phase 1 complete] Threshold keys established")
    print()

    # ================================================================
    # PHASE 2: Proximity Match
    # ================================================================

    print("[Phase 2] Proximity match")
    print("-" * 40)

    if is_lead:
        # ---- LEAD / INITIATOR PATH ----

        input("  Press ENTER to start the match...")
        print()

        # Step 8: Encrypt own coords and start match
        print("  Encrypting coordinates under JOINT key...")
        start_match(party_id, lat, lon, joint_pk)
        print("  Match started — encrypted coords sent to server")
        print("  (Other parties can now compute distances locally)")
        print()

        # Step 9: Wait for all distances
        expected_distances = poll_server_status()["parties_joined"] - 1
        print(f"  Waiting for {expected_distances} parties to submit distances...")
        wait_for_distances(expected_distances)
        print("  All distances received")
        print()

        # Step 10: Tell server to compute pairwise scoring
        print("  Requesting server to compute pairwise scoring...")
        resp = requests.post(f"{SERVER_URL}/compute_nearest", json={})
        if resp.status_code != 200:
            raise RuntimeError(f"compute_nearest failed: {resp.json()}")
        print("  Scoring complete (on encrypted data — server learns nothing)")
        print()

        # Step 11: Partial decryption (lead)
        print("  Producing partial decryption (MultipartyDecryptLead)...")
        do_partial_decrypt(party_id, kp, is_lead=True)
        print("  Partial decryption submitted")

        # Wait for all partials
        total_parties = poll_server_status()["parties_joined"]
        print(f"  Waiting for all {total_parties} partial decryptions...")
        wait_for_partials(total_parties)
        print()

        # Step 12: Get fused result
        print("  Retrieving fused plaintext result...")
        values = get_decrypted_result(party_id)

        # Find winner
        initiator_slot = slot
        active_slots = [i for i in range(len(values)) if i != initiator_slot and abs(values[i]) > 0.01]
        if not active_slots:
            # Fallback: check all slots except initiator
            all_slots = list(range(poll_server_status()["slots_assigned"]))
            active_slots = [s for s in all_slots if s != initiator_slot]

        print("  Score vector (non-initiator slots):")
        for s in active_slots:
            print(f"    Slot {s}: {values[s]:.4f}")

        winner_slot = max(active_slots, key=lambda s: values[s])

        # Step 13: Resolve identity
        winner_name = resolve_slot(winner_slot)
        print()
        print("=" * 55)
        print(f"  RESULT: Nearest party is '{winner_name}' (slot {winner_slot})")
        print(f"  Score: {values[winner_slot]:.4f}")
        print(f"  Your location was NEVER revealed to anyone.")
        print(f"  Their location was NEVER revealed to you.")
        print(f"  The server learned NOTHING about anyone's location.")
        print("=" * 55)

    else:
        # ---- JOIN / RESPONDER PATH ----

        # Step 8: Wait for a match to start
        print("  Waiting for lead to start a match...")
        match_info = wait_for_match(party_id)
        print()
        print(f"  Match started by: {match_info['initiator']}")

        # Step 9: Compute distance LOCALLY
        print("  Computing encrypted distance LOCALLY...")
        print(f"    Using own coords ({lat:.5f}, {lon:.5f}) — stays on this device")
        print(f"    Subtracting from initiator's ENCRYPTED coords")

        enc_lat = deserialize_ciphertext(match_info["enc_lat"])
        enc_lon = deserialize_ciphertext(match_info["enc_lon"])

        dist_ct = compute_distance_local(enc_lat, enc_lon, lat, lon, joint_pk)
        id_ct = encrypt_onehot_id(slot, joint_pk)
        print("  Encrypted distance² computed (depth +1)")

        # Step 10: Submit distance to server
        print("  Submitting encrypted distance + one-hot ID...")
        submit_resp = submit_distance(party_id, dist_ct, id_ct)
        print(f"  Submitted ({submit_resp['received']}/{submit_resp['expected']} distances)")
        print()

        # Step 11: Wait for scoring to complete, then get result
        print("  Waiting for server to compute scoring...")
        while True:
            try:
                resp = requests.post(f"{SERVER_URL}/get_result", json={"party_id": party_id})
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        print("  Encrypted result available")

        # Step 12: Partial decryption (main)
        print("  Producing partial decryption (MultipartyDecryptMain)...")
        do_partial_decrypt(party_id, kp, is_lead=False)
        print("  Partial decryption submitted")
        print()
        print("=" * 55)
        print("  Done. Your partial decryption has been submitted.")
        print("  Only the initiator receives the final result.")
        print("  Your location was NEVER sent to anyone.")
        print("=" * 55)

