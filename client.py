# ==========================================
# client.py
# Privacy-preserving FHE proximity client
#
# Flow:
#   1. Get a slot index from server (for one-hot ID encoding)
#   2. Encrypt location (x, y replicated) + one-hot ID vector
#   3. Register with server
#   4. Request nearest → get encrypted blended ID vector
#   5. Decrypt, find argmax → winning slot
#   6. Send slot to /resolve → get actual user identity
#
# Client never sees anyone else's location.
# Server never sees plaintext locations or distances.
# ==========================================

import requests
import uuid
import time
import base64
import tempfile
import os
from openfhe import *

SERVER_URL = "http://localhost:8000"
KEYS_DIR = "fhe_keys"
BATCH_SIZE = 32
MAX_COORD = 180.0  # Longitude range: covers [-180, 180] decimal degrees

# -----------------------------
# Load shared context + keys
# -----------------------------

def load_context():
    cc, success = DeserializeCryptoContext(f"{KEYS_DIR}/cryptocontext.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load crypto context. Run setup_keys.py first!")

    pubkey, success = DeserializePublicKey(f"{KEYS_DIR}/publickey.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load public key")

    seckey, success = DeserializePrivateKey(f"{KEYS_DIR}/secretkey.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load secret key")

    return cc, pubkey, seckey


cc, public_key, secret_key = load_context()
print("Loaded shared crypto context")


# -----------------------------
# Epoch-based ephemeral ID
# -----------------------------

def generate_epoch_id():
    epoch = int(time.time() // 60)
    base = f"{epoch}_{uuid.uuid4()}"
    return str(abs(hash(base)))[:12]


# -----------------------------
# Serialization
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


# -----------------------------
# Encrypt location (replicated)
#
# Each coord is replicated across ALL BATCH_SIZE slots.
# Coords are decimal degrees (lat/lon) normalized by 180
# so they fit in [-1, 1]. This ensures the selector in
# the tournament broadcasts naturally to all slots of
# the one-hot ID vector.
# -----------------------------

def encrypt_location(lat, lon):
    nlat = float(lat) / MAX_COORD  # normalize to ~[-0.5, 0.5]
    nlon = float(lon) / MAX_COORD  # normalize to [-1, 1]
    lat_pt = cc.MakeCKKSPackedPlaintext([nlat] * BATCH_SIZE)
    lon_pt = cc.MakeCKKSPackedPlaintext([nlon] * BATCH_SIZE)
    lat_ct = cc.Encrypt(public_key, lat_pt)
    lon_ct = cc.Encrypt(public_key, lon_pt)
    return lat_ct, lon_ct


def encrypt_onehot_id(slot_index):
    """Encrypt a one-hot vector with 1.0 at the given slot."""
    vec = [0.0] * BATCH_SIZE
    vec[slot_index] = 1.0
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(public_key, pt)


# =============================================
# Main flow
# =============================================

if __name__ == "__main__":
    import sys
    import random

    # Actual user identity
    actual_user = f"user_{uuid.uuid4().hex[:8]}"

    # Location: from command line (lat lon) or random (roughly continental US)
    if len(sys.argv) >= 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
    else:
        lat = random.uniform(25.0, 48.0)     # ~US latitude range
        lon = random.uniform(-125.0, -70.0)   # ~US longitude range

    epoch_id = generate_epoch_id()

    print(f"=== FHE Proximity Client ===")
    print(f"  Actual user:  {actual_user}")
    print(f"  Epoch ID:     {epoch_id}")
    print(f"  Location:     ({lat:.5f}, {lon:.5f})  [plaintext, only client knows]")
    print()

    # ---- Step 1: Get slot assignment ----
    print("[1] Getting slot assignment from server...")
    resp = requests.post(
        SERVER_URL + "/get_slot",
        json={"actual_user": actual_user}
    )
    if resp.status_code != 200:
        print(f"    Failed: {resp.json().get('error', resp.status_code)}")
        sys.exit(1)

    slot = resp.json()["slot"]
    print(f"    Assigned slot: {slot}")
    print()

    # ---- Step 2: Encrypt and register ----
    print("[2] Encrypting location + one-hot ID and registering...")
    lat_ct, lon_ct = encrypt_location(lat, lon)
    id_ct = encrypt_onehot_id(slot)

    resp = requests.post(
        SERVER_URL + "/register",
        json={
            "epoch_id": epoch_id,
            "slot": slot,
            "x_ct": serialize_ciphertext(lat_ct),
            "y_ct": serialize_ciphertext(lon_ct),
            "id_ct": serialize_ciphertext(id_ct),
        }
    )
    if resp.status_code != 200:
        print("    Registration failed:", resp.status_code)
        sys.exit(1)
    print("    Registered successfully.")
    print()

    # ---- Step 3: Request nearest ----
    print("[3] Requesting nearest user (server computes on encrypted data)...")
    response = requests.post(
        SERVER_URL + "/nearest",
        json={"epoch_id": epoch_id}
    )

    if response.status_code == 400:
        print("    No other users registered yet. Run another client first.")
        sys.exit(0)
    elif response.status_code != 200:
        print("    Error:", response.status_code)
        sys.exit(1)

    data = response.json()
    encrypted_result = deserialize_ciphertext(data["encrypted_nearest_id"])

    # ---- Step 4: Decrypt and find argmax ----
    print("[4] Decrypting result vector and finding argmax...")
    pt = cc.Decrypt(secret_key, encrypted_result)
    pt.SetLength(BATCH_SIZE)
    values = pt.GetRealPackedValue()

    # Find the slot with the highest value (one-hot argmax)
    winner_slot = max(range(len(values)), key=lambda i: values[i])
    print(f"    Result vector (first 8 slots): {[f'{v:.3f}' for v in values[:8]]}")
    print(f"    Winner slot: {winner_slot} (value: {values[winner_slot]:.3f})")
    print()

    # ---- Step 5: Resolve to actual user ----
    print("[5] Resolving slot to actual user via server...")
    resolve_resp = requests.post(
        SERVER_URL + "/resolve",
        json={"slot": winner_slot}
    )

    if resolve_resp.status_code == 200:
        resolved = resolve_resp.json()
        print(f"    >>> Nearest user: {resolved['actual_user']}")
    else:
        print(f"    Failed to resolve: {resolve_resp.json().get('error', resolve_resp.status_code)}")

    print()
    print("Done. Server never saw plaintext locations or distances.")
