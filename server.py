# ==========================================
# server.py
# Privacy-preserving FHE proximity server
#
# Flow:
#   1. /get_slot  - server assigns a slot index for one-hot ID encoding
#   2. /register  - client sends encrypted location + encrypted one-hot ID
#   3. /nearest   - server computes encrypted distances, pairwise
#                   scoring to find nearest, returns encrypted
#                   blended ID vector
#   4. /resolve   - client sends back the winning slot (from argmax),
#                   server maps to actual user, returns it
#
# Key insight: IDs are ONE-HOT vectors across CKKS slots.
# Even with an imperfect sign polynomial, the argmax of the
# blended vector always points to the tournament winner —
# because the sign polynomial preserves sign (f(x)*x > 0),
# so the selector is always on the correct side of 0.5.
#
# Coordinates are replicated across all slots so the selector
# broadcasts naturally to multiply the one-hot vectors.
#
# Depth budget (depth 7):
#   - Distance: 1 ct-ct mult level
#   - Selector polynomial: 2 ct-ct + 1 pt-ct = 3 levels
#   - Score × ID: 1 ct-ct level
#   - Total: 5 levels ≤ 7 ✓  (independent of user count!)
#
# Server NEVER sees plaintext locations or distances.
# ==========================================

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import base64
import tempfile
import os
from openfhe import *

PORT = 8000
KEYS_DIR = "fhe_keys"
BATCH_SIZE = 32
MAX_USERS = 20

# -----------------------------
# Load shared crypto context
# -----------------------------

def load_context():
    cc, success = DeserializeCryptoContext(f"{KEYS_DIR}/cryptocontext.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load crypto context. Run setup_keys.py first!")

    if not cc.DeserializeEvalMultKey(f"{KEYS_DIR}/evalmultkey.bin", BINARY):
        raise RuntimeError("Failed to load eval mult key")

    return cc


cc = load_context()
print("Loaded shared crypto context")

# ----- State -----
NEXT_SLOT = 0
SLOT_TO_USER = {}   # slot_index -> actual_user
ACTIVE_USERS = {}   # epoch_id -> (x_ct, y_ct, id_ct)
EPOCH_TO_SLOT = {}  # epoch_id -> slot_index

# -----------------------------
# Serialization
# -----------------------------

def serialize_ciphertext(ct):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
    SerializeToFile(fname, ct, BINARY)
    with open(fname, 'rb') as f:
        data = f.read()
    os.unlink(fname)
    return base64.b64encode(data).decode()

def deserialize_ciphertext(data):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
        f.write(base64.b64decode(data))
    ct, success = DeserializeCiphertext(fname, BINARY)
    os.unlink(fname)
    if not success:
        raise ValueError("Deserialize failed")
    return ct

# -------------------------------------------------
# Squared Euclidean Distance
#
# Each user stores x_ct and y_ct, both with the value
# REPLICATED across all BATCH_SIZE slots.
# Output is also replicated — the selector in the
# tournament broadcasts to all slots naturally.
#
# Depth cost: 1 level (ct-ct mult; dx² and dy² are parallel)
# -------------------------------------------------

def compute_distance(x1, y1, x2, y2):
    dx = cc.EvalSub(x1, x2)          # all slots: dx
    dy = cc.EvalSub(y1, y2)          # all slots: dy
    dx2 = cc.EvalMult(dx, dx)        # all slots: dx²  (depth +1)
    dy2 = cc.EvalMult(dy, dy)        # all slots: dy²  (depth +1, parallel)
    return cc.EvalAdd(dx2, dy2)      # all slots: dx² + dy²

# -------------------------------------------------
# Selector polynomial (Horner form, coefficients
# pre-scaled to fold in the 0.15 normalization factor)
#
# selector(diff) ≈ 1 if diff > 0 (d_j > d_i),
#                ≈ 0 if diff < 0.
#
# Derived from:
#   sign(0.15*x) = 1.5*(0.15x) - 0.5*(0.15x)³
#   selector = (sign + 1) / 2
#            = 0.5 + 0.1125*x - 0.00084375*x³
#
# Horner: 0.5 + x * (0.1125 - 0.00084375 * x²)
#
# Valid for |diff| ≤ ~6.7 (our max ≈ 5). Always
# returns a value on the correct side of 0.5.
#
# Depth cost: 2 ct-ct + 1 pt-ct = 3 levels
# -------------------------------------------------

def compute_selector(diff):
    x2 = cc.EvalMult(diff, diff)                 # ct-ct: depth +1
    inner = cc.EvalMult(x2, -0.00084375)         # pt-ct: depth +1
    inner = cc.EvalAdd(inner, 0.1125)            # add: free
    product = cc.EvalMult(inner, diff)            # ct-ct: depth +1
    return cc.EvalAdd(product, 0.5)              # add: free

# -------------------------------------------------
# Pairwise scoring — finds nearest neighbor
#
# For each opponent i, compute:
#   score_i = sum_{j != i} selector(d_j - d_i)
#
# selector ≈ 1 when d_j > d_i, meaning i is closer.
# The opponent with the HIGHEST score has the smallest
# distance (it wins the most pairwise comparisons).
#
# result = sum_i  score_i * id_i
# argmax of result vector → winning slot
#
# Unlike the tournament approach, this compares every
# pair using ORIGINAL distances (no blending), so
# results are always correct regardless of user count.
#
# Depth: 1 (distance) + 3 (selector) + 1 (score*id)
#      = 5 levels total, independent of n.
# Compute: O(n²) selector evaluations.
# -------------------------------------------------

def find_nearest_by_scoring(target_x, target_y, opponents):
    """
    opponents: list of (x_ct, y_ct, id_ct) tuples
    Returns: encrypted one-hot ID of the nearest opponent
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
    # Compute selector for each ordered pair (i < j) once,
    # derive the reverse via 1 - selector.
    scores = [None] * n
    pairs_computed = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff = cc.EvalSub(dists[j], dists[i])
            sel_ji = compute_selector(diff)        # ≈ 1 if d_j > d_i
            sel_ij = cc.EvalSub(1.0, sel_ji)       # ≈ 1 if d_i > d_j
            pairs_computed += 1

            # Accumulate into score_i (benefits when d_j > d_i)
            scores[i] = cc.EvalAdd(scores[i], sel_ji) if scores[i] is not None else sel_ji
            # Accumulate into score_j (benefits when d_i > d_j)
            scores[j] = cc.EvalAdd(scores[j], sel_ij) if scores[j] is not None else sel_ij

    print(f"  [scoring] {pairs_computed} pairwise comparisons computed")

    # Step 3: Weighted sum of one-hot IDs
    result = cc.EvalMult(scores[0], opponents[0][2])   # ct-ct: depth +1
    for i in range(1, n):
        result = cc.EvalAdd(result, cc.EvalMult(scores[i], opponents[i][2]))

    return result


# ==========================================
# HTTP Handler
# ==========================================

class FHEHandler(BaseHTTPRequestHandler):

    def _read_body(self):
        length = int(self.headers["Content-Length"])
        body = b""
        while len(body) < length:
            chunk = self.rfile.read(min(length - len(body), 1024 * 1024))
            if not chunk:
                break
            body += chunk
        return body

    def _json_response(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = self._read_body()
        data = json.loads(body)

        # ---- /get_slot ----
        if self.path == "/get_slot":
            global NEXT_SLOT

            if NEXT_SLOT >= MAX_USERS:
                self._json_response(400, {"error": f"max {MAX_USERS} users reached"})
                return

            actual_user = data["actual_user"]
            slot = NEXT_SLOT
            NEXT_SLOT += 1
            SLOT_TO_USER[slot] = actual_user

            print(f"[get_slot] slot={slot} -> {actual_user}")
            self._json_response(200, {"slot": slot})
            return

        # ---- /register ----
        if self.path == "/register":
            epoch_id = data["epoch_id"]
            slot     = data["slot"]
            x_ct     = deserialize_ciphertext(data["x_ct"])
            y_ct     = deserialize_ciphertext(data["y_ct"])
            id_ct    = deserialize_ciphertext(data["id_ct"])

            ACTIVE_USERS[epoch_id] = (x_ct, y_ct, id_ct)
            EPOCH_TO_SLOT[epoch_id] = slot

            print(f"[register] epoch={epoch_id} slot={slot}  (all data encrypted)")
            self._json_response(200, {"status": "ok"})
            return

        # ---- /nearest ----
        if self.path == "/nearest":
            target_id = data["epoch_id"]

            if target_id not in ACTIVE_USERS:
                self._json_response(404, {"error": "user not found"})
                return

            target_x, target_y, _ = ACTIVE_USERS[target_id]

            opponents = []
            for uid, (x_ct, y_ct, id_ct) in ACTIVE_USERS.items():
                if uid == target_id:
                    continue
                opponents.append((x_ct, y_ct, id_ct))

            if not opponents:
                self._json_response(400, {"error": "no other users to compare"})
                return

            print(f"[nearest] Scoring {len(opponents)} opponents (pairwise comparisons)...")
            result_id = find_nearest_by_scoring(target_x, target_y, opponents)

            self._json_response(200, {
                "encrypted_nearest_id": serialize_ciphertext(result_id)
            })
            return

        # ---- /resolve ----
        if self.path == "/resolve":
            slot = data["slot"]
            actual_user = SLOT_TO_USER.get(slot)

            if actual_user is None:
                self._json_response(404, {"error": f"slot {slot} not found"})
                return

            print(f"[resolve] slot={slot} -> {actual_user}")
            self._json_response(200, {"actual_user": actual_user})
            return

        self._json_response(404, {"error": "unknown endpoint"})


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    print(f"Server running on port {PORT}  (max {MAX_USERS} users, batch {BATCH_SIZE})")
    HTTPServer(("", PORT), FHEHandler).serve_forever()
