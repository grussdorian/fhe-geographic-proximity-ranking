# ==========================================
# server.py — Threshold FHE Proximity Server
#
# Architecture: Multi-party threshold FHE
#   - No single party can decrypt alone
#   - Clients compute distances locally on their devices
#   - Server only sees encrypted distances, never coords
#   - Decryption requires ALL parties to contribute
#
# Protocol flow:
#   Phase 1 — Interactive key generation:
#     1. /join          - each party generates a key share
#     2. /finalize_keys - server combines eval mult keys
#
#   Phase 2 — Match (per query):
#     3. /start_match       - initiator sends encrypted coords
#     4. /get_match         - other parties poll for match request
#     5. /submit_distance   - each party submits enc(distance²) + enc(id)
#     6. /compute_nearest   - server runs pairwise scoring
#     7. /get_result        - parties get encrypted result
#     8. /submit_partial_decrypt - each party provides partial decrypt
#     9. /get_decrypted     - initiator gets fused plaintext result
#    10. /resolve           - initiator resolves slot → user identity
#
# Depth budget (depth 7):
#   - Distance: 1 ct-ct mult (done by clients)
#   - Selector polynomial: 2 ct-ct + 1 pt-ct = 3 levels (server)
#   - Score × ID: 1 ct-ct level (server)
#   - Total: 5 levels ≤ 7 ✓
#
# Server NEVER sees plaintext locations or distances.
# Server CANNOT decrypt — no secret key.
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
MAX_COORD = 180.0

# -----------------------------
# Load crypto context (no keys — those come from parties)
# -----------------------------

def load_context():
    cc, success = DeserializeCryptoContext(f"{KEYS_DIR}/cryptocontext.bin", BINARY)
    if not success:
        raise RuntimeError("Failed to load crypto context. Run setup_keys.py first!")
    return cc


cc = load_context()
print("Loaded crypto context (MULTIPARTY mode)")

# ----- State -----
# Key generation state
PARTIES = {}           # party_id -> {"keypair": kp, "order": int}
PARTY_ORDER = []       # ordered list of party_ids (join order)
KEYS_FINALIZED = False
JOINT_PUBLIC_KEY = None  # final combined public key

# User/slot state
NEXT_SLOT = 0
SLOT_TO_USER = {}      # slot_index -> actual_user
PARTY_TO_SLOT = {}     # party_id -> slot_index

# Eval mult key state (round 1 combined key — needed by non-lead parties)
COMBINED_EVAL_KEY = None  # serialized combined eval key from round 1
FINAL_EVAL_MULT_KEY = None  # serialized final eval mult key (after round 2)
LEAD_EVAL_SHARE = None  # serialized lead party's KeySwitchGen result (needed by others)

# Match state
CURRENT_MATCH = None   # {"initiator": party_id, "enc_lat": ct, "enc_lon": ct,
                        #  "distances": {party_id: (dist_ct, id_ct)},
                        #  "result_ct": ciphertext or None,
                        #  "partials": [partial_ct, ...],
                        #  "fused_values": list or None}


# -----------------------------
# Serialization helpers
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


def serialize_key(key):
    """Serialize a public key or eval key to base64."""
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


# -------------------------------------------------
# Pairwise scoring — finds nearest neighbor
# (operates on encrypted distances, NOT coordinates)
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
# Depth consumed on server:
#   3 levels (selector) + 1 level (score * id) = 4 levels
#   Distance was already computed by clients (1 level).
# -------------------------------------------------

def compute_selector(diff):
    """Selector polynomial in Horner form.
    0.5 + x * (0.1125 - 0.00084375 * x²)
    Depth: 2 ct-ct + 1 pt-ct = 3 levels.
    """
    x2 = cc.EvalMult(diff, diff)
    inner = cc.EvalMult(x2, -0.00084375)
    inner = cc.EvalAdd(inner, 0.1125)
    product = cc.EvalMult(inner, diff)
    return cc.EvalAdd(product, 0.5)


def find_nearest_by_scoring(opponents):
    """
    opponents: list of (dist_ct, id_ct) tuples — encrypted distances and one-hot IDs.
    Returns: encrypted one-hot ID of the nearest opponent.
    """
    n = len(opponents)

    if n == 1:
        return opponents[0][1]  # id_ct

    dists = [d for d, _ in opponents]

    # Pairwise selectors → accumulate scores
    scores = [None] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = cc.EvalSub(dists[j], dists[i])
            sel_ji = compute_selector(diff)
            sel_ij = cc.EvalSub(1.0, sel_ji)

            scores[i] = cc.EvalAdd(scores[i], sel_ji) if scores[i] is not None else sel_ji
            scores[j] = cc.EvalAdd(scores[j], sel_ij) if scores[j] is not None else sel_ij

    # Weighted sum of one-hot IDs
    result = cc.EvalMult(scores[0], opponents[0][1])
    for i in range(1, n):
        result = cc.EvalAdd(result, cc.EvalMult(scores[i], opponents[i][1]))

    return result


# ==========================================
# HTTP Handler
# ==========================================

class FHEHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle GET requests (status, combined eval key)."""

        if self.path == "/status":
            self._json_response(200, {
                "parties_joined": len(PARTY_ORDER),
                "parties_with_keys": len(PARTIES),
                "keys_finalized": KEYS_FINALIZED,
                "slots_assigned": NEXT_SLOT,
                "match_active": CURRENT_MATCH is not None,
                "match_distances_received": len(CURRENT_MATCH["distances"]) if CURRENT_MATCH else 0,
                "match_partials_received": len(CURRENT_MATCH["partials"]) if CURRENT_MATCH else 0,
            })
            return

        if self.path == "/combined_eval_key":
            if COMBINED_EVAL_KEY is None:
                self._json_response(404, {"error": "combined eval key not yet available"})
                return
            self._json_response(200, {
                "combined_eval_key": COMBINED_EVAL_KEY,
                "joint_pk_tag": JOINT_PUBLIC_KEY.GetKeyTag() if JOINT_PUBLIC_KEY else None,
            })
            return

        if self.path == "/eval_mult_key":
            if FINAL_EVAL_MULT_KEY is None:
                self._json_response(404, {"error": "eval mult key not yet finalized"})
                return
            self._json_response(200, {
                "eval_mult_key": FINAL_EVAL_MULT_KEY,
            })
            return

        if self.path == "/joint_public_key":
            if JOINT_PUBLIC_KEY is None:
                self._json_response(404, {"error": "joint public key not yet available"})
                return
            self._json_response(200, {
                "joint_public_key": serialize_key(JOINT_PUBLIC_KEY),
            })
            return

        if self.path == "/lead_eval_share":
            if LEAD_EVAL_SHARE is None:
                self._json_response(404, {"error": "lead eval share not yet available"})
                return
            self._json_response(200, {
                "lead_eval_share": LEAD_EVAL_SHARE,
            })
            return

        self._json_response(404, {"error": "unknown endpoint"})

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
        global NEXT_SLOT, JOINT_PUBLIC_KEY, KEYS_FINALIZED, CURRENT_MATCH, COMBINED_EVAL_KEY, FINAL_EVAL_MULT_KEY, LEAD_EVAL_SHARE
        body = self._read_body()
        data = json.loads(body)

        # ============================================================
        # Phase 1: Interactive Threshold Key Generation
        # ============================================================

        # ---- /join ----
        # Each party joins and contributes a key share.
        # First party calls cc.KeyGen(); subsequent parties
        # call cc.MultipartyKeyGen(prev_public_key).
        if self.path == "/join":

            if KEYS_FINALIZED:
                self._json_response(400, {"error": "keys already finalized"})
                return

            if NEXT_SLOT >= MAX_USERS:
                self._json_response(400, {"error": f"max {MAX_USERS} parties"})
                return

            party_id = data["party_id"]
            actual_user = data["actual_user"]

            if party_id in PARTIES:
                self._json_response(400, {"error": "party already joined"})
                return

            order = len(PARTY_ORDER)
            slot = NEXT_SLOT
            NEXT_SLOT += 1
            SLOT_TO_USER[slot] = actual_user
            PARTY_TO_SLOT[party_id] = slot
            PARTY_ORDER.append(party_id)

            # Server returns the current joint public key (or None for first party)
            resp = {
                "slot": slot,
                "order": order,
                "is_lead": (order == 0),
            }

            if JOINT_PUBLIC_KEY is not None:
                resp["prev_public_key"] = serialize_key(JOINT_PUBLIC_KEY)
            else:
                resp["prev_public_key"] = None

            print(f"[join] party={party_id} slot={slot} order={order}")
            self._json_response(200, resp)
            return

        # ---- /submit_key_share ----
        # Each party sends back: their public key, their eval mult key share
        if self.path == "/submit_key_share":

            party_id = data["party_id"]
            pub_key = deserialize_public_key(data["public_key"])

            JOINT_PUBLIC_KEY = pub_key

            order = PARTY_ORDER.index(party_id)

            # Store the lead's eval share so non-lead parties can fetch it
            if order == 0:
                LEAD_EVAL_SHARE = data["eval_mult_share"]

            # Store the eval mult key share
            PARTIES[party_id] = {
                "public_key": pub_key,
                "eval_mult_share": data["eval_mult_share"],  # base64 serialized
                "order": order,
            }

            print(f"[submit_key_share] party={party_id}  "
                  f"({len(PARTIES)}/{len(PARTY_ORDER)} shares received)")
            self._json_response(200, {"status": "ok"})
            return

        # ---- /finalize_keys ----
        # Server combines all eval mult key shares into the joint eval mult key.
        # Triggered after all parties have submitted their shares.
        if self.path == "/finalize_keys":

            if KEYS_FINALIZED:
                self._json_response(400, {"error": "already finalized"})
                return

            if len(PARTIES) != len(PARTY_ORDER):
                self._json_response(400, {
                    "error": f"waiting for {len(PARTY_ORDER) - len(PARTIES)} more key shares"
                })
                return

            if len(PARTIES) < 2:
                self._json_response(400, {"error": "need at least 2 parties"})
                return

            # Combine eval mult keys
            # Step 1: deserialize all eval mult shares
            ordered_parties = sorted(PARTIES.items(),
                                     key=lambda x: x[1]["order"])

            # The lead party's share is the base
            lead_id, lead_info = ordered_parties[0]
            lead_share_data = base64.b64decode(lead_info["eval_mult_share"])
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(lead_share_data)
                fname = f.name
            lead_share, ok = DeserializeEvalKey(fname, BINARY)
            os.unlink(fname)
            if not ok:
                self._json_response(500, {"error": "failed to deserialize lead eval mult share"})
                return

            combined = lead_share

            for party_id, info in ordered_parties[1:]:
                share_data = base64.b64decode(info["eval_mult_share"])
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(share_data)
                    fname = f.name
                share, ok = DeserializeEvalKey(fname, BINARY)
                os.unlink(fname)
                if not ok:
                    self._json_response(500, {"error": f"failed to deserialize share for {party_id}"})
                    return

                # Accumulate: MultiAddEvalKeys (must use JOINT pk tag, not individual)
                combined = cc.MultiAddEvalKeys(combined, share,
                                               JOINT_PUBLIC_KEY.GetKeyTag())

            # Now we need each party to compute s_i * combined
            # and then combine those. For the CKKS threshold model we need
            # MultiMultEvalKey from each party, then MultiAddEvalMultKeys.
            # Store the combined eval key for the next round.
            # We'll need a second round where each party calls
            # MultiMultEvalKey with their secret key.
            # For simplicity in this HTTP model, we store it and let
            # parties fetch + process it.

            # Save the combined (but not yet finalized) eval mult key
            COMBINED_EVAL_KEY = serialize_key(combined)
            KEYS_FINALIZED = True

            # For now with the HTTP model, we do a simplified approach:
            # Each party already sent their KeySwitchGen / MultiKeySwitchGen share.
            # The full multi-round protocol requires parties to come back.
            # We'll handle this with /get_combined_eval_key and /submit_mult_eval_key.

            print(f"[finalize_keys] Combined {len(PARTIES)} eval mult key shares")
            print(f"  Joint public key ready (tag={JOINT_PUBLIC_KEY.GetKeyTag()[:12]}...). Parties must now complete round 2.")
            self._json_response(200, {
                "status": "keys_combined_round1",
                "parties": len(PARTIES),
                "combined_eval_key": serialize_key(combined),
                "joint_pk_tag": JOINT_PUBLIC_KEY.GetKeyTag(),
            })
            return

        # ---- /submit_mult_eval_key ----
        # Round 2: each party sends s_i * combined_eval_key
        if self.path == "/submit_mult_eval_key":
            party_id = data["party_id"]
            mult_share_data = base64.b64decode(data["mult_eval_share"])

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(mult_share_data)
                fname = f.name
            mult_share, ok = DeserializeEvalKey(fname, BINARY)
            os.unlink(fname)
            if not ok:
                self._json_response(500, {"error": "failed to deserialize mult eval share"})
                return

            if not hasattr(self.server, 'mult_eval_shares'):
                self.server.mult_eval_shares = {}
            self.server.mult_eval_shares[party_id] = mult_share

            print(f"[submit_mult_eval_key] party={party_id}  "
                  f"({len(self.server.mult_eval_shares)}/{len(PARTIES)} received)")

            # When all shares received, combine and install
            if len(self.server.mult_eval_shares) == len(PARTIES):
                ordered = sorted(self.server.mult_eval_shares.items(),
                                 key=lambda x: PARTIES[x[0]]["order"])

                final = ordered[0][1]
                for _, share in ordered[1:]:
                    final = cc.MultiAddEvalMultKeys(final, share,
                                                    final.GetKeyTag())

                cc.InsertEvalMultKey([final])
                FINAL_EVAL_MULT_KEY = serialize_key(final)
                print("[finalize] ✓ Joint eval mult key installed!")
                self._json_response(200, {"status": "eval_mult_key_finalized"})
            else:
                self._json_response(200, {"status": "waiting_for_more_shares"})
            return

        # ============================================================
        # Phase 2: Match Protocol
        # ============================================================

        # ---- /start_match ----
        # Initiator sends their encrypted coordinates.
        # Server stores them for distribution to other parties.
        if self.path == "/start_match":

            party_id = data["party_id"]

            if party_id not in PARTIES:
                self._json_response(404, {"error": "party not registered"})
                return

            CURRENT_MATCH = {
                "initiator": party_id,
                "enc_lat": data["enc_lat"],  # serialized ciphertexts
                "enc_lon": data["enc_lon"],
                "distances": {},
                "result_ct": None,
                "partials": {},
                "fused_values": None,
            }

            print(f"[start_match] initiator={party_id}")
            self._json_response(200, {
                "status": "match_started",
                "num_parties": len(PARTIES),
            })
            return

        # ---- /get_match ----
        # Other parties poll to see if there's an active match.
        # Returns the initiator's encrypted coords so they can
        # compute distance locally.
        if self.path == "/get_match":
            party_id = data["party_id"]

            if CURRENT_MATCH is None:
                self._json_response(404, {"error": "no active match"})
                return

            if party_id == CURRENT_MATCH["initiator"]:
                self._json_response(400, {"error": "you are the initiator"})
                return

            self._json_response(200, {
                "initiator": CURRENT_MATCH["initiator"],
                "enc_lat": CURRENT_MATCH["enc_lat"],
                "enc_lon": CURRENT_MATCH["enc_lon"],
            })
            return

        # ---- /submit_distance ----
        # Each party (including initiator for self-exclusion purposes)
        # submits their encrypted distance + one-hot ID.
        # The party computes dist = (enc_lat - my_lat)² + (enc_lon - my_lon)²
        # on their device, consuming 1 depth level. The server never
        # sees the party's own lat/lon.
        if self.path == "/submit_distance":
            party_id = data["party_id"]

            if CURRENT_MATCH is None:
                self._json_response(404, {"error": "no active match"})
                return

            # Don't accept from the initiator (they are the target)
            if party_id == CURRENT_MATCH["initiator"]:
                self._json_response(400, {"error": "initiator doesn't submit distance"})
                return

            dist_ct = deserialize_ciphertext(data["dist_ct"])
            id_ct = deserialize_ciphertext(data["id_ct"])

            CURRENT_MATCH["distances"][party_id] = (dist_ct, id_ct)

            expected = len(PARTIES) - 1  # everyone except initiator
            received = len(CURRENT_MATCH["distances"])
            print(f"[submit_distance] party={party_id}  ({received}/{expected})")

            self._json_response(200, {
                "status": "distance_received",
                "received": received,
                "expected": expected,
            })
            return

        # ---- /compute_nearest ----
        # Once all distances are in, server runs pairwise scoring.
        # Returns status (for polling).
        if self.path == "/compute_nearest":
            if CURRENT_MATCH is None:
                self._json_response(404, {"error": "no active match"})
                return

            expected = len(PARTIES) - 1
            received = len(CURRENT_MATCH["distances"])

            if received < expected:
                self._json_response(400, {
                    "error": f"waiting for {expected - received} more distances"
                })
                return

            if CURRENT_MATCH["result_ct"] is not None:
                self._json_response(200, {"status": "already_computed"})
                return

            # Build opponents list from submitted distances
            opponents = []
            for pid in PARTY_ORDER:
                if pid == CURRENT_MATCH["initiator"]:
                    continue
                if pid in CURRENT_MATCH["distances"]:
                    opponents.append(CURRENT_MATCH["distances"][pid])

            print(f"[compute_nearest] Scoring {len(opponents)} opponents...")
            result_ct = find_nearest_by_scoring(opponents)
            CURRENT_MATCH["result_ct"] = result_ct

            print(f"[compute_nearest] ✓ Result computed")
            self._json_response(200, {
                "status": "computed",
                "encrypted_result": serialize_ciphertext(result_ct),
            })
            return

        # ---- /get_result ----
        # Any party can fetch the encrypted result for partial decryption.
        if self.path == "/get_result":
            if CURRENT_MATCH is None or CURRENT_MATCH["result_ct"] is None:
                self._json_response(404, {"error": "no result yet"})
                return

            self._json_response(200, {
                "encrypted_result": serialize_ciphertext(CURRENT_MATCH["result_ct"]),
            })
            return

        # ---- /submit_partial_decrypt ----
        # Each party provides their partial decryption of the result.
        # Lead party uses MultipartyDecryptLead, others use MultipartyDecryptMain.
        if self.path == "/submit_partial_decrypt":
            party_id = data["party_id"]

            if CURRENT_MATCH is None:
                self._json_response(404, {"error": "no active match"})
                return

            partial_data = base64.b64decode(data["partial_ct"])
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(partial_data)
                fname = f.name
            partial_ct, ok = DeserializeCiphertext(fname, BINARY)
            os.unlink(fname)
            if not ok:
                self._json_response(500, {"error": "failed to deserialize partial decryption"})
                return

            CURRENT_MATCH["partials"][party_id] = partial_ct

            received = len(CURRENT_MATCH["partials"])
            expected = len(PARTIES)
            print(f"[submit_partial_decrypt] party={party_id}  ({received}/{expected})")

            self._json_response(200, {
                "status": "partial_received",
                "received": received,
                "expected": expected,
            })
            return

        # ---- /get_decrypted ----
        # Once all partial decryptions are in, server fuses them
        # and returns the plaintext result to the initiator.
        if self.path == "/get_decrypted":
            party_id = data.get("party_id")

            if CURRENT_MATCH is None:
                self._json_response(404, {"error": "no active match"})
                return

            if party_id != CURRENT_MATCH["initiator"]:
                self._json_response(403, {"error": "only the initiator can get the result"})
                return

            expected = len(PARTIES)
            received = len(CURRENT_MATCH["partials"])

            if received < expected:
                self._json_response(400, {
                    "error": f"waiting for {expected - received} more partial decryptions"
                })
                return

            if CURRENT_MATCH["fused_values"] is not None:
                # Already fused
                self._json_response(200, {"values": CURRENT_MATCH["fused_values"]})
                return

            # Fuse partial decryptions — must be in party order
            # (lead first, then the rest)
            partial_vec = []
            lead_party = PARTY_ORDER[0]
            partial_vec.append(CURRENT_MATCH["partials"][lead_party])
            for pid in PARTY_ORDER[1:]:
                partial_vec.append(CURRENT_MATCH["partials"][pid])

            plaintext = cc.MultipartyDecryptFusion(partial_vec)
            plaintext.SetLength(BATCH_SIZE)
            values = list(plaintext.GetRealPackedValue())

            CURRENT_MATCH["fused_values"] = values

            print(f"[get_decrypted] ✓ Fused {len(partial_vec)} partial decryptions")
            self._json_response(200, {"values": values})
            return

        # ---- /resolve ----
        # Initiator sends the winning slot, server returns the actual user.
        if self.path == "/resolve":
            slot = data["slot"]
            actual_user = SLOT_TO_USER.get(slot)

            if actual_user is None:
                self._json_response(404, {"error": f"slot {slot} not found"})
                return

            print(f"[resolve] slot={slot} -> {actual_user}")
            self._json_response(200, {"actual_user": actual_user})
            return

        # ---- /status ----
        # For debugging: show current state
        if self.path == "/status":
            self._json_response(200, {
                "parties_joined": len(PARTY_ORDER),
                "parties_with_keys": len(PARTIES),
                "keys_finalized": KEYS_FINALIZED,
                "slots_assigned": NEXT_SLOT,
                "match_active": CURRENT_MATCH is not None,
                "match_distances_received": len(CURRENT_MATCH["distances"]) if CURRENT_MATCH else 0,
                "match_partials_received": len(CURRENT_MATCH["partials"]) if CURRENT_MATCH else 0,
            })
            return

        self._json_response(404, {"error": "unknown endpoint"})


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    print(f"Threshold FHE Proximity Server")
    print(f"  Port: {PORT}")
    print(f"  Max parties: {MAX_USERS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Depth: 7 (5 used: 1 client + 4 server)")
    print(f"  Decryption: threshold (ALL parties required)")
    print()
    HTTPServer(("", PORT), FHEHandler).serve_forever()
