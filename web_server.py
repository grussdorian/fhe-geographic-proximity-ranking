#!/usr/bin/env python3
"""
web_server.py — Multi-device FHE Proximity Demo

Setup:  pip install flask flask-socketio
Run:    python web_server.py
Open:   http://localhost:5000  (or network IP for other devices)

Each player opens the URL on their device, picks a character,
and places themselves on the map. Others see mystery sprites.
The host controls the protocol visualization.
"""

import math
import random
import socket
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit

# ── App ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "threshold-fhe-demo"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Constants ────────────────────────────────────────────
DRESDEN_LAT, DRESDEN_LNG = 51.0504, 13.7373
CHARS = {
    "alice":   {"name": "Alice",   "color": "#51cf66"},
    "bob":     {"name": "Bob",     "color": "#c0934b"},
    "charlie": {"name": "Charlie", "color": "#ff6b6b"},
    "diana":   {"name": "Diana",   "color": "#E79B40"},
}

# ── Game State ───────────────────────────────────────────
players = {}            # sid → {char, pos, placed, host}
state   = "lobby"       # lobby → placing → ready → running → done
step    = -1
initiator_char = None
auto_mode = False
result_data = {}


# ── Helpers ──────────────────────────────────────────────
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def taken():
    return [p["char"] for p in players.values()]


def everyone_placed():
    return len(players) >= 2 and all(p.get("placed") for p in players.values())


def fake_pos():
    return (
        DRESDEN_LAT + random.uniform(-0.013, 0.013),
        DRESDEN_LNG + random.uniform(-0.019, 0.019),
    )


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dLon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── HTTP Routes ──────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/images/<path:name>")
def serve_image(name):
    return send_from_directory("images", name)


# ── Socket.IO Events ────────────────────────────────────
@sio.on("connect")
def on_connect():
    host = len(players) == 0 and state == "lobby"
    emit(
        "welcome",
        {
            "host": host,
            "taken": taken(),
            "state": state,
            "auto": auto_mode,
            "playerCount": len(players),
        },
    )


@sio.on("disconnect")
def on_disconnect():
    sid = request.sid
    if sid not in players:
        return
    p = players.pop(sid)
    was_host = p.get("host", False)
    sio.emit("player_left", {"char": p["char"]})
    if was_host and players:
        new_sid = next(iter(players))
        players[new_sid]["host"] = True
        sio.emit("new_host", {}, to=new_sid)


@sio.on("pick")
def on_pick(data):
    global state
    sid = request.sid
    ch = data.get("char")

    if ch not in CHARS or ch in taken():
        emit("pick_fail", {"msg": "Unavailable"})
        return

    host = len(players) == 0
    players[sid] = {"char": ch, "pos": None, "placed": False, "host": host}
    state = "placing"

    emit(
        "picked",
        {
            "char": ch,
            "name": CHARS[ch]["name"],
            "color": CHARS[ch]["color"],
            "host": host,
        },
    )
    sio.emit(
        "joined",
        {
            "char": ch,
            "name": CHARS[ch]["name"],
            "taken": taken(),
            "count": len(players),
        },
    )

    # Send existing placements as mysteries to this new player
    for osid, op in players.items():
        if osid != sid and op.get("placed") and op["pos"]:
            fp = fake_pos()
            emit("other_placed", {"char": op["char"], "lat": fp[0], "lng": fp[1]})


@sio.on("place")
def on_place(data):
    global state
    sid = request.sid
    if sid not in players:
        return

    p = players[sid]
    lat, lng = data["lat"], data["lng"]
    p["pos"] = (lat, lng)
    p["placed"] = True

    # Real position to self
    emit("self_placed", {"char": p["char"], "lat": lat, "lng": lng})

    # Fake position to each other player
    for osid in players:
        if osid != sid:
            fp = fake_pos()
            sio.emit(
                "other_placed",
                {"char": p["char"], "lat": fp[0], "lng": fp[1]},
                to=osid,
            )

    if everyone_placed():
        state = "ready"
        sio.emit("all_ready", {"chars": taken()})


@sio.on("set_mode")
def on_mode(data):
    global auto_mode
    auto_mode = data.get("auto", False)
    sio.emit("mode_changed", {"auto": auto_mode})


@sio.on("set_initiator")
def on_init(data):
    global initiator_char
    initiator_char = data["char"]
    sio.emit("initiator_set", {"char": initiator_char})


@sio.on("start")
def on_start():
    global state, step, result_data
    state = "running"
    step = -1

    # Find initiator's real position
    init_pos = None
    for p in players.values():
        if p["char"] == initiator_char:
            init_pos = p["pos"]
            break

    # Compute distances
    dists = {}
    for p in players.values():
        if p["char"] != initiator_char and p["pos"]:
            d = haversine(init_pos[0], init_pos[1], p["pos"][0], p["pos"][1])
            dists[p["char"]] = round(d, 2)

    nearest = min(dists, key=dists.get) if dists else None
    result_data = {
        "dists": dists,
        "nearest": nearest,
        "initiator": initiator_char,
    }

    sio.emit(
        "started",
        {"initiator": initiator_char, "auto": auto_mode, "totalSteps": 12},
    )


@sio.on("next")
def on_next():
    global step, state
    step += 1
    if step >= 12:
        state = "done"
        sio.emit("done", result_data)
    else:
        sio.emit("step", {"n": step, "auto": auto_mode})


@sio.on("reset")
def on_reset():
    global state, step, initiator_char, result_data
    for p in players.values():
        p["pos"] = None
        p["placed"] = False
    state = "placing"
    step = -1
    initiator_char = None
    result_data = {}
    sio.emit("game_reset", {"taken": taken()})


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    ip = get_ip()
    port = 5000
    print(
        f"""
{'='*52}
  🔐 FHE PROXIMITY DEMO — MULTI-DEVICE SERVER
{'='*52}
  Local:   http://localhost:{port}
  Network: http://{ip}:{port}

  Share the network URL with other players!
{'='*52}
"""
    )
    sio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
