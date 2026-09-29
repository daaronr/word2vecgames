/*
 * Word Bocce — online rooms.
 *
 * A thin wrapper over PeerJS: browsers talk to each other directly (WebRTC data
 * channels), introduced by the free PeerJS cloud broker. There is no game server
 * of our own: the host's browser *is* the room. Guests connect to it, it keeps the
 * authoritative state and rebroadcasts it. If the host closes the tab, the room ends.
 *
 * The PeerJS library is only fetched when someone opens or joins a room.
 */
(function (root) {
  "use strict";
  const PEERJS = "https://cdn.jsdelivr.net/npm/peerjs@1.5.5/dist/peerjs.min.js";
  const PREFIX = "wordbocce-v1-";
  const ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // no I, L, O, 0, 1
  const JOIN_TIMEOUT_MS = 15000;

  let loading = null;
  function loadPeer() {
    if (root.Peer) return Promise.resolve(root.Peer);
    if (!loading) {
      loading = new Promise((resolve, reject) => {
        const el = document.createElement("script");
        el.src = PEERJS;
        el.onload = () => (root.Peer ? resolve(root.Peer) : reject(new Error("The online-play library didn't start.")));
        el.onerror = () => { loading = null; reject(new Error("Couldn't load the online-play library. Check your connection.")); };
        document.head.append(el);
      });
    }
    return loading;
  }

  const newCode = () => Array.from({ length: 5 }, () => ALPHABET[Math.floor(Math.random() * ALPHABET.length)]).join("");
  const cleanCode = (s) => String(s || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);

  function explain(err) {
    const t = err && err.type;
    if (t === "peer-unavailable") return "There's no open room with that code. Check the link, or ask the host to open the room again.";
    if (t === "unavailable-id") return "That room code is taken. Try again for a new one.";
    if (t === "browser-incompatible") return "This browser can't do online play. Try an up-to-date Chrome, Safari or Firefox.";
    if (t === "network" || t === "server-error" || t === "socket-error" || t === "socket-closed") return "Couldn't reach the online-play service. Check your connection and try again.";
    return (err && err.message) || "Something went wrong with the connection.";
  }

  /**
   * Open a room. handlers: onOpen(code), onJoin(peerId), onMessage(peerId, msg), onLeave(peerId), onError(message)
   * Returns { code, send(peerId, msg), broadcast(msg), close() }.
   */
  async function host(handlers, code = newCode()) {
    const Peer = await loadPeer();
    const peer = new Peer(PREFIX + code);
    const conns = new Map();
    peer.on("open", () => handlers.onOpen(code));
    peer.on("connection", (c) => {
      c.on("open", () => { conns.set(c.peer, c); handlers.onJoin(c.peer); });
      c.on("data", (msg) => handlers.onMessage(c.peer, msg));
      c.on("close", () => { if (conns.delete(c.peer)) handlers.onLeave(c.peer); });
      c.on("error", () => { if (conns.delete(c.peer)) handlers.onLeave(c.peer); });
    });
    peer.on("error", (e) => handlers.onError(explain(e)));
    // Losing the broker doesn't break open data channels; reconnect so new guests can still find us.
    peer.on("disconnected", () => { if (!peer.destroyed) peer.reconnect(); });
    return {
      code,
      send(peerId, msg) { const c = conns.get(peerId); if (c && c.open) c.send(msg); },
      broadcast(msg) { for (const c of conns.values()) if (c.open) c.send(msg); },
      close() { peer.destroy(); },
    };
  }

  /**
   * Join a room. handlers: onOpen(myId), onMessage(msg), onClose(), onError(message)
   * Returns { send(msg), close() }.
   */
  async function join(code, handlers) {
    const Peer = await loadPeer();
    const peer = new Peer();
    let conn = null, opened = false, closed = false;
    const fail = (msg) => { if (!closed) { closed = true; handlers.onError(msg); peer.destroy(); } };
    const timer = setTimeout(() => { if (!opened) fail("Couldn't reach the room. The host may have closed it, or a network is blocking the connection."); }, JOIN_TIMEOUT_MS);
    peer.on("open", () => {
      conn = peer.connect(PREFIX + cleanCode(code), { reliable: true });
      conn.on("open", () => { opened = true; clearTimeout(timer); handlers.onOpen(peer.id); });
      conn.on("data", (msg) => handlers.onMessage(msg));
      conn.on("close", () => { if (!closed) { closed = true; handlers.onClose(); } });
      conn.on("error", (e) => fail(explain(e)));
    });
    peer.on("error", (e) => fail(explain(e)));
    return {
      send(msg) { if (conn && conn.open) conn.send(msg); },
      close() { closed = true; clearTimeout(timer); peer.destroy(); },
    };
  }

  root.BocceNet = { host, join, newCode, cleanCode };
})(typeof self !== "undefined" ? self : this);
