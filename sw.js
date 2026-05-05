/* sw.js — F1 2026 Service Worker
   Handles background polling and push notifications
*/

const CACHE = "f1-2026-v3";
const POLL_INTERVAL = 10 * 60 * 1000; // 10 minutes
const API_BASE = "https://api.jolpi.ca/ergast/f1/2026";

// ── Install & cache shell ────────────────────────────────────────────────────
self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(["./", "./index.html"]))
  );
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(clients.claim());
});

// ── Fetch: cache-first for app shell ────────────────────────────────────────
self.addEventListener("fetch", e => {
  // Don't intercept Jolpica API calls — always fresh
  if (e.request.url.includes("jolpi.ca")) return;
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});

// ── Background sync: periodic result check ──────────────────────────────────
self.addEventListener("periodicsync", e => {
  if (e.tag === "f1-sync") {
    e.waitUntil(checkForNewResults());
  }
});

// Fallback: message from page to trigger a check
self.addEventListener("message", e => {
  if (e.data?.type === "CHECK_RESULTS") {
    checkForNewResults();
  }
});

// ── Core check logic ─────────────────────────────────────────────────────────
async function checkForNewResults() {
  try {
    // Get latest completed round from Jolpica
    const res = await fetch(`${API_BASE}/results.json?limit=5`, { cache: "no-store" });
    if (!res.ok) return;

    const data = await res.json();
    const races = data?.MRData?.RaceTable?.Races ?? [];
    if (!races.length) return;

    const latest = races[races.length - 1];
    const key = `f1_last_round_${latest.round}_${latest.season}`;

    // Check if we've already notified for this round
    const cache = await caches.open(CACHE);
    const seen = await cache.match(key);
    if (seen) return; // already notified

    // New result! Save marker and fire notification
    await cache.put(key, new Response("seen"));

    const winner = latest.Results?.[0];
    const winnerName = winner
      ? `${winner.Driver.givenName} ${winner.Driver.familyName}`
      : "Unknown";
    const team = winner?.Constructor?.name ?? "";

    await self.registration.showNotification(`🏁 F1 Result: ${latest.raceName}`, {
      body: `Winner: ${winnerName} (${team})\nTap to see full results`,
      icon: "./icon-192.png",
      badge: "./icon-192.png",
      tag: `f1-result-r${latest.round}`,
      renotify: true,
      data: { round: latest.round },
      actions: [
        { action: "view", title: "View Results" },
        { action: "dismiss", title: "Dismiss" }
      ]
    });
  } catch (err) {
    console.warn("[SW] checkForNewResults error:", err);
  }
}

// ── Notification click ───────────────────────────────────────────────────────
self.addEventListener("notificationclick", e => {
  e.notification.close();
  if (e.action === "dismiss") return;

  const round = e.notification.data?.round;
  e.waitUntil(
    clients.matchAll({ type: "window" }).then(list => {
      const url = round ? `/?round=${round}` : "/";
      if (list.length) {
        list[0].focus();
        list[0].postMessage({ type: "OPEN_ROUND", round });
      } else {
        clients.openWindow(url);
      }
    })
  );
});
