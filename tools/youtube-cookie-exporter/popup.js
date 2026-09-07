// YouTube Cookie Exporter for yt-dlp & Antigravity Pipeline
// Formats cookies in standard Netscape format: domain, flag, path, secure, expiration, name, value

let cachedNetscapeData = "";

function toNetscapeFormat(cookies) {
  const header = [
    "# Netscape HTTP Cookie File",
    "# http://curl.haxx.se/rfc/cookie_spec.html",
    "# Exported by YouTube Cookie Exporter for yt-dlp & Antigravity Clipper",
    ""
  ].join("\n");

  const lines = cookies.map((c) => {
    const domain = c.domain || ".youtube.com";
    const flag = domain.startsWith(".") ? "TRUE" : "FALSE";
    const path = c.path || "/";
    const secure = c.secure ? "TRUE" : "FALSE";
    const expiration = c.expirationDate ? Math.round(c.expirationDate) : 0;
    const name = c.name;
    const value = c.value || "";

    return [domain, flag, path, secure, expiration, name, value].join("\t");
  });

  return header + "\n" + lines.join("\n") + "\n";
}

async function loadCookies() {
  const sessionStatusEl = document.getElementById("session-status");
  const countEl = document.getElementById("cookie-count");

  try {
    const ytCookies = await chrome.cookies.getAll({ domain: "youtube.com" });
    let googleCookies = [];
    try {
      googleCookies = await chrome.cookies.getAll({ domain: "google.com" });
    } catch (_) {}

    const authNames = new Set([
      "SAPISID", "APISID", "SSID", "HSID", "SID",
      "__Secure-1PSID", "__Secure-3PSID", "__Secure-1PAPISID", "__Secure-3PAPISID",
      "LOGIN_INFO", "PREF", "VISITOR_INFO1_LIVE", "YSC"
    ]);
    const relevantGoogle = (googleCookies || []).filter((c) => authNames.has(c.name));

    const cookies = [...(ytCookies || [])];
    const existing = new Set(cookies.map((c) => `${c.domain}:${c.name}`));
    for (const gc of relevantGoogle) {
      if (!existing.has(`${gc.domain}:${gc.name}`)) {
        cookies.push(gc);
      }
    }

    if (!cookies || cookies.length === 0) {
      sessionStatusEl.innerHTML = '<span class="badge-warn">⚠️ No cookies found</span>';
      countEl.textContent = "0";
      return;
    }

    countEl.textContent = cookies.length.toString();

    // Check if user is logged into YouTube (has LOGIN_INFO, SID, or __Secure-3PSID)
    const hasAuth = cookies.some((c) =>
      ["LOGIN_INFO", "SID", "SSID", "__Secure-3PSID", "__Secure-1PSID", "SAPISID"].includes(c.name)
    );

    if (hasAuth) {
      sessionStatusEl.innerHTML = '<span class="badge-ok">✅ Active & Logged In</span>';
    } else {
      sessionStatusEl.innerHTML = '<span class="badge-warn">⚠️ Guest Session (Login recommended)</span>';
    }

    cachedNetscapeData = toNetscapeFormat(cookies);
  } catch (err) {
    sessionStatusEl.innerHTML = `<span class="badge-warn">Error: ${err.message}</span>`;
  }
}

document.getElementById("btn-copy").addEventListener("click", async () => {
  if (!cachedNetscapeData) {
    await loadCookies();
  }

  if (!cachedNetscapeData) {
    alert("No YouTube cookies found. Please visit youtube.com and log in first!");
    return;
  }

  try {
    await navigator.clipboard.writeText(cachedNetscapeData);
    const btn = document.getElementById("btn-copy");
    const originalText = btn.innerHTML;

    btn.classList.add("btn-success");
    btn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
      Copied! Paste into YOUTUBE_COOKIES
    `;

    setTimeout(() => {
      btn.classList.remove("btn-success");
      btn.innerHTML = originalText;
    }, 3000);
  } catch (err) {
    alert("Failed to copy to clipboard: " + err);
  }
});

document.getElementById("btn-download").addEventListener("click", async () => {
  if (!cachedNetscapeData) {
    await loadCookies();
  }

  if (!cachedNetscapeData) {
    alert("No YouTube cookies found. Please visit youtube.com and log in first!");
    return;
  }

  const blob = new Blob([cachedNetscapeData], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "youtube_cookies.txt";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// Load on popup open
document.addEventListener("DOMContentLoaded", loadCookies);
