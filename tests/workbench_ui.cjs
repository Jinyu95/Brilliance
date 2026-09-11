/* Offline browser smoke test. Set BRILLIANCE_PLAYWRIGHT_PACKAGE to an installed
 * Playwright package if it is not on Node's module path. No downloads required. */
const { chromium } = require(process.env.BRILLIANCE_PLAYWRIGHT_PACKAGE || "playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => {
      window.streams = [];
      window.EventSource = class extends EventTarget {
        constructor() { super(); window.streams.push(this); }
        close() {}
      };
    });
    const snapshot = {
      version: "1", project: { id: "test", title: "JuTrack study", status: "researching", runtime: "codex", revision: 1,
        backend: { model: "local-test-model" }, research_network: { enabled: false, local_mcp_enabled: true } },
      messages: [{ id: "u1", role: "user", content: "Study the optics", timestamp: "2026-09-08T00:00:00Z" },
        { id: "a1", role: "assistant", content: "## Computed evidence\n\n| Quantity | Value |\n|---|---:|\n| βx | 12 m |\n\n```python\nprint('JuTrack')\n```", attachments: [
          { type: "image", title: "One cell optics", size: 100, content_sha256: "before", url: "/figures/before.svg" },
          { type: "image", title: "One cell optics", size: 100, content_sha256: "after", url: "/figures/after.svg" },
          { type: "image", title: "One cell optics copy", size: 100, content_sha256: "before", url: "/figures/copy.svg" },
          { type: "evidence_reference", title: "Earlier optics", message_id: "u1" },
          { type: "evidence_summary", title: "Baseline orbit diagnostic", summary: "Measured before correction.",
            scientific_passed: true, request_satisfied: false, limitations: [],
            pending_acceptance: ["orbit_check: numerical acceptance has not yet been evaluated."] },
        ], timestamp: "2026-09-08T00:00:01Z" }],
      live: { state: "running", label: "Computing optics", backend: { model: "local-test-model" }, progress: {}, pending_count: 0 },
      computations: [], session: { can_pause: true, programs: [{ name: "study.py", size: 40 }], activities: [{ id: "t1", title: "run_program", status: "failed", detail: "Example exact error" }] }
    };
    let submitted = "";
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== "brilliance.test") throw new Error("Unexpected network request: " + url);
      const pathname = url.pathname;
      if (pathname.startsWith("/figures/")) return route.fulfill({ body: '<svg xmlns="http://www.w3.org/2000/svg"></svg>', contentType: "image/svg+xml" });
      let json;
      if (pathname === "/api/projects") json = { projects: [snapshot.project] };
      else if (pathname.endsWith("/messages")) { submitted = route.request().postDataJSON().content; json = { action: "queued_for_codex" }; }
      else if (pathname.endsWith("/pause")) json = { detail: "Pause requested. Numerical jobs continue." };
      else if (pathname.endsWith("/programs/study.py")) json = { name: "study.py", source: "# Actual JuTrack research source\n", sha256: "test" };
      else if (pathname === "/api/projects/test") json = snapshot;
      if (json) return route.fulfill({ json });
      const file = pathname.startsWith("/static/") ? pathname.slice(8) : "index.html";
      const body = fs.readFileSync(path.join(__dirname, "../brilliance/web/static", file));
      return route.fulfill({ body, contentType: file.endsWith(".js") ? "application/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto("http://brilliance.test/p/test");
    await page.locator(".message-table").waitFor();
    assert.equal(await page.locator(".message-content pre code").textContent(), "print('JuTrack')");
    assert.equal(await page.locator(".generated-figure").count(), 2);
    assert((await page.locator(".evidence-summary-card").textContent()).includes("Pending final acceptance"));
    assert(!(await page.locator(".evidence-summary-card").textContent()).includes("Limitations"));
    await page.getByRole("button", { name: "Previously shown: Earlier optics" }).click();
    await page.evaluate(() => { window.originalMessage = document.querySelector('[data-message-id="a1"]'); });
    await page.evaluate(s => {
      window.streams.at(-1).dispatchEvent(new MessageEvent("delta", { data: JSON.stringify({ ...s, version: "2", upsert: [], message_order: ["u1", "a1"] }) }));
    }, snapshot);
    assert(await page.evaluate(() => window.originalMessage === document.querySelector('[data-message-id="a1"]')));
    await page.locator("#prompt").fill("Change the study to phase-space tracking");
    await page.locator("#send").click();
    await page.waitForFunction(() => !document.querySelector("#send").disabled || !document.querySelector("#prompt").value);
    await page.locator("#workspace-toggle").click();
    await page.getByRole("button", { name: "study.py", exact: true }).click();
    await page.locator("#program-source").waitFor();
    assert((await page.locator("#program-source").inputValue()).includes("Actual JuTrack"));
    await page.locator("#show-activity").click();
    assert((await page.locator("#activity-log").textContent()).includes("Example exact error"));
    await page.locator("#workspace-close").click();
    await page.locator("#pause-agent").click();
    assert(submitted.includes("phase-space"));
    assert.equal(await page.locator('[data-message-id="a1"]').count(), 1);
    assert.deepEqual(errors, []);
    if (process.env.BRILLIANCE_UI_SCREENSHOT) await page.screenshot({ path: process.env.BRILLIANCE_UI_SCREENSHOT, fullPage: true });
    console.log("PASS: offline browser rendering, incremental DOM, live steering, source viewer, errors, pause");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
