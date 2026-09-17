/* Read-only, offline rendering of one real saved project's selected result.
 * Usage: node tests/scientific_history_ui.cjs PROJECT_ID OUTPUT.png [PROJECT_ROOT]
 * Uses the configured workspace venv; requires an already installed Playwright
 * package via BRILLIANCE_PLAYWRIGHT_PACKAGE. Never starts research or downloads.
 */
const { chromium } = require(process.env.BRILLIANCE_PLAYWRIGHT_PACKAGE || "playwright");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const digest = file => createHash("sha256").update(fs.readFileSync(file)).digest("hex");

(async () => {
  const projectId = process.argv[2];
  const output = process.argv[3];
  if (!/^project-[a-z0-9]+$/.test(projectId || "") || !output) throw new Error("Specify a saved project ID and screenshot path.");
  const repository = path.resolve(__dirname, "..");
  const projectRoot = path.resolve(repository, process.argv[4] || "research_projects");
  const python = path.join(repository, ".venv", "Scripts", "python.exe");
  // Do not invoke snapshot/jobs.active: its worker-liveness reconciliation may
  // write job state. This replay only uses the actual presentation assembler.
  const source = "import json,sys; from brilliance.chat_service import ResearchChatService; s=ResearchChatService(sys.argv[2]); p=s.store.load(sys.argv[1]); j=s.jobs.list(p.id); print(json.dumps({'project':s._project_meta(p),'messages':s._messages(p,j[-1] if j else None),'live':None,'computations':[],'session':{},'version':'offline-replay'},ensure_ascii=False))";
  const snapshot = JSON.parse(execFileSync(python, ["-X", "utf8", "-c", source, projectId, projectRoot], { cwd: repository, encoding: "utf8", maxBuffer: 32 * 1024 * 1024 }));
  const final = [...snapshot.messages].reverse().find(message => message.attachments?.some(item => item.presentation_role === "selected_evidence"));
  assert(final, "A committed selected-result receipt is required.");
  const expectedFigures = final.attachments.filter(item => item.type === "image").length;
  const proof = final.meta?.completion_evidence;
  assert(proof?.evidence_run_id, "The rendered selection needs an exact completion proof.");
  const summary = final.attachments.find(item => item.type === "evidence_summary" && item.presentation_role === "selected_evidence");
  assert(summary?.lattice_id, "Selected result needs candidate identity.");
  assert(expectedFigures > 0, "A release candidate needs an actually loaded plot.");
  const expectedTables = (summary.tables || []).filter(table => table.rows?.length).length;
  assert(expectedTables > 0, "A release candidate needs real numerical tables.");
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => { window.EventSource = class extends EventTarget { close() {} }; });
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      assert.equal(url.hostname, "brilliance.test", "All browser requests must stay inside the local fixture.");
      const pathname = decodeURIComponent(url.pathname);
      if (pathname === "/api/projects") return route.fulfill({ json: { projects: [snapshot.project] } });
      if (pathname === `/api/projects/${projectId}`) return route.fulfill({ json: snapshot });
      const prefix = `/api/projects/${projectId}/workspace-artifacts/`;
      if (pathname.startsWith(prefix)) {
        const artifactRoot = path.join(projectRoot, projectId, "workspace", "artifacts");
        const artifact = path.resolve(artifactRoot, pathname.slice(prefix.length));
        assert(artifact.startsWith(artifactRoot + path.sep), "Artifact must belong to the selected project.");
        return route.fulfill({ body: fs.readFileSync(artifact), contentType: artifact.endsWith(".svg") ? "image/svg+xml" : "image/png" });
      }
      const file = pathname.startsWith("/static/") ? pathname.slice(8) : "index.html";
      assert(!file.includes(".."));
      return route.fulfill({ body: fs.readFileSync(path.join(repository, "brilliance/web/static", file)),
        contentType: file.endsWith(".js") ? "application/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto(`http://brilliance.test/p/${projectId}`);
    const article = page.locator(`[data-message-id="${final.id}"]`);
    await article.locator(".selected-evidence-label").waitFor();
    assert.equal(await article.locator(".generated-figure").count(), expectedFigures);
    await article.locator(".generated-figure img").evaluateAll(images => Promise.all(images.map(image => image.decode())));
    assert.equal(await article.locator(".alternative-results[open]").count(), 0);
    assert.deepEqual(errors, []);
    assert(await article.locator("table").count() >= expectedTables, "Each selected numerical table must render as a real table.");
    const imageHashes = final.attachments.filter(item => item.type === "image").map(item => {
      const url = new URL(item.url, "http://brilliance.test");
      const relative = decodeURIComponent(url.pathname).split("/workspace-artifacts/")[1];
      assert(relative, "Selected images must be project artifacts.");
      const artifactRoot = path.resolve(projectRoot, projectId, "workspace/artifacts");
      const file = path.resolve(artifactRoot, relative);
      assert(file.startsWith(artifactRoot + path.sep));
      return digest(file);
    });
    assert.equal(new Set(imageHashes).size, expectedFigures, "Do not show duplicate image bytes in the selected answer.");
    // Hide the fixed page controls in the screenshot so they cannot occlude a
    // long scientific result. The rendered conversation remains unchanged.
    await page.addStyleTag({ content: ".composer-dock, .topbar { visibility: hidden !important; }" });
    await article.screenshot({ path: output });
    const resultPath = path.join(projectRoot, projectId, "workspace/artifacts/programs", proof.evidence_run_id, "result.json");
    const receipt = {
      schema: "brilliance_rendered_selection/v1", project_directory: path.resolve(projectRoot, projectId),
      candidate_id: summary.lattice_id, run_id: proof.evidence_run_id, result_sha256: digest(resultPath),
      assets: Object.fromEntries(["app.js", "styles.css", "workbench.css", "index.html"].map(name => [name, digest(path.join(repository, "brilliance/web/static", name))])),
      checks: { exact_selected_result: true, images_loaded: true, real_tables_rendered: true,
        duplicate_images_absent: true, browser_errors_absent: true, external_requests_absent: true },
      figures: expectedFigures, tables: expectedTables, screenshot: path.resolve(output), screenshot_sha256: digest(output),
      limitations: ["Offline rendering replay verifies delivery, not live SSE timing or the scientific correctness of narrative/plot scope."]
    };
    fs.writeFileSync(output + ".json", JSON.stringify(receipt, null, 2) + "\n");
    console.log(`PASS: exact selected result rendered with ${expectedFigures} figures; history read-only. Screenshot: ${output}`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
