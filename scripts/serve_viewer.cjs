#!/usr/bin/env node
// Optional localhost-only preview. Serves the viewer and exactly one selected trace.
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const args = process.argv.slice(2);
const arg = (key, fallback) =>
  args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const port = Number(arg("--port", "8765"));
const root = path.resolve(__dirname, "../viewer");
const trace = arg("--trace", null);
const selected = trace ? path.resolve(trace) : null;
if (selected && !fs.statSync(selected).isFile())
  throw new Error("Trace is not a file");
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
};
http
  .createServer((req, res) => {
    try {
      if (req.method !== "GET" && req.method !== "HEAD") {
        res.writeHead(405);
        return res.end();
      }
      const pathname = decodeURIComponent(
        new URL(req.url, "http://localhost").pathname,
      );
      let file;
      if (pathname === "/trace.jsonl" && selected) file = selected;
      else if (pathname === "/summary.json" && selected)
        file = path.join(path.dirname(selected), "summary.json");
      else {
        file = path.resolve(
          root,
          "." + (pathname === "/" ? "/index.html" : pathname),
        );
        if (!file.startsWith(root + path.sep)) {
          res.writeHead(403);
          return res.end();
        }
      }
      if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
        res.writeHead(404);
        return res.end("Not found");
      }
      res.writeHead(200, {
        "Content-Type": types[path.extname(file)] || "application/octet-stream",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
      });
      if (req.method === "HEAD") return res.end();
      const stream = fs.createReadStream(file);
      stream.on("error", () => res.destroy());
      stream.pipe(res);
    } catch (_) {
      res.writeHead(400);
      res.end("Invalid request");
    }
  })
  .listen(port, "127.0.0.1", () =>
    console.log(
      `Viewer: http://127.0.0.1:${port}/${selected ? "?trace=/trace.jsonl&summary=/summary.json" : ""}`,
    ),
  );
