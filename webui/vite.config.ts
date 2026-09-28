import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// License notice prepended to every script after minification (which strips comments).
const NOTICE = "/*! Lab2Shot — GNU AGPL-3.0-or-later (see LICENSE in the repository). */";

/** Prepends the same notice to every style sheet. */
const cssNotice = (): Plugin => ({
  name: "lab2shot-css-notice",
  enforce: "post",
  generateBundle(_, bundle) {
    for (const f of Object.values(bundle))
      if (f.type === "asset" && f.fileName.endsWith(".css") && typeof f.source === "string") f.source = `${NOTICE}\n${f.source}`;
  },
});

/** Writes a brotli (.br) and a gzip (.gz) copy of every script and style sheet at maximum compression. The server
 * serves the encoding the browser accepts (lab2shot/server/wire.py Assets), which reduces transfer size and avoids
 * per-request compression. Node's zlib is loaded untyped because the page's type check does not include Node types. */
const precompress = (): Plugin => ({
  name: "lab2shot-precompress",
  apply: "build",
  async writeBundle(options, bundle) {
    // Runs on the final written files (after all plugins) and writes the compressed copies alongside them.
    const node = (name: string) => import(/* @vite-ignore */ `node:${name}`);
    const [zlib, fs] = await Promise.all([node("zlib"), node("fs")]);
    for (const name of Object.keys(bundle)) {
      if (!/\.(js|css)$/.test(name)) continue;
      const file = `${options.dir}/${name}`;
      const data = fs.readFileSync(file);
      const br = zlib.brotliCompressSync(data, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11, [zlib.constants.BROTLI_PARAM_SIZE_HINT]: data.length } });
      fs.writeFileSync(`${file}.br`, br);
      fs.writeFileSync(`${file}.gz`, zlib.gzipSync(data, { level: 9 }));
    }
  },
});

export default defineConfig({
  plugins: [react(), cssNotice(), precompress()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
  build: {
    chunkSizeWarningLimit: 2000,
    sourcemap: false, // Source maps are not published: the site is public, the source code is not.
    // The server serves each file according to its access level (lab2shot/server/access.py): the gate's to everyone,
    // the admin page's to administrators, everything else to a logged-in user. It reads the file-to-level mapping
    // from this manifest.
    manifest: true,
    rolldownOptions: { output: { postBanner: NOTICE } },
  },
  // Page workers (e.g. the EXR decoder for local preview): each is an ES module with the license notice prepended.
  worker: { format: "es", rolldownOptions: { output: { postBanner: NOTICE } } },
});
