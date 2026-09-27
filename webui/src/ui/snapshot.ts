import { MessageError } from "../messages/message";

/** A picture of the page as it looks now, taken inside the page (no screen-sharing permission to ask for): the page
 * is copied into an SVG with the page's own style sheets, its canvases (the node graph's minimap, the viewers) as
 * pictures, then drawn into a PNG. What the browser does not allow to copy (a 3D view that does not keep its last
 * frame, pictures from elsewhere) comes out empty; the user sees the picture before sending it, and can attach their
 * own instead. `skip`: elements left out (the feedback dialog itself). */

const MAX_SIDE = 2400;

function styles(): string {
  let css = "";
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      css += Array.from(sheet.cssRules, (r) => r.cssText).join("\n");
    } catch {
      /* a style sheet from elsewhere: not readable, left out */
    }
  }
  return css;
}

function pictureOf(el: HTMLCanvasElement | HTMLImageElement): string {
  try {
    if (el instanceof HTMLCanvasElement) return el.toDataURL("image/png");
    if (!el.complete || !el.naturalWidth) return "";
    const c = document.createElement("canvas");
    c.width = el.naturalWidth;
    c.height = el.naturalHeight;
    c.getContext("2d")!.drawImage(el, 0, 0);
    return c.toDataURL("image/png");
  } catch {
    return ""; // tainted (from elsewhere)
  }
}

export async function snapshotPage(skip: string): Promise<Blob | null> {
  const width = innerWidth;
  const height = innerHeight;
  const body = document.body.cloneNode(true) as HTMLElement;

  // what cloning loses: form values, canvases' pixels, pictures (an SVG image loads nothing from outside); the copy
  // is exact, so its elements come in the same order as the page's
  const which = "input, textarea, select, canvas, img";
  const copies = body.querySelectorAll(which);
  document.body.querySelectorAll(which).forEach((src, i) => {
    const dst = copies[i];
    if (src instanceof HTMLInputElement) {
      if (src.type === "checkbox" || src.type === "radio") (src.checked ? dst.setAttribute("checked", "") : dst.removeAttribute("checked"));
      else dst.setAttribute("value", src.value);
    } else if (src instanceof HTMLTextAreaElement) dst.textContent = src.value;
    else if (src instanceof HTMLSelectElement) dst.querySelectorAll("option").forEach((o, j) => (j === src.selectedIndex ? o.setAttribute("selected", "") : o.removeAttribute("selected")));
    else if ((src instanceof HTMLCanvasElement || src instanceof HTMLImageElement) && !src.closest(skip)) {
      const img = document.createElement("img");
      const r = src.getBoundingClientRect();
      img.src = pictureOf(src);
      img.className = src.className;
      img.setAttribute("style", `${src.getAttribute("style") ?? ""};width:${r.width}px;height:${r.height}px`);
      dst.replaceWith(img);
    }
  });
  body.querySelectorAll(`${skip}, script, noscript`).forEach((el) => el.remove());
  const style = document.createElement("style");
  style.textContent = styles();
  body.prepend(style);
  body.setAttribute("style", `${document.body.getAttribute("style") ?? ""};margin:0;width:${width}px;height:${height}px;overflow:hidden`);

  const html = new XMLSerializer().serializeToString(body); // XHTML: it names its namespace itself
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" class="${document.documentElement.className}">` +
    `<foreignObject x="0" y="0" width="100%" height="100%">${html}</foreignObject></svg>`;
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`; // (a blob: URL would taint the canvas)
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new MessageError("E-SNAPSHOT-FAILED"));
      img.src = url;
    });
    const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = getComputedStyle(document.body).backgroundColor || "#1c1c1e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  } catch {
    return null;
  }
}
