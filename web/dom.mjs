// Tiny DOM builders. Report cells are untrusted text (tensor names come from a downloaded model),
// so nothing here ever parses a string as markup: children become text nodes.

const SVG_NS = 'http://www.w3.org/2000/svg';

function build(el, props, children) {
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (key === 'style') Object.assign(el.style, value); // CSSOM, so the page needs no inline-style CSP exception
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  return append(el, ...children);
}

/** Append nodes and strings, flattening arrays and skipping null and false, so callers can write `cond && node`. */
export function append(el, ...children) {
  el.append(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return el;
}

export function h(tag, props, ...children) {
  return build(document.createElement(tag), props, children);
}

export function svg(tag, props, ...children) {
  return build(document.createElementNS(SVG_NS, tag), props, children);
}
