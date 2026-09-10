/**
 * Copy Panel Widget: frontend module (anywidget ESM, no build step).
 *
 * Renders a header bar (label + Copy button) above a word-wrapping, monospace,
 * bounded-height scrollable body. The Copy button writes the full body text to
 * the clipboard and shows transient "Copied!" / "Copy failed" feedback.
 */

/** Copy text to the clipboard, preferring the async Clipboard API and falling
 *  back to a hidden-textarea execCommand for non-secure contexts.
 *  Returns a Promise<boolean> indicating success. */
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      // fall through to the legacy path
    }
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (err) {
    ok = false;
  }
  document.body.removeChild(textarea);
  return ok;
}

function render({ model, el }) {
  el.innerHTML = "";
  el.className = "copy-panel-widget";

  // Header bar: label on the left, Copy button on the right.
  const header = document.createElement("div");
  header.className = "copy-panel-header";

  const labelEl = document.createElement("span");
  labelEl.className = "copy-panel-label";
  labelEl.textContent = model.get("label");
  header.appendChild(labelEl);

  const button = document.createElement("button");
  button.className = "copy-panel-button";
  button.type = "button";
  button.textContent = "Copy";
  header.appendChild(button);

  // Body: word-wrapping, monospace, bounded and scrollable.
  const body = document.createElement("pre");
  body.className = "copy-panel-body";
  body.textContent = model.get("text");

  el.appendChild(header);
  el.appendChild(body);

  // Transient copy feedback (purely visual, no Python round-trip).
  let feedbackTimer = null;
  function flashFeedback(message, stateClass) {
    if (feedbackTimer !== null) {
      clearTimeout(feedbackTimer);
    }
    button.textContent = message;
    button.classList.add(stateClass);
    feedbackTimer = setTimeout(() => {
      button.textContent = "Copy";
      button.classList.remove("copy-panel-button--ok", "copy-panel-button--err");
      feedbackTimer = null;
    }, 1500);
  }

  async function onClick() {
    const ok = await copyText(model.get("text"));
    flashFeedback(
      ok ? "Copied!" : "Copy failed",
      ok ? "copy-panel-button--ok" : "copy-panel-button--err",
    );
  }
  button.addEventListener("click", onClick);

  // Teardown: clear any pending feedback timer and detach the click listener.
  // anywidget invokes this when the widget view is destroyed / re-rendered.
  return () => {
    if (feedbackTimer !== null) {
      clearTimeout(feedbackTimer);
    }
    button.removeEventListener("click", onClick);
  };
}

export default { render };
