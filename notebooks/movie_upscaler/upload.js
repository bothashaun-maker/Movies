import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function request(route, options) {
  const response = await api.fetchApi(route, options);
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || `Upload failed (${response.status})`);
  return result;
}

app.registerExtension({
  name: "MovieStudio.Upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MovieStudioUpload") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      const node = this;
      let busy = false;
      const status = document.createElement("div");
      status.textContent = "Choose a clip from your computer. Then press Run.";
      Object.assign(status.style, {color: "#ddd", font: "13px sans-serif", padding: "8px", whiteSpace: "normal", overflowWrap: "anywhere"});
      const button = this.addWidget("button", "Upload clip", null, () => {
        if (busy) return;
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".mp4,.mkv,.mov,.webm,.avi,.m4v";
        picker.addEventListener("change", async () => {
          const file = picker.files?.[0];
          if (!file) return;
          busy = true;
          button.name = "Uploading…";
          let upload;
          try {
            upload = await request("/movie-studio/upload/start", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name: file.name, size: file.size})});
            let offset = 0;
            while (offset < file.size) {
              const end = Math.min(offset + upload.chunk_bytes, file.size);
              const result = await request(`/movie-studio/upload/${upload.id}/chunk`, {method: "POST", headers: {"Content-Type": "application/octet-stream", "X-Upload-Offset": String(offset)}, body: file.slice(offset, end)});
              if (result.offset !== end) throw new Error("Upload byte count did not match.");
              offset = result.offset;
              status.textContent = `${file.name} — ${Math.round(offset / file.size * 100)}% uploaded`;
            }
            status.textContent = "Checking uploaded clip…";
            const result = await request(`/movie-studio/upload/${upload.id}/finish`, {method: "POST"});
            const path = node.widgets.find(widget => widget.name === "source_path");
            path.value = result.path;
            path.callback?.(result.path);
            status.textContent = `Ready: ${result.name}. Press Run to upscale.`;
            app.graph?.setDirtyCanvas(true, true);
          } catch (error) {
            status.textContent = `Upload failed: ${error.message}. Choose the clip again to retry.`;
            if (upload) {
              try { await api.fetchApi(`/movie-studio/upload/${upload.id}`, {method: "DELETE"}); } catch {}
            }
          } finally {
            busy = false;
            button.name = "Upload clip";
          }
        }, {once: true});
        picker.click();
      }, {serialize: false});
      this.addDOMWidget("upload_status", "status", status, {serialize: false});
      this.setSize([420, 240]);
    };
  }
});
