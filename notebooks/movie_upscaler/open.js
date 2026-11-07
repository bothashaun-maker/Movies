import { app } from "../../scripts/app.js";
app.registerExtension({
  name: "MovieStudio.Open",
  setup() {
    const url = new URL(window.location.href);
    if (url.searchParams.get("movie_studio") !== "v1") return;
    void (async () => {
      try {
        const deadline = Date.now() + 60000;
        const canvasReady = () => { try { return !!app.canvas?.canvas; } catch { return false; } };
        while (!app.vueAppReady || !app.positionConversion || !canvasReady() || !window.LiteGraph?.registered_node_types?.MovieStudioChunk || !window.LiteGraph?.registered_node_types?.MovieStudioCommit) {
          if (Date.now() > deadline) throw new Error("ComfyUI did not finish starting");
          await new Promise(r => setTimeout(r, 100));
        }
        await new Promise(r => setTimeout(r, 500));
        const response = await fetch(new URL("default.json", import.meta.url), {cache: "no-store"});
        if (!response.ok) throw new Error("Workflow could not be loaded");
        await app.loadGraphData(await response.json(), true, true, "Movie Studio.json");
        url.searchParams.delete("movie_studio");
        history.replaceState(null, "", url);
      } catch (error) { window.alert(`Movie Studio: ${error.message}`); }
    })();
  }
});
