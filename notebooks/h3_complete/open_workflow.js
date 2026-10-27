import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "H3Kaggle.OpenFullWorkflow",
    setup() {
        const url = new URL(window.location.href);
        if (url.searchParams.get("h3_workflow") !== "full-v1") return;
        const open = async () => {
        try {
            // Let ComfyUI finish mounting and restoring its previous tab first.
            const deadline = Date.now() + 60000;
            while (!app.vueAppReady || !app.positionConversion) {
                if (Date.now() > deadline) throw new Error("ComfyUI did not finish starting.");
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            await new Promise(resolve => setTimeout(resolve, 500));
            const response = await fetch(new URL("full-workflow.json", import.meta.url), { cache: "no-store" });
            if (!response.ok) throw new Error(`Workflow download failed: ${response.status}`);
            const graph = await response.json();
            if (!([16, 17].includes(graph.nodes?.length)) || !([17, 19].includes(graph.links?.length)) ||
                !graph.nodes.some(node => node.type === "H3IsolatedSaveVideo")) {
                throw new Error("The installed full H3 workflow is incomplete.");
            }
            await app.loadGraphData(graph, true, true, (graph.extra?.h3_title || "MiniMax H3 - Full T4 Workflow") + ".json");
            // Refreshes must preserve subsequent edits rather than reload the template.
            url.searchParams.delete("h3_workflow");
            window.history.replaceState(null, "", url);
        } catch (error) {
            console.error("Could not open the full H3 workflow", error);
            window.alert(`Could not open the full H3 workflow: ${error.message}`);
        }
        };
        // Do not block ComfyUI setup while waiting for its mounted state.
        void open();
    }
});
