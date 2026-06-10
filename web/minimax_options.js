import { app } from "../../scripts/app.js";

const COMBO = {
    "MiniMax-Hailuo-2.3": { "768P": ["6", "10"], "1080P": ["6"] },
    "MiniMax-Hailuo-2.3-Fast": { "768P": ["6", "10"], "1080P": ["6"] },
    "MiniMax-Hailuo-02": { "512P": ["6", "10"], "768P": ["6", "10"], "1080P": ["6"] },
};

app.registerExtension({
    name: "haoee.minimax.dynamicOptions",
    nodeCreated(node) {
        if (node.comfyClass !== "Comfly_HaoeeVideo_MiniMax") return;
        const modelW = node.widgets?.find(w => w.name === "model");
        const resW = node.widgets?.find(w => w.name === "resolution");
        const durW = node.widgets?.find(w => w.name === "duration");
        if (!modelW || !resW || !durW) return;

        const apply = () => {
            const resMap = COMBO[modelW.value] || {};
            const resVals = Object.keys(resMap);
            resW.options.values = resVals;
            if (!resVals.includes(resW.value)) resW.value = resVals[0];

            const durVals = resMap[resW.value] || ["6"];
            durW.options.values = durVals;
            if (!durVals.includes(durW.value)) durW.value = durVals[0];
        };

        for (const w of [modelW, resW]) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                apply();
                return r;
            };
        }
        apply();
    },
});
