import { app } from "../../scripts/app.js";

const SIZE_SUPPORT = {
    "doubao-seedream-5-0-260128": ["2K", "3K", "4K"],
    "doubao-seedream-4-5-251128": ["2K", "4K"],
    "doubao-seedream-4-0-250828": ["1K", "2K", "4K"],
};

app.registerExtension({
    name: "haoee.seedream.dynamicSize",
    nodeCreated(node) {
        if (node.comfyClass !== "Comfly_HaoeeImage_Doubao_Seedream") return;
        const modelW = node.widgets?.find(w => w.name === "model");
        const sizeW = node.widgets?.find(w => w.name === "size");
        if (!modelW || !sizeW) return;

        const apply = () => {
            const values = SIZE_SUPPORT[modelW.value] || ["2K"];
            sizeW.options.values = values;
            if (!values.includes(sizeW.value)) sizeW.value = values[0];
        };
        const orig = modelW.callback;
        modelW.callback = function (...args) {
            const r = orig?.apply(this, args);
            apply();
            return r;
        };
        apply();
    },
});
