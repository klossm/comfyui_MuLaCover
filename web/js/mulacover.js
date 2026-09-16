import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

async function uploadMIDI(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("subfolder", "");
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status === 200 || resp.status === 201) {
        return await resp.json();
    }
    throw new Error(`${resp.status}: ${await resp.text()}`);
}

app.registerExtension({
    name: "MuLaCover.UploadMIDI",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MuLaCoverLoadMIDI") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const combo = this.widgets?.find(w => w.name === "midi_file");
            if (!combo) return r;

            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = ".mid,.midi";
            fileInput.style.display = "none";
            document.body.appendChild(fileInput);

            fileInput.onchange = async () => {
                if (!fileInput.files?.length) return;
                try {
                    const data = await uploadMIDI(fileInput.files[0]);
                    const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                    if (!combo.options.values.includes(name)) combo.options.values.push(name);
                    combo.value = name;
                    app.graph.setDirtyCanvas(true, true);
                } catch (err) {
                    alert(`[MuLaCover] MIDI 上传失败: ${err.message}`);
                }
                fileInput.value = "";
            };

            this.addWidget("button", "📁 上传 MIDI 文件", null, () => fileInput.click());
            return r;
        };
    },
});
