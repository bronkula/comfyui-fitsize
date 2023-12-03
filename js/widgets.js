import { app } from '../../scripts/app.js'

// from: https://github.com/melMass/comfy_mtb

export const setupDynamicConnections = (nodeType, prefix, inputType) => {
    const onNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
        console.log('onNodeCreated', `${prefix}_1`)
        this.addInput(`${prefix}_1`, inputType)
        return r
    }

    const onConnectionsChange = nodeType.prototype.onConnectionsChange
    nodeType.prototype.onConnectionsChange = function (
        type,
        index,
        connected,
        link_info
    ) {
        const r = onConnectionsChange
            ? onConnectionsChange.apply(this, arguments)
            : undefined
        dynamic_connection(this, index, connected, `${prefix}_`, inputType)
    }
}

export const dynamic_connection = (
    node,
    index,
    connected,
    connectionPrefix = 'input_',
    connectionType = 'PSDLAYER',
    nameArray = []
) => {
    if (!node.inputs[index].name.startsWith(connectionPrefix)) {
        return
    }
    // remove all non connected inputs
    if (!connected && node.inputs.length > 1) {
        if (node.widgets) {
            const w = node.widgets.find((w) => w.name === node.inputs[index].name)
            if (w) {
                w.onRemoved?.()
                node.widgets.length = node.widgets.length - 1
            }
        }
        node.removeInput(index)

        // make inputs sequential again
        for (let i = 0; i < node.inputs.length; i++) {
            const name =
                i < nameArray.length ? nameArray[i] : `${connectionPrefix}${i + 1}`
            node.inputs[i].label = name
            node.inputs[i].name = name
        }
    }

    // add an extra input
    if (node.inputs[node.inputs.length - 1].link != undefined) {
        const nextIndex = node.inputs.length
        const name =
            nextIndex < nameArray.length
                ? nameArray[nextIndex]
                : `${connectionPrefix}${nextIndex + 1}`

        node.addInput(name, connectionType)
    }
}







app.registerExtension({
    name: "Comfy.Fitsize.PickImageFromBatches",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.startsWith('FS:')) {
            return
        }
        if(nodeData.name === "FS: Pick Image From Batches") {
            setupDynamicConnections(nodeType, 'batches', 'IMAGE')
        }
    }
})