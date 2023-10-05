import computeShaderCode from './ticket_lock.wgsl';

async function main() {
    const canvas = document.querySelector('canvas')!;
    const context = canvas.getContext('webgpu');

    if (!context) {
        console.error("WebGPU is not supported!");
        throw new Error("WebGPU is not supported!");
    }

    console.log("Successfully created WebGPU context!");

    // Initial WebGPU setup
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error('WebGPU not supported.');
    }
    const device = await adapter.requestDevice();
    console.log('Max workgroups: %d', device.limits.maxComputeWorkgroupsPerDimension);
    console.log('Max workgroup size: %d', device.limits.maxComputeWorkgroupSizeX);

    // Create buffers
    const numWorkgroups = 64;
    const counter = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const nowServing = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE
    });
    const nextTicket = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE
    });

    // Create compute pipeline.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: computeShaderCode,
            }),
            entryPoint: 'main',
        }
    });

    // Queue commands.
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: counter}},
            {binding: 1, resource: {buffer: nextTicket}},
            {binding: 2, resource: {buffer: nowServing}}
        ],
    }));
    computePass.dispatchWorkgroups(numWorkgroups);
    computePass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        counter, // src
        0,
        gpuReadBuffer, // dst
        0,
        4, // size
    );

    // Submit the commands.
    device.queue.submit([commandEncoder.finish()]);

    // Read the result back from the resultBuffer
    device.queue.onSubmittedWorkDone().then(async () => {
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Int32Array(gpuReadBuffer.getMappedRange());
        console.log(resultArray)
        gpuReadBuffer.unmap();
    });
}

main();
