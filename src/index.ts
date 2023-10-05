import ticketLockCode from './ticket_lock.wgsl';
import occupancyDiscoveryCode from './occupancy_discovery.wgsl';
import globalBarrierCode from './global_barrier.wgsl'

const NUM_ITERS = 512;

function coefficientOfVariation(arr: number[]) {
    // Check if array is empty or has a length of 1
    if (arr.length <= 1) {
        throw new Error('Array should have at least two elements for meaningful CV computation.');
    }

    // Calculate mean
    const mean = arr.reduce((sum: any, value: any) => sum + value, 0) / arr.length;

    // Calculate standard deviation
    const variance = arr.reduce((sum: any, value: any) => sum + Math.pow(value - mean, 2), 0) / arr.length;
    const stdDeviation = Math.sqrt(variance);

    // Calculate coefficient of variation
    const cv = (stdDeviation / mean);

    return cv;
}

async function ticketLockTest(device: GPUDevice) {
    // Create buffers
    const numWorkgroups = 1024 * 1;
    console.log("Running the ticket lock test with %d workgroups for %d iterations", 
        numWorkgroups, NUM_ITERS);

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

    const numItersData = new Int32Array(1);
    numItersData[0] = NUM_ITERS;
    const numItersBuf = device.createBuffer({
        size: numItersData.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true
    });
    new Int32Array(numItersBuf.getMappedRange()).set(numItersData);
    numItersBuf.unmap();

    const histogramBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create compute pipeline.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: ticketLockCode,
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
            {binding: 2, resource: {buffer: nowServing}},
            {binding: 3, resource: {buffer: numItersBuf}},
            {binding: 4, resource: {buffer: histogramBuf}}
        ],
    }));
    computePass.dispatchWorkgroups(numWorkgroups);
    computePass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const counterReadBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        counter, // src
        0,
        counterReadBuf, // dst
        0,
        4, // size
    );
    const histReadBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(
        histogramBuf, // src
        0,
        histReadBuf, // dst
        0,
        numWorkgroups * 4, // size
    );

    // Submit the commands.
    const start = performance.now();
    device.queue.submit([commandEncoder.finish()]);

    // Read the result back from the resultBuffer
    device.queue.onSubmittedWorkDone().then(async () => {
        await counterReadBuf.mapAsync(GPUMapMode.READ);
        const resultArray = new Int32Array(counterReadBuf.getMappedRange());
        console.log('Counter: %d', resultArray[0]);
        if (resultArray[0] != numWorkgroups * NUM_ITERS) {
            console.error("Mismatch in ticket lock results!");
        } else {
            console.log('Ticket lock test succeeded!');
        }

        await histReadBuf.mapAsync(GPUMapMode.READ);
        const histArray = new Int32Array(histReadBuf.getMappedRange());
        console.log('Coefficient of variation: %f', coefficientOfVariation(Array.from(histArray)));
        console.log('Elapsed time: %dms\n', performance.now() - start);
        counterReadBuf.unmap();
    });
}

async function occupancyDiscoveryTest(device: GPUDevice) {
    // Create buffers
    const numWorkgroups = 1024;
    // console.log('Starting the occupancy discovery test with %d workgroups', 
    //     numWorkgroups);

    const countBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    const pollOpenBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const MBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const nextTicketBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const nowServingBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    // Create compute pipeline.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: occupancyDiscoveryCode,
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
            {binding: 0, resource: {buffer: countBuf}},
            {binding: 1, resource: {buffer: pollOpenBuf}},
            {binding: 2, resource: {buffer: MBuf}},
            {binding: 3, resource: {buffer: nextTicketBuf}},
            {binding: 4, resource: {buffer: nowServingBuf}},
        ],
    }));
    computePass.dispatchWorkgroups(numWorkgroups);
    computePass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const countReadBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        countBuf, // src
        0,
        countReadBuf, // dst
        0,
        4, // size
    );

    // Submit the commands.
    const start = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read the result back from the resultBuffer
    await countReadBuf.mapAsync(GPUMapMode.READ);
    const occupancy = new Int32Array(countReadBuf.getMappedRange());
    const res = occupancy[0];
    // console.log('Estimated occupancy bound: %d', occupancy[0]);
    // console.log('Elapsed time: %dms\n', performance.now() - start);
    countReadBuf.unmap();
    return res;
}

async function globalBarrierTest(device: GPUDevice) {
    // Create buffers
    const numWorkgroups = 29;
    // console.log('Starting the occupancy discovery test with %d workgroups', 
    //     numWorkgroups);

    const countBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    
    const pollOpenBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const MBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const nextTicketBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const nowServingBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const flagBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const outputBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const numItersBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true
    });
    new Uint32Array(numItersBuf.getMappedRange()).set([NUM_ITERS]);
    numItersBuf.unmap();

    // Create compute pipeline.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: globalBarrierCode,
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
            {binding: 0, resource: {buffer: countBuf}},
            {binding: 1, resource: {buffer: pollOpenBuf}},
            {binding: 2, resource: {buffer: MBuf}},
            {binding: 3, resource: {buffer: nextTicketBuf}},
            {binding: 4, resource: {buffer: nowServingBuf}},
            {binding: 5, resource: {buffer: flagBuf}},
            {binding: 6, resource: {buffer: outputBuf}},
            {binding: 7, resource: {buffer: numItersBuf}},
        ],
    }));
    computePass.dispatchWorkgroups(numWorkgroups);
    computePass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const countReadBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        countBuf, // src
        0,
        countReadBuf, // dst
        0,
        4, // size
    );
    const outputReadBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        outputBuf, // src
        0,
        outputReadBuf, // dst
        0,
        numWorkgroups * 4, // size
    );
    const MReadBuf = device.createBuffer({
        size: numWorkgroups * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        MBuf, // src
        0,
        MReadBuf, // dst
        0,
        numWorkgroups * 4, // size
    );

    // Submit the commands.
    const start = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read the result back from the resultBuffer
    await countReadBuf.mapAsync(GPUMapMode.READ);
    const occupancy = new Int32Array(countReadBuf.getMappedRange());
    const res = occupancy[0];
    console.log('Estimated occupancy bound: %d', occupancy[0]);
    console.log('Elapsed time: %dms\n', performance.now() - start);
    countReadBuf.unmap();
    await outputReadBuf.mapAsync(GPUMapMode.READ);
    const outputArray = Array.from(new Int32Array(outputReadBuf.getMappedRange()));
    outputReadBuf.unmap();
    console.log(outputArray);
    
    await MReadBuf.mapAsync(GPUMapMode.READ);
    const MArray = Array.from(new Int32Array(MReadBuf.getMappedRange()));
    console.log('M Array: ');
    console.log(MArray);
    return res;
}

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

    // await ticketLockTest(device);

    // const numTrials = 256;
    // let results: any[] = []
    // for (let i = 0; i < numTrials; i++) {
    //     let res = await occupancyDiscoveryTest(device);
    //     results.push(res);
    // }
    // let maxOccupancy = Math.max(...results);
    // let average = results.reduce((acc, curr) => acc + curr, 0) / numTrials;
    // let variance = coefficientOfVariation(results);
    // console.log('Max occupancy bound: %f', maxOccupancy);
    // console.log('Average occupancy bound', average);
    // console.log('Occupancy bound coeff. of variation: %f', variance);

    await globalBarrierTest(device);

}

main();
