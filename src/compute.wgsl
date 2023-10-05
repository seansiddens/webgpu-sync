// shader.wgsl content
struct Data {
    numbers: array<f32>,
};

@group(0) @binding(0) var<storage, read> a: Data;
@group(0) @binding(1) var<storage, read> b: Data;
@group(0) @binding(2) var<storage, read_write> result: Data;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    result.numbers[global_id.x] = a.numbers[global_id.x] + b.numbers[global_id.x];
}