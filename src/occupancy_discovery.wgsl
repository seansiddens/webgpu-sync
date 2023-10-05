@group(0) @binding(0) var<storage, read_write> count: i32;
@group(0) @binding(1) var<storage, read_write> poll_open: i32;
@group(0) @binding(2) var<storage, read_write> M: array<i32>;
@group(0) @binding(3) var<storage, read_write> next_ticket: atomic<i32>;
@group(0) @binding(4) var<storage, read_write> now_serving: atomic<i32>;

var<workgroup> scratch_pad: i32;

fn lock() {
    // Loop until we acquire a ticket.
    var my_ticket: i32; 
    loop {
        var old = atomicLoad(&next_ticket);
        var res = atomicCompareExchangeWeak(&next_ticket, old, old + 1);
        if (res.exchanged) {
            my_ticket = res.old_value;
            break;
        }
    }

    // Wait until it's our turn.
    loop {
        if (atomicLoad(&now_serving) == my_ticket) {
            break;
        }
    }
}

fn unlock() {
    atomicStore(&now_serving, atomicLoad(&now_serving) + 1);
}

fn occupancy_discovery(workgroup_id: u32) -> bool {
    lock();

    // Polling phase.
    if (poll_open == 0) {
        M[workgroup_id] = count;
        count++;
        unlock();
    } else {
        // Poll is no longer open. Workgroup is not participating.
        unlock();
        return false;
    }

    // // Do some extra work to give other workgroups a chance to participate.
    // for (var i = 0; i < 1024 * 16; i++) {
    //     scratch_pad++;
    // }

    // Closing phase.
    lock();
    if (poll_open == 0) {
        // First workgroup to reach this point closes the poll.
        poll_open = 1;
    }
    unlock();

    return true;
}

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_index) local_id : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
    if (local_id == 0) {
        occupancy_discovery(workgroup_id.x);
    }

    storageBarrier();
}