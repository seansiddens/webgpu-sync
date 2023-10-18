@group(0) @binding(0) var<storage, read_write> count: i32;
@group(0) @binding(1) var<storage, read_write> poll_open: i32;
@group(0) @binding(2) var<storage, read_write> M: array<i32>;
@group(0) @binding(3) var<storage, read_write> next_ticket: atomic<i32>;
@group(0) @binding(4) var<storage, read_write> now_serving: atomic<i32>;
@group(0) @binding(5) var<storage, read_write> flag: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> output_buf: array<i32>;
@group(0) @binding(7) var<storage, read> num_iters: u32;

var<workgroup> participating: bool;
var<workgroup> num_arrived: atomic<i32>; 
var<workgroup> released: atomic<i32>;

const WORKGROUP_SIZE: u32 = 1;

// Occupancy-Bound Execution Environment
var<private> p_num_groups: u32; 
var<private> p_group_id: i32;   
// var<private> p_global_id: u32;
// var<private> p_global_size: u32;

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
        M[workgroup_id] = -1;
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



@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_index) local_id : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
    if (local_id == 0) {
        participating = occupancy_discovery(workgroup_id.x);
    }

    storageBarrier();

    // Workgroups found to not be participating immediately exit. 
    if (!participating) {
        return;
    }

    // Participating workgroups continue with kernel computation. 
    // From here we can assume fair scheduling of workgroups.    
    // Here we can set up the execution environment variables.
    p_num_groups = u32(count);
    p_group_id = M[workgroup_id.x];
    // p_global_id = (p_group_id * WORKGROUP_SIZE) + local_id;
    // p_global_size = u32(count) * WORKGROUP_SIZE;

    const use_barrier: bool = true;
    
    if (!use_barrier) {
        return;
    }

    for (var i = 0; i < i32(num_iters); i++) {
        // Each workgroup will write to a location in a global buffer, offset each
        // iteration. A safe and correct barrier is necessary to ensure that there 
        // are no data races and every iteration each workgroup is writing to it's 
        // own memory location.
        if (local_id == 0) {
            output_buf[(p_group_id + i) % i32(p_num_groups)]++;
        }

        if (!use_barrier) {
            continue;
        }

        if (p_group_id == 0) {
            if (local_id == 0) {
                // Representative thread in controller group waits for all follower groups 
                // to arrive at the barrier.
                for (var j = 1; j < i32(p_num_groups); j++) {
                    while(atomicLoad(&flag[j]) == 0) {}
                    atomicAdd(&num_arrived, 1);
                }
            }

            // Wait for all follower groups to arrive.
            while (atomicLoad(&num_arrived) < i32(p_num_groups) - 1) {};

            if (local_id == 0) {
                atomicStore(&num_arrived, 0); // Reset counter.
                for (var j = 1; j < i32(p_num_groups); j++) {
                    // Relase follower workgroups from the barrier.
                    atomicStore(&flag[j], 0);
                }
            }
        } else if (participating) {
            atomicStore(&released, 0);
            if (local_id == 0) {
                // Update flag to signal arrival to the barrier.
                atomicStore(&flag[p_group_id], 1);
                while (atomicLoad(&flag[p_group_id]) == 1) {};
                atomicStore(&released, 1);
            }

            while (atomicLoad(&released) == 0) {}; // Sync follower threads before continuing.
        }
        // END GLOBAL BARRIER -----------------------------------------------------------
    }
}