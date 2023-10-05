@group(0) @binding(0) var<storage, read_write> counter: i32;
@group(0) @binding(1) var<storage, read_write> next_ticket: atomic<i32>;
@group(0) @binding(2) var<storage, read_write> now_serving: atomic<i32>;

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

@compute @workgroup_size(1)
fn main() {
    for (var i = 0; i < 4; i++) {
        lock();

        counter++;

        unlock();
    }
}