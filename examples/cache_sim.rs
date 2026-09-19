//! Replay a routing trace (`JOSHUA_ROUTE_TRACE=trace.csv joshua …`) against
//! the expert-cache policies and print the hit rates, plus the two-tier
//! (VRAM pool + host page cache) disk-read estimate.
//!
//!   cargo run --release --example cache_sim -- trace.csv \
//!       --slots 2067 --host-slots 7400 --hot-share 3/4 --expert-mib 6.75
//!
//! `--slots` is the device budget in experts (the load log prints it);
//! `--host-slots` is the RAM left for experts divided by the expert size
//! (64 GiB host − dense set − OS ≈ 50 GiB → ~7,400 experts of 6.75 MiB).

use joshua::cache_sim::{simulate, simulate_two_tier, Config, Policy, Trace, TwoTierConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut path: Option<String> = None;
    let mut slots = 2067usize;
    let mut host_slots: Option<usize> = None;
    let mut hot_share = (3usize, 4usize);
    let mut expert_mib = 6.75f64;
    let mut prefill_inserts = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--slots" => slots = args.next().expect("--slots N").parse().expect("slots"),
            "--host-slots" => {
                host_slots = Some(
                    args.next()
                        .expect("--host-slots N")
                        .parse()
                        .expect("host slots"),
                )
            }
            "--hot-share" => {
                let v = args.next().expect("--hot-share n/d");
                let (n, d) = v.split_once('/').expect("n/d");
                hot_share = (n.parse().expect("n"), d.parse().expect("d"));
                if hot_share.1 == 0 || hot_share.0 > hot_share.1 {
                    eprintln!("--hot-share must be a fraction n/d with 0 < d and n <= d");
                    std::process::exit(2);
                }
            }
            "--expert-mib" => {
                expert_mib = args.next().expect("--expert-mib F").parse().expect("MiB")
            }
            "--prefill-inserts" => prefill_inserts = true,
            other if path.is_none() => path = Some(other.to_string()),
            other => panic!("unexpected argument {other}"),
        }
    }
    let path = path.unwrap_or_else(|| {
        eprintln!("usage: cache_sim <trace.csv> [--slots N] [--host-slots M] [--hot-share n/d] [--expert-mib F] [--prefill-inserts]");
        std::process::exit(2);
    });
    let trace = Trace::from_path(std::path::Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let steps = trace.decode_steps();
    println!(
        "trace: {} calls, {steps} decode steps, {} distinct (layer, expert) keys",
        trace.calls.len(),
        trace.distinct_keys()
    );
    let cfg = Config {
        slots,
        prefill_inserts,
    };
    println!(
        "\ndevice pool of {slots} slots ({:.2} GiB):",
        slots as f64 * expert_mib / 1024.0
    );
    println!(
        "{:<20} {:>10} {:>10} {:>9} {:>10} {:>10} {:>8}",
        "policy", "visits", "hits", "hit%", "uploads", "evictions", "refused"
    );
    for policy in [
        Policy::Lru,
        Policy::LruHot { share: hot_share },
        Policy::StaticTop,
        Policy::Belady,
    ] {
        let r = simulate(&trace, policy, &cfg);
        println!(
            "{:<20} {:>10} {:>10} {:>8.1}% {:>10} {:>10} {:>8}",
            r.policy,
            r.decode_visits,
            r.decode_hits,
            100.0 * r.decode_hit_rate(),
            r.uploads,
            r.evictions,
            r.refused
        );
    }
    if let Some(host) = host_slots {
        println!(
            "\ntwo tiers: {slots} device slots + {host} host slots ({:.1} GiB of RAM), per decode step:",
            host as f64 * expert_mib / 1024.0
        );
        println!(
            "{:<10} {:>9} {:>9} {:>9} {:>12} {:>14}",
            "tiers", "device%", "ram%", "disk%", "disk reads", "disk MiB/step"
        );
        for exclusive in [true, false] {
            let r = simulate_two_tier(
                &trace,
                &TwoTierConfig {
                    device_slots: slots,
                    host_slots: host,
                    hot_share,
                    exclusive,
                },
            );
            let v = r.decode_visits.max(1) as f64;
            println!(
                "{:<10} {:>8.1}% {:>8.1}% {:>8.1}% {:>12.1} {:>14.1}",
                if exclusive { "exclusive" } else { "inclusive" },
                100.0 * r.decode_device_hits as f64 / v,
                100.0 * r.decode_host_hits as f64 / v,
                100.0 * r.decode_disk as f64 / v,
                r.disk_reads_per_decode_step(),
                r.disk_reads_per_decode_step() * expert_mib,
            );
        }
    }
}
