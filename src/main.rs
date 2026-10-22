//! Command line: the benchmark corpus and the quality report.
//!
//! ```text
//! pedroso-mesher parts
//! pedroso-mesher quality mesh.json [--json]
//! ```

use std::process::ExitCode;

use pedroso_mesher::bench::PARTS;
use pedroso_mesher::io::read_mesh_json;
use pedroso_mesher::quality::{report, Verdict};

const USAGE: &str = "\
pedroso-mesher — unstructured quad/hex meshing (P0: model, quality, corpus)

USAGE:
    pedroso-mesher parts
    pedroso-mesher quality <mesh.json> [--json]

COMMANDS:
    parts      list the benchmark corpus and the bar each part must clear
    quality    read a mesh, validate it, and report its quality
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("parts") => {
            print_parts();
            ExitCode::SUCCESS
        }
        Some("quality") => {
            let path = match args.get(1) {
                Some(p) => p,
                None => {
                    eprintln!("error: quality needs a mesh file\n\n{USAGE}");
                    return ExitCode::from(2);
                }
            };
            quality(path, args.iter().any(|a| a == "--json"))
        }
        Some("-h" | "--help" | "help") | None => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("error: unknown command {other:?}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn print_parts() {
    println!("{:<18}{:>3}  {:<10}  requirement", "id", "dim", "min SJ");
    println!("{}", "-".repeat(96));
    for part in PARTS {
        println!(
            "{:<18}{:>3}  {:<10.2}  {}",
            part.id, part.dim, part.min_scaled_jacobian, part.requirement
        );
    }
    let n2 = PARTS.iter().filter(|p| p.dim == 2).count();
    println!(
        "\n{} parts: {} planar (P1 gate), {} solid (P4 acceptance: 6 of 8 at 0.1)",
        PARTS.len(),
        n2,
        PARTS.len() - n2
    );
}

/// Non-finite numbers have no JSON representation; report them as null.
fn finite_or_null(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::Value::from(value)
    } else {
        serde_json::Value::Null
    }
}

fn quality(path: &str, as_json: bool) -> ExitCode {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(error) => {
            eprintln!("error: cannot read {path}: {error}");
            return ExitCode::from(2);
        }
    };
    let mesh = match read_mesh_json(&text) {
        Ok(m) => m,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let result = report(&mesh);

    if as_json {
        let payload = serde_json::json!({
            "elements": result.elements,
            "min_scaled_jacobian": result.min_scaled_jacobian,
            "mean_scaled_jacobian": result.mean_scaled_jacobian,
            "max_scaled_jacobian": result.max_scaled_jacobian,
            "worst_element": result.worst_element,
            "inverted": result.inverted,
            "below_target": result.below_target,
            "min_dihedral_deg": finite_or_null(result.min_dihedral_deg),
            "max_dihedral_deg": finite_or_null(result.max_dihedral_deg),
            "max_aspect_ratio": result.max_aspect_ratio,
            "verdict": match result.verdict {
                Verdict::Acceptable => "acceptable",
                Verdict::BelowTarget => "below_target",
                Verdict::Refused => "refused",
            },
            "reasons": result.reasons,
            "topology": {
                "boundary": mesh.topology().boundary,
                "interior": mesh.topology().interior,
                "non_manifold": mesh.topology().non_manifold,
                "inconsistent": mesh.topology().inconsistent,
            },
            "volume": mesh.volume(),
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("error: cannot serialise the report: {error}");
                return ExitCode::from(2);
            }
        }
    } else {
        println!("{}", result.summary());
        for reason in &result.reasons {
            println!("  - {reason}");
        }
        let topo = mesh.topology();
        println!(
            "  topology: {} boundary, {} interior, {} non-manifold, {} inconsistent",
            topo.boundary, topo.interior, topo.non_manifold, topo.inconsistent
        );
    }

    if result.verdict == Verdict::Refused {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
