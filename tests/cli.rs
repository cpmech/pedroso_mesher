//! The CLI is the P0 contract made executable: `parts` lists the corpus,
//! `quality` refuses an inverted mesh with a non-zero exit code.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pedroso-mesher"))
        .args(args)
        .output()
        .expect("the binary starts")
}

fn text(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn parts_lists_the_twelve_part_corpus() {
    let out = run(&["parts"]);
    assert!(out.status.success());
    let body = text(&out);
    assert!(body.contains("plate-hole"));
    assert!(body.contains("sliver-feature"));
    assert!(body.contains("12 parts"));
}

#[test]
fn quality_accepts_a_unit_cube() {
    let out = run(&["quality", "examples/cube.json"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body = text(&out);
    assert!(body.contains("min SJ 1.0000"), "{body}");
    assert!(body.contains("acceptable"));
}

#[test]
fn quality_refuses_an_inverted_element_with_exit_code_1() {
    let out = run(&["quality", "tests/fixtures/inverted_hex.json"]);
    assert_eq!(out.status.code(), Some(1));
    let body = text(&out);
    assert!(body.contains("REFUSED"), "{body}");
    assert!(body.contains("inverted"));
}

#[test]
fn a_missing_file_is_a_usage_error() {
    let out = run(&["quality", "tests/fixtures/nope.json"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
}

#[test]
fn json_output_is_parseable() {
    let out = run(&["quality", "examples/cube.json", "--json"]);
    assert!(out.status.success());
    let body = text(&out);
    assert!(body.contains("\"verdict\": \"acceptable\""));
    assert!(body.contains("\"elements\": 1"));
}
