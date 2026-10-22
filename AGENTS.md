# Pedroso-Mesher - My Mesh Generator

This is a clean-room, open-source, pure-Rust, quadrilateral and hexahedral mesh generator.
The code is implemented "from scratch" by reading public Journal papers and University Theses.
No code is ported from anywhere and non-MIT code is not even looked, ever.

## Non-negotiables
- **Independent.** Pedroso-Mesher is independent. Behavior is designed from public documentation.
  Never copy another product's code, icons, artwork, help text, tooltips, templates or
  materials. File formats come from public specifications.
- **Public knowledge only.** If a Journal paper or an University MSc or PhD thesis mention an MIT-licensed code,
  the algorithms implemented in such MIT-licensed code may be observed, for learning and better understanding
  the paper/thesis, but the code must not be copied and no code may be ported.
- **BSD-licensed code requires cpmech's authorization.** If a Journal paper or thesis mentions
  source code under BSD-licenses, cpmech (Dorival Pedroso) must be contacted first for authorization
  to take a peek at such codes.
- **No commercial products.** Agents must not drive, script, automate or extract data from
  commercial products.
- **Never read, copy or link GPL/LGPL/AGPL code**.
- **Never break wasm**: everything must build for `wasm32-unknown-unknown`

## Never crash
- **No panics in non-test code:** no `unwrap()`, `expect()`, `panic!`, `unreachable!`, `todo!`,
  `unimplemented!`; no `unsafe` (`unsafe_code = "forbid"`). The exceptions, in the app crate and in
  `crates/threedx-client` (which deny it instead), have no safe API: the macOS-only
  `crates/threedx-client/src/macos.rs` (loads the 3Dconnexion driver's client framework at run time
  and calls through its function pointers), two functions in
  `apps/solidcraft/src/open_documents.rs` (macOS hands Finder files only to an Objective-C
  method), the Windows-only `apps/solidcraft/src/native_crash.rs` (the exception filter that
  records crashes inside native code such as graphics drivers) and the Windows-only
  `apps/solidcraft/src/native_frame.rs` (the window procedure that answers the resize-border hit
  test of the window without OS decorations).
- **Errors are `Result<T, E>`** through the crate's error type and `?`. An unfinished feature
  returns an error ("not supported yet: …"); it never panics.
