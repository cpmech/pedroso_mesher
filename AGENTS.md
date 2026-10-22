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
  `unimplemented!`; no `unsafe` (`unsafe_code = "forbid"`).
- **Errors are `Result<T, E>`** through the crate's error type and `?`. An unfinished feature
  returns an error ("not supported yet: …"); it never panics.
